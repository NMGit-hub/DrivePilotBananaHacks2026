"""The voice pipeline: microphone → wake word → end of question (VAD) → Whisper → the assistant.

One thread feeds 80 ms chunks (`feed`, also used directly by tests and the synthetic benchmark):
- the wake word model sees every chunk (also while a question is being collected, so its audio buffer has no
  gaps); a detection starts the endpointer with ~0.4 s of pre-roll and calls `on_wake` (the app stops any
  speech, so "Hey DrivePilot, stop" works mid-answer, and plays a soft "listening" blip);
- the endpointer decides when the question is over; at ~0.25 s of silence it asks for a *speculative*
  transcription, which is used if the silence lasts to the end (~0.6 s), saving most of Whisper's time;
- transcription runs on one worker thread; each transcript is answered on its own thread (`on_question`), so a
  long streamed answer never stops the pipeline from hearing "Hey DrivePilot, stop";
- if the wake word fired early (inside "...pilot"), the rest of the wake phrase can pass for the question and the
  pause after it for its end: the transcript is then empty (the wake phrase is stripped). Audio is held from the
  end of every question until its transcript is known; an empty one resumes listening on that audio, so the
  question said after the pause is still heard (within `no_speech_timeout_s` of the wake word, or of a repeat of
  it: a wake word during a question restarts that window but not the question).
Timing for every question (wake, end of speech, end detected, transcript ready) goes with it, for latency
measurements against PLAN.md's targets.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from core.config import VoiceConfig
from core.logging_setup import get_logger
from voice.audio import CHUNK, SAMPLE_RATE, MicStream
from voice.endpoint import Endpointer, EndpointEvent, StreamingVad
from voice.stt import Transcriber
from voice.wakeword import WakeWordDetector

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HeardQuestion:
    """A transcribed question with its timing."""

    text: str
    wake_t: float  # stream time the wake word fired (s)
    speech_end_t: float | None  # stream time the speaker stopped talking
    end_t: float  # stream time the end of the question was detected
    end_detected_at: float  # perf_counter when the audio that ended the question arrived
    transcript_at: float  # perf_counter when the transcript was ready
    stt_ms: float
    speculative: bool  # the speculative transcript was used
    reason: str  # "silence" | "maximum length"
    speculated_t: float | None = None  # stream time the (used) speculative transcription started
    resumed: bool = False  # heard after a first end that turned out to be only the wake phrase


@dataclass(slots=True)
class _Held:
    """Audio since a question ended, kept until its transcript is known (it may have been only the wake phrase)."""

    since: float  # stream time the question ended
    # (chunk, stream time at its end, perf_counter when it arrived: an end found on replay is dated by its audio)
    chunks: list[tuple[np.ndarray, float, float]] = field(default_factory=list)
    state: str = "waiting"  # "waiting" | "resume" (the transcript was empty) | "superseded" (a new wake word)


class VoicePipeline:
    """Wake word → question → transcript → `on_question`."""

    def __init__(
        self,
        cfg: VoiceConfig,
        wake: WakeWordDetector,
        stt: Transcriber,
        on_question: Callable[[HeardQuestion], None],
        on_wake: Callable[[], None] | None = None,
        vad: StreamingVad | None = None,
        answer_in_thread: bool = True,
        on_nothing: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.wake = wake
        self.stt = stt
        self.on_question = on_question
        self.on_wake = on_wake
        self.on_nothing = on_nothing  # the wake word fired but no question came: "timeout" | "empty"
        self.answer_in_thread = answer_in_thread
        self._endpointer = Endpointer(cfg.endpoint, vad or StreamingVad())
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, round(cfg.endpoint.pre_roll_s * SAMPLE_RATE / CHUNK) + 1))
        self._stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._speculation: Future[tuple[str, float]] | None = None
        self._wake_t = 0.0  # stream time of the wake word that started the question
        self._window_t = 0.0  # ...or of a repeat of it: the listening window counts from the latest one
        self._samples = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._speculated_t: float | None = None
        self._resumed = False  # the question being collected was resumed after a wake-phrase-only transcript
        self._lock = threading.Lock()  # `_held` is shared with the transcription thread
        self._held: _Held | None = None
        self.wakes = 0
        self.timeouts = 0
        self.empty_transcripts = 0
        self.resumes = 0

    # ---------------------------------------------------------------------------------- feeding

    @property
    def t(self) -> float:
        """Stream time (s) of the audio fed so far."""
        return self._samples / SAMPLE_RATE

    @property
    def listening(self) -> bool:
        """Still waiting for (the rest of) a question: benchmarks feed silence until this is False, so the next
        clip starts with the pipeline idle."""
        with self._lock:
            resuming = self._held is not None and self._held.state == "resume"
        return self._endpointer.active or resuming

    def feed(self, chunk: np.ndarray) -> None:
        """Process one 80 ms int16 chunk at 16 kHz."""
        arrived = time.perf_counter()
        self._samples += len(chunk)
        t = self.t
        fired = self.wake.process(chunk, t)
        # Before the pre-roll is taken: the chunk the wake word fired in belongs to the question's audio (missing
        # it left an 80 ms hole right after "Hey DrivePilot" that garbled the first word, on real voices).
        self._ring.append(chunk)
        if fired is not None and not self._endpointer.active:
            self._woke(t)
            return
        if fired is not None:
            self._rewoke(t)
        self._take(chunk, t, arrived)

    def _woke(self, t: float) -> None:
        with self._lock:
            if self._held is not None:  # a new wake word replaces a question that may have been only the wake phrase
                self._held.state = "superseded"
                self._held = None
        self.wakes += 1
        self._wake_t = self._window_t = t
        self._resumed = False
        pre_roll = (
            np.concatenate(self._ring)[-round(self.cfg.endpoint.pre_roll_s * SAMPLE_RATE) :]
            if self._ring
            else np.zeros(0, np.int16)
        )
        self._endpointer.start(pre_roll, t)
        self._speculation = None
        if self.on_wake is not None:
            try:
                self.on_wake()
            except Exception:
                log.exception("on_wake failed")

    def _rewoke(self, t: float) -> None:
        """The wake word fired again while a question is being collected: the driver repeated it (maybe after an
        early wake). It doesn't start a new question: a wake at the end of one ("is it safe, DrivePilot?") would
        lose it, and the app would count the question as superseded. The listening window restarts from it."""
        log.info("wake word again while listening", t=round(t, 2))
        self._window_t = t
        self._endpointer.rewoke(t)

    def _take(self, chunk: np.ndarray, t: float, arrived: float) -> None:
        """Give a chunk to the question being collected, or hold it while the last transcript is pending."""
        if self._endpointer.active:
            event = self._endpointer.feed(chunk, t)
            if event is not None:
                self._handle(event, t, arrived)
            return
        with self._lock:
            held = self._held
            if held is None:
                return
            held.chunks.append((chunk, t, arrived))
            if held.state != "resume":
                if t - held.since > self.cfg.endpoint.max_utterance_s:  # the transcript never came: stop holding
                    self._held = None
                return
            self._held = None
        self._resume(held)

    def _resume(self, held: _Held) -> None:
        """The question so far was only the wake phrase: listen again, from the audio held since it ended."""
        self.resumes += 1
        self._resumed = True
        log.info("only the wake phrase so far, still listening", t=round(held.since, 2))
        self._endpointer.resume(held.since, self._window_t)
        for chunk, t, arrived in held.chunks:  # catches up at once; chunks after a new end are held again
            self._take(chunk, t, arrived)

    def _transcribe(self, audio: np.ndarray) -> Future[tuple[str, float]]:
        if self._resumed:  # the audio starts after the wake phrase: only a repeat of it is stripped
            return self._stt_pool.submit(self.stt.transcribe, audio, pre_roll=False)
        return self._stt_pool.submit(self.stt.transcribe, audio)

    def _handle(self, event: EndpointEvent, t: float, arrived: float) -> None:
        if event.kind == "speculate":
            self._speculation = self._transcribe(event.audio)
            self._speculated_t = t
            return
        if event.kind == "timeout":
            self.timeouts += 1
            log.info("no question after the wake word", t=round(t, 2))
            self._nothing("timeout")
            return
        end_detected_at = arrived  # (replayed audio: when it arrived, not when the replay reached it)
        speculative = self._speculation is not None and self._endpointer.speculation_still_valid()
        future = self._speculation if speculative and self._speculation is not None else self._transcribe(event.audio)
        self._speculation = None
        wake_t, speech_end_t, resumed = self._wake_t, event.speech_end_t, self._resumed
        speculated_t = self._speculated_t if speculative else None
        # Until the transcript is known, keep what comes next: if it was only the wake phrase, the question follows.
        held = _Held(t) if t - self._window_t < self.cfg.endpoint.no_speech_timeout_s else None
        with self._lock:
            self._held = held

        def release() -> None:
            with self._lock:
                if held is not None and self._held is held:
                    self._held = None

        def done(f: Future[tuple[str, float]]) -> None:
            try:
                text, stt_ms = f.result()
            except Exception:
                release()
                log.exception("transcription failed")
                return
            heard = HeardQuestion(
                text,
                wake_t,
                speech_end_t,
                t,
                end_detected_at,
                time.perf_counter(),
                stt_ms,
                speculative,
                event.reason,
                speculated_t,
                resumed,
            )
            log.info("heard", text=text, stt_ms=round(stt_ms), speculative=speculative, reason=event.reason)
            if not text:
                with self._lock:
                    resume = superseded = False
                    if held is not None:
                        resume = self._held is held
                        if resume:
                            held.state = "resume"  # picked up with the next chunk (or by `flush`)
                        superseded = held.state == "superseded"
                if resume or superseded:
                    return  # still listening for this question, or a new wake word took over
                self.empty_transcripts += 1
                self._nothing("empty")
                return
            release()
            if self.answer_in_thread:
                threading.Thread(target=self._answer, args=(heard,), name="answer", daemon=True).start()
            else:
                self._answer(heard)

        future.add_done_callback(done)

    def _nothing(self, reason: str) -> None:
        if self.on_nothing is not None:
            try:
                self.on_nothing(reason)
            except Exception:
                log.exception("on_nothing failed")

    def _answer(self, heard: HeardQuestion) -> None:
        try:
            self.on_question(heard)
        except Exception:
            log.exception("answering failed", text=heard.text)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for pending transcriptions (tests, benchmarks). Call it on the thread that feeds: a question that was
        only the wake phrase so far is resumed on the audio already fed, as the next chunk would have done."""
        deadline = time.perf_counter() + timeout
        while True:
            self._stt_pool.submit(lambda: None).result(timeout=max(0.0, deadline - time.perf_counter()))
            with self._lock:
                held = self._held
                if held is None or held.state != "resume":
                    return
                self._held = None
            self._resume(held)

    # -------------------------------------------------------------------------------- live audio

    def start(self, mic: MicStream) -> None:
        """Run on the microphone in a background thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(mic,), name="voice", daemon=True)
        self._thread.start()

    def _run(self, mic: MicStream) -> None:
        while not self._stop.is_set():
            chunk = mic.read(timeout=0.5)
            if chunk is not None:
                try:
                    self.feed(chunk)
                except Exception:
                    log.exception("voice pipeline error")

    def stop(self) -> None:
        """Stop the background thread and the transcription worker."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stt_pool.shutdown(wait=False, cancel_futures=True)
