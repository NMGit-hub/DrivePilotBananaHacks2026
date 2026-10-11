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

Without the full wake word (`voice.wake_mode` "hey" or "none", switchable while running with `set_mode`), speech
itself starts a question: a second VAD watches the idle microphone, and `open_mic.onset_s` of speech starts the
endpointer (with a little pre-roll, no wake-phrase tail to wait out). Whisper's transcript then decides whether it
was meant for us: "hey" mode needs it to start with "hey" (a lone "hey" opens the listening window like the wake
word, and the question can follow), "none" mode takes questions and requests (`router.meant_for_us`). Anything
else is dropped silently (a passenger talking). While DrivePilot speaks, and for `echo_tail_s` after, speech
doesn't start a question (its own voice comes back through the microphone). The "Hey DrivePilot" model keeps
running in every mode, so "Hey DrivePilot, stop" still interrupts an answer.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from core.config import WAKE_MODES, VoiceConfig
from core.logging_setup import get_logger
from voice.audio import CHUNK, SAMPLE_RATE, MicStream
from voice.endpoint import VAD_CHUNK, Endpointer, EndpointEvent, StreamingVad
from voice.router import meant_for_us, strip_hey
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
    reply: bool = False  # heard in a reply window (no wake word): only a plain yes/no is taken (answers/parking.py)
    open: bool = False  # started by speech, not the "Hey DrivePilot" model (`voice.wake_mode` "hey" / "none")


@dataclass(slots=True)
class _Held:
    """Audio since a question ended, kept until its transcript is known (it may have been only the wake phrase)."""

    since: float  # stream time the question ended
    # (chunk, stream time at its end, perf_counter when it arrived: an end found on replay is dated by its audio)
    chunks: list[tuple[np.ndarray, float, float]] = field(default_factory=list)
    state: str = "waiting"  # "waiting" | "resume" (the transcript was empty) | "superseded" (a new wake word)
    open: bool = False  # the question was started by speech (a lone "hey" so far)


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
        speaking: Callable[[], bool] | None = None,
        onset_vad: StreamingVad | None = None,
    ) -> None:
        self.cfg = cfg
        self.wake = wake
        self.stt = stt
        self.on_question = on_question
        self.on_wake = on_wake
        self.on_nothing = on_nothing  # the wake word fired but no question came: "timeout" | "empty"
        self.answer_in_thread = answer_in_thread
        self.speaking = speaking  # True while DrivePilot talks: its voice must not start a question (open modes)
        self._endpointer = Endpointer(cfg.endpoint, vad or StreamingVad())
        ring_s = max(cfg.endpoint.pre_roll_s, cfg.open_mic.onset_s + cfg.open_mic.pre_roll_s)
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, round(ring_s * SAMPLE_RATE / CHUNK) + 1))
        self._mode: str = cfg.wake_mode
        self._onset_vad = onset_vad  # created on first use (only the open modes need it)
        self._onset_run = 0  # VAD frames of speech in a row while idle
        self._last_spoke_t = -1e9  # stream time DrivePilot was last heard talking
        self._open = False  # the question being collected was started by speech, not the wake word
        self.open_starts = 0
        self.ignored = 0  # open-mode transcripts that weren't meant for us
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
        self._reply_request: float | None = None  # seconds to listen without the wake word (any thread sets it)
        self._replying = False  # the question being collected is a reply (no wake word)
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

    @property
    def mode(self) -> str:
        """The wake mode: "hey_drivepilot", "hey" or "none" (`voice.wake_mode`)."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch the wake mode while running (any thread; applies from the next chunk)."""
        if mode not in WAKE_MODES:
            raise ValueError(f"unknown wake mode {mode!r} (one of {', '.join(WAKE_MODES)})")
        self._mode = mode
        log.info("wake mode", mode=mode)

    def expect_reply(self, seconds: float) -> None:
        """Listen for a short reply without the wake word, for `seconds` (DrivePilot just asked a yes/no question:
        the parking-lot offer). Any thread; starts with the next chunk, unless a question is being collected."""
        with self._lock:
            self._reply_request = seconds

    def feed(self, chunk: np.ndarray) -> None:
        """Process one 80 ms int16 chunk at 16 kHz."""
        arrived = time.perf_counter()
        self._samples += len(chunk)
        t = self.t
        with self._lock:
            reply_s, self._reply_request = self._reply_request, None
        if reply_s is not None and not self._endpointer.active:
            self._listen_for_reply(t, reply_s)
        fired = self.wake.process(chunk, t)
        # Before the pre-roll is taken: the chunk the wake word fired in belongs to the question's audio (missing
        # it left an 80 ms hole right after "Hey DrivePilot" that garbled the first word, on real voices).
        self._ring.append(chunk)
        if fired is not None and not self._endpointer.active:
            self._woke(t)
            return
        if fired is not None:
            self._rewoke(t)
        if self._watch_onset(chunk, t):
            self._open_start(t)
            return
        self._take(chunk, t, arrived)

    def _watch_onset(self, chunk: np.ndarray, t: float) -> bool:
        """Open modes, while idle: has speech started (`open_mic.onset_s` of it in a row)? Never while DrivePilot is
        talking or just after (its own voice), or while a transcript that may resume a question is pending."""
        if self._mode == "hey_drivepilot" or self._endpointer.active:
            self._onset_run = 0
            return False
        with self._lock:
            pending = self._held is not None
        if pending:
            self._onset_run = 0
            return False
        try:
            talking = self.speaking is not None and self.speaking()
        except Exception:
            log.exception("speaking check failed")
            talking = False
        if talking:
            self._last_spoke_t = t
        if self._onset_vad is None:
            self._onset_vad = StreamingVad()
        probs = self._onset_vad.process(chunk)  # (fed while DrivePilot talks too: its state follows the audio)
        if talking or t - self._last_spoke_t < self.cfg.open_mic.echo_tail_s:
            self._onset_run = 0
            return False
        for prob in probs:
            self._onset_run = self._onset_run + 1 if prob >= self.cfg.endpoint.speech_threshold else 0
        return self._onset_run * VAD_CHUNK / SAMPLE_RATE >= self.cfg.open_mic.onset_s

    def _open_start(self, t: float) -> None:
        """Speech started without the wake word: collect it as a question (whether it was meant for us is decided
        on its transcript). No `on_wake`: no blip and no interruption for every word said in the car."""
        self.open_starts += 1
        self._onset_run = 0
        self._wake_t = self._window_t = t
        self._resumed = self._replying = False
        self._open = True
        keep = round((self.cfg.open_mic.onset_s + self.cfg.open_mic.pre_roll_s) * SAMPLE_RATE)
        pre_roll = np.concatenate(self._ring)[-keep:] if self._ring else np.zeros(0, np.int16)
        self._endpointer.start_open(pre_roll, t)
        self._speculation = None
        log.info("speech without the wake word, listening", t=round(t, 2), mode=self._mode)

    def _woke(self, t: float) -> None:
        with self._lock:
            if self._held is not None:  # a new wake word replaces a question that may have been only the wake phrase
                self._held.state = "superseded"
                self._held = None
        self.wakes += 1
        self._wake_t = self._window_t = t
        self._resumed = self._replying = self._open = False
        pre_roll = (
            np.concatenate(self._ring)[-round(self.cfg.endpoint.pre_roll_s * SAMPLE_RATE) :]
            if self._ring
            else np.zeros(0, np.int16)
        )
        self._endpointer.start(pre_roll, t)
        self._speculation = None
        self._call_on_wake()

    def _call_on_wake(self) -> None:
        if self.on_wake is not None:
            try:
                self.on_wake()
            except Exception:
                log.exception("on_wake failed")

    def _listen_for_reply(self, t: float, seconds: float) -> None:
        with self._lock:
            if self._held is not None:  # (a pending wake-phrase-only question is dropped)
                self._held.state = "superseded"
                self._held = None
        self._wake_t = self._window_t = t
        self._resumed, self._replying, self._open = False, True, False
        self._speculation = None
        self._endpointer.resume(t, t, timeout_s=seconds)
        log.info("listening for a reply", seconds=seconds)

    def _rewoke(self, t: float) -> None:
        """The wake word fired again while a question is being collected: the driver repeated it (maybe after an
        early wake). It doesn't start a new question: a wake at the end of one ("is it safe, DrivePilot?") would
        lose it, and the app would count the question as superseded. The listening window restarts from it."""
        log.info("wake word again while listening", t=round(t, 2))
        self._window_t = t
        self._replying = False  # "Hey DrivePilot, ..." during a reply window is a question, not a reply
        self._endpointer.rewoke(t)
        if self._open:  # speech started a question, and it was "Hey DrivePilot": a question like any other
            self._open = False
            self.wakes += 1
            self._call_on_wake()

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
        self._open = held.open
        log.info("only the wake phrase so far, still listening", t=round(held.since, 2))
        self._endpointer.resume(held.since, self._window_t)
        for chunk, t, arrived in held.chunks:  # catches up at once; chunks after a new end are held again
            self._take(chunk, t, arrived)

    def _transcribe(self, audio: np.ndarray) -> Future[tuple[str, float]]:
        if self._resumed or self._replying or self._open:  # no wake phrase at the start: only a repeat is stripped
            return self._stt_pool.submit(self.stt.transcribe, audio, pre_roll=False)
        return self._stt_pool.submit(self.stt.transcribe, audio)

    def _handle(self, event: EndpointEvent, t: float, arrived: float) -> None:
        if event.kind == "speculate":
            self._speculation = self._transcribe(event.audio)
            self._speculated_t = t
            return
        if event.kind == "timeout":
            if self._replying:
                self._replying = False
                log.info("no reply")
                self._nothing("no reply")
                return
            if self._open:  # a lone "hey" and then nothing (or a VAD blip): nobody asked us anything
                self._open = False
                log.info("nothing more said", t=round(t, 2))
                return
            self.timeouts += 1
            log.info("no question after the wake word", t=round(t, 2))
            self._nothing("timeout")
            return
        end_detected_at = arrived  # (replayed audio: when it arrived, not when the replay reached it)
        speculative = self._speculation is not None and self._endpointer.speculation_still_valid()
        future = self._speculation if speculative and self._speculation is not None else self._transcribe(event.audio)
        self._speculation = None
        wake_t, speech_end_t, resumed, reply = self._wake_t, event.speech_end_t, self._resumed, self._replying
        open_ = self._open
        self._replying = self._open = False
        speculated_t = self._speculated_t if speculative else None
        # Until the transcript is known, keep what comes next: if it was only the wake phrase (or a lone "hey"), the
        # question follows. (No wake phrase in "none" mode: nothing to resume.)
        held = (
            _Held(t, open=open_)
            if not reply
            and not (open_ and (resumed or self._mode == "none"))
            and t - self._window_t < self.cfg.endpoint.no_speech_timeout_s
            else None
        )
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
            called = False  # a lone "hey": we were called, the question comes next
            if open_ and text:
                addressed = self._addressed(text, resumed)
                if addressed is None:
                    release()
                    self.ignored += 1
                    log.info("not for us", text=text, mode=self._mode)
                    return
                called, text = not addressed, addressed
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
                reply,
                open_,
            )
            log.info("heard", text=text, stt_ms=round(stt_ms), speculative=speculative, reason=event.reason, open=open_)
            if not text:
                with self._lock:
                    resume = superseded = False
                    if held is not None:
                        resume = self._held is held
                        if resume:
                            held.state = "resume"  # picked up with the next chunk (or by `flush`)
                        superseded = held.state == "superseded"
                if reply:
                    self._nothing("no reply")
                    return
                if resume or superseded:
                    if called and resume:
                        self._call_on_wake()  # "hey": the blip and the listening light, as for the wake word
                    return  # still listening for this question, or a new wake word took over
                if open_:
                    return  # noise or silence, and nobody called us: nothing to say
                self.empty_transcripts += 1
                self._nothing("empty")
                return
            release()
            if self.answer_in_thread:
                threading.Thread(target=self._answer, args=(heard,), name="answer", daemon=True).start()
            else:
                self._answer(heard)

        future.add_done_callback(done)

    def _addressed(self, text: str, resumed: bool) -> str | None:
        """An open-mode transcript: the question meant for us, "" for a lone "hey", or None if it wasn't for us."""
        if resumed:
            return text  # the question after a lone "hey"
        if self._mode == "hey":
            return strip_hey(text)
        if self._mode == "none":
            return text if not self.cfg.open_mic.only_questions or meant_for_us(text) else None
        return None  # switched back to the wake word meanwhile

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
