"""The voice pipeline: microphone → wake word → end of question (VAD) → Whisper → the assistant.

One thread feeds 80 ms chunks (`feed`, also used directly by tests and the synthetic benchmark):
- the wake word model sees every chunk (also while a question is being collected, so its audio buffer has no
  gaps); a detection starts the endpointer with ~0.4 s of pre-roll and calls `on_wake` (the app stops any
  speech, so "Hey DrivePilot, stop" works mid-answer, and plays a soft "listening" blip);
- the endpointer decides when the question is over; at ~0.25 s of silence it asks for a *speculative*
  transcription, which is used if the silence lasts to the end (~0.6 s), saving most of Whisper's time;
- transcription runs on one worker thread; each transcript is answered on its own thread (`on_question`), so a
  long streamed answer never stops the pipeline from hearing "Hey DrivePilot, stop".
Timing for every question (wake, end of speech, end detected, transcript ready) goes with it, for latency
measurements against PLAN.md's targets.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

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
    end_detected_at: float  # perf_counter at that moment
    transcript_at: float  # perf_counter when the transcript was ready
    stt_ms: float
    speculative: bool  # the speculative transcript was used
    reason: str  # "silence" | "maximum length"
    speculated_t: float | None = None  # stream time the (used) speculative transcription started


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
        self._wake_t = 0.0
        self._samples = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._speculated_t: float | None = None
        self.wakes = 0
        self.timeouts = 0
        self.empty_transcripts = 0

    # ---------------------------------------------------------------------------------- feeding

    @property
    def t(self) -> float:
        """Stream time (s) of the audio fed so far."""
        return self._samples / SAMPLE_RATE

    def feed(self, chunk: np.ndarray) -> None:
        """Process one 80 ms int16 chunk at 16 kHz."""
        self._samples += len(chunk)
        t = self.t
        fired = self.wake.process(chunk, t)
        if self._endpointer.active:
            event = self._endpointer.feed(chunk, t)
            if event is not None:
                self._handle(event, t)
        elif fired is not None:
            self.wakes += 1
            self._wake_t = t
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
        self._ring.append(chunk)

    def _handle(self, event: EndpointEvent, t: float) -> None:
        if event.kind == "speculate":
            self._speculation = self._stt_pool.submit(self.stt.transcribe, event.audio)
            self._speculated_t = t
            return
        if event.kind == "timeout":
            self.timeouts += 1
            log.info("no question after the wake word", t=round(t, 2))
            self._nothing("timeout")
            return
        end_detected_at = time.perf_counter()
        speculative = self._speculation is not None and self._endpointer.speculation_still_valid()
        future = (
            self._speculation
            if speculative and self._speculation is not None
            else self._stt_pool.submit(self.stt.transcribe, event.audio)
        )
        self._speculation = None
        wake_t, speech_end_t = self._wake_t, event.speech_end_t
        speculated_t = self._speculated_t if speculative else None

        def done(f: Future[tuple[str, float]]) -> None:
            try:
                text, stt_ms = f.result()
            except Exception:
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
            )
            log.info("heard", text=text, stt_ms=round(stt_ms), speculative=speculative, reason=event.reason)
            if not text:
                self.empty_transcripts += 1
                self._nothing("empty")
                return
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
        """Wait for pending transcriptions (tests, benchmarks)."""
        self._stt_pool.submit(lambda: None).result(timeout=timeout)

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
