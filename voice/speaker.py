"""The audio `Speaker`: answers via Piper, alerts as chime + pre-rendered clip, with priorities.

PLAN.md "Rules for alerts": a chime before every alert; alerts play right away (pre-rendered); safety alerts
interrupt anything and an interrupted answer resumes after. Here every alert interrupts an answer (a "you're
up" that waits for a long answer to finish would miss the 0.3 s target); the interrupted sentence is said again
afterwards. `stop()` silences everything at once. Each utterance's latency (queued → first sound at the
speaker, device latency included) is recorded for the benchmarks.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from answers.speech import Utterance
from core.logging_setup import get_logger
from voice.audio import AudioOutput
from voice.tts import ClipCache, Tts, alert_chime

log = get_logger(__name__)


@dataclass(slots=True)
class _Item:
    utterance: Utterance
    queued_at: float
    audio: np.ndarray | None = None  # synthesized already (an interrupted answer keeps its audio)


@dataclass(frozen=True, slots=True)
class SpeechRecord:
    """One utterance as it was spoken."""

    kind: str
    text: str
    queued_at: float  # perf_counter when say() was called
    sound_at: float | None  # perf_counter when its first sample reached the speaker (None: never played)
    interrupted: bool = False

    @property
    def latency_ms(self) -> float | None:
        """From say() to sound."""
        return None if self.sound_at is None else (self.sound_at - self.queued_at) * 1000.0


@dataclass
class _State:
    alerts: deque[_Item] = field(default_factory=deque)
    answers: deque[_Item] = field(default_factory=deque)
    playing: _Item | None = None
    interrupted: bool = False
    closed: bool = False
    generation: int = 0  # bumped by stop(): items from before it are dropped


class AudioSpeaker:
    """Speaks utterances on an `AudioOutput` (implements `answers.speech.Speaker`)."""

    def __init__(self, tts: Tts, output: AudioOutput, clips: ClipCache, chime_volume: float = 0.25) -> None:
        self.tts = tts
        self.output = output
        self.clips = clips
        self._chime = alert_chime(tts.rate, chime_volume)
        self._gap = np.zeros(round(0.04 * tts.rate), np.float32)
        self._cond = threading.Condition()
        self._state = _State()
        self.records: deque[SpeechRecord] = deque(maxlen=500)
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)

    def start(self) -> AudioSpeaker:
        """Start the playback thread."""
        self._thread.start()
        return self

    def say(self, utterance: Utterance) -> None:
        """Queue an utterance; an alert interrupts an answer that is playing."""
        item = _Item(utterance, time.perf_counter())
        with self._cond:
            state = self._state
            if utterance.kind == "alert":
                state.alerts.append(item)
                if state.playing is not None and state.playing.utterance.kind != "alert":
                    state.interrupted = True
                    self.output.stop()
            else:
                state.answers.append(item)
            self._cond.notify()

    def stop(self) -> None:
        """Stop speaking now and drop everything queued."""
        with self._cond:
            self._state.alerts.clear()
            self._state.answers.clear()
            self._state.interrupted = False
            self._state.generation += 1
            if self._state.playing is not None:
                self.output.stop()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until nothing is queued or playing (tests, benchmarks)."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._state.alerts or self._state.answers or self._state.playing is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self) -> None:
        """Stop the thread."""
        with self._cond:
            self._state.closed = True
            self._cond.notify_all()
        self.output.stop()
        self._thread.join(timeout=2.0)

    # ---------------------------------------------------------------------------------- playback

    def _run(self) -> None:
        while True:
            with self._cond:
                state = self._state
                while not state.alerts and not state.answers and not state.closed:
                    self._cond.wait()
                if state.closed:
                    return
                item = state.alerts.popleft() if state.alerts else state.answers.popleft()
                state.playing, state.interrupted = item, False
                generation = state.generation
            try:
                audio = self._render(item)
            except Exception:
                log.exception("speech synthesis failed", text=item.utterance.text)
                audio = None
            with self._cond:
                if self._state.generation != generation:  # stop() while we were synthesizing
                    self._finish(item, None, interrupted=True)
                    continue
                if item.utterance.kind != "alert" and self._state.alerts:
                    item.audio = audio  # an alert arrived during synthesis: it goes first, this answer right after
                    self._state.answers.appendleft(item)
                    self._state.playing = None
                    self._cond.notify_all()
                    continue
            sound_at = self.output.play(audio, self.tts.rate) if audio is not None and len(audio) else None
            with self._cond:
                stopped = self._state.generation != generation
                interrupted = stopped or self._state.interrupted
                if interrupted and not stopped and item.utterance.kind != "alert":
                    self._state.answers.appendleft(_Item(item.utterance, item.queued_at, item.audio))
                self._finish(item, sound_at, interrupted)

    def _render(self, item: _Item) -> np.ndarray:
        if item.audio is not None:
            return item.audio
        utterance = item.utterance
        if utterance.kind == "alert":
            audio = np.concatenate((self._chime, self._gap, self.clips.get(utterance.text)))
        else:
            audio = self.tts.synthesize(utterance.text)
        item.audio = audio
        return audio

    def _finish(self, item: _Item, sound_at: float | None, interrupted: bool) -> None:
        """Record the utterance (caller holds the lock)."""
        record = SpeechRecord(item.utterance.kind, item.utterance.text, item.queued_at, sound_at, interrupted)
        self.records.append(record)
        if self._state.playing is item:
            self._state.playing = None
        self._cond.notify_all()
        log.info(
            "spoke",
            kind=record.kind,
            latency_ms=None if record.latency_ms is None else round(record.latency_ms),
            interrupted=interrupted,
            text=record.text,
        )
