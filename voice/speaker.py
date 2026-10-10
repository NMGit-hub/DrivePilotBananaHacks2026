"""The audio `Speaker`: answers via Piper, alerts as chime + pre-rendered clip, with priorities.

PLAN.md "Rules for alerts": a chime before every alert; alerts play right away (pre-rendered); safety alerts
interrupt anything and an interrupted answer resumes after. Here every alert interrupts an answer (a "you're
up" that waits for a long answer to finish would miss the 0.3 s target); the interrupted sentence is said again
afterwards. `stop()` silences everything at once; `listening()` (wake word heard) does too, then plays a
short blip. Alerts whose text was pre-rendered (`ClipCache.prepare`) play from memory; others (navigation,
whose distances vary) are synthesized when said. Each utterance's latency (queued → first sound at the
speaker, device latency included) is recorded for the benchmarks and passed to `on_record`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from answers.speech import Priority, Utterance
from core.logging_setup import get_logger
from voice.audio import AudioOutput
from voice.tts import ClipCache, Tts, alert_chime, listen_tone

log = get_logger(__name__)


@dataclass(slots=True)
class _Item:
    utterance: Utterance
    queued_at: float
    audio: np.ndarray | None = None  # synthesized already (an interrupted answer keeps its audio)
    epoch: int = 0  # answers from before an alert that made them stale are not resumed


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
    answer_epoch: int = 0  # bumped by an alert with drops_answers: answers from before it are not resumed


class AudioSpeaker:
    """Speaks utterances on an `AudioOutput` (implements `answers.speech.Speaker`)."""

    def __init__(
        self,
        tts: Tts,
        output: AudioOutput,
        clips: ClipCache,
        chime_volume: float = 0.25,
        listen_chime: bool = True,
        on_record: Callable[[SpeechRecord], None] | None = None,
    ) -> None:
        self.tts = tts
        self.output = output
        self.clips = clips
        self.on_record = on_record
        self._chime = alert_chime(tts.rate, chime_volume)
        self._listen = listen_tone(tts.rate, chime_volume) if listen_chime else None
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
                if utterance.drops_answers:
                    state.answer_epoch += 1
                    state.answers.clear()
                state.alerts.append(item)
                if state.playing is not None and state.playing.utterance.kind != "alert":
                    state.interrupted = True
                    self.output.stop()
            else:
                item.epoch = state.answer_epoch
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

    def listening(self) -> None:
        """The wake word was heard: stop speaking (so "Hey DrivePilot, stop" works mid-answer) and blip."""
        self.stop()
        if self._listen is None:
            return
        cue = Utterance(text="", kind="cue", priority=Priority.SAFETY)
        with self._cond:
            self._state.alerts.append(_Item(cue, time.perf_counter(), self._listen))
            self._cond.notify()

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
            dropped: SpeechRecord | None = None
            with self._cond:
                if self._state.generation != generation:  # stop() while we were synthesizing
                    dropped = self._finish(item, None, interrupted=True)
                elif item.utterance.kind == "answer" and self._state.alerts:
                    item.audio = audio  # an alert arrived during synthesis: it goes first, this answer right after
                    if item.epoch == self._state.answer_epoch:
                        self._state.answers.appendleft(item)
                        self._state.playing = None
                        self._cond.notify_all()
                        continue
                    dropped = self._finish(item, None, interrupted=True)  # the alert made it stale
            if dropped is not None:
                self._report(dropped)
                continue
            sound_at = self.output.play(audio, self.tts.rate) if audio is not None and len(audio) else None
            with self._cond:
                stopped = self._state.generation != generation
                interrupted = stopped or self._state.interrupted
                fresh = item.epoch == self._state.answer_epoch
                if interrupted and not stopped and item.utterance.kind == "answer" and fresh:
                    self._state.answers.appendleft(_Item(item.utterance, item.queued_at, item.audio, item.epoch))
                record = self._finish(item, sound_at, interrupted)
            self._report(record)

    def _render(self, item: _Item) -> np.ndarray:
        if item.audio is not None:
            return item.audio
        utterance = item.utterance
        if utterance.kind == "alert":
            clip = self.clips.cached(utterance.text)
            speech = clip if clip is not None else self.tts.synthesize(utterance.text)
            audio = np.concatenate((self._chime, self._gap, speech))
        else:
            audio = self.tts.synthesize(utterance.text)
        item.audio = audio
        return audio

    def _finish(self, item: _Item, sound_at: float | None, interrupted: bool) -> SpeechRecord:
        """Record the utterance (caller holds the lock)."""
        record = SpeechRecord(item.utterance.kind, item.utterance.text, item.queued_at, sound_at, interrupted)
        self.records.append(record)
        if self._state.playing is item:
            self._state.playing = None
        self._cond.notify_all()
        if record.kind == "cue":
            return record
        log.info(
            "spoke",
            kind=record.kind,
            latency_ms=None if record.latency_ms is None else round(record.latency_ms),
            interrupted=interrupted,
            text=record.text,
        )
        return record

    def _report(self, record: SpeechRecord) -> None:
        """Hand a finished record to `on_record` (outside the lock)."""
        if self.on_record is not None:
            try:
                self.on_record(record)
            except Exception:
                log.exception("on_record failed")
