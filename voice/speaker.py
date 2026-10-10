"""The audio `Speaker`: answers via Piper, alerts as chime + pre-rendered clip, with priorities.

PLAN.md "Rules for alerts": a chime before every alert; alerts play right away (pre-rendered); safety alerts
interrupt anything and an interrupted answer resumes after. Here:
- Two threads. A synthesis thread renders queued utterances ahead of time (alerts first; Piper is only ever used
  from this thread), and the playback thread plays them: alerts first, then answers in order. So a pre-rendered
  alert starts at once even while a long answer sentence is still being synthesized.
- Every alert interrupts an answer that is playing (a "you're up" that waits for a long answer would miss the
  0.3 s target); the interrupted sentence is said again afterwards, unless the alert made it stale
  (`Utterance.drops_answers`: "You're up", "Light's green", route changes), in which case queued answers are
  dropped too.
- A SAFETY alert (a pedestrian in our path) also goes ahead of queued alerts that aren't, and cuts off one that is
  playing (a navigation sentence takes seconds); that alert is said again right after it.
- `stop()` silences everything ("stop"); `stop_answers()` stops only answers (the wake word, "repeat that"), so
  a false wake can't lose an alert; `listening()` is `stop_answers()` plus a short blip.
- A stop or alert that lands just before a sound would start still prevents it: the start is checked under the
  output's lock.
- Every utterance ends in a `SpeechRecord` (played, interrupted, or dropped with no sound), passed to
  `on_record` with the utterance's id, for latency measurement. Alert clips from `ClipCache.prepare` play from
  memory; other alert texts (navigation, with varying distances) are synthesized when said.
"""

from __future__ import annotations

import itertools
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
    audio: np.ndarray | None = None  # rendered (an interrupted answer keeps its audio)
    epoch: int = 0  # answers from before an alert that made them stale are not resumed
    rendering: bool = False


@dataclass(frozen=True, slots=True)
class SpeechRecord:
    """One utterance as it was spoken (or dropped)."""

    kind: str
    text: str
    queued_at: float  # perf_counter when say() was called
    sound_at: float | None  # perf_counter when its first sample reached the speaker (None: never played)
    interrupted: bool = False
    uid: int = 0  # the Utterance's id

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
    generation: int = 0  # bumped by stop(): what was playing then must not start or resume
    answer_epoch: int = 0  # bumped when answers become stale: an interrupted answer from before is not resumed


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
        self._synth_thread = threading.Thread(target=self._synth_loop, name="speaker-synth", daemon=True)
        self._play_thread = threading.Thread(target=self._play_loop, name="speaker", daemon=True)

    def start(self) -> AudioSpeaker:
        """Start the synthesis and playback threads."""
        self._synth_thread.start()
        self._play_thread.start()
        return self

    # --------------------------------------------------------------------------------- interface

    def say(self, utterance: Utterance) -> None:
        """Queue an utterance; an alert interrupts an answer that is playing."""
        item = _Item(utterance, time.perf_counter())
        if utterance.kind == "alert":
            clip = self.clips.cached(utterance.text)
            if clip is not None:  # pre-rendered: ready now, without waiting for the synthesis thread
                item.audio = np.concatenate((self._chime, self._gap, clip))
        dropped: list[_Item] = []
        with self._cond:
            state = self._state
            if utterance.kind == "alert":
                if utterance.drops_answers:
                    state.answer_epoch += 1
                    dropped += state.answers
                    state.answers.clear()
                playing = state.playing.utterance if state.playing is not None else None
                if utterance.priority >= Priority.SAFETY:
                    state.alerts.insert(self._after_safety(), item)  # ahead of alerts that aren't safety
                    if playing is not None and playing.kind == "alert" and playing.priority < Priority.SAFETY:
                        state.interrupted = True  # (said again after this one: see _play_loop)
                        self.output.stop()
                else:
                    state.alerts.append(item)
                if playing is not None and playing.kind != "alert":
                    state.interrupted = True
                    self.output.stop()
            else:
                item.epoch = state.answer_epoch
                state.answers.append(item)
            self._cond.notify_all()
        self._drop(dropped)

    def stop(self) -> None:
        """Stop speaking now and drop everything queued."""
        with self._cond:
            state = self._state
            dropped = [*state.alerts, *state.answers]
            state.alerts.clear()
            state.answers.clear()
            state.interrupted = False
            state.generation += 1
            if state.playing is not None:
                self.output.stop()
            self._cond.notify_all()
        self._drop(dropped)

    def stop_answers(self) -> None:
        """Stop an answer playing now and drop queued answers; alerts keep playing."""
        with self._cond:
            state = self._state
            dropped = list(state.answers)
            state.answers.clear()
            state.answer_epoch += 1
            if state.playing is not None and state.playing.utterance.kind == "answer":
                state.interrupted = True
                self.output.stop()
            self._cond.notify_all()
        self._drop(dropped)

    def listening(self) -> None:
        """The wake word was heard: stop answering (so "Hey DrivePilot, stop" works mid-answer) and blip."""
        self.stop_answers()
        if self._listen is None:
            return
        cue = Utterance(text="", kind="cue", priority=Priority.SAFETY)
        with self._cond:
            self._state.alerts.append(_Item(cue, time.perf_counter(), self._listen))
            self._cond.notify_all()

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
        """Stop the threads."""
        with self._cond:
            self._state.closed = True
            self._cond.notify_all()
        self.output.stop()
        for thread in (self._play_thread, self._synth_thread):
            if thread.is_alive():
                thread.join(timeout=2.0)

    # --------------------------------------------------------------------------------- synthesis

    def _synth_loop(self) -> None:
        while True:
            with self._cond:
                item = None
                while not self._state.closed:
                    queued = itertools.chain(self._state.alerts, self._state.answers)  # alerts first
                    item = next((i for i in queued if i.audio is None and not i.rendering), None)
                    if item is not None:
                        break
                    self._cond.wait()
                if self._state.closed or item is None:
                    return
                item.rendering = True
            try:
                audio = self._render(item.utterance)
            except Exception:
                log.exception("speech synthesis failed", text=item.utterance.text)
                audio = np.zeros(0, np.float32)  # played as nothing: recorded as not heard
            with self._cond:
                item.audio, item.rendering = audio, False
                self._cond.notify_all()

    def _render(self, utterance: Utterance) -> np.ndarray:
        if utterance.kind == "alert":
            clip = self.clips.cached(utterance.text)
            speech = clip if clip is not None else self.tts.synthesize(utterance.text)
            return np.concatenate((self._chime, self._gap, speech))
        return self.tts.synthesize(utterance.text)

    # ---------------------------------------------------------------------------------- playback

    def _after_safety(self) -> int:
        """Where a safety alert (or an alert it cut off) goes in the alert queue: after the safety alerts and cues
        already at its front (caller holds the lock)."""
        alerts = self._state.alerts
        return next((i for i, it in enumerate(alerts) if it.utterance.priority < Priority.SAFETY), len(alerts))

    def _next_ready(self) -> _Item | None:
        """The next item to play, if it is rendered (caller holds the lock): alerts before answers."""
        state = self._state
        if state.alerts:
            return state.alerts[0] if state.alerts[0].audio is not None else None
        if state.answers and state.answers[0].audio is not None:
            return state.answers[0]
        return None

    def _play_loop(self) -> None:
        while True:
            with self._cond:
                state = self._state
                item = self._next_ready()
                while not state.closed and item is None:
                    self._cond.wait()
                    item = self._next_ready()
                if state.closed or item is None:
                    return
                (state.answers if item.utterance.kind == "answer" else state.alerts).popleft()
                state.playing, state.interrupted = item, False
                generation = state.generation
            audio = item.audio
            sound_at = None
            if audio is not None and len(audio):

                def may_start(generation: int = generation) -> bool:
                    return self._state.generation == generation and not self._state.interrupted

                sound_at = self.output.play(audio, self.tts.rate, should_start=may_start)
            ended_at = time.perf_counter()
            with self._cond:
                stopped = self._state.generation != generation
                interrupted = stopped or self._state.interrupted
                fresh = item.epoch == self._state.answer_epoch
                u = item.utterance
                if interrupted and not stopped and u.kind == "answer" and fresh:
                    resumed = _Item(item.utterance, item.queued_at, item.audio, item.epoch)
                    self._state.answers.appendleft(resumed)
                elif interrupted and not stopped and u.kind == "alert" and u.priority < Priority.SAFETY:
                    # Cut off by a safety alert: said again right after it (unless it had just ended by itself).
                    duration = 0.0 if audio is None else len(audio) / self.tts.rate
                    if sound_at is None or ended_at - sound_at < duration - 0.05:
                        resumed = _Item(item.utterance, item.queued_at, item.audio, item.epoch)
                        self._state.alerts.insert(self._after_safety(), resumed)
                record = self._finish(item, sound_at, interrupted)
            self._report(record)

    def _finish(self, item: _Item, sound_at: float | None, interrupted: bool) -> SpeechRecord:
        """Record the utterance (caller holds the lock)."""
        u = item.utterance
        record = SpeechRecord(u.kind, u.text, item.queued_at, sound_at, interrupted, u.uid)
        self.records.append(record)
        if self._state.playing is item:
            self._state.playing = None
        self._cond.notify_all()
        if record.kind != "cue":
            log.info(
                "spoke",
                kind=record.kind,
                latency_ms=None if record.latency_ms is None else round(record.latency_ms),
                interrupted=interrupted,
                text=record.text,
            )
        return record

    def _drop(self, items: list[_Item]) -> None:
        """Record utterances dropped before they played (outside the lock)."""
        for item in items:
            u = item.utterance
            record = SpeechRecord(u.kind, u.text, item.queued_at, None, True, u.uid)
            self.records.append(record)
            self._report(record)

    def _report(self, record: SpeechRecord) -> None:
        """Hand a finished record to `on_record` (outside the lock)."""
        if self.on_record is not None:
            try:
                self.on_record(record)
            except Exception:
                log.exception("on_record failed")
