"""Latency bookkeeping for PLAN.md's targets: question end → first spoken word, detection → alert sound.

Answers and alerts are handed to the speaker on the thread that produced them, so each utterance's origin is
known there: the end of the spoken question (set around `Assistant.ask` with `question`), or the capture time of
the frame whose world update raised the alert (set around the world update with `detection`). `on_say` files
the utterance under its id with that origin; when the audio speaker reports that utterance (`on_played`, same
id) the latency is the moment its first sample reached the speaker minus the origin. Utterances that never
played (stopped, dropped, or a text-only speaker) are not counted, and at most `MAX_PENDING` wait for a report.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

MAX_PENDING = 500  # a text-only speaker never reports playback: forget the oldest


@dataclass(frozen=True, slots=True)
class Measured:
    """One measured latency."""

    kind: str  # "answer" | "alert"
    key: str  # question id, or the alert's kind ("your_turn", "nav_lanes", ...)
    seconds: float


@dataclass(slots=True)
class _Question:
    qid: int
    origin: float
    first_said: bool = False


class LatencyBook:
    """Thread-safe; one per app run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._pending: OrderedDict[int, tuple[str, str, float]] = OrderedDict()
        self.answers: dict[int, float] = {}  # question id → seconds from the end of the question to first audio
        self.alerts: list[Measured] = []

    @contextlib.contextmanager
    def question(self, qid: int, origin: float) -> Iterator[None]:
        """Answers said on this thread inside the block belong to question `qid`, which ended at `origin`
        (perf_counter)."""
        self._local.question = _Question(qid, origin)
        try:
            yield
        finally:
            self._local.question = None

    @contextlib.contextmanager
    def detection(self, origin: float) -> Iterator[None]:
        """Alerts said on this thread inside the block were detected in a frame captured at `origin`."""
        self._local.detected_at = origin
        try:
            yield
        finally:
            self._local.detected_at = None

    def on_say(self, uid: int, kind: str, clip: str | None = None) -> None:
        """Call before handing utterance `uid` (`kind` "answer" | "alert") to the speaker."""
        if kind == "answer":
            question: _Question | None = getattr(self._local, "question", None)
            if question is None or question.first_said:
                return
            question.first_said = True
            entry = ("answer", str(question.qid), question.origin)
        elif kind == "alert":
            origin = getattr(self._local, "detected_at", None)
            entry = ("alert", clip or "alert", origin if origin is not None else time.perf_counter())
        else:
            return
        with self._lock:
            self._pending[uid] = entry
            while len(self._pending) > MAX_PENDING:
                self._pending.popitem(last=False)

    def on_played(self, uid: int, sound_at: float | None) -> Measured | None:
        """The speaker played (or dropped: `sound_at` None) utterance `uid`: its latency, if it was being timed.
        Only the first report counts (an interrupted answer that resumes is reported twice)."""
        with self._lock:
            entry = self._pending.pop(uid, None)
            if entry is None or sound_at is None:
                return None
            kind, key, origin = entry
            measured = Measured(kind, key, sound_at - origin)
            if kind == "answer":
                self.answers[int(key)] = measured.seconds
            else:
                self.alerts.append(measured)
            return measured
