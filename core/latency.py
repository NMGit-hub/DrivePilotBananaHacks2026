"""Latency bookkeeping for PLAN.md's targets: question end → first spoken word, detection → alert sound.

Answers and alerts are handed to the speaker on the thread that produced them, so each utterance's origin is
known there: the end of the spoken question (set around `Assistant.ask` with `question`), or the capture time of
the frame whose world update raised the alert (set around the world update with `detection`). `on_say` files
the utterance under its text with that origin; when the audio speaker reports it as played (`on_record`), the
latency is the moment its first sample reached the speaker minus the origin. Utterances never played (stopped,
or a text-only speaker) are not counted.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass

MAX_PENDING_PER_TEXT = 20  # a text-only speaker never reports playback: don't grow forever


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
        self._pending: defaultdict[str, deque[tuple[str, str, float]]] = defaultdict(
            lambda: deque(maxlen=MAX_PENDING_PER_TEXT)
        )
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

    def on_say(self, kind: str, text: str, clip: str | None = None) -> None:
        """Call before handing an utterance (`kind` "answer" | "alert") to the speaker."""
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
            self._pending[text].append(entry)

    def on_played(self, kind: str, text: str, sound_at: float | None) -> Measured | None:
        """The speaker played (or dropped, `sound_at` None) an utterance: its latency, if it was being timed."""
        if kind not in ("answer", "alert"):
            return None
        with self._lock:
            queue = self._pending.get(text)
            if not queue:
                return None
            entry_kind, key, origin = queue.popleft()
            if sound_at is None:
                return None
            measured = Measured(entry_kind, key, sound_at - origin)
            if entry_kind == "answer":
                self.answers[int(key)] = measured.seconds
            else:
                self.alerts.append(measured)
            return measured
