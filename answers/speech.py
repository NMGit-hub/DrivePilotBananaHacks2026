"""What the co-pilot says: utterances, their priority, and the speaker interface (audio lives in voice/, M4)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from core.logging_setup import get_logger

log = get_logger(__name__)


class Priority(IntEnum):
    """Who may interrupt whom, and what quiet mode keeps (PLAN.md "Rules for alerts")."""

    INFO = 0  # spoken on its own ("You're up"): muted in quiet mode
    ANSWER = 1  # a reply to a question
    REQUESTED = 2  # something the user asked to be told ("tell me when it's green"): kept in quiet mode
    SAFETY = 3  # interrupts anything; the interrupted answer resumes afterwards


@dataclass(frozen=True, slots=True)
class Utterance:
    """One thing to say."""

    text: str
    kind: str  # "answer" | "alert"
    priority: Priority
    clip: str | None = None  # pre-rendered audio clip for alerts (instant playback, M4)
    t: float = 0.0  # media time it refers to
    drops_answers: bool = False  # an alert that changes the situation ("You're up"): answers in progress are stale


class Speaker(Protocol):
    """Speech output. M4 implements it with Piper TTS + a chime; text mode and tests record."""

    def say(self, utterance: Utterance) -> None:
        """Queue (or, by priority, interrupt with) an utterance. Must not block."""

    def stop(self) -> None:
        """Stop speaking now and drop anything queued."""


class TextSpeaker:
    """A speaker that logs and keeps what it was told to say (text mode, the V1 screen, tests)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.said: list[Utterance] = []
        self.stops = 0

    def say(self, utterance: Utterance) -> None:
        """Record the utterance."""
        log.info("say", kind=utterance.kind, priority=utterance.priority.name, text=utterance.text)
        with self._lock:
            self.said.append(utterance)

    def stop(self) -> None:
        """Count stop requests."""
        with self._lock:
            self.stops += 1

    def texts(self) -> list[str]:
        """Everything said so far."""
        with self._lock:
            return [u.text for u in self.said]
