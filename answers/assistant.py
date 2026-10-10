"""The assistant: question in, spoken answer out; world events in, alerts out.

`Assistant.ask(text)` routes a transcribed (or typed) question and answers it:
- voice controls are handled here (quiet / normal mode, repeat that, stop, never mind);
- right of way, watches and split-second refusals are answered locally from the latest `WorldSnapshot`
  (instant, PLAN.md target < 1 s end to end);
- navigation and lane questions go to the navigation handler (M6), everything else to the vision model (M5),
  both behind small interfaces so they can be swapped or mocked.
`Assistant.on_world_event` is registered with `WorldState` and feeds the alert engine. Everything said goes
through `say`, which remembers the last message for "repeat that".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from answers import local
from answers.alerts import AlertEngine
from answers.speech import Priority, Speaker, Utterance
from core.config import AppConfig
from core.logging_setup import get_logger
from state.world import WorldEvent, WorldSnapshot
from voice.router import Route, RouteKind, route

log = get_logger(__name__)

QUIET_ON = "Quiet mode. I'll only speak up for safety, or when you ask."
QUIET_OFF = "Normal mode. All alerts are back on."
NOTHING_TO_REPEAT = "I haven't said anything yet."
STARTING = "I'm still starting up. Ask me again in a moment."
NO_MODEL = "I can't look at the scene right now: the vision model isn't connected."
NO_NAV = "Navigation isn't available right now."
CANCELLED = "Okay, cancelled."
OKAY = "Okay."


class NavAnswerer(Protocol):
    """Navigation handler (M6): destinations, route questions, lane guidance."""

    def answer(self, route: Route, snapshot: WorldSnapshot) -> str:
        """Answer a DESTINATION / NAV / LANE question."""


class SceneAnswerer(Protocol):
    """Vision-model handler (M5): questions about what the camera sees."""

    def answer(self, route: Route, snapshot: WorldSnapshot) -> str:
        """Answer a MODEL (or stopped GAP_CHECK) question from recent frames and the world state."""


@dataclass(frozen=True, slots=True)
class Reply:
    """The outcome of one question."""

    route: Route
    text: str | None  # None: nothing was said (stop, empty)
    source: str  # "control" | "local" | "nav" | "model" | "none"
    elapsed_ms: float  # routing + answering (excludes speech)


class Assistant:
    """Routes questions, speaks answers and alerts, and handles the voice controls."""

    def __init__(
        self,
        cfg: AppConfig,
        snapshot: Callable[[], WorldSnapshot | None],
        speaker: Speaker,
        nav: NavAnswerer | None = None,
        model: SceneAnswerer | None = None,
    ) -> None:
        self.cfg = cfg
        self._snapshot = snapshot
        self.speaker = speaker
        self.nav = nav
        self.model = model
        self.alerts = AlertEngine(cfg.alerts, self.say)
        self._lock = threading.Lock()
        self._last: Utterance | None = None

    # ------------------------------------------------------------------------------------ output

    def say(self, utterance: Utterance) -> None:
        """Speak, and remember it for "repeat that"."""
        with self._lock:
            self._last = utterance
        self.speaker.say(utterance)

    @property
    def last(self) -> Utterance | None:
        """The last thing said."""
        with self._lock:
            return self._last

    def on_world_event(self, event: WorldEvent) -> None:
        """WorldState event callback (perception thread): alerts."""
        if event.kind == "segment_reset":
            self.alerts.reset()
            return
        self.alerts.on_world_event(event)

    # -------------------------------------------------------------------------------- questions

    def ask(self, text: str) -> Reply:
        """Answer one question (voice thread or typed). Speaks the answer and returns it."""
        start = time.perf_counter()
        r = route(text)
        snap = self._snapshot()
        t = snap.t if snap is not None else 0.0
        answer, source = self._answer(r, snap, t)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "question",
            kind=r.kind.value,
            intent=r.intent,
            question=r.question,
            answer=answer,
            source=source,
            ms=round(elapsed_ms, 2),
            t=round(t, 2),
        )
        if answer:
            self.say(Utterance(text=answer, kind="answer", priority=Priority.ANSWER, t=t))
        return Reply(route=r, text=answer, source=source, elapsed_ms=elapsed_ms)

    def _answer(self, r: Route, snap: WorldSnapshot | None, t: float) -> tuple[str | None, str]:
        if r.kind is RouteKind.EMPTY:
            return None, "none"
        if r.kind is RouteKind.CONTROL:
            return self._control(r.intent), "control"
        if snap is None:
            return STARTING, "local"
        if r.kind in (RouteKind.DESTINATION, RouteKind.NAV, RouteKind.LANE):
            if self.nav is None:
                return NO_NAV, "nav"
            return self.nav.answer(r, snap), "nav"
        if r.kind is RouteKind.GAP_CHECK:
            local_answer = local.gap_check(snap, r.text)
            if local_answer is not None:
                return self._local(local_answer, t), "local"
            return self._model(r, snap), "model"
        if r.kind is RouteKind.RIGHT_OF_WAY:
            handler = {"who_first": local.who_first, "can_i_go": local.can_i_go, "position": local.position}[r.intent]
            return self._local(handler(snap), t), "local"
        if r.kind is RouteKind.WATCH:
            handler = local.light_watch if r.intent == "light_watch" else local.go_watch
            return self._local(handler(snap), t), "local"
        return self._model(r, snap), "model"

    def _local(self, answer: local.LocalAnswer, t: float) -> str:
        for watch in answer.watches:
            self.alerts.arm(watch, t)
        return answer.text

    def _model(self, r: Route, snap: WorldSnapshot) -> str:
        if self.model is None:
            return NO_MODEL
        return self.model.answer(r, snap)

    def _control(self, intent: str) -> str | None:
        if intent == "quiet":
            self.alerts.set_quiet(True)
            return QUIET_ON
        if intent == "normal":
            self.alerts.set_quiet(False)
            return QUIET_OFF
        if intent == "stop":
            self.speaker.stop()
            return None
        if intent == "cancel":
            self.speaker.stop()
            return CANCELLED if self.alerts.cancel() else OKAY
        if intent == "repeat":
            last = self.last
            if last is None:
                return NOTHING_TO_REPEAT
            self.speaker.stop()
            return last.text  # spoken again as an answer (also shown on screen)
        raise ValueError(f"unknown control intent {intent!r}")
