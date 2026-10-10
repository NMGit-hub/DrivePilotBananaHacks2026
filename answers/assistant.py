"""The assistant: question in, spoken answer out; world events in, alerts out.

`Assistant.ask(text)` routes a transcribed (or typed) question and answers it:
- voice controls are handled here (quiet / normal mode, repeat that, stop, never mind);
- right of way, watches and split-second refusals are answered locally from the latest `WorldSnapshot`
  (instant, PLAN.md target < 1 s end to end);
- navigation and lane questions go to the navigation handler (M6), everything else to the vision model (M5),
  both behind small interfaces so they can be swapped or mocked. Model answers stream: each sentence is spoken
  as soon as it arrives, and "stop" (or a newer question) cuts the rest off.
- a last safety net: a gap-like question that reached the model while we are moving gets the refusal.
`Assistant.on_world_event` is registered with `WorldState` and feeds the alert engine. Everything said goes
through `say`, which remembers the last message for "repeat that".
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from answers import local
from answers.alerts import LIGHT_GREEN, YOUR_TURN, AlertEngine, ArmResult
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
HANDLER_ERROR = "Sorry, something went wrong answering that."
CANCELLED = "Okay, cancelled."
OKAY = "Okay."
EARLIER = "Earlier I said: "
REPEAT_STALE_S = 30.0  # repeating something older than this says so
# Gap-like wording that must not reach the model while we move, whatever route it took.
GAP_HINT = re.compile(
    r"\b(?:merge|merging|overtake|overtaking|blind spot|make (?:it|the light)|stop or go|room to|gap|cut in|"
    r"squeeze in|change lanes?|lane change|switch lanes?|(?:safe|clear|ok|okay) to|is (?:it|that|everything) clear|"
    r"(?:lane|side|road|way) (?:is )?clear)\b"
)

Answer = str | Iterable[str] | None


class NavAnswerer(Protocol):
    """Navigation handler (M6): destinations, route questions, lane guidance."""

    def answer(self, route: Route, snapshot: WorldSnapshot) -> str:
        """Answer a DESTINATION / NAV / LANE question."""


class SceneAnswerer(Protocol):
    """Vision-model handler (M5): questions about what the camera sees."""

    def answer(self, route: Route, snapshot: WorldSnapshot) -> str | Iterable[str]:
        """Answer a MODEL (or stopped GAP_CHECK) question; a stream of sentences, or one string."""


@dataclass(frozen=True, slots=True)
class Reply:
    """The outcome of one question."""

    route: Route
    text: str | None  # None: nothing was said (stop, empty)
    source: str  # "control" | "local" | "nav" | "model" | "error" | "none"
    elapsed_ms: float  # routing + answering, until the whole answer was handed to the speaker
    first_ms: float | None = None  # until the first sentence was handed to the speaker


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
        self._turn = 0  # bumped by every question and by stop/cancel: a streaming answer stops when it changes

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
        with self._lock:
            self._turn += 1
            turn = self._turn
        r = route(text)
        generation = self.alerts.generation
        snap = self._snapshot()
        t = snap.t if snap is not None else 0.0
        try:
            answer, source = self._answer(r, snap, generation)
        except Exception:
            log.exception("answer failed", kind=r.kind.value, intent=r.intent)
            answer, source = HANDLER_ERROR, "error"

        first_ms: float | None = None
        if isinstance(answer, str):
            self.say(Utterance(text=answer, kind="answer", priority=Priority.ANSWER, t=t))
            first_ms = (time.perf_counter() - start) * 1000.0
            spoken: str | None = answer
        elif answer is None:
            spoken = None
        else:
            said: list[str] = []
            try:
                for sentence in answer:
                    if self._turn != turn:  # "stop", "never mind" or a newer question
                        log.info("answer cut off", after_sentences=len(said))
                        break
                    self.say(Utterance(text=sentence, kind="answer", priority=Priority.ANSWER, t=t))
                    if first_ms is None:
                        first_ms = (time.perf_counter() - start) * 1000.0
                    said.append(sentence)
            except Exception:
                log.exception("streamed answer failed", kind=r.kind.value)
                if not said:
                    self.say(Utterance(text=HANDLER_ERROR, kind="answer", priority=Priority.ANSWER, t=t))
                    said.append(HANDLER_ERROR)
            finally:
                if isinstance(answer, Iterator) and hasattr(answer, "close"):
                    answer.close()  # stop the model stream if we cut it off
            spoken = " ".join(said) or None
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "question",
            kind=r.kind.value,
            intent=r.intent,
            question=r.question,
            answer=spoken,
            source=source,
            ms=round(elapsed_ms, 2),
            first_ms=None if first_ms is None else round(first_ms, 1),
            t=round(t, 2),
        )
        return Reply(route=r, text=spoken, source=source, elapsed_ms=elapsed_ms, first_ms=first_ms)

    def _answer(self, r: Route, snap: WorldSnapshot | None, generation: int) -> tuple[Answer, str]:
        if r.kind is RouteKind.EMPTY:
            return None, "none"
        if r.kind is RouteKind.CONTROL:
            return self._control(r.intent, snap), "control"
        if snap is None:
            return STARTING, "local"
        sure_stopped = snap.ego_stopped and snap.ego_motion_confident
        if r.kind in (RouteKind.DESTINATION, RouteKind.NAV, RouteKind.LANE):
            if self.nav is None:
                return NO_NAV, "nav"
            return self.nav.answer(r, snap), "nav"
        if r.kind is RouteKind.GAP_CHECK:
            if r.intent == "gap_watch":
                return self._local(lambda s: local.gap_watch(s), snap, generation), "local"
            answer = local.gap_check(snap, r.text)
            if answer is not None:
                return self._local(lambda s: local.gap_check(s, r.text) or answer, snap, generation), "local"
            return self._model(r, snap), "model"
        if r.kind is RouteKind.RIGHT_OF_WAY:
            if r.intent == "should_wait":
                wait = local.should_wait(snap, r.text)
                if wait is None:
                    return self._model(r, snap), "model"
                return self._local(lambda s: local.should_wait(s, r.text) or wait, snap, generation), "local"
            handler = {"who_first": local.who_first, "can_i_go": local.can_i_go, "position": local.position}[r.intent]
            return self._local(handler, snap, generation), "local"
        if r.kind is RouteKind.WATCH:
            handler = local.light_watch if r.intent == "light_watch" else local.go_watch
            return self._local(handler, snap, generation), "local"
        if not sure_stopped and GAP_HINT.search(r.text):
            log.info("gap-like question refused while moving", question=r.question)
            return local.GAP_REFUSAL, "local"
        return self._model(r, snap), "model"

    def _local(
        self, compute: Callable[[WorldSnapshot], local.LocalAnswer], snap: WorldSnapshot, generation: int
    ) -> str:
        """Answer locally and arm what it promises; if the promised event already happened in the meantime,
        answer again from the latest state rather than promise an alert that won't come."""
        answer = compute(snap)
        for watch in answer.watches:
            result = self.alerts.arm(watch, snap.t, moving=not snap.ego_stopped, generation=generation)
            if result is ArmResult.MISSED:
                fresh = self._snapshot()
                if fresh is not None and fresh is not snap:
                    retry = compute(fresh)
                    for again in retry.watches:
                        self.alerts.arm(again, fresh.t, moving=not fresh.ego_stopped, generation=generation)
                    return retry.text
                # No newer state to answer from: say what the alert would have said, not "I'll tell you".
                return {local.Watch.TURN: YOUR_TURN, local.Watch.LIGHT: LIGHT_GREEN}[watch]
        return answer.text

    def _model(self, r: Route, snap: WorldSnapshot) -> Answer:
        if self.model is None:
            return NO_MODEL
        return self.model.answer(r, snap)

    def _control(self, intent: str, snap: WorldSnapshot | None) -> str | None:
        if intent == "quiet":
            self.alerts.set_quiet(True)
            return QUIET_ON
        if intent == "normal":
            self.alerts.set_quiet(False)
            return QUIET_OFF
        if intent == "stop":
            self._interrupt()
            return None
        if intent == "cancel":
            self._interrupt()
            return CANCELLED if self.alerts.cancel(snap.t if snap is not None else None) else OKAY
        if intent == "repeat":
            last = self.last
            if last is None:
                return NOTHING_TO_REPEAT
            self.speaker.stop()
            stale = snap is not None and (snap.t - last.t > REPEAT_STALE_S or snap.t < last.t)
            return EARLIER + last.text if stale else last.text
        raise ValueError(f"unknown control intent {intent!r}")

    def _interrupt(self) -> None:
        """Stop speaking now and cut off any answer still streaming."""
        with self._lock:
            self._turn += 1
        self.speaker.stop()
