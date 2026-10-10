"""Alert engine: turns world events into spoken alerts, once per event (PLAN.md "Rules for alerts").

- "You're up, check and go." when it becomes our turn at a 4-way stop after waiting for other cars. Spoken on
  its own (INFO); if the user asked ("can I go?" → "I'll tell you"), it is REQUESTED and survives quiet mode.
- "Light's green, check the intersection and go." only when the user asked ("tell me when it's green").
- Once per event: the world state emits each event once; on top of that each alert kind has a cooldown, so a
  flickering state can't repeat it.
- Watches expire: after `watch_timeout_s`, and when we drive away from the intersection (a stale request must
  not fire at the next light).
- Quiet mode: only REQUESTED and SAFETY alerts.

Called from the perception thread (world events) and the voice thread (arming, controls): thread-safe.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from answers.local import Watch
from answers.speech import Priority, Utterance
from core.config import AlertsConfig
from core.logging_setup import get_logger
from state.world import WorldEvent

log = get_logger(__name__)

YOUR_TURN = "You're up, check and go."
LIGHT_GREEN = "Light's green, check the intersection and go."


class AlertEngine:
    """Decides which world events are spoken, and how urgently."""

    def __init__(self, cfg: AlertsConfig, emit: Callable[[Utterance], None]) -> None:
        self.cfg = cfg
        self._emit = emit
        self._lock = threading.Lock()
        self._quiet = False
        self._watches: dict[Watch, float] = {}  # armed watch -> media time armed
        self._last_fired: dict[str, float] = {}  # alert kind -> media time

    # ---------------------------------------------------------------------------------- controls

    @property
    def quiet(self) -> bool:
        """Quiet mode: only alerts the user asked for, and safety alerts."""
        with self._lock:
            return self._quiet

    def set_quiet(self, quiet: bool) -> None:
        """Turn quiet mode on or off."""
        with self._lock:
            self._quiet = quiet
        log.info("quiet mode", on=quiet)

    def arm(self, watch: Watch, t: float) -> None:
        """The user asked to be told (e.g. "tell me when it's green")."""
        with self._lock:
            self._watches[watch] = t
        log.info("watch armed", watch=watch.value, t=round(t, 2))

    def cancel(self) -> bool:
        """Drop every watch ("never mind"). Returns True if any was armed."""
        with self._lock:
            had = bool(self._watches)
            self._watches.clear()
        return had

    def armed(self) -> set[Watch]:
        """Watches currently armed."""
        with self._lock:
            return set(self._watches)

    # ------------------------------------------------------------------------------------ events

    def on_world_event(self, event: WorldEvent) -> Utterance | None:
        """Handle one world event; returns (and emits) the alert it caused, if any."""
        utterance: Utterance | None = None
        with self._lock:
            self._expire(event.t)
            if event.kind == "your_turn":
                requested = self._watches.pop(Watch.TURN, None) is not None
                utterance = self._fire(
                    "your_turn", event.t, YOUR_TURN, Priority.REQUESTED if requested else Priority.INFO
                )
            elif event.kind == "light_green":
                if self._watches.pop(Watch.LIGHT, None) is not None:
                    utterance = self._fire("light_green", event.t, LIGHT_GREEN, Priority.REQUESTED)
            elif event.kind == "left_intersection":
                self._watches.pop(Watch.TURN, None)
            elif event.kind == "drove_off":
                # We drove away from where we were waiting: requests made before we started moving were about
                # that place (one made while approaching the next light stays).
                moving_since = event.detail.get("moving_since", event.t)
                for watch, armed_at in list(self._watches.items()):
                    if isinstance(moving_since, int | float) and armed_at < moving_since:
                        del self._watches[watch]
                        log.info("watch dropped: drove off", watch=watch.value)
        if utterance is not None:
            self._emit(utterance)
        return utterance

    def _fire(self, kind: str, t: float, text: str, priority: Priority) -> Utterance | None:
        last = self._last_fired.get(kind)
        if last is not None and 0.0 <= t - last < self.cfg.cooldown_s:
            log.info("alert suppressed: cooldown", kind=kind, t=round(t, 2))
            return None
        if self._quiet and priority < Priority.REQUESTED:
            log.info("alert suppressed: quiet mode", kind=kind, t=round(t, 2))
            return None
        self._last_fired[kind] = t
        log.info("alert", kind=kind, t=round(t, 2), priority=priority.name)
        return Utterance(text=text, kind="alert", priority=priority, clip=kind, t=t)

    def _expire(self, t: float) -> None:
        for watch, armed_at in list(self._watches.items()):
            if t - armed_at > self.cfg.watch_timeout_s or t < armed_at - 1.0:  # timed out, or the video jumped back
                del self._watches[watch]
                log.info("watch expired", watch=watch.value)

    def reset(self) -> None:
        """New video segment: forget watches and cooldowns (quiet mode is a user setting and stays)."""
        with self._lock:
            self._watches.clear()
            self._last_fired.clear()
