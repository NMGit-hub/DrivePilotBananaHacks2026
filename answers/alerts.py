"""Alert engine: turns world events into spoken alerts, once per event (PLAN.md "Rules for alerts").

- "You're up, check and go." when it becomes our turn at a 4-way stop after waiting for other cars. Spoken on
  its own (INFO); if the user asked ("can I go?" → "I'll tell you"), it is REQUESTED and survives quiet mode.
  "Never mind" silences it for the current stop.
- "Light's green, check the intersection and go." when the user asked ("tell me when it's green"); otherwise just
  "Light's green." (INFO) if `alerts.auto.light_green` is on.
- Unrequested heads-ups (INFO, each behind its `AutoSwitches` switch, answers/auto_alerts.py): "The car ahead is
  moving." (stopped behind a car that pulls away; not right after "Light's green", which says it already) and "Stop
  sign ahead." (state/heads_up.py raises both).
- Once per event: the world state emits each event once per stop; on top of that each alert kind has a
  cooldown (reset when we leave the intersection or drive off), so a flickering state can't repeat it. A watch
  is used up only when its alert is actually spoken.
- Watches expire: after `watch_timeout_s`; when the video jumps back before the request; when we drive away
  from where we waited; and, if requested while driving, when we haven't stopped within `approach_timeout_s`
  (the light turned green before we got there: a later, unrelated light must not be announced).
- Races: the answer that arms a watch was computed from a snapshot; if the event fired in between, `arm`
  reports "missed" so the assistant answers again from fresh state instead of promising an alert that won't
  come. A request made before a seek (`generation` changed) is ignored.
- Quiet mode: only REQUESTED and SAFETY alerts.
- Pedestrians and cyclists in (or stepping into) our path while we move (state/pedestrians.py): a SAFETY alert
  from a small fixed set of sentences (pre-rendered at startup, so they play at once), spoken at most once per
  track. One that would repeat the last warning within `safety_gap_s` (the same kind, on the same side or ahead: a
  group crossing) is not said again; another kind, or someone on the other side, is. Kept in quiet mode; it
  interrupts answers, which resume afterwards (it doesn't make them stale).

Called from the perception thread (world events) and the voice thread (arming, controls): thread-safe.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from answers.auto_alerts import CAR_AHEAD_MOVING, LIGHT_GREEN_AUTO, STOP_SIGN_AHEAD, AutoSwitches
from answers.local import Watch
from answers.speech import Priority, Utterance
from core.config import AlertsConfig
from core.logging_setup import get_logger
from state.world import WorldEvent

log = get_logger(__name__)

YOUR_TURN = "You're up, check and go."
LIGHT_GREEN = "Light's green, check the intersection and go."
_EVENT_OF = {Watch.TURN: "your_turn", Watch.LIGHT: "light_green"}

# (who, how, side) of a "pedestrian_in_path" event -> what is said. "entering" names the side they come from.
PEDESTRIAN_TEXTS: dict[tuple[str, str, str], str] = {
    ("pedestrian", "in_path", "ahead"): "Pedestrian ahead.",
    ("pedestrian", "in_path", "left"): "Pedestrian ahead on your left.",
    ("pedestrian", "in_path", "right"): "Pedestrian ahead on your right.",
    ("pedestrian", "entering", "left"): "Pedestrian stepping into the road on your left.",
    ("pedestrian", "entering", "right"): "Pedestrian stepping into the road on your right.",
    ("cyclist", "in_path", "ahead"): "Cyclist ahead.",
    ("cyclist", "in_path", "left"): "Cyclist ahead on your left.",
    ("cyclist", "in_path", "right"): "Cyclist ahead on your right.",
    ("cyclist", "entering", "left"): "Cyclist cutting in on your left.",
    ("cyclist", "entering", "right"): "Cyclist cutting in on your right.",
}
SAFETY_ALERT_TEXTS = tuple(PEDESTRIAN_TEXTS.values())  # pre-rendered at startup with the other fixed alerts
_WARNED_MEMORY_S = 120.0  # how long a warned track id is remembered (ids only grow within a segment)
GREEN_COVERS_S = 5.0  # "The car ahead is moving" this soon after "Light's green" adds nothing


class ArmResult(StrEnum):
    """What happened to a watch request."""

    ARMED = "armed"
    MISSED = "missed"  # the event already happened after the snapshot the answer was built from
    STALE = "stale"  # the request was made before a seek / segment reset


@dataclass(slots=True)
class _Watch:
    armed_at: float
    while_moving: bool
    saw_stop: bool = False


class AlertEngine:
    """Decides which world events are spoken, and how urgently."""

    def __init__(
        self, cfg: AlertsConfig, emit: Callable[[Utterance], None], switches: AutoSwitches | None = None
    ) -> None:
        self.cfg = cfg
        self._emit = emit
        self.switches = switches or AutoSwitches.from_config(cfg.auto)  # unrequested alerts on/off (the page)
        self._lock = threading.Lock()
        self._quiet = False
        self._watches: dict[Watch, _Watch] = {}
        self._last_fired: dict[str, float] = {}  # alert kind -> media time it was spoken
        self._last_event: dict[str, float] = {}  # event kind -> media time it happened (spoken or not)
        self._turn_silenced = False  # "never mind" at this stop
        self._generation = 0  # bumped on every reset (seek / new segment)
        self._warned: dict[int, float] = {}  # pedestrian/cyclist track id -> when we warned about them
        self._last_warning: tuple[float, str, str] | None = None  # (t, kind, side) of the last one spoken

    # ---------------------------------------------------------------------------------- controls

    @property
    def quiet(self) -> bool:
        """Quiet mode: only alerts the user asked for, and safety alerts."""
        with self._lock:
            return self._quiet

    @property
    def generation(self) -> int:
        """Changes on every reset; pass it back to `arm` to detect requests made before a seek."""
        with self._lock:
            return self._generation

    def set_quiet(self, quiet: bool) -> None:
        """Turn quiet mode on or off."""
        with self._lock:
            self._quiet = quiet
        log.info("quiet mode", on=quiet)

    def arm(self, watch: Watch, t: float, *, moving: bool = False, generation: int | None = None) -> ArmResult:
        """The user asked to be told (e.g. "tell me when it's green"). `t`: media time of the snapshot the
        answer was built from; `moving`: we were driving when asked."""
        with self._lock:
            if generation is not None and generation != self._generation:
                log.info("watch ignored: made before a seek", watch=watch.value)
                return ArmResult.STALE
            last = self._last_event.get(_EVENT_OF[watch])
            if last is not None and last >= t:
                log.info("watch not armed: it already happened", watch=watch.value, event_t=round(last, 2))
                return ArmResult.MISSED
            self._watches[watch] = _Watch(armed_at=t, while_moving=moving)
            if watch is Watch.TURN:
                self._turn_silenced = False  # asking again undoes an earlier "never mind"
        log.info("watch armed", watch=watch.value, t=round(t, 2), while_moving=moving)
        return ArmResult.ARMED

    def cancel(self, t: float | None = None) -> bool:
        """ "Never mind": drop every watch and keep "you're up" quiet at this stop. True if a watch was armed."""
        with self._lock:
            if t is not None:
                self._expire(t)
            had = bool(self._watches)
            self._watches.clear()
            self._turn_silenced = True
        return had

    def armed(self) -> set[Watch]:
        """Watches currently armed."""
        with self._lock:
            return set(self._watches)

    # ------------------------------------------------------------------------------------ events

    def on_world_event(self, event: WorldEvent) -> Utterance | None:
        """Handle one world event; returns (and emits) the alert it caused, if any."""
        utterance: Utterance | None = None
        t = event.t
        with self._lock:
            self._expire(t)
            if event.kind == "pedestrian_in_path":
                utterance = self._pedestrian(event)
            elif event.kind == "your_turn":
                self._last_event["your_turn"] = t
                if self._turn_silenced:
                    log.info("alert silenced: never mind", kind="your_turn")
                else:
                    priority = Priority.REQUESTED if Watch.TURN in self._watches else Priority.INFO
                    utterance = self._fire("your_turn", t, YOUR_TURN, priority)
                    if utterance is not None:
                        self._watches.pop(Watch.TURN, None)
            elif event.kind == "light_green":
                self._last_event["light_green"] = t
                if Watch.LIGHT in self._watches:
                    utterance = self._fire("light_green", t, LIGHT_GREEN, Priority.REQUESTED)
                    if utterance is not None:
                        self._watches.pop(Watch.LIGHT, None)
                elif self.switches.enabled("light_green"):
                    utterance = self._fire("light_green", t, LIGHT_GREEN_AUTO, Priority.INFO)
            elif event.kind == "car_ahead_moving":
                green = self._last_fired.get("light_green")
                if green is not None and 0.0 <= t - green < GREEN_COVERS_S:
                    log.info("alert suppressed: the light just turned green", kind="car_ahead")
                elif self.switches.enabled("car_ahead"):
                    utterance = self._fire("car_ahead", t, CAR_AHEAD_MOVING, Priority.INFO)
            elif event.kind == "stop_sign_ahead":
                if self.switches.enabled("stop_sign"):
                    utterance = self._fire("stop_sign", t, STOP_SIGN_AHEAD, Priority.INFO, drops_answers=False)
            elif event.kind == "left_intersection":
                # The next stop is a new event: its "you're up" must not be eaten by this one's cooldown.
                self._watches.pop(Watch.TURN, None)
                self._last_fired.pop("your_turn", None)
                self._last_event.pop("your_turn", None)
                self._turn_silenced = False
            elif event.kind == "ego_stopped":
                for record in self._watches.values():
                    record.saw_stop = True
            elif event.kind == "drove_off":
                # We drove away from where we were waiting: requests made before we started moving were about
                # that place (one made while approaching the next light stays).
                moving_since = event.detail.get("moving_since", t)
                for watch, record in list(self._watches.items()):
                    if isinstance(moving_since, int | float) and record.armed_at < moving_since:
                        del self._watches[watch]
                        log.info("watch dropped: drove off", watch=watch.value)
                self._last_fired.pop("light_green", None)
                self._last_event.pop("light_green", None)
                self._turn_silenced = False
        if utterance is not None:
            self._emit(utterance)
        return utterance

    def _pedestrian(self, event: WorldEvent) -> Utterance | None:
        """A pedestrian or cyclist in our path (caller holds the lock): once per track, and not a repeat of the
        warning just given (`safety_gap_s`). Safety: spoken in quiet mode too."""
        d, t = event.detail, event.t
        cls, side = str(d.get("cls")), str(d.get("side"))
        text = PEDESTRIAN_TEXTS.get((cls, str(d.get("how")), side))
        track = d.get("track")
        if text is None or not isinstance(track, int):
            log.warning("pedestrian event without a known phrase", **d)
            return None
        if track in self._warned:
            log.info("alert suppressed: already warned about this person", kind="pedestrian", track=track)
            return None
        self._warned[track] = t
        if len(self._warned) > 500:
            self._warned = {k: v for k, v in self._warned.items() if 0.0 <= t - v <= _WARNED_MEMORY_S}
        last = self._last_warning
        recent = last is not None and 0.0 <= t - last[0] < self.cfg.safety_gap_s
        if recent and last is not None and last[1] == cls and (side == last[2] or "ahead" in (side, last[2])):
            # The same warning a moment ago (someone in the same group): said once.
            log.info("alert suppressed: just warned", kind="pedestrian", track=track, t=round(t, 2))
            return None
        self._last_warning = (t, cls, side)
        log.info("alert", kind="pedestrian", t=round(t, 2), priority=Priority.SAFETY.name, track=track, text=text)
        return Utterance(text=text, kind="alert", priority=Priority.SAFETY, clip="pedestrian", t=t)

    def _fire(self, kind: str, t: float, text: str, priority: Priority, drops_answers: bool = True) -> Utterance | None:
        last = self._last_fired.get(kind)
        if last is not None and 0.0 <= t - last < self.cfg.cooldown_s:
            log.info("alert suppressed: cooldown", kind=kind, t=round(t, 2))
            return None
        if self._quiet and priority < Priority.REQUESTED:
            log.info("alert suppressed: quiet mode", kind=kind, t=round(t, 2))
            return None
        self._last_fired[kind] = t
        log.info("alert", kind=kind, t=round(t, 2), priority=priority.name)
        # "You're up" / "Light's green": whatever was being said about waiting is now out of date.
        return Utterance(text=text, kind="alert", priority=priority, clip=kind, t=t, drops_answers=drops_answers)

    def _expire(self, t: float) -> None:
        cfg = self.cfg
        for watch, record in list(self._watches.items()):
            reason = None
            if t - record.armed_at > cfg.watch_timeout_s:
                reason = "timed out"
            elif t < record.armed_at - cfg.seek_back_tolerance_s:
                reason = "video jumped back"
            elif record.while_moving and not record.saw_stop and t - record.armed_at > cfg.approach_timeout_s:
                reason = "we never stopped"
            if reason is not None:
                del self._watches[watch]
                log.info("watch expired", watch=watch.value, reason=reason)

    def reset(self) -> None:
        """New video segment: forget watches, cooldowns and events (quiet mode is a user setting and stays)."""
        with self._lock:
            self._watches.clear()
            self._last_fired.clear()
            self._last_event.clear()
            self._turn_silenced = False
            self._warned.clear()  # track ids start over with the new segment
            self._last_warning = None
            self._generation += 1
