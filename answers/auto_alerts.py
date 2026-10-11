"""Spoken alerts nobody asked for (`alerts.auto`): the switches, and the watches the app feeds every GPS tick.

- `AutoSwitches`: which of them are on. Starts from config.yaml; the page switches them while running (any thread).
  "light_green", "car_ahead" and "stop_sign" come from world events (answers/alerts.py, state/heads_up.py),
  "turns" from navigation (nav/handler.py), "wrong_lane" and "speeding" from the watches below.
- `WrongLaneWatch`: navigating, lane detection puts us in a lane that doesn't work for the next maneuver
  (`ui.overlay.lane_target`: the route's lane data and lane detection agree on the lane count) for
  `wrong_lane_hold_s`, within `wrong_lane_within_m` of it: "You're in the wrong lane for your turn. Move one lane
  left." Said once per maneuver, once more if we're still in a wrong lane within `wrong_lane_repeat_m`; after a
  warning, moving into a good lane gets "That's the right lane." (once).
- `SpeedingWatch`: GPS speed over the posted limit the sign detector read (not a school limit: those depend on
  the hours; not one older than `speeding_limit_max_age_s`) by `speeding_margin_mph` for `speeding_hold_s`:
  "You're over the limit. It's 30 here." Once per limit, again only after we slowed to the limit for
  `speeding_repeat_s` or a new limit is read.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from core.config import AutoAlertsConfig

AUTO_NAMES = ("light_green", "car_ahead", "turns", "wrong_lane", "speeding", "stop_sign")
LIGHT_GREEN_AUTO = "Light's green."
CAR_AHEAD_MOVING = "The car ahead is moving."
STOP_SIGN_AHEAD = "Stop sign ahead."
RIGHT_LANE = "That's the right lane."
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
MPS_TO_MPH = 2.23694


def move_text(side: str, moves: int) -> str:
    """ "move one lane left", "move two lanes right"."""
    return f"move {_WORDS.get(moves, str(moves))} lane{'s' if moves > 1 else ''} {side}"


def wrong_lane_text(side: str, moves: int, again: bool = False) -> str:
    """The wrong-lane warning."""
    lead = "You're still in the wrong lane" if again else "You're in the wrong lane for your turn"
    return f"{lead}. {move_text(side, moves).capitalize()}."


def speeding_text(limit: int) -> str:
    """The speeding warning."""
    return f"You're over the limit. It's {limit} here."


FIXED_TEXTS = (  # pre-rendered at startup (instant playback)
    LIGHT_GREEN_AUTO,
    CAR_AHEAD_MOVING,
    STOP_SIGN_AHEAD,
    RIGHT_LANE,
    *(wrong_lane_text(side, n, again) for side in ("left", "right") for n in (1, 2) for again in (False, True)),
    *(speeding_text(v) for v in range(15, 80, 5)),
)


@dataclass
class AutoSwitches:
    """Which unrequested alerts are on (thread-safe)."""

    on: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(AUTO_NAMES, True))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_config(cls, cfg: AutoAlertsConfig) -> AutoSwitches:
        """The switches as config.yaml sets them."""
        return cls({name: bool(getattr(cfg, name)) for name in AUTO_NAMES})

    def enabled(self, name: str) -> bool:
        """Is this alert on?"""
        with self._lock:
            return self.on.get(name, False)

    def set(self, name: str, value: bool) -> None:
        """Switch one alert on or off."""
        if name not in AUTO_NAMES:
            raise ValueError(f"unknown alert {name!r} (one of {', '.join(AUTO_NAMES)})")
        with self._lock:
            self.on[name] = value

    def as_dict(self) -> dict[str, bool]:
        """All switches (for the page)."""
        with self._lock:
            return dict(self.on)


class WrongLaneWatch:
    """Navigating in a lane that doesn't work for the next maneuver."""

    def __init__(self, cfg: AutoAlertsConfig) -> None:
        self.cfg = cfg
        self._step: int | None = None
        self._wrong_since: float | None = None
        self._warned = 0  # warnings given for this maneuver
        self._first_at_m: float | None = None  # distance to the maneuver at the first warning
        self._confirmed = True  # "That's the right lane." said (or not owed)

    def update(self, t: float, target: dict[str, Any] | None, step: int | None, to_next_m: float | None) -> str | None:
        """One tick. `target`: `lane_target`'s answer ({"lane": "ours"} or {"lane": side, "moves": n}, None when it
        can't tell); `step`: the next maneuver's index; `to_next_m`: our distance to it. What to say, or None."""
        if target is None or step is None or to_next_m is None:
            self._wrong_since = None
            return None
        if step != self._step:
            self._step, self._warned, self._first_at_m, self._confirmed = step, 0, None, True
            self._wrong_since = None
        if target.get("lane") == "ours":
            self._wrong_since = None
            if not self._confirmed:
                self._confirmed = True
                return RIGHT_LANE
            return None
        side, moves = str(target.get("lane")), int(target.get("moves") or 1)
        if self._wrong_since is None:
            self._wrong_since = t
        cfg = self.cfg
        if t - self._wrong_since < cfg.wrong_lane_hold_s or to_next_m > cfg.wrong_lane_within_m:
            return None
        if self._warned == 0:
            self._warned, self._first_at_m, self._confirmed = 1, to_next_m, False
            return wrong_lane_text(side, moves)
        repeat = cfg.wrong_lane_repeat_m
        first = self._first_at_m if self._first_at_m is not None else 0.0
        if self._warned == 1 and repeat > 0 and to_next_m <= repeat < first:
            self._warned, self._confirmed = 2, False
            return wrong_lane_text(side, moves, again=True)
        return None


class SpeedingWatch:
    """Over the posted speed limit."""

    def __init__(self, cfg: AutoAlertsConfig) -> None:
        self.cfg = cfg
        self._limit: int | None = None
        self._over_since: float | None = None
        self._slow_since: float | None = None
        self._said = False

    def update(self, t: float, speed_mps: float | None, limit: int | None, limit_t: float | None) -> str | None:
        """One tick: our GPS speed, the last limit the sign detector read and when (media time). What to say, or
        None."""
        cfg = self.cfg
        if limit is None or limit_t is None or t - limit_t > cfg.speeding_limit_max_age_s:
            self._limit, self._over_since, self._slow_since, self._said = None, None, None, False
            return None
        if limit != self._limit:
            self._limit, self._over_since, self._slow_since, self._said = limit, None, None, False
        if speed_mps is None:
            self._over_since = None
            return None
        mph = speed_mps * MPS_TO_MPH
        if mph > limit + cfg.speeding_margin_mph:
            self._slow_since = None
            if self._over_since is None:
                self._over_since = t
            if not self._said and t - self._over_since >= cfg.speeding_hold_s:
                self._said = True
                return speeding_text(limit)
            return None
        self._over_since = None
        if mph <= limit:
            if self._slow_since is None:
                self._slow_since = t
            if self._said and t - self._slow_since >= cfg.speeding_repeat_s:
                self._said = False
        return None
