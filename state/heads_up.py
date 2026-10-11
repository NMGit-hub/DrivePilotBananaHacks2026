"""Heads-up events the world state raises for spoken alerts nobody asked for (`alerts.auto`):

- `CarAheadWatch`: we have been stopped behind a car for `car_ahead_wait_s`, it pulls away (its box shrinks by
  `car_ahead_shrink` from the biggest it was while we waited), and we are still stopped `car_ahead_delay_s` later:
  "The car ahead is moving." Once per stop. A car that left the lead zone after it started moving still counts
  (it drove off); a different car becoming the lead before ours moved (a cut-in, a lost track) starts over.
- `StopSignWatch`: driving (GPS speed at least `stop_sign_min_speed_mps`, or ego motion moving when there's no
  GPS) toward a stop sign that is close (`stop_sign_min_height` of the frame) and on our side of the road (right
  of `stop_sign_min_x`): "Stop sign ahead." Once per sign, and not again within `stop_sign_repeat_s`.

Both are pure (fed by `WorldState.on_perception`, tested on their own).
"""

from __future__ import annotations

from collections.abc import Sequence

from core.config import AutoAlertsConfig
from perception.detector import Track


class CarAheadWatch:
    """Stopped behind a car that pulls away while we stay put."""

    def __init__(self, cfg: AutoAlertsConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Forget the current stop."""
        self._track: int | None = None
        self._tallest = 0.0  # the lead car's biggest box height while we waited (px)
        self._moving_since: float | None = None
        self._fired = False

    def update(
        self,
        t: float,
        stopped: bool,
        stopped_since: float | None,
        confident: bool,
        lead: tuple[int, float] | None,
    ) -> bool:
        """One frame. `lead`: (track id, box height px) of the car ahead in our lane, or None. True when the alert
        is due (once per stop)."""
        if not stopped or not confident or stopped_since is None:
            self.reset()
            return False
        if self._fired:
            return False
        if lead is not None and lead[0] != self._track and self._moving_since is None:
            # A (new) car ahead: follow it once we've waited behind it (a cut-in or a re-found track starts over).
            self._track, self._tallest = None, 0.0
            if t - stopped_since >= self.cfg.car_ahead_wait_s:
                self._track, self._tallest = lead
            return False
        if self._track is None:
            return False
        if lead is not None and lead[0] == self._track:
            height = lead[1]
            if height > self._tallest:
                self._tallest = height
            if height <= self._tallest * (1.0 - self.cfg.car_ahead_shrink):
                if self._moving_since is None:
                    self._moving_since = t
            elif self._moving_since is not None and height > self._tallest * (1.0 - self.cfg.car_ahead_shrink / 2):
                self._moving_since = None  # it was a jittery box, not a car pulling away
        if self._moving_since is not None and t - self._moving_since >= self.cfg.car_ahead_delay_s:
            self._fired = True
            return True
        return False


class StopSignWatch:
    """Driving toward a stop sign on our side of the road."""

    def __init__(self, cfg: AutoAlertsConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Forget announced signs (new segment)."""
        self._announced: set[int] = set()
        self._last: float | None = None

    def update(
        self,
        t: float,
        moving: bool,
        speed_mps: float | None,
        signs: Sequence[Track],
        width: int,
        height: int,
    ) -> int | None:
        """One frame: the track id of a stop sign to announce now, or None."""
        cfg = self.cfg
        fast_enough = speed_mps >= cfg.stop_sign_min_speed_mps if speed_mps is not None else moving
        if not fast_enough or width <= 0 or height <= 0:
            return None
        for sign in signs:
            if sign.track_id in self._announced:
                continue
            if sign.height / height < cfg.stop_sign_min_height or sign.center[0] / width < cfg.stop_sign_min_x:
                continue
            self._announced.add(sign.track_id)
            if len(self._announced) > 200:
                self._announced = {sign.track_id}
            if self._last is not None and 0.0 <= t - self._last < cfg.stop_sign_repeat_s:
                return None  # (the same intersection's other sign, or one just after)
            self._last = t
            return sign.track_id
        return None
