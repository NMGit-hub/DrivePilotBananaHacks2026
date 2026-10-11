"""Heads-up events the world state raises for spoken alerts nobody asked for (`alerts.auto`):

- `CarAheadWatch`: we have been stopped behind a car that was itself stopped in front of us for
  `car_ahead_wait_s`, it pulls away (its box shrinks by `car_ahead_shrink` from the biggest it was while we
  waited), and we are still stopped `car_ahead_delay_s` later: "The car ahead is moving." Once per stop. A car that
  left the lead zone after it started moving still counts (it drove off); a different car becoming the lead before
  ours moved (a cut-in, a lost track) starts over. Never at a red light: if our light read red or yellow at any time
  during this stop, nothing is said ("Light's green." is the alert there; a car pulling away on red may be turning
  right, and "the car ahead is moving" must not sound like "go"). Seen on the SF replay before these two rules: a
  car that crossed in front of us and drove off up the street was announced while we waited first at a red light.
- `StopSignWatch`: driving (GPS speed at least `stop_sign_min_speed_mps`, or ego motion moving when there's no
  GPS) toward a stop sign that is close (`stop_sign_min_height` of the frame), on our side of the road (right
  of `stop_sign_min_x`) and facing us: its box about square, and its face red (`face_redness`: the middle of
  the box clearly redder than green). YOLO also finds the grey backs of the cross street's signs (seen on the SF
  replay: "Stop sign ahead." for a sign that wasn't ours). Faces on earlier runs' frames, washed-out pink to
  maroon at dawn, measured +15..+77; backs -16..-5. "Stop sign ahead." Once per sign, and not again within
  `stop_sign_repeat_s`.

Both are fed by `WorldState.on_perception` and tested on their own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.config import AutoAlertsConfig
from perception.detector import Track

MIN_SIGN_ASPECT, MAX_SIGN_ASPECT = 0.65, 1.5  # width / height of an octagon's box (with some slack)


def face_redness(image: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """How much redder than green the middle of a sign's box is (median R - G, 0-255 scale; BGR image). A stop
    sign's face is red (pink when washed out, maroon at dawn: still red above green); its back is grey (about 0)."""
    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, y1 = max(0, int(box[0] + bw / 5)), max(0, int(box[1] + bh / 5))  # the middle 60 %: no sky or pole
    x2, y2 = min(w, round(box[2] - bw / 5)), min(h, round(box[3] - bh / 5))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    inner = image[y1:y2, x1:x2].astype(np.int16)
    return float(np.median(inner[..., 2] - inner[..., 1]))


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
        self._red_seen = False  # our light read red or yellow during this stop

    def update(
        self,
        t: float,
        stopped: bool,
        stopped_since: float | None,
        confident: bool,
        lead: tuple[int, float, float | None] | None,
        red_light: bool = False,
    ) -> bool:
        """One frame. `lead`: (track id, box height px, since when it has been stopped or None while it moves) of
        the car ahead in our lane, or None; `red_light`: our light reads red or yellow now. True when the alert is
        due (once per stop)."""
        if not stopped or not confident or stopped_since is None:
            self.reset()
            return False
        self._red_seen = self._red_seen or red_light
        if self._fired or self._red_seen:
            return False
        wait = self.cfg.car_ahead_wait_s
        if lead is not None and lead[0] != self._track and self._moving_since is None:
            # A (new) car ahead: follow it once we've both waited, it in front of us (cross traffic never stops
            # there; a cut-in or a re-found track starts over).
            self._track, self._tallest = None, 0.0
            lead_since = lead[2]
            if t - stopped_since >= wait and lead_since is not None and t - lead_since >= wait:
                self._track, self._tallest = lead[0], lead[1]
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
        image: np.ndarray,
    ) -> int | None:
        """One frame (`image`: BGR): the track id of a stop sign to announce now, or None."""
        cfg = self.cfg
        height, width = image.shape[:2]
        fast_enough = speed_mps >= cfg.stop_sign_min_speed_mps if speed_mps is not None else moving
        if not fast_enough or width <= 0 or height <= 0:
            return None
        for sign in signs:
            if sign.track_id in self._announced:
                continue
            if sign.height / height < cfg.stop_sign_min_height or sign.center[0] / width < cfg.stop_sign_min_x:
                continue
            if not MIN_SIGN_ASPECT <= sign.width / max(sign.height, 1.0) <= MAX_SIGN_ASPECT:
                continue  # a sign with its pole, a sliver: not a clear view of the face
            if face_redness(image, sign.box) < cfg.stop_sign_min_redness:
                continue  # the back of a sign (grey), facing the cross street
            self._announced.add(sign.track_id)
            if len(self._announced) > 200:
                self._announced = {sign.track_id}
            if self._last is not None and 0.0 <= t - self._last < cfg.stop_sign_repeat_s:
                return None  # (the same intersection's other sign, or one just after)
            self._last = t
            return sign.track_id
        return None
