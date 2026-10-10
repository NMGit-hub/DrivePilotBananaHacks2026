"""Route progress: snap each GPS fix to the route, distance to the next maneuver, off route → reroute.

PLAN.md "Tracking progress": each GPS point is snapped to the route line; more than ~50 m off the route for 5 s
means we left it (reroute and say so). Details that keep this honest:
- Snapping only searches the stretch of route we could have driven since the last fix (speed × time, with a
  margin; a GPS gap widens it), so a noisy fix where the route crosses itself (a cloverleaf) can't jump us ahead.
  A match elsewhere on the route is only taken after a second fix agrees with it.
- A fix that moves us backwards a little (GPS jitter while stopped) is accepted; a big jump back is noise.
- Driving the wrong way along the route (heading against it at speed) counts as off route.
- Starting off the road (a parking lot) is not "off route": until we first reach the route, only moving well
  away from it is.
- Near the end, leaving the route means we turned in at the destination: that is arrival, not a reroute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.config import NavConfig
from nav.geo import angle_diff, haversine_m
from nav.gps import GpsFix
from nav.route import Route, Step

MAX_SPEED_MPS = 45.0  # ~100 mph: how far we might have gone when the fix has no speed
REACH_MARGIN_M = 40.0
NEAR_END_ALONG_M = 500.0  # "near the end" also needs this little route left (a route can start near its end)


@dataclass(frozen=True, slots=True)
class Progress:
    """Where we are on the route."""

    t: float
    along_m: float  # metres from the start of the route
    offset_m: float  # metres from the route line
    step: Step  # the step we are on
    next_step: Step | None  # the next maneuver ahead
    to_next_m: float | None  # metres to it
    remaining_m: float
    remaining_s: float  # Mapbox's duration, scaled to what is left
    off_route: bool  # confirmed: far from the route for long enough (never near the end)
    speed_mps: float | None
    arrived: bool
    joined: bool = True  # we have been on the route line (False while still leaving a parking lot)
    heading_diff: float | None = None  # our heading vs the route's direction here (degrees), when moving


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


class RouteTracker:
    """Follows one route."""

    def __init__(self, route: Route, cfg: NavConfig) -> None:
        self.route = route
        self.cfg = cfg
        self._along: float | None = None
        self._last_t: float | None = None
        self._off_since: float | None = None
        self._joined = False
        self._start_offset: float | None = None
        self._candidate: tuple[float, float] | None = None  # (along, t) of a far match awaiting confirmation
        self._end = route.line.points[-1]

    def _reach_m(self, fix: GpsFix, since_t: float | None) -> float:
        """How far along the route we could have driven since `since_t`."""
        if since_t is None:
            return self.cfg.snap_window_m
        speed = _finite(fix.speed_mps)
        speed = MAX_SPEED_MPS if speed is None else max(speed * 1.5, 5.0)
        return speed * max(fix.t - since_t, 0.0) + REACH_MARGIN_M

    def _confirmed(self, along: float, fix: GpsFix) -> bool:
        """A match far from where we were: take it only if the previous fix pointed to the same place."""
        candidate, self._candidate = self._candidate, (along, fix.t)
        if candidate is None:
            return False
        before, t = candidate
        return before - self.cfg.max_backtrack_m <= along <= before + self._reach_m(fix, t)

    def update(self, fix: GpsFix) -> Progress:
        """Advance with a fix."""
        cfg, line = self.cfg, self.route.line
        if self._along is None:
            snap = line.snap(fix.latlon)
        else:
            low = self._along - cfg.max_backtrack_m
            high = self._along + self._reach_m(fix, self._last_t)
            snap = line.snap(fix.latlon, near_m=(low + high) / 2, window_m=(high - low) / 2)
            if snap.offset_m > cfg.off_route_m:
                # Nothing close where we could be: after a GPS jump we may be elsewhere on the route.
                anywhere = line.snap(fix.latlon)
                if anywhere.offset_m <= cfg.off_route_m and self._confirmed(anywhere.along_m, fix):
                    snap, self._along = anywhere, anywhere.along_m
            else:
                self._candidate = None
        self._last_t = fix.t

        heading, speed = _finite(fix.heading), _finite(fix.speed_mps)
        moving = heading is not None and speed is not None and speed > 3.0
        heading_diff = angle_diff(heading, snap.segment_bearing) if moving and heading is not None else None
        wrong_way = heading_diff is not None and heading_diff > 120.0
        on_line = snap.offset_m <= cfg.off_route_m and not wrong_way
        if self._start_offset is None:
            self._start_offset = snap.offset_m
        if on_line:
            self._joined = True
            self._along = snap.along_m if self._along is None else max(snap.along_m, self._along - cfg.max_backtrack_m)
        limit = cfg.off_route_m if self._joined else max(cfg.off_route_m, self._start_offset) + cfg.join_slack_m
        # On the road going against the route counts at once; in a parking lot headings mean nothing.
        leaving = snap.offset_m > limit or (wrong_way and snap.offset_m <= cfg.off_route_m)
        if leaving:
            if self._off_since is None:
                self._off_since = fix.t
        else:
            self._off_since = None
        left_route = self._off_since is not None and fix.t - self._off_since >= cfg.off_route_s

        position = self._along if self._along is not None else snap.along_m
        remaining = max(line.length_m - position, 0.0)
        to_end = haversine_m(fix.latlon, self._end)
        near_end = to_end <= cfg.near_end_m and remaining <= NEAR_END_ALONG_M
        arrived = (
            (remaining <= cfg.arrival_m and snap.offset_m <= cfg.off_route_m)
            or to_end <= cfg.arrival_radius_m
            or (near_end and left_route)  # turned in at the destination
        )
        next_step = self.route.next_maneuver(position)
        fraction = remaining / line.length_m if line.length_m > 0 else 0.0
        return Progress(
            t=fix.t,
            along_m=position,
            offset_m=snap.offset_m,
            step=self.route.current_step(position),
            next_step=next_step,
            to_next_m=None if next_step is None else next_step.start_m - position,
            remaining_m=remaining,
            remaining_s=self.route.duration_s * fraction,
            off_route=left_route and not near_end,
            speed_mps=speed,
            arrived=arrived,
            joined=self._joined,
            heading_diff=heading_diff,
        )
