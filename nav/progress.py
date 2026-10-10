"""Route progress: snap each GPS fix to the route, distance to the next maneuver, off route → reroute.

PLAN.md "Tracking progress": each GPS point is snapped to the route line; more than ~50 m off the route for 5 s
means we left it (reroute and say so). Snapping searches near where we were (the route may pass the same place
twice), and a fix that moves us a long way backwards is treated as noise rather than progress lost. Driving
the wrong way along the route (heading against it at speed) also counts as off route.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import NavConfig
from nav.geo import angle_diff
from nav.gps import GpsFix
from nav.route import Route, Step


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
    off_route: bool  # confirmed: far from the route for long enough
    speed_mps: float | None
    arrived: bool


class RouteTracker:
    """Follows one route."""

    def __init__(self, route: Route, cfg: NavConfig) -> None:
        self.route = route
        self.cfg = cfg
        self._along: float | None = None
        self._off_since: float | None = None

    def update(self, fix: GpsFix) -> Progress:
        """Advance with a fix."""
        cfg, route = self.cfg, self.route
        snap = route.line.snap(fix.latlon, near_m=self._along, window_m=cfg.snap_window_m)
        if self._along is not None and snap.offset_m > cfg.off_route_m:
            # Nothing close near where we were: after a GPS gap (tunnel, restart) we may be further along. If
            # the whole route has a close match, continue from there instead of calling it off route.
            anywhere = route.line.snap(fix.latlon)
            if anywhere.offset_m <= cfg.off_route_m:
                snap, self._along = anywhere, anywhere.along_m
        along = snap.along_m
        if self._along is not None and along < self._along - cfg.max_backtrack_m:
            along = self._along  # a wild jump backwards: GPS noise, keep our place
        wrong_way = (
            fix.heading is not None
            and fix.speed_mps is not None
            and fix.speed_mps > 3.0
            and angle_diff(fix.heading, snap.segment_bearing) > 120.0
        )
        if snap.offset_m > cfg.off_route_m or wrong_way:
            if self._off_since is None:
                self._off_since = fix.t
        else:
            self._off_since = None
            self._along = along if self._along is None else max(along, self._along - cfg.max_backtrack_m)
        off_route = self._off_since is not None and fix.t - self._off_since >= cfg.off_route_s
        position = self._along if self._along is not None else along
        next_step = route.next_maneuver(position)
        remaining = max(route.line.length_m - position, 0.0)
        fraction = remaining / route.line.length_m if route.line.length_m > 0 else 0.0
        return Progress(
            t=fix.t,
            along_m=position,
            offset_m=snap.offset_m,
            step=route.current_step(position),
            next_step=next_step,
            to_next_m=None if next_step is None else next_step.start_m - position,
            remaining_m=remaining,
            remaining_s=route.duration_s * fraction,
            off_route=off_route,
            speed_mps=fix.speed_mps,
            arrived=remaining <= cfg.arrival_m and snap.offset_m <= cfg.off_route_m,
        )
