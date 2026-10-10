"""The route model, parsed from a Mapbox Directions v5 response (driving-traffic, steps + lanes + banners).

PLAN.md "Getting the route": the app keeps the step list: each maneuver, where it is along the route, its lanes
(left to right, which are valid for our route) and, for roundabouts, the exit number. Positions are metres
along the route line from the start, so progress is one number.

Mapbox details used here:
- step i's maneuver happens where step i begins; its first intersection is the maneuver point and its `lanes`
  are the lanes approaching that point, left to right, `valid` for the ones our route can use;
- with `roundabout_exits=true` a roundabout/rotary step covers just the ring (an "exit roundabout" step starts
  where we leave it); `maneuver.exit` is the exit to take, and the ring's later intersections that have a
  road we could leave onto are the exits we pass first (roads that only feed into the ring don't count);
- highway exits carry `exits` (the exit number on the sign) and `destinations` ("I-405 N: Everett").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nav.geo import LatLon, Polyline

ROUNDABOUT_TYPES = {"roundabout", "rotary", "roundabout turn"}


class RouteError(ValueError):
    """The directions response can't be used."""


@dataclass(frozen=True, slots=True)
class Lane:
    """One lane at a maneuver, as Mapbox reports it (left to right)."""

    indications: tuple[str, ...]  # "left", "straight", "slight right", ...
    valid: bool  # usable for our route's maneuver


@dataclass(frozen=True, slots=True)
class Step:
    """One maneuver and the road after it."""

    index: int
    maneuver: str  # Mapbox maneuver type: "turn", "off ramp", "roundabout", "arrive", ...
    modifier: str | None  # "left", "slight right", ...
    instruction: str  # Mapbox's text, e.g. "Turn left onto 148th Avenue Northeast"
    name: str  # road after the maneuver
    ref: str | None  # road number, e.g. "SR 520"
    destinations: str | None  # signed destinations of a ramp
    exit_ref: str | None  # highway exit number on the sign, e.g. "14B"
    roundabout_exit: int | None  # which roundabout exit to take
    start_m: float  # where the maneuver happens, metres along the route
    distance_m: float  # length of this step
    duration_s: float
    location: LatLon
    lanes: tuple[Lane, ...] = ()
    roundabout_exits_m: tuple[float, ...] = ()  # positions of the exits passed in the roundabout (ours last)

    @property
    def is_roundabout(self) -> bool:
        """A roundabout or rotary entry."""
        return self.maneuver in ROUNDABOUT_TYPES

    @property
    def needs_lanes(self) -> bool:
        """Only some lanes lead to this maneuver."""
        return (
            bool(self.lanes) and not all(lane.valid for lane in self.lanes) and any(lane.valid for lane in self.lanes)
        )


@dataclass(slots=True)
class Route:
    """A route to a destination."""

    line: Polyline
    steps: list[Step]
    distance_m: float
    duration_s: float
    destination: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def current_step(self, along_m: float) -> Step:
        """The step we are on (the last maneuver at or before `along_m`)."""
        current = self.steps[0]
        for step in self.steps:
            if step.start_m <= along_m + 1.0:
                current = step
            else:
                break
        return current

    def next_maneuver(self, along_m: float, after_m: float = 5.0) -> Step | None:
        """The first maneuver ahead (more than `after_m` past us)."""
        return next((s for s in self.steps if s.start_m > along_m + after_m), None)

    def upcoming(self, along_m: float, within_m: float) -> list[Step]:
        """Maneuvers ahead within a distance."""
        return [s for s in self.steps if along_m < s.start_m <= along_m + within_m]


def _is_exit(intersection: dict[str, Any]) -> bool:
    """A point on the ring where we could leave it: some road other than the way in and the way on accepts
    traffic (roads that only feed into the roundabout don't count as exits)."""
    entries = intersection.get("entry") or []
    used = {intersection.get("in"), intersection.get("out")}
    return any(ok for i, ok in enumerate(entries) if i not in used)


def _latlon(coordinate: list[float]) -> LatLon:
    lon, lat = coordinate[:2]
    return float(lat), float(lon)


def parse_directions(response: dict[str, Any], destination: str = "") -> Route:
    """Build a `Route` from a Mapbox Directions v5 JSON response (geometries=geojson, steps=true)."""
    if response.get("code") not in (None, "Ok"):
        raise RouteError(f"directions failed: {response.get('code')}: {response.get('message', '')}")
    routes = response.get("routes") or []
    if not routes:
        raise RouteError("no route found")
    route = routes[0]
    coordinates = (route.get("geometry") or {}).get("coordinates") or []
    if len(coordinates) < 2:
        raise RouteError("route has no geometry (request geometries=geojson&overview=full)")
    line = Polyline([_latlon(c) for c in coordinates])
    raw_steps = [s for leg in route.get("legs", []) for s in leg.get("steps", [])]
    if not raw_steps:
        raise RouteError("route has no steps (request steps=true)")

    steps: list[Step] = []
    start = 0.0
    for i, s in enumerate(raw_steps):
        maneuver = s.get("maneuver", {})
        location = _latlon(maneuver.get("location", coordinates[0]))
        # Mapbox's step distances add up to the route length, but snap the maneuver point onto the line to avoid
        # rounding drift over a long route.
        start_m = line.snap(location, near_m=start, window_m=300.0).along_m if i else 0.0
        intersections = s.get("intersections") or []
        lanes: tuple[Lane, ...] = ()
        if intersections and intersections[0].get("lanes"):
            lanes = tuple(
                Lane(tuple(lane.get("indications") or ()), bool(lane.get("valid")))
                for lane in intersections[0]["lanes"]
            )
        exits_m: tuple[float, ...] = ()
        if maneuver.get("type") in ROUNDABOUT_TYPES:
            exits_m = tuple(
                line.snap(_latlon(x["location"]), near_m=start_m, window_m=300.0).along_m
                for x in intersections[1:]
                if _is_exit(x)
            )
        steps.append(
            Step(
                index=i,
                maneuver=str(maneuver.get("type", "")),
                modifier=maneuver.get("modifier"),
                instruction=str(maneuver.get("instruction", "")),
                name=str(s.get("name", "")),
                ref=s.get("ref") or None,
                destinations=s.get("destinations") or None,
                exit_ref=s.get("exits") or None,
                roundabout_exit=maneuver.get("exit"),
                start_m=start_m,
                distance_m=float(s.get("distance", 0.0)),
                duration_s=float(s.get("duration", 0.0)),
                location=location,
                lanes=lanes,
                roundabout_exits_m=exits_m,
            )
        )
        start = start_m + float(s.get("distance", 0.0))
    name = destination or str((response.get("waypoints") or [{}])[-1].get("name", ""))
    return Route(
        line, steps, float(route.get("distance", line.length_m)), float(route.get("duration", 0.0)), name, response
    )
