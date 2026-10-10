"""A hand-made route in Mapbox Directions v5 format (no stored Mapbox data), and synthetic GPS tracks along it.

Start → 800 m east → turn left onto 148th Avenue Northeast (only the left lane turns) → 2400 m north → exit 14
onto SR 520 East (two right lanes of four) → 2000 m → roundabout, 2nd exit onto Main Street (one exit passed) →
300 m → arrive. Total ~5.53 km.
"""

from __future__ import annotations

from typing import Any

from nav.geo import LatLon, Polyline, destination

START: LatLon = (47.6100, -122.2000)


def _leg(points: list[LatLon], bearing: float, metres: float, step_m: float = 50.0) -> None:
    for _ in range(round(metres / step_m)):
        points.append(destination(points[-1], bearing, step_m))


def _lanes(pattern: str, valid_ind: str) -> list[dict[str, Any]]:
    return [
        {"valid": c == "V", "active": False, "indications": [valid_ind if c == "V" else "straight"]} for c in pattern
    ]


def _lonlat(p: LatLon) -> list[float]:
    return [p[1], p[0]]


def build() -> tuple[dict[str, Any], Polyline]:
    """The directions response and its line."""
    pts: list[LatLon] = [START]
    _leg(pts, 90.0, 800.0)  # east
    p1 = pts[-1]
    _leg(pts, 0.0, 2400.0)  # north on 148th
    p2 = pts[-1]
    _leg(pts, 45.0, 2000.0)  # northeast on SR 520
    p3 = pts[-1]
    _leg(pts, 60.0, 30.0, step_m=5.0)  # round the ring
    exit_passed = Polyline(pts).point_at(Polyline(pts).length_m - 15.0)
    p4 = pts[-1]
    _leg(pts, 90.0, 300.0)  # east on Main Street
    p5 = pts[-1]
    line = Polyline(pts)

    def step(
        kind: str, where: LatLon, distance: float, instruction: str, name: str = "", **extra: Any
    ) -> dict[str, Any]:
        maneuver = {"type": kind, "location": _lonlat(where), "instruction": instruction}
        if "modifier" in extra:
            maneuver["modifier"] = extra.pop("modifier")
        if "exit" in extra:
            maneuver["exit"] = extra.pop("exit")
        intersections = extra.pop("intersections", [{"location": _lonlat(where), "entry": [True], "out": 0}])
        return {
            "maneuver": maneuver,
            "distance": distance,
            "duration": distance / 13.4,
            "name": name,
            "intersections": intersections,
            **extra,
        }

    steps = [
        step("depart", START, 800.0, "Drive east on Northeast 8th Street.", "Northeast 8th Street"),
        step(
            "turn",
            p1,
            2400.0,
            "Turn left onto 148th Avenue Northeast.",
            "148th Avenue Northeast",
            modifier="left",
            intersections=[
                {"location": _lonlat(p1), "lanes": _lanes("Vxx", "left"), "entry": [True, True], "in": 0, "out": 1}
            ],
        ),
        step(
            "off ramp",
            p2,
            2000.0,
            "Take exit 14 onto SR 520 East toward Redmond.",
            "SR 520",
            modifier="slight right",
            ref="SR 520",
            destinations="SR 520 East: Redmond",
            exits="14",
            intersections=[
                {
                    "location": _lonlat(p2),
                    "lanes": _lanes("xxVV", "slight right"),
                    "entry": [True, True],
                    "in": 0,
                    "out": 1,
                }
            ],
        ),
        step(
            "roundabout",
            p3,
            30.0,
            "Enter the roundabout and take the 2nd exit onto Main Street.",
            "Main Street",
            modifier="right",
            exit=2,
            intersections=[
                {"location": _lonlat(p3), "in": 0, "out": 2, "entry": [False, False, True]},
                {"location": _lonlat(exit_passed), "in": 0, "out": 1, "entry": [False, True, True]},
            ],
        ),
        step("exit roundabout", p4, 300.0, "Exit the roundabout onto Main Street.", "Main Street", modifier="right"),
        step("arrive", p5, 0.0, "You have arrived at your destination.", "Main Street"),
    ]
    response = {
        "code": "Ok",
        "routes": [
            {
                "distance": line.length_m,
                "duration": line.length_m / 13.4,
                "geometry": {"coordinates": [_lonlat(p) for p in pts]},
                "legs": [{"steps": steps}],
            }
        ],
        "waypoints": [{"name": "Northeast 8th Street"}, {"name": "Main Street"}],
    }
    return response, line


def track(
    line: Polyline,
    speed_mps: float = 15.0,
    start_m: float = 0.0,
    until_m: float | None = None,
    t0: float = 0.0,
    off_route: tuple[float, float, float] | None = None,
) -> list[tuple[float, LatLon, float]]:
    """GPS fixes once a second: (t, position, heading). `off_route` = (from_s, to_s, metres to the side)."""
    end = line.length_m if until_m is None else until_m
    out = []
    t, along = t0, start_m
    while along <= end:
        point = line.point_at(along)
        heading = line.bearing_at(along)
        if off_route is not None and off_route[0] <= t - t0 < off_route[1]:
            point = destination(point, (heading + 90.0) % 360.0, off_route[2])
        out.append((t, point, heading))
        t += 1.0
        along += speed_mps
    return out
