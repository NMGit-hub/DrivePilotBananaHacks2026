"""Small geodesy for navigation: distances, bearings, snapping a GPS fix onto the route polyline.

Routes are short (city driving), so each snap works in a local equirectangular projection around the fix:
metres east/north of it, accurate to well under a metre over a few kilometres. Points are (lat, lon) in degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
LatLon = tuple[float, float]


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: LatLon, b: LatLon) -> float:
    """Initial bearing from a to b, degrees clockwise from north (0..360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings (0..180)."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def destination(origin: LatLon, bearing: float, distance_m: float) -> LatLon:
    """The point `distance_m` from `origin` along `bearing` (for synthetic GPS tracks)."""
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    theta, delta = math.radians(bearing), distance_m / EARTH_RADIUS_M
    lat2 = math.asin(math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(theta))
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(lat1), math.cos(delta) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class Snap:
    """A fix projected onto a polyline."""

    along_m: float  # distance from the start of the line to the projected point
    offset_m: float  # distance from the fix to the line
    segment: int  # index of the segment (point i to i+1) it projected onto
    point: LatLon  # the projected point
    segment_bearing: float  # direction of that segment


class Polyline:
    """A route line with cumulative distances, for fast snapping."""

    def __init__(self, points: Sequence[LatLon]) -> None:
        if len(points) < 2:
            raise ValueError("a polyline needs at least two points")
        self.points = [(float(lat), float(lon)) for lat, lon in points]
        self._lat = np.array([p[0] for p in self.points])
        self._lon = np.array([p[1] for p in self.points])
        steps = [haversine_m(self.points[i], self.points[i + 1]) for i in range(len(self.points) - 1)]
        self.cumulative = np.concatenate(([0.0], np.cumsum(steps)))

    @property
    def length_m(self) -> float:
        """Total length."""
        return float(self.cumulative[-1])

    def point_at(self, along_m: float) -> LatLon:
        """The point at a distance along the line (clamped to the ends)."""
        along_m = min(max(along_m, 0.0), self.length_m)
        i = int(np.searchsorted(self.cumulative, along_m, side="right") - 1)
        i = min(i, len(self.points) - 2)
        seg = self.cumulative[i + 1] - self.cumulative[i]
        f = 0.0 if seg <= 0 else (along_m - self.cumulative[i]) / seg
        a, b = self.points[i], self.points[i + 1]
        return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f

    def bearing_at(self, along_m: float) -> float:
        """Direction of travel at a distance along the line."""
        i = int(np.searchsorted(self.cumulative, min(max(along_m, 0.0), self.length_m), side="right") - 1)
        i = min(max(i, 0), len(self.points) - 2)
        return bearing_deg(self.points[i], self.points[i + 1])

    def snap(self, fix: LatLon, near_m: float | None = None, window_m: float = 400.0) -> Snap:
        """Project a fix onto the line. With `near_m`, only segments within `window_m` of that distance along
        the line are considered (keeps a fix on the right part of a route that passes the same place twice)."""
        lat0 = math.radians(fix[0])
        kx = math.radians(1.0) * EARTH_RADIUS_M * math.cos(lat0)  # metres per degree of longitude here
        ky = math.radians(1.0) * EARTH_RADIUS_M
        x = (self._lon - fix[1]) * kx
        y = (self._lat - fix[0]) * ky
        x0, y0, x1, y1 = x[:-1], y[:-1], x[1:], y[1:]
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(np.where(length2 > 0, -(x0 * dx + y0 * dy) / length2, 0.0), 0.0, 1.0)
        px, py = x0 + t * dx, y0 + t * dy
        dist = np.hypot(px, py)
        if near_m is not None:
            starts = self.cumulative[:-1]
            ends = self.cumulative[1:]
            outside = (ends < near_m - window_m) | (starts > near_m + window_m)
            if not outside.all():
                dist = np.where(outside, np.inf, dist)
        i = int(np.argmin(dist))
        along = float(self.cumulative[i] + t[i] * (self.cumulative[i + 1] - self.cumulative[i]))
        point = (fix[0] + py[i] / ky, fix[1] + px[i] / kx)
        return Snap(along, float(dist[i]), i, point, bearing_deg(self.points[i], self.points[i + 1]))
