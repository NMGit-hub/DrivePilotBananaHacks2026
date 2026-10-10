"""GPS input: a GPX track replayed in sync with the video, or live fixes from the phone's web page.

`GpxReplay.fix_at(t)` interpolates the recorded track at video time `t`. The GPX clock is aligned with
`video.file.gpx_offset_s` = GPX time at video t=0 minus the first point's time, or, when the file has a waypoint
named `video_start` (written by `scripts/bdd_info_to_gpx.py` and `scripts/route_to_gpx.py`), video t=0 is that
waypoint's time (plus `gpx_offset_s` as a further correction). Speed and heading come from the file when
recorded, otherwise from the neighbouring points; nothing is interpolated across track segments (a gap in the
recording is reported as no fix). `write_gpx` writes such files (GPX 1.0, which has <speed> and <course>).
`PhoneGps` keeps the latest fix posted by the phone (M7 serves the page over `adb reverse`) and reports nothing
once it is stale.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from nav.geo import LatLon, bearing_deg, haversine_m

VIDEO_START = "video_start"  # name of a GPX waypoint whose time is the video's first frame


class GpxPoint(Protocol):
    """A timed position for `write_gpx` (read-only fields, so frozen dataclasses fit)."""

    @property
    def time(self) -> datetime:
        """When (timezone-aware)."""

    @property
    def lat(self) -> float:
        """Latitude in degrees."""

    @property
    def lon(self) -> float:
        """Longitude in degrees."""

    @property
    def speed_mps(self) -> float | None:
        """Speed in m/s (None: unknown)."""

    @property
    def course_deg(self) -> float | None:
        """Course over ground, degrees clockwise from north (None: unknown)."""


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """A plain `GpxPoint`."""

    time: datetime
    lat: float
    lon: float
    speed_mps: float | None = None
    course_deg: float | None = None


def write_gpx(
    segments: Sequence[Sequence[GpxPoint]], video_start: datetime, name: str, description: str, track_name: str
) -> str:
    """GPX 1.0 text: one track with a segment per list of points (a replay never joins two segments), and a
    `video_start` waypoint at the first point's position whose time is the video's t=0."""
    import gpxpy.gpx

    if not segments or not segments[0]:
        raise ValueError("a GPX track needs at least one point")
    gpx = gpxpy.gpx.GPX()
    gpx.name = name
    gpx.description = description
    first = segments[0][0]
    gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(first.lat, first.lon, time=video_start, name=VIDEO_START))
    gpx_track = gpxpy.gpx.GPXTrack(name=track_name)
    gpx.tracks.append(gpx_track)
    for points in segments:
        segment = gpxpy.gpx.GPXTrackSegment()
        for p in points:
            point = gpxpy.gpx.GPXTrackPoint(p.lat, p.lon, time=p.time, speed=p.speed_mps)
            point.course = p.course_deg  # type: ignore[assignment]  # (gpxpy annotates it None; GPX 1.0 writes it)
            segment.points.append(point)
        gpx_track.segments.append(segment)
    return str(gpx.to_xml(version="1.0"))


@dataclass(frozen=True, slots=True)
class GpsFix:
    """One position."""

    t: float  # time on the app's clock (video time in file mode, wall clock live)
    lat: float
    lon: float
    speed_mps: float | None = None
    heading: float | None = None  # degrees clockwise from north
    accuracy_m: float | None = None

    @property
    def latlon(self) -> LatLon:
        """(lat, lon)."""
        return self.lat, self.lon

    @property
    def speed_mph(self) -> float | None:
        """Speed in miles per hour."""
        return None if self.speed_mps is None else self.speed_mps * 2.23694


class GpsSource(Protocol):
    """Where fixes come from."""

    def fix_at(self, t: float) -> GpsFix | None:
        """The position at app time `t`, or None if unknown."""


class GpxReplay:
    """A recorded GPX track, played back on the video's clock."""

    def __init__(self, path: Path, offset_s: float = 0.0) -> None:
        import gpxpy

        with path.open(encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        points, segment_of = [], []  # segment_of[i]: which track segment point i is in
        segments = [s for track in gpx.tracks for s in track.segments]
        for index, segment in enumerate(segments):
            timed = [p for p in segment.points if p.time is not None]
            points += timed
            segment_of += [index] * len(timed)
        if len(points) < 2:
            raise ValueError(f"{path}: needs at least two timed track points")
        anchors = [w.time for w in gpx.waypoints if w.name == VIDEO_START and w.time is not None]
        t0 = anchors[0] if anchors else points[0].time
        assert t0 is not None
        self._t = np.array([(p.time - t0).total_seconds() for p in points if p.time is not None]) - offset_s
        self._segment = np.array(segment_of)
        self._lat = np.array([p.latitude for p in points])
        self._lon = np.array([p.longitude for p in points])
        self._speed = np.array([p.speed if p.speed is not None else np.nan for p in points], dtype=float)
        courses = [getattr(p, "course", None) for p in points]  # (0.0 is due north, not "unknown")
        self._course = np.array([c if c is not None else np.nan for c in courses], dtype=float)

    @classmethod
    def from_points(cls, times: list[float], points: list[LatLon], speeds: list[float] | None = None) -> GpxReplay:
        """A replay from raw arrays (tests and synthetic tracks)."""
        replay = cls.__new__(cls)
        replay._t = np.asarray(times, dtype=float)
        replay._lat = np.array([p[0] for p in points])
        replay._lon = np.array([p[1] for p in points])
        replay._speed = np.asarray(speeds if speeds is not None else [np.nan] * len(points), dtype=float)
        replay._course = np.full(len(points), np.nan)
        replay._segment = np.zeros(len(points), dtype=int)
        return replay

    @property
    def duration_s(self) -> float:
        """Track length in time."""
        return float(self._t[-1] - self._t[0])

    def fix_at(self, t: float) -> GpsFix | None:
        """Interpolated position at video time `t` (None outside the track or in a gap between its segments)."""
        if t < self._t[0] or t > self._t[-1]:
            return None
        i = int(np.searchsorted(self._t, t, side="right") - 1)
        i = min(max(i, 0), len(self._t) - 2)
        t0, t1 = self._t[i], self._t[i + 1]
        if self._segment[i] != self._segment[i + 1]:  # between two segments: the recording has a gap here
            if t != t0:
                return None
            speed_i = None if np.isnan(self._speed[i]) else float(self._speed[i])
            course_i = None if np.isnan(self._course[i]) else float(self._course[i])
            return GpsFix(t, float(self._lat[i]), float(self._lon[i]), speed_i, course_i)
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        lat = float(self._lat[i] + (self._lat[i + 1] - self._lat[i]) * f)
        lon = float(self._lon[i] + (self._lon[i + 1] - self._lon[i]) * f)
        a, b = (self._lat[i], self._lon[i]), (self._lat[i + 1], self._lon[i + 1])
        distance = haversine_m(a, b)
        s0, s1 = self._speed[i], self._speed[i + 1]
        if not np.isnan(s0) and not np.isnan(s1):  # interpolated like the position (holding s0 lagged up to 1 s)
            speed = float(s0 + (s1 - s0) * f)
        elif not np.isnan(s0):
            speed = float(s0)
        else:
            speed = distance / (t1 - t0) if t1 > t0 else 0.0
        heading = (
            float(self._course[i]) if not np.isnan(self._course[i]) else (bearing_deg(a, b) if distance > 0.5 else None)
        )
        return GpsFix(t, lat, lon, speed, heading)


class PhoneGps:
    """The latest fix posted by the phone's GPS page (thread-safe)."""

    def __init__(self, stale_s: float = 3.0, clock: Callable[[], float] = time.perf_counter) -> None:
        self.stale_s = stale_s
        self._clock = clock
        self._lock = threading.Lock()
        self._fix: GpsFix | None = None
        self._received_at = 0.0
        self.received = 0

    def push(
        self,
        lat: float,
        lon: float,
        speed_mps: float | None = None,
        heading: float | None = None,
        accuracy_m: float | None = None,
    ) -> None:
        """A fix from the phone (HTTP handler thread)."""
        now = self._clock()
        with self._lock:
            self._fix = GpsFix(now, lat, lon, speed_mps, heading, accuracy_m)
            self._received_at = now
            self.received += 1

    def fix_at(self, t: float) -> GpsFix | None:
        """The latest fix if it is fresh (live mode: `t` is ignored, the phone's clock is now)."""
        with self._lock:
            if self._fix is None or self._clock() - self._received_at > self.stale_s:
                return None
            return self._fix
