"""GPS input: a GPX track replayed in sync with the video, or live fixes from the phone's web page.

`GpxReplay.fix_at(t)` interpolates the recorded track at video time `t` (the GPX clock is aligned with
`video.file.gpx_offset_s` = GPX time at video t=0 minus the first point's time); speed and heading come from the
file when recorded, otherwise from the neighbouring points. `PhoneGps` keeps the latest fix posted by the phone
(M7 serves the page over `adb reverse`) and reports nothing once it is stale.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nav.geo import LatLon, bearing_deg, haversine_m


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
        points = [p for track in gpx.tracks for segment in track.segments for p in segment.points]
        points = [p for p in points if p.time is not None]
        if len(points) < 2:
            raise ValueError(f"{path}: needs at least two timed track points")
        t0 = points[0].time
        assert t0 is not None
        self._t = np.array([(p.time - t0).total_seconds() for p in points if p.time is not None]) - offset_s
        self._lat = np.array([p.latitude for p in points])
        self._lon = np.array([p.longitude for p in points])
        self._speed = np.array([p.speed if p.speed is not None else np.nan for p in points], dtype=float)
        self._course = np.array([getattr(p, "course", None) or np.nan for p in points], dtype=float)

    @classmethod
    def from_points(cls, times: list[float], points: list[LatLon], speeds: list[float] | None = None) -> GpxReplay:
        """A replay from raw arrays (tests and synthetic tracks)."""
        replay = cls.__new__(cls)
        replay._t = np.asarray(times, dtype=float)
        replay._lat = np.array([p[0] for p in points])
        replay._lon = np.array([p[1] for p in points])
        replay._speed = np.asarray(speeds if speeds is not None else [np.nan] * len(points), dtype=float)
        replay._course = np.full(len(points), np.nan)
        return replay

    @property
    def duration_s(self) -> float:
        """Track length in time."""
        return float(self._t[-1] - self._t[0])

    def fix_at(self, t: float) -> GpsFix | None:
        """Interpolated position at video time `t` (None outside the track)."""
        if t < self._t[0] or t > self._t[-1]:
            return None
        i = int(np.searchsorted(self._t, t, side="right") - 1)
        i = min(max(i, 0), len(self._t) - 2)
        t0, t1 = self._t[i], self._t[i + 1]
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        lat = float(self._lat[i] + (self._lat[i + 1] - self._lat[i]) * f)
        lon = float(self._lon[i] + (self._lon[i + 1] - self._lon[i]) * f)
        a, b = (self._lat[i], self._lon[i]), (self._lat[i + 1], self._lon[i + 1])
        distance = haversine_m(a, b)
        speed = float(self._speed[i]) if not np.isnan(self._speed[i]) else (distance / (t1 - t0) if t1 > t0 else 0.0)
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
