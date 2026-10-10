"""BDD100K "info" files (GPS + IMU per clip) → GPX tracks that the app replays with the clip's video.

Each BDD100K clip has `bdd100k/info/100k/{train,val}/<clip id>.json` in `bdd100k_info.zip` (4 GB, 47.7 GB
unpacked, so entries are read straight from the zip). The fields used, checked on real files (2026-10-10):
- `startTime` / `endTime`: the video's first and last frame, Unix ms;
- `locations`: ~1 Hz fused fixes: `timestamp` (ms), `latitude`, `longitude`, `speed` (m/s), `course` (degrees),
  `accuracy` (m). -1 marks an unknown speed or course. A few files have no fixes, repeated or out-of-order
  timestamps, or gaps of up to ~23 s; the first fix is usually within ±1 s of the video start (up to ~20 s late);
- `timelapse`: true for time-lapse recordings, whose video time isn't real time (rejected);
- `gps` (raw fixes without course, only in some files) is not used.

The GPX (version 1.0, which has <speed> and <course>) carries a `video_start` waypoint at `startTime`, so
`nav.gps.GpxReplay` lines the track up with the video exactly. Fixes less accurate than `max_accuracy_m` are
dropped, and the track is split into segments where fixes are more than `gap_s` apart, so a replay never draws a
straight line through a gap.
"""

from __future__ import annotations

import itertools
import json
import math
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nav.gps import write_gpx

INFO_PREFIX = "bdd100k/info/100k/"
SPLITS = ("val", "train")
EDGE_S = 3.0  # fixes kept beyond the video's ends (interpolation at t=0 and at the end needs a neighbor)


class BddInfoError(ValueError):
    """An info file that can't become a usable track."""


@dataclass(frozen=True, slots=True)
class InfoFix:
    """One GPS fix from an info file."""

    time: datetime
    lat: float
    lon: float
    speed_mps: float | None  # None: unknown
    course_deg: float | None  # None: unknown
    accuracy_m: float | None


@dataclass(frozen=True, slots=True)
class InfoTrack:
    """A clip's GPS track, split at gaps, with the video's time span."""

    clip_id: str
    video_start: datetime
    video_end: datetime
    segments: list[list[InfoFix]]
    dropped: Counter[str] = field(default_factory=Counter)  # reason → fixes dropped

    @property
    def fixes(self) -> int:
        """Fixes kept."""
        return sum(len(s) for s in self.segments)

    @property
    def first_fix_after_start_s(self) -> float:
        """Seconds from the video's first frame to the first fix (negative: the fix is earlier)."""
        return (self.segments[0][0].time - self.video_start).total_seconds()


def _ms(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def read_info(zip_path: Path, clip_id: str) -> dict[str, Any]:
    """The info JSON of one clip (searched in the val and train splits) without unpacking the zip."""
    with zipfile.ZipFile(zip_path) as archive:
        for split in SPLITS:
            name = f"{INFO_PREFIX}{split}/{clip_id}.json"
            try:
                raw = archive.read(name)
            except KeyError:
                continue
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise BddInfoError(f"{name}: not a JSON object")
            return data
    raise BddInfoError(f"clip {clip_id} has no info file in {zip_path}")


def parse_info(info: Mapping[str, Any], clip_id: str, max_accuracy_m: float = 100.0, gap_s: float = 5.0) -> InfoTrack:
    """Validate and clean an info file's fixes (see the module docstring)."""
    if info.get("timelapse"):
        raise BddInfoError(f"{clip_id}: a time-lapse recording (video time isn't real time)")
    start, end = _number(info.get("startTime")), _number(info.get("endTime"))
    if start is None or end is None or end <= start:
        raise BddInfoError(f"{clip_id}: no valid startTime/endTime")
    dropped: Counter[str] = Counter()
    fixes: list[InfoFix] = []
    for raw in info.get("locations") or []:
        if not isinstance(raw, Mapping):
            dropped["malformed"] += 1
            continue
        t, lat, lon = _number(raw.get("timestamp")), _number(raw.get("latitude")), _number(raw.get("longitude"))
        if t is None or lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
            dropped["no position"] += 1
            continue
        if t < start - EDGE_S * 1000 or t > end + EDGE_S * 1000:
            dropped["outside the video"] += 1
            continue
        accuracy = _number(raw.get("accuracy"))
        if accuracy is not None and (accuracy < 0 or accuracy > max_accuracy_m):
            dropped["inaccurate"] += 1
            continue
        when = _ms(t)
        if fixes and when <= fixes[-1].time:
            dropped["not after the previous fix"] += 1
            continue
        speed = _number(raw.get("speed"))
        course = _number(raw.get("course"))
        fixes.append(
            InfoFix(
                when,
                lat,
                lon,
                speed if speed is not None and speed >= 0 else None,
                course % 360.0 if course is not None and 0 <= course <= 360 else None,
                accuracy,
            )
        )
    if len(fixes) < 2:
        raise BddInfoError(f"{clip_id}: fewer than two usable GPS fixes ({sum(dropped.values())} dropped)")
    segments: list[list[InfoFix]] = [[fixes[0]]]
    for previous, fix in itertools.pairwise(fixes):
        if (fix.time - previous.time).total_seconds() > gap_s:
            segments.append([])
        segments[-1].append(fix)
    return InfoTrack(clip_id, _ms(start), _ms(end), segments, dropped)


def to_gpx(track: InfoTrack) -> str:
    """GPX 1.0 text: one track, a segment per stretch without gaps, and the `video_start` waypoint."""
    return write_gpx(
        track.segments,
        track.video_start,
        name=f"BDD100K {track.clip_id}",
        description="Converted from the BDD100K info file by scripts/bdd_info_to_gpx.py",
        track_name=track.clip_id,
    )
