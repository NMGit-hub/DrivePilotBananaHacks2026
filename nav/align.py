"""Line a GPX track up with a phone video: find the video's start time and write the `video_start` waypoint.

`nav.gps.GpxReplay` plays a GPX on the video's clock from a waypoint named `video_start` whose time is the
video's first frame. A drive recorded with the phone's camera app and the GPSLogger app has no such waypoint;
`scripts/align_gpx.py` works it out in two steps and writes an aligned copy of the GPX.

1. From timestamps (`decide_start`). Two sources, both with 1 s resolution:
   - the file name `VID_YYYYMMDD_HHMMSS` (OnePlus camera app; `VID20261011_143022`, `..._01` are tolerated):
     the local wall time (America/Los_Angeles by default, DST-aware) at which the recording started;
   - the MP4's `creation_time` (UTC), read with PyAV (ffprobe only as a fallback). Writers differ: Android's
     MP4 writer stamps it when the file is finalized, i.e. at the end of the recording, others at the start.
     When it is within `END_MATCH_S` of name + duration it is taken as the end (start = creation_time -
     duration), when within `END_MATCH_S` of the name as the start. Without a usable name it is assumed to
     be the end (`creation_is` overrides).
   Default (`prefer="auto"`): the file name when it parses, else creation_time. The name is written by the
   camera app when recording starts, so it measures the start directly. creation_time only becomes a start
   through the start/end guess and the duration, adds the delay between the last frame and the file being
   finalized, and is rewritten or dropped when a file is re-encoded or edited. Both are whole seconds
   (truncated), so `TRUNCATION_S` (the middle of that second) is added to each. When both exist and disagree
   by more than `DISAGREE_WARN_S` the summary says so loudly.

2. Refinement from motion (`refine_shift`): the first clean pull-away from a stop, seen on both sides.
   - Video: ego motion's "stopped" flag per frame (`perception.ego_motion.EgoMotionEstimator` with the tracked
     objects' boxes masked out, as cached by `eval.perception_cache`; only the first `search_s` seconds are
     processed). Flag runs separated by short "moving" glitches (< `merge_gap_s`) are one stop. The flag turns
     on `stopped_after_s` after the car stops, so the arrival is backdated by that much.
   - GPS: the recorded (Doppler) speed when the file has it, else speed from positions (centered over
     ±`POSITION_HALF_WINDOW_S`, median-filtered: 1 Hz position noise is 1-2 m/s). Stopped = at or below the
     stop speed; a single fast-ish sample inside a stop is a glitch. Edges are where the speed crosses the
     stop speed, interpolated between the 1 Hz samples.
   - A pull-away counts when the stop lasted >= `min_stop_s` on that side and the motion is clean: the car keeps
     moving for `min_move_s` (video and GPS) and the GPS speed reaches `go_speed_mps` within `go_within_s`.
     Parking-lot creeps are skipped: phone speed reads 0 below ~0.5 m/s while the video already sees motion,
     so their edges disagree by seconds.
   - Candidates are GPS pull-aways within ±`max_shift_s` of the video's pull-away on the timestamp clock. Each
     is checked against the same stop's arrival (when both sides saw it) and against the stopped/moving
     agreement of the two whole series under that shift. A contradicted candidate is dropped; two candidates
     that fit equally well are ambiguous (that pull-away is skipped).
   - The shift comes from the pull-away alone, minus `pull_away_bias_s` (the GPS-minus-video pull-away time on
     aligned BDD100K phone clips: the phone's speed lags 0.5-1 s and ego motion's edges are late too, so they
     nearly cancel). Confidence "high": the stop's arrival agrees within `corroborate_s` (the stop began
     after both recordings did). "medium": only the pull-away could be checked, typically parked when the
     recording began. On BDD100K parking-lot clips those were off by up to 6 s (pulling out of a space at
     walking pace, the phone's speed and position stay at ~0 while the video sees motion), so by default
     only the first "high" match is applied (`min_confidence`). Nothing usable: the timestamp start is kept
     and the summary says so.

The aligned copy (`write_aligned_gpx`) keeps every track point (time, position, speed, course) as GPX 1.0, the
version whose <speed>/<course> gpxpy (so `GpxReplay`) reads; speed and course found only in a GPX 1.1 file's
<extensions> (or as non-standard children) are moved there. An existing `video_start` waypoint is replaced.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from core.logging_setup import get_logger
from nav.geo import haversine_m
from nav.gps import VIDEO_START, TrackPoint

if TYPE_CHECKING:
    from core.config import AppConfig
    from eval.perception_cache import CachedFrame

log = get_logger(__name__)

DEFAULT_TZ = "America/Los_Angeles"
TRUNCATION_S = 0.5  # whole-second timestamps are truncated: the true instant is on average half a second later
END_MATCH_S = 3.0  # creation_time within this of name + duration = it marks the end of the recording
DISAGREE_WARN_S = 2.0  # name vs creation_time starts further apart than this: warn
POSITION_HALF_WINDOW_S = 1.5  # speed from positions: displacement over ±this (1 Hz position noise is 1-2 m/s)
RECORDED_SPEED_MIN_FRACTION = 0.8  # use the file's speeds when at least this share of the points have one

StartSource = Literal["name", "creation", "manual"]
CreationRole = Literal["start", "end"]
Confidence = Literal["high", "medium", "none"]

_NAME_RE = re.compile(r"VID_?(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})(?!\d)", re.IGNORECASE)


class AlignError(ValueError):
    """The inputs can't be aligned (no start time, unreadable GPX, output would overwrite the input...)."""


# ------------------------------------------------------------------------------------------ timestamps


@dataclass(frozen=True, slots=True)
class NameTime:
    """The recording start written in a camera file name (whole seconds, local wall time)."""

    local: datetime  # naive wall time from the name
    tz: str
    candidates: tuple[datetime, ...]  # UTC instants it can mean: two in the hour repeated when clocks fall back
    nonexistent: bool  # in the hour skipped when clocks spring forward (wrong phone clock or time zone)

    @property
    def utc(self) -> datetime:
        """The (first) UTC instant."""
        return self.candidates[0]

    @property
    def ambiguous(self) -> bool:
        """True in the repeated hour after clocks fall back."""
        return len(self.candidates) > 1


def time_zone(tz: str) -> ZoneInfo:
    """The IANA time zone `tz` ("America/Los_Angeles"), or AlignError for a name that isn't one."""
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:  # (OSError: a folder of tzdata, "Pacific")
        raise AlignError(f"unknown time zone {tz!r} (e.g. America/Los_Angeles)") from exc


def parse_video_name(name: str, tz: str = DEFAULT_TZ) -> NameTime | None:
    """The start time in a camera file name `VID_YYYYMMDD_HHMMSS...` (None if the name has none)."""
    match = _NAME_RE.search(Path(name).name)
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    try:
        local = datetime(year, month, day, hour, minute, second)
    except ValueError:
        log.warning("file name has an impossible date or time", name=name)
        return None
    zone = time_zone(tz)
    valid: list[datetime] = []
    for fold in (0, 1):
        instant = local.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if instant.astimezone(zone).replace(tzinfo=None) == local and instant not in valid:
            valid.append(instant)
    if valid:
        return NameTime(local, tz, tuple(sorted(valid)), nonexistent=False)
    return NameTime(local, tz, (local.replace(tzinfo=zone).astimezone(UTC),), nonexistent=True)


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """What the container says about a video."""

    creation_time: datetime | None  # UTC, as written by the recorder
    duration_s: float | None
    reader: str  # "pyav", "ffprobe" or "none"


def _parse_iso(value: object) -> datetime | None:
    """An ISO time from container metadata (naive = UTC); None if unset (MP4 zero dates) or unreadable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed if parsed.year >= 2000 else None  # 0 in the header reads as 1904 or 1970: not set


def _read_meta_pyav(path: Path) -> VideoMeta:
    import av

    with av.open(str(path)) as container:
        video = container.streams.video[0] if container.streams.video else None
        creation = _parse_iso(container.metadata.get("creation_time"))
        if creation is None and video is not None:
            creation = _parse_iso(video.metadata.get("creation_time"))
        duration: float | None = None
        if video is not None and video.duration and video.time_base:
            duration = float(video.duration * video.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
    return VideoMeta(creation, duration, "pyav")


def _read_meta_ffprobe(path: Path, exe: str) -> VideoMeta:
    args = [exe, "-v", "error", "-show_entries", "format=duration:format_tags=creation_time"]
    args += ["-show_entries", "stream=codec_type,duration:stream_tags=creation_time", "-of", "json", str(path)]
    done = subprocess.run(args, capture_output=True, text=True, timeout=60, check=True)
    data = json.loads(done.stdout)
    fmt = data.get("format", {})
    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    creation = _parse_iso(fmt.get("tags", {}).get("creation_time"))
    if creation is None and videos:
        creation = _parse_iso(videos[0].get("tags", {}).get("creation_time"))
    raw = videos[0].get("duration") if videos and videos[0].get("duration") else fmt.get("duration")
    return VideoMeta(creation, float(raw) if raw else None, "ffprobe")


def read_video_meta(path: Path) -> VideoMeta:
    """creation_time and duration via PyAV; ffprobe (if on PATH) when PyAV can't read the file."""
    import av

    try:
        return _read_meta_pyav(path)
    except (av.FFmpegError, OSError, ValueError, IndexError) as exc:
        exe = shutil.which("ffprobe")
        if exe is None:
            log.warning("could not read the video's metadata", path=str(path), error=str(exc))
            return VideoMeta(None, None, "none")
        log.warning("PyAV could not read the video; trying ffprobe", path=str(path), error=str(exc))
        try:
            return _read_meta_ffprobe(path, exe)
        except (OSError, ValueError, subprocess.SubprocessError) as exc2:
            log.warning("ffprobe could not read the video either", path=str(path), error=str(exc2))
            return VideoMeta(None, None, "none")


@dataclass(frozen=True, slots=True)
class StartDecision:
    """The video's start time from its timestamps, with everything the summary reports."""

    start: datetime  # UTC: the first frame, before any refinement
    source: StartSource
    name: NameTime | None
    name_start: datetime | None  # UTC, with the half-second truncation correction
    creation_time: datetime | None  # UTC, as written in the file
    creation_role: CreationRole | None
    creation_role_reason: str
    creation_start: datetime | None  # UTC: creation_time turned into a start (+ truncation correction)
    duration_s: float | None
    disagreement_s: float | None  # creation_start - name_start
    warnings: tuple[str, ...]


def _whole_second(t: datetime) -> bool:
    return t.microsecond == 0


def decide_start(
    name: NameTime | None,
    meta: VideoMeta,
    prefer: Literal["auto", "name", "creation"] = "auto",
    creation_is: Literal["auto", "start", "end"] = "auto",
    manual: datetime | None = None,
) -> StartDecision:
    """Pick the video's start from the file name, creation_time or a manual time (see the module docstring)."""
    warnings: list[str] = []
    creation = meta.creation_time
    duration = meta.duration_s
    name_utc = name.utc if name is not None else None
    role: CreationRole | None = None
    reason = ""
    if creation is not None:
        if creation_is != "auto":
            role, reason = creation_is, "set with --creation-is"
        elif name is not None and duration is not None:
            # Which reading of creation_time fits the name best (also picks the hour when the name is ambiguous).
            fits = []
            for candidate in name.candidates:
                gap = (creation - candidate).total_seconds()
                fits.append((abs(gap - duration), "end", candidate))
                fits.append((abs(gap), "start", candidate))
            off, best_role, name_utc = min(fits, key=lambda f: f[0])
            role = "end" if best_role == "end" else "start"
            if off <= END_MATCH_S:
                where = "name + duration" if role == "end" else "the name"
                reason = f"matches {where} within {off:.1f} s"
            else:
                reason = f"matches neither the start nor the end (closest: the {role}, {off:.1f} s off)"
        elif name is not None:
            role, reason = "start", "duration unknown: assumed to be the start"
        else:
            role, reason = "end", "assumed (Android's MP4 writer stamps the end); --creation-is start if not"
    if name is not None and name.ambiguous:
        if creation is None or duration is None or creation_is != "auto":
            warnings.append(
                f"the file name's time {name.local:%H:%M:%S} is in the hour repeated when clocks fell back: "
                "took the first (daylight-time) one"
            )
        else:
            warnings.append(
                f"the file name's time is in the repeated hour: creation_time picks {name_utc:%H:%M:%S} UTC"
            )
    if name is not None and name.nonexistent:
        warnings.append(
            f"the file name's time {name.local:%Y-%m-%d %H:%M:%S} does not exist in {name.tz} (clocks sprang "
            "forward): wrong phone clock or --tz?"
        )
    name_start = name_utc + timedelta(seconds=TRUNCATION_S) if name_utc is not None else None
    creation_start: datetime | None = None
    if creation is not None and role is not None:
        if role == "end" and duration is None:
            warnings.append("creation_time marks the end but the duration is unknown: creation_time not used")
        else:
            creation_start = creation - timedelta(seconds=duration if role == "end" and duration else 0.0)
            if _whole_second(creation):
                creation_start += timedelta(seconds=TRUNCATION_S)
    disagreement = None
    if name_start is not None and creation_start is not None:
        disagreement = (creation_start - name_start).total_seconds()
        if abs(disagreement) > DISAGREE_WARN_S:
            warnings.append(
                f"file name and creation_time disagree by {disagreement:+.1f} s (creation_time {reason}): "
                "check the phone's clock and --tz"
            )
    source: StartSource
    if manual is not None:
        start, source = manual.astimezone(UTC), "manual"
    elif prefer == "name" or (prefer == "auto" and name_start is not None):
        if name_start is None:
            raise AlignError("the video's name has no VID_YYYYMMDD_HHMMSS time: use --start-from creation or --start")
        start, source = name_start, "name"
    elif creation_start is not None:
        start, source = creation_start, "creation"
    elif prefer == "creation":
        raise AlignError("the video has no usable creation_time: use --start-from name or --start")
    else:
        raise AlignError("no start time: the file name has none and the video has no creation_time: give --start")
    for warning in warnings:
        log.warning(warning)
    return StartDecision(
        start=start,
        source=source,
        name=name,
        name_start=name_start,
        creation_time=creation,
        creation_role=role,
        creation_role_reason=reason,
        creation_start=creation_start,
        duration_s=duration,
        disagreement_s=disagreement,
        warnings=tuple(warnings),
    )


def parse_manual_start(text: str, tz: str = DEFAULT_TZ) -> datetime:
    """An ISO time for --start: with an offset or Z as given, without one in the local time zone `tz`."""
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlignError(f"--start: not an ISO time: {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=time_zone(tz))
    return parsed.astimezone(UTC)


# ------------------------------------------------------------------------------------------------- GPX


@dataclass(frozen=True, slots=True)
class GpxTrack:
    """A GPX file's track points (speed and course resolved, including GPX 1.1 extensions)."""

    segments: tuple[tuple[TrackPoint, ...], ...]
    version: str
    speed_origin: str  # where the speeds came from: "fields", "extensions", "fields+extensions" or "none"

    @property
    def points(self) -> list[TrackPoint]:
        """All points in file order."""
        return [p for segment in self.segments for p in segment]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _number(text: str | None) -> float | None:
    try:
        value = float(text) if text is not None else None
    except ValueError:
        return None
    return value if value is not None and np.isfinite(value) else None


def _raw_speed_course(path: Path) -> list[tuple[float | None, float | None, bool]]:
    """(speed, course, speed came from <extensions>) per <trkpt> in file order, read from the XML itself:
    gpxpy reads <speed>/<course> only in GPX 1.0 and never looks inside <extensions>."""
    out = []
    for element in ET.parse(path).getroot().iter():
        if _local_name(element.tag) != "trkpt":
            continue
        speed = course = None
        from_extensions = False
        for child in element:
            name = _local_name(child.tag)
            if name == "speed":
                speed = _number(child.text)
            elif name == "course":
                course = _number(child.text)
        for child in element:
            if _local_name(child.tag) != "extensions":
                continue
            for item in child.iter():
                name = _local_name(item.tag)
                if speed is None and name == "speed":
                    speed = _number(item.text)
                    from_extensions = speed is not None
                elif course is None and name in ("course", "bearing", "heading"):
                    course = _number(item.text)
        if speed is not None and speed < 0:
            speed = None  # -1 = unknown in some writers
        if course is not None and course < 0:
            course = None
        out.append((speed, course, from_extensions))
    return out


def _aware(t: datetime) -> datetime:
    """A GPX time in UTC (GPX times are UTC; a time without a zone is taken as such)."""
    return (t if t.tzinfo is not None else t.replace(tzinfo=UTC)).astimezone(UTC)


def _parse_gpx(path: Path) -> Any:
    """gpxpy's parse of a GPX file (gpxpy has no type hints)."""
    import gpxpy
    import gpxpy.gpx

    try:
        with path.open(encoding="utf-8") as fh:
            return gpxpy.parse(fh)
    except (OSError, ValueError, gpxpy.gpx.GPXException) as exc:
        raise AlignError(f"{path}: not a readable GPX file ({exc})") from exc


def read_gpx(path: Path) -> GpxTrack:
    """The GPX's timed track points, speed and course resolved from fields or extensions."""
    gpx = _parse_gpx(path)
    raw = _raw_speed_course(path)
    all_points = [p for track in gpx.tracks for segment in track.segments for p in segment.points]
    if len(raw) != len(all_points):
        log.warning("GPX speed/course scan doesn't line up with the track points; extensions ignored", path=str(path))
        raw = [(None, None, False)] * len(all_points)
    origins: set[str] = set()
    segments: list[tuple[TrackPoint, ...]] = []
    index = 0
    for track in gpx.tracks:
        for segment in track.segments:
            points: list[TrackPoint] = []
            for p in segment.points:
                raw_speed, raw_course, from_extensions = raw[index]
                index += 1
                if p.time is None:
                    continue
                speed = p.speed if p.speed is not None and p.speed >= 0 else raw_speed
                course = getattr(p, "course", None)
                course = course if course is not None else raw_course
                if speed is not None:
                    origins.add("extensions" if p.speed is None and from_extensions else "fields")
                points.append(TrackPoint(_aware(p.time), p.latitude, p.longitude, speed, course))
            if points:
                segments.append(tuple(points))
    if sum(len(s) for s in segments) < 2:
        raise AlignError(f"{path}: needs at least two timed track points")
    origin = "+".join(sorted(origins)) if origins else "none"
    return GpxTrack(tuple(segments), str(gpx.version or "?"), origin)


def write_aligned_gpx(src: Path, dst: Path, video_start: datetime, description: str) -> None:
    """Write a copy of `src` with the `video_start` waypoint (replacing any) as GPX 1.0; `src` is never touched."""
    import gpxpy.gpx

    if dst.resolve() == src.resolve():
        raise AlignError(f"refusing to overwrite the input GPX {src}: choose another --out")
    gpx = _parse_gpx(src)
    raw = _raw_speed_course(src)
    points = [p for track in gpx.tracks for segment in track.segments for p in segment.points]
    if len(raw) == len(points):
        for point, (speed, course, _) in zip(points, raw, strict=True):
            if point.speed is None and speed is not None:
                point.speed = speed
            if getattr(point, "course", None) is None and course is not None:
                point.course = course  # GPX 1.0 writes it (gpxpy annotates the attribute as None)
    timed = [p for p in points if p.time is not None]
    if not timed:
        raise AlignError(f"{src}: no timed track points")
    nearest = min(timed, key=lambda p: abs((_aware(p.time) - video_start).total_seconds()))
    gpx.waypoints = [w for w in gpx.waypoints if w.name != VIDEO_START]
    gpx.waypoints.insert(
        0,
        gpxpy.gpx.GPXWaypoint(
            nearest.latitude, nearest.longitude, time=video_start, name=VIDEO_START, description=description
        ),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(str(gpx.to_xml(version="1.0")), encoding="utf-8")


def default_output(gpx: Path) -> Path:
    """`<gpx stem>.aligned.gpx` next to the input."""
    return gpx.with_name(f"{gpx.stem}.aligned.gpx")


# ------------------------------------------------------------------------------------- motion: stops


@dataclass(frozen=True, slots=True)
class SpeedSeries:
    """GPS speed on a clock in seconds from a reference time (the timestamp start)."""

    t: np.ndarray
    speed: np.ndarray  # m/s
    segment: np.ndarray  # track segment of each sample (no edge is taken across two)
    source: Literal["recorded", "positions"]


def speed_series(track: GpxTrack, ref: datetime) -> SpeedSeries:
    """Recorded speeds when most points have one, else speeds from positions (centered, median-filtered)."""
    points = track.points
    have = sum(p.speed_mps is not None for p in points)
    t_all, v_all, seg_all = [], [], []
    if have >= RECORDED_SPEED_MIN_FRACTION * len(points):
        for index, segment in enumerate(track.segments):
            for p in segment:
                if p.speed_mps is not None:
                    t_all.append((p.time - ref).total_seconds())
                    v_all.append(p.speed_mps)
                    seg_all.append(index)
        source: Literal["recorded", "positions"] = "recorded"
    else:
        for index, segment in enumerate(track.segments):
            t = np.array([(p.time - ref).total_seconds() for p in segment])
            v = _speed_from_positions(t, np.array([p.lat for p in segment]), np.array([p.lon for p in segment]))
            t_all += list(t)
            v_all += list(v)
            seg_all += [index] * len(segment)
        source = "positions"
    return SpeedSeries(np.array(t_all, dtype=float), np.array(v_all, dtype=float), np.array(seg_all, dtype=int), source)


def _speed_from_positions(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Displacement over ±POSITION_HALF_WINDOW_S around each fix / time (positions interpolated in time), then a
    3-sample median: centered, so the edges aren't shifted, and much less noisy than fix-to-fix speeds."""
    if len(t) < 2:
        return np.zeros(len(t))
    half = POSITION_HALF_WINDOW_S
    lo = np.clip(t - half, t[0], t[-1])
    hi = np.clip(t + half, t[0], t[-1])
    lat_lo, lon_lo = np.interp(lo, t, lat), np.interp(lo, t, lon)
    lat_hi, lon_hi = np.interp(hi, t, lat), np.interp(hi, t, lon)
    speed = np.array(
        [
            haversine_m((a, b), (c, d)) / (h - low) if h > low else 0.0
            for a, b, c, d, low, h in zip(lat_lo, lon_lo, lat_hi, lon_hi, lo, hi, strict=True)
        ]
    )
    if len(speed) >= 3:
        padded = np.concatenate([speed[:1], speed, speed[-1:]])
        speed = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    return speed


@dataclass(frozen=True, slots=True)
class RefineSettings:
    """Thresholds of the refinement (see the module docstring)."""

    max_shift_s: float = 10.0  # search the GPS pull-away within ± this of the timestamp start
    min_stop_s: float = 2.0  # a stop lasts at least this long (both sides)
    min_move_s: float = 3.0  # after a pull-away the car keeps moving this long (else it was a glitch)
    merge_gap_s: float = 1.5  # video: "moving" flag gaps shorter than this inside a stop are glitches
    gps_merge_gap_s: float = 2.0  # GPS: one moving sample (1 Hz) between stopped ones is a glitch
    stop_speed_mps: float = 0.5  # GPS (recorded speed): at or below = stopped
    stop_speed_positions_mps: float = 0.8  # GPS (speed from positions, noisier): at or below = stopped
    go_speed_mps: float = 2.0  # a clean pull-away reaches this speed...
    go_within_s: float = 4.0  # ...within this long of leaving the stop
    # GPS edge minus video edge on aligned clips (the phone's speed lags 0.5-1 s, ego motion's edges are late
    # too, and the GPS edge is where the speed passes stop_speed_mps). Measured on BDD100K phone clips: +0.25 s
    # (mean) at the full stops with clean pull-aways, +0.1 s median over all pull-aways; -0.1 s median at
    # arrivals (+0.5 s when braking hard at lights, -0.2 s creeping into parking-lot stops).
    pull_away_bias_s: float = 0.25
    pull_away_bias_positions_s: float = 0.0  # the same with speed from positions (median -0.07 s on BDD100K)
    arrival_bias_s: float = 0.0
    corroborate_s: float = 1.0  # the stop's arrival within this of the pull-away's shift: confidence "high"
    contradict_s: float = 2.5  # ... further off than this: that GPS stop is not this stop
    min_agreement: float = 0.7  # stopped/moving must agree on at least this share of the overlap (edges excluded)
    edge_margin_s: float = 1.0  # agreement ignores this much around every edge
    min_confidence: Literal["high", "medium"] = "high"  # apply a "medium" match (pull-away only) too?


@dataclass(frozen=True, slots=True)
class Stop:
    """A stop on one side's clock (video seconds, or GPS seconds from the timestamp start)."""

    begin: float  # arrival, or where the data begins if already stopped then
    end: float  # pull-away, or where the data ends if still stopped then
    arrive: float | None  # None: already stopped when the data begins
    depart: float | None  # None: still stopped when the data ends
    clean_departure: bool  # sustained motion after the pull-away (GPS: and a quick rise in speed)

    @property
    def duration_s(self) -> float:
        """How long the stop lasted (as far as the data shows)."""
        return self.end - self.begin


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def video_stops(
    t: Sequence[float] | np.ndarray,
    stopped: Sequence[bool] | np.ndarray,
    stopped_after_s: float,
    settings: RefineSettings | None = None,
) -> list[Stop]:
    """Stops from ego motion's per-frame "stopped" flag (the flag turns on `stopped_after_s` after the stop)."""
    cfg = settings or RefineSettings()
    times = np.asarray(t, dtype=float)
    flags = np.asarray(stopped, dtype=bool)
    if len(times) == 0:
        return []
    runs: list[list[int]] = []
    for first, last in _true_runs(flags):
        if runs and runs[-1][1] + 1 < len(times) and times[first] - times[runs[-1][1] + 1] < cfg.merge_gap_s:
            runs[-1][1] = last  # a short "moving" glitch inside a stop
        else:
            runs.append([first, last])
    stops: list[Stop] = []
    for first, last in runs:
        # Already stopped at the start: the flag can't turn on before stopped_after_s plus a little history.
        censored = times[first] - times[0] <= stopped_after_s + 0.6
        arrive = None if censored else float(times[first] - stopped_after_s)
        depart = float(times[last + 1]) if last + 1 < len(times) else None
        stops.append(
            Stop(
                begin=float(times[0]) if arrive is None else arrive,
                end=depart if depart is not None else float(times[-1]),
                arrive=arrive,
                depart=depart,
                clean_departure=False,
            )
        )
    out = []
    for i, stop in enumerate(stops):
        clean = False
        if stop.depart is not None:
            next_begin = stops[i + 1].begin if i + 1 < len(stops) else float(times[-1])
            clean = next_begin - stop.depart >= cfg.min_move_s
        out.append(Stop(stop.begin, stop.end, stop.arrive, stop.depart, clean))
    return out


def _crossing(t0: float, v0: float, t1: float, v1: float, level: float) -> float:
    """When the speed crosses `level` between two samples (linear)."""
    if v1 == v0:
        return (t0 + t1) / 2
    f = min(max((level - v0) / (v1 - v0), 0.0), 1.0)
    return t0 + f * (t1 - t0)


def gps_stops(series: SpeedSeries, settings: RefineSettings | None = None) -> list[Stop]:
    """Stops in the GPS speed (per track segment; an edge at a segment's end is unknown)."""
    cfg = settings or RefineSettings()
    level = cfg.stop_speed_mps if series.source == "recorded" else cfg.stop_speed_positions_mps
    stops: list[Stop] = []
    for segment in np.unique(series.segment):
        sel = series.segment == segment
        t, v = series.t[sel], series.speed[sel]
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        runs: list[list[int]] = []
        for first, last in _true_runs(v <= level):
            if runs:
                prev_last = runs[-1][1]
                between = v[prev_last + 1 : first]
                if t[first] - t[prev_last] <= cfg.gps_merge_gap_s and float(between.max()) < cfg.go_speed_mps:
                    runs[-1][1] = last  # a single faster sample inside a stop
                    continue
            runs.append([first, last])
        for first, last in runs:
            arrive = _crossing(t[first - 1], v[first - 1], t[first], v[first], level) if first > 0 else None
            depart = _crossing(t[last], v[last], t[last + 1], v[last + 1], level) if last + 1 < len(t) else None
            clean = False
            if depart is not None:
                after = slice(last + 1, None)
                ta, va = t[after], v[after]
                reached = bool(np.any((va >= cfg.go_speed_mps) & (ta - depart <= cfg.go_within_s)))
                restopped = bool(np.any((va <= level) & (ta <= depart + cfg.min_move_s)))
                clean = reached and not restopped
            stops.append(
                Stop(
                    begin=float(arrive if arrive is not None else t[first]),
                    end=float(depart if depart is not None else t[last]),
                    arrive=None if arrive is None else float(arrive),
                    depart=None if depart is None else float(depart),
                    clean_departure=clean,
                )
            )
    return sorted(stops, key=lambda s: s.begin)


# -------------------------------------------------------------------------------- motion: matching


@dataclass(frozen=True, slots=True)
class EdgeCheck:
    """Another edge looked at under the chosen shift."""

    kind: Literal["pull-away", "arrival"]
    video_s: float
    gps_s: float | None  # nearest GPS edge of that kind, bias-corrected, on the timestamp clock (None: none)
    residual_s: float | None  # (gps_s - video_s) - shift


@dataclass(frozen=True, slots=True)
class Refinement:
    """The motion-based correction to the timestamp start."""

    shift_s: float | None  # add to the timestamp start (None: not applied)
    confidence: Confidence  # of the match found ("none": no match)
    video_pull_away_s: float | None  # video time of the matched pull-away
    gps_pull_away_s: float | None  # GPS pull-away, seconds from the timestamp start (raw, before the bias)
    arrival: EdgeCheck | None  # the same stop's arrival under the shift (None: one side didn't see it)
    checks: tuple[EdgeCheck, ...]  # the other stops' edges under the shift
    agreement: float | None  # share of the overlap where both sides agree on stopped/moving
    speed_source: str
    notes: tuple[str, ...]  # skipped pull-aways and why
    candidate_shift_s: float | None = None  # the match's shift, also when it was not applied

    @property
    def applied(self) -> bool:
        """True if the start was refined."""
        return self.shift_s is not None


@dataclass(frozen=True, slots=True)
class _Fit:
    """One GPS pull-away tried against a video pull-away."""

    gps: Stop
    shift: float
    arrival: EdgeCheck | None
    agreement: float | None
    level: Literal["high", "medium"]
    problem: str  # why it can't be this stop ("" = viable)
    weakness: str  # why it is only "medium"


def _agreement(
    video: Sequence[Stop],
    video_range: tuple[float, float],
    gps: Sequence[Stop],
    gps_ranges: Sequence[tuple[float, float]],
    shift: float,
    cfg: RefineSettings,
) -> float | None:
    """Share of the overlap (edges excluded) where both sides say the same (stopped or moving) under `shift`."""
    tau = np.arange(video_range[0], video_range[1], 0.1)
    covered = np.zeros(len(tau), dtype=bool)
    for lo, hi in gps_ranges:
        covered |= (tau + shift >= lo) & (tau + shift <= hi)
    near_edge = np.zeros(len(tau), dtype=bool)
    v_stopped = np.zeros(len(tau), dtype=bool)
    g_stopped = np.zeros(len(tau), dtype=bool)
    for s in video:
        v_stopped |= (tau >= s.begin) & (tau <= s.end)
        for edge in (s.arrive, s.depart):
            if edge is not None:
                near_edge |= np.abs(tau - edge) < cfg.edge_margin_s
    for s in gps:
        g_stopped |= (tau + shift >= s.begin) & (tau + shift <= s.end)
        for edge in (s.arrive, s.depart):
            if edge is not None:
                near_edge |= np.abs(tau + shift - edge) < cfg.edge_margin_s
    use = covered & ~near_edge
    if use.sum() < 50:  # less than 5 s to compare
        return None
    return float(np.mean(v_stopped[use] == g_stopped[use]))


def _nearest(edges: Sequence[float], target: float) -> float | None:
    return min(edges, key=lambda e: abs(e - target)) if edges else None


def _fit(
    v: Stop,
    g: Stop,
    video: Sequence[Stop],
    video_range: tuple[float, float],
    gps: Sequence[Stop],
    gps_ranges: Sequence[tuple[float, float]],
    cfg: RefineSettings,
) -> _Fit:
    """How well GPS stop `g` explains video stop `v` (shift from the pull-aways, checked with the arrivals)."""
    assert v.depart is not None and g.depart is not None
    shift = g.depart - cfg.pull_away_bias_s - v.depart
    arrival = None
    problem = ""
    level: Literal["high", "medium"] = "medium"
    weakness = "only the pull-away could be checked (already stopped when the video or the GPS began)"
    if v.arrive is not None and g.arrive is not None:
        gps_arrive = g.arrive - cfg.arrival_bias_s
        residual = gps_arrive - v.arrive - shift
        arrival = EdgeCheck("arrival", v.arrive, gps_arrive, residual)
        if abs(residual) > cfg.contradict_s:
            problem = f"the stop's arrival is {residual:+.1f} s off"
        elif abs(residual) <= cfg.corroborate_s:
            level, weakness = "high", ""
        else:
            weakness = f"the stop's arrival is {residual:+.1f} s off (more than {cfg.corroborate_s:.0f} s)"
    elif v.arrive is None and g.arrive is not None:
        if g.arrive - cfg.arrival_bias_s - shift > v.begin + cfg.contradict_s:
            problem = "the GPS still shows motion where the video's first frames show the car stopped"
    elif g.arrive is None and v.arrive is not None and g.begin - shift < v.arrive - cfg.contradict_s:
        # (a GPS track that starts after the video's stop began just missed the arrival: that's consistent)
        problem = "the GPS shows the car stopped before the video's stop began"
    agreement = _agreement(video, video_range, gps, gps_ranges, shift, cfg)
    if not problem and agreement is not None and agreement < cfg.min_agreement:
        problem = f"stopped/moving agree on only {agreement:.0%} of the overlap"
    return _Fit(g, shift, arrival, agreement, level, problem, weakness)


def _other_edges(
    video: Sequence[Stop], anchor: Stop, gps: Sequence[Stop], shift: float, cfg: RefineSettings
) -> tuple[EdgeCheck, ...]:
    """The other video stops' edges, each against the nearest GPS edge of its kind under `shift`."""
    departs = [g.depart - cfg.pull_away_bias_s for g in gps if g.depart is not None]
    arrivals = [g.arrive - cfg.arrival_bias_s for g in gps if g.arrive is not None]
    checks: list[EdgeCheck] = []
    for other in video:
        if other is anchor or other.duration_s < cfg.min_stop_s:
            continue
        for kind, edge, pool in (("arrival", other.arrive, arrivals), ("pull-away", other.depart, departs)):
            if edge is None:
                continue
            near = _nearest(pool, edge + shift)
            residual = None if near is None else near - edge - shift
            checks.append(EdgeCheck("arrival" if kind == "arrival" else "pull-away", edge, near, residual))
    return tuple(checks)


def refine_shift(
    video: Sequence[Stop],
    video_range: tuple[float, float],
    gps: Sequence[Stop],
    gps_ranges: Sequence[tuple[float, float]],
    speed_source: str = "recorded",
    settings: RefineSettings | None = None,
) -> Refinement:
    """Match the first clean video pull-away to a GPS pull-away (see the module docstring).

    `video` is on the video's clock; `gps` and `gps_ranges` (the track segments' time spans) on the timestamp
    clock: seconds from the timestamp start. The result's shift is added to that start.

    A match is "high" when the stop's arrival agrees too (the stop began after both recordings did), "medium"
    when only the pull-away could be checked (typically: parked when the recording started; the phone's speed
    reads 0 while the car creeps out of a parking space, so these were off by 1.4-6.5 s on BDD100K parking-lot
    clips). The first "high" match wins; a "medium" one is applied only with `min_confidence="medium"`.
    """
    cfg = settings or RefineSettings()
    if speed_source == "positions":  # no Doppler-filter lag in speeds from positions
        cfg = replace(cfg, pull_away_bias_s=cfg.pull_away_bias_positions_s)
    notes: list[str] = []
    usable_gps = [g for g in gps if g.depart is not None and g.clean_departure and g.duration_s >= cfg.min_stop_s]
    first_medium: Refinement | None = None
    for v in video:
        if v.depart is None or v.duration_s < cfg.min_stop_s:
            continue
        where = f"video pull-away at {v.depart:.1f} s"
        if not v.clean_departure:
            notes.append(f"{where} skipped: stopped again within {cfg.min_move_s:.0f} s")
            continue
        candidates = [
            g for g in usable_gps if abs((g.depart or 0.0) - cfg.pull_away_bias_s - v.depart) <= cfg.max_shift_s
        ]
        if not candidates:
            notes.append(f"{where}: no clean GPS pull-away within ±{cfg.max_shift_s:.0f} s (a slow creep?)")
            continue
        fits = [_fit(v, g, video, video_range, gps, gps_ranges, cfg) for g in candidates]
        viable = [f for f in fits if not f.problem]
        if not viable:
            reasons = "; ".join(f"shift {f.shift:+.1f} s: {f.problem}" for f in fits)
            notes.append(f"{where}: no consistent GPS pull-away ({reasons})")
            continue
        viable.sort(key=lambda f: (f.level == "high", f.agreement or 0.0), reverse=True)
        best = viable[0]
        if len(viable) > 1:
            other = viable[1]
            if (
                other.level == best.level
                and abs((best.agreement or 0.0) - (other.agreement or 0.0)) < 0.05
                and abs(best.shift - other.shift) > cfg.corroborate_s
            ):
                notes.append(f"{where}: ambiguous, shifts {best.shift:+.1f} s and {other.shift:+.1f} s fit equally")
                continue
        result = Refinement(
            shift_s=best.shift,
            confidence=best.level,
            video_pull_away_s=v.depart,
            gps_pull_away_s=best.gps.depart,
            arrival=best.arrival,
            checks=_other_edges(video, v, gps, best.shift, cfg),
            agreement=best.agreement,
            speed_source=speed_source,
            notes=(),
            candidate_shift_s=best.shift,
        )
        if best.level == "high":
            return _with_notes(result, notes)
        if first_medium is None:
            first_medium = result
            if cfg.min_confidence == "high":
                notes.append(
                    f"{where}: {best.weakness}: {best.shift:+.2f} s, not applied (--min-confidence medium applies it)"
                )
    if first_medium is not None and cfg.min_confidence == "medium":
        return _with_notes(first_medium, notes)
    if not any(v.depart is not None and v.duration_s >= cfg.min_stop_s for v in video):
        notes.append(f"the video shows no stop of {cfg.min_stop_s:.0f} s or more followed by a pull-away")
    if first_medium is not None:
        return _with_notes(first_medium, notes, applied=False)
    return Refinement(None, "none", None, None, None, (), None, speed_source, tuple(notes))


def _with_notes(result: Refinement, notes: list[str], applied: bool = True) -> Refinement:
    """`result` with the notes gathered so far (and its shift dropped when not applied)."""
    return Refinement(
        shift_s=result.shift_s if applied else None,
        confidence=result.confidence,
        video_pull_away_s=result.video_pull_away_s,
        gps_pull_away_s=result.gps_pull_away_s,
        arrival=result.arrival,
        checks=result.checks,
        agreement=result.agreement,
        speed_source=result.speed_source,
        notes=tuple(notes),
        candidate_shift_s=result.candidate_shift_s,
    )


def refine_from_frames(
    times: Sequence[float] | np.ndarray,
    stopped: Sequence[bool] | np.ndarray,
    stopped_after_s: float,
    track: GpxTrack,
    nominal_start: datetime,
    settings: RefineSettings | None = None,
) -> Refinement:
    """Refinement from a per-frame stopped flag (video clock) and the GPX track (`nominal_start` = video t=0)."""
    cfg = settings or RefineSettings()
    t = np.asarray(times, dtype=float)
    series = speed_series(track, nominal_start)
    gps = gps_stops(series, cfg)
    ranges = []
    for segment in np.unique(series.segment):
        seg_t = series.t[series.segment == segment]
        ranges.append((float(seg_t.min()), float(seg_t.max())))
    video = video_stops(t, stopped, stopped_after_s, cfg)
    video_range = (float(t[0]), float(t[-1])) if len(t) else (0.0, 0.0)
    return refine_shift(video, video_range, gps, ranges, series.source, cfg)


# ---------------------------------------------------------------------------- perception (video side)


def _cache_covers(frames: Sequence[CachedFrame], need_s: float) -> bool:
    return bool(frames) and frames[-1].t >= need_s - 1.0


def perception_frames(
    cfg: AppConfig,
    clip: Path,
    search_s: float,
    duration_s: float | None,
    cache: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[list[CachedFrame], Path, bool]:
    """Cached perception (tracks + ego motion) for the first `search_s` seconds: (frames, cache file, built).

    Uses an explicit `cache` file, else the full cache of `scripts/cache_perception.py`
    (data/cache/perception/<stem>.pkl), else data/cache/align/<stem>.pkl; builds the missing one on the GPU
    (never overwriting data/cache/perception/, whose files must cover whole clips).
    """
    from eval.perception_cache import load_cache

    need = min(search_s, duration_s) if duration_s else search_s
    full = cfg.root / "data" / "cache" / "perception" / f"{clip.stem}.pkl"
    partial = cfg.root / "data" / "cache" / "align" / f"{clip.stem}.pkl"
    candidates = [cache] if cache is not None else [full, partial]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        try:
            frames = load_cache(path, cfg)
        except (ValueError, OSError, pickle.UnpicklingError, EOFError) as exc:
            log.warning("perception cache not usable; rebuilding", path=str(path), error=str(exc))
            continue
        if _cache_covers(frames, need):
            return [f for f in frames if f.t <= search_s], path, False
        cached_s = round(frames[-1].t, 1) if frames else 0.0
        log.info("perception cache too short; rebuilding", path=str(path), cached_s=cached_s)
    out = cache if cache is not None else partial
    if out.resolve() == full.resolve() or out.resolve().parent == full.resolve().parent:
        # (data/cache/perception/ holds whole-clip caches the evals trust: a partial one there would be truncated)
        raise AlignError(f"won't write a partial perception cache into {full.parent}: give --cache another path")
    build_partial_cache(cfg, clip, out, search_s, progress)
    frames = load_cache(out, cfg)
    return frames, out, True


def build_partial_cache(
    cfg: AppConfig, clip: Path, out: Path, max_s: float, progress: Callable[[float], None] | None = None
) -> int:
    """`eval.perception_cache.build_cache` for the first `max_s` seconds only (a long drive needs a few minutes
    to find a pull-away). Same file format, so `load_cache` reads it. Returns the number of frames."""
    from eval.perception_cache import build_cache

    return build_cache(cfg, clip, out, max_s=max_s, progress=progress)


# ------------------------------------------------------------------------------------------ the whole job


@dataclass(frozen=True, slots=True)
class AlignResult:
    """Everything `align` decided and wrote."""

    video: Path
    gpx: Path
    out: Path
    decision: StartDecision
    refinement: Refinement | None  # None: skipped (--no-refine)
    refine_error: str | None  # why the refinement could not run (e.g. no GPU to build the perception cache)
    video_start: datetime
    track: GpxTrack
    search_s: float
    cache_path: Path | None
    warnings: tuple[str, ...]  # about the result (e.g. the GPX doesn't cover the video)


def coverage_warning(track: GpxTrack, start: datetime, duration_s: float | None) -> str | None:
    """A warning when the GPX track doesn't cover the video's time span (wrong files, clock or --tz)."""
    first, last = track.points[0].time, track.points[-1].time
    end = start + timedelta(seconds=duration_s or 0.0)
    overlap = (min(end, last) - max(start, first)).total_seconds()
    span = f"track {_utc(first, 0)} to {_utc(last, 0)}, video {_utc(start, 0)} to {_utc(end, 0)}"
    if overlap <= 0:
        return f"the GPX does not cover the video at all ({span}): wrong files, phone clock or --tz?"
    if duration_s and overlap < 0.9 * duration_s:
        return f"the GPX covers only {overlap / duration_s:.0%} of the video ({span})"
    return None


def align(
    video: Path,
    gpx: Path,
    out: Path | None = None,
    *,
    tz: str = DEFAULT_TZ,
    prefer: Literal["auto", "name", "creation"] = "auto",
    creation_is: Literal["auto", "start", "end"] = "auto",
    manual_start: datetime | None = None,
    frames: Callable[[float | None], tuple[Sequence[CachedFrame], Path | None]] | None = None,
    stopped_after_s: float = 0.5,
    search_s: float = 180.0,
    settings: RefineSettings | None = None,
    meta_reader: Callable[[Path], VideoMeta] = read_video_meta,
) -> AlignResult:
    """Decide the video's start, refine it from motion (when `frames` is given), and write the aligned GPX.

    `frames(duration_s)` returns the cached perception frames covering the first `search_s` seconds and the
    cache file (None skips the refinement, as `--no-refine` does).
    """
    out = out if out is not None else default_output(gpx)
    if out.resolve() == gpx.resolve():
        raise AlignError(f"refusing to overwrite the input GPX {gpx}: choose another --out")
    meta = meta_reader(video)
    decision = decide_start(parse_video_name(video.name, tz), meta, prefer, creation_is, manual_start)
    track = read_gpx(gpx)
    refinement: Refinement | None = None
    refine_error: str | None = None
    cache_path: Path | None = None
    if frames is not None:
        try:
            cached, cache_path = frames(meta.duration_s)
        except Exception as exc:  # noqa: BLE001  (GPU missing, decode failure...: keep the timestamp start)
            refine_error = f"{type(exc).__name__}: {exc}"
            log.warning("refinement skipped: perception failed", error=refine_error)
        else:
            usable = [f for f in cached if f.t <= search_s]
            refinement = refine_from_frames(
                [f.t for f in usable],
                [f.ego_stopped for f in usable],
                stopped_after_s,
                track,
                decision.start,
                settings,
            )
            if not refinement.applied:
                log.warning("no usable pull-away: keeping the timestamp start", notes=list(refinement.notes))
    start = decision.start
    if refinement is not None and refinement.shift_s is not None:
        start = start + timedelta(seconds=refinement.shift_s)
    how = {"name": "file name", "creation": "creation_time", "manual": "--start"}[decision.source]
    if refinement is not None and refinement.applied:
        how += f" {refinement.shift_s:+.2f} s from the first pull-away (confidence {refinement.confidence})"
    write_aligned_gpx(gpx, out, start, f"video's first frame: {how} (scripts/align_gpx.py)")
    warning = coverage_warning(track, start, meta.duration_s)
    if warning is not None:
        log.warning(warning)
    warnings = () if warning is None else (warning,)
    return AlignResult(
        video, gpx, out, decision, refinement, refine_error, start, track, search_s, cache_path, warnings
    )


def _utc(t: datetime | None, digits: int = 1) -> str:
    if t is None:
        return "-"
    text = t.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
    return text[: len(text) - (6 - digits)] + " UTC"


def _local(t: datetime, tz: str) -> str:
    return t.astimezone(ZoneInfo(tz)).strftime("%H:%M:%S %Z")


def replay_command(video: Path, gpx: Path, root: Path) -> str:
    """The command that replays the video with its aligned GPX (the phone's ultrawide: hfov from config)."""

    def show(path: Path) -> str:
        try:
            text = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            text = str(path)
        return f'"{text}"' if " " in text else text

    return f".venv/Scripts/python -m app --video {show(video)} --gpx {show(gpx)} --set video.file.hfov_deg=null"


def format_summary(result: AlignResult, tz: str, root: Path) -> str:
    """The short report `scripts/align_gpx.py` prints."""
    d = result.decision
    lines = [f"video        {result.video}" + (f"  ({d.duration_s:.1f} s)" if d.duration_s else "")]
    track = result.track
    lines.append(
        f"gpx          {result.gpx}  (GPX {track.version}, {len(track.points)} points, speed: {track.speed_origin})"
    )
    if d.name is not None:
        local = f"{d.name.local:%Y-%m-%d %H:%M:%S} {tz}"
        lines.append(f"file name    {local} -> start {_utc(d.name_start)} (+{TRUNCATION_S} s: whole seconds)")
    else:
        lines.append("file name    no VID_YYYYMMDD_HHMMSS time")
    if d.creation_time is not None:
        role = f"{d.creation_role} of the recording ({d.creation_role_reason})"
        lines.append(f"creation     {_utc(d.creation_time, 0)} = {role}")
        lines.append(f"             -> start {_utc(d.creation_start)}")
    else:
        lines.append("creation     none in the file")
    if d.disagreement_s is not None:
        flag = "  <-- WARNING" if abs(d.disagreement_s) > DISAGREE_WARN_S else " (ok)"
        lines.append(f"disagreement creation - name = {d.disagreement_s:+.1f} s{flag}")
    source = {"name": "file name", "creation": "creation_time", "manual": "--start"}[d.source]
    lines.append(f"timestamps   start {_utc(d.start)} ({_local(d.start, tz)}) from the {source}")
    r = result.refinement
    if r is None:
        why = result.refine_error or "--no-refine"
        lines.append(f"refinement   NOT RUN ({why}): the timestamp start is used")
    elif r.applied:
        assert r.shift_s is not None and r.video_pull_away_s is not None and r.gps_pull_away_s is not None
        gps_at = d.start + timedelta(seconds=r.gps_pull_away_s)
        lines.append(f"refinement   {r.shift_s:+.2f} s (confidence {r.confidence}), from the first clean pull-away:")
        lines.append(f"             video stopped -> moving at {r.video_pull_away_s:.2f} s (ego motion)")
        lines.append(f"             GPS speed rising at {_utc(gps_at, 2)} ({r.speed_source} speed)")
        if r.arrival is not None and r.arrival.residual_s is not None:
            arrival_at = d.start + timedelta(seconds=r.arrival.gps_s or 0.0)
            lines.append(
                f"             the stop's arrival: video {r.arrival.video_s:.2f} s, GPS {_utc(arrival_at, 2)} "
                f"(residual {r.arrival.residual_s:+.2f} s)"
            )
        good = [c for c in r.checks if c.residual_s is not None and abs(c.residual_s) <= 1.0]
        if good:
            lines.append(f"             {len(good)} more edge(s) agree within 1 s")
        if r.agreement is not None:
            lines.append(f"             stopped/moving agree on {r.agreement:.0%} of the overlap")
    else:
        lines.append("refinement   NONE: no usable pull-away, the TIMESTAMP START IS KEPT (±1 s, check it)")
        if r.candidate_shift_s is not None:
            lines.append(f"             (a {r.confidence}-confidence match said {r.candidate_shift_s:+.2f} s)")
    if r is not None:
        lines += [f"             note: {n}" for n in r.notes]
    lines.append(f"video_start  {_utc(result.video_start, 3)}")
    for warning in (*d.warnings, *result.warnings):
        lines.append(f"WARNING      {warning}")
    lines.append(f"wrote        {result.out}")
    lines.append(f"replay       {replay_command(result.video, result.out, root)}")
    return "\n".join(lines)
