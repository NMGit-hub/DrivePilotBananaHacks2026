"""nav/align.py + scripts/align_gpx.py: lining a GPSLogger track up with a phone video.

Synthetic data only (no GPU, no hardware): camera file names (variants, DST on both sides of a change, bad names),
creation_time read through PyAV (a tiny MP4 written here, and a fake container) and interpreted as the start or
the end of the recording, GPX 1.0 / 1.1 round trips through `GpxReplay`, and the motion refinement on a simulated
drive: ego motion's stopped flag at 30 fps (edges 0.4 s late, as measured) against 1 Hz GPS speed (lagging
0.6 s, noisy), with a known true start.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nav import align as A
from nav.align import (
    AlignError,
    GpxTrack,
    RefineSettings,
    VideoMeta,
    decide_start,
    parse_manual_start,
    parse_video_name,
    read_gpx,
    read_video_meta,
    refine_from_frames,
    write_aligned_gpx,
)
from nav.geo import destination, haversine_m
from nav.gps import VIDEO_START, GpxReplay, TrackPoint, write_gpx

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = (37.7641, -122.4353)
TRUE_START = datetime(2026, 10, 11, 21, 30, 22, 700000, tzinfo=UTC)  # the video's first frame


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)  # type: ignore[misc]


# ------------------------------------------------------------------------------------------ file names


@pytest.mark.parametrize(
    "name",
    [
        "VID_20261011_143022.mp4",
        "VID20261011_143022.mp4",
        "VID_20261011_143022_01.mp4",
        "VID_20261011_143022~2.mp4",
        "vid_20261011_143022.MP4",
        "data/videos/own/VID_20261011_143022.mp4",
    ],
)
def test_name_variants(name: str) -> None:
    parsed = parse_video_name(name)
    assert parsed is not None
    assert parsed.local == datetime(2026, 10, 11, 14, 30, 22)
    assert parsed.utc == utc(2026, 10, 11, 21, 30, 22)  # PDT = UTC-7
    assert not parsed.ambiguous and not parsed.nonexistent


@pytest.mark.parametrize(
    "name",
    [
        "video.mp4",
        "VID_20261341_143022.mp4",  # month 13
        "VID_20261011_256022.mp4",  # hour 25
        "VID_20261011_1430221.mp4",  # 7 digits: not this format
        "PXL_20261011_143022123.mp4",  # Pixel names are UTC with milliseconds: not taken as local time
        "VID_2026101_143022.mp4",
    ],
)
def test_bad_names(name: str) -> None:
    assert parse_video_name(name) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("VID_20260307_120000.mp4", utc(2026, 3, 7, 20, 0, 0)),  # PST (UTC-8), the day before clocks spring forward
        ("VID_20260309_120000.mp4", utc(2026, 3, 9, 19, 0, 0)),  # PDT (UTC-7), the day after
        ("VID_20261031_120000.mp4", utc(2026, 10, 31, 19, 0, 0)),  # PDT, the day before clocks fall back
        ("VID_20261102_120000.mp4", utc(2026, 11, 2, 20, 0, 0)),  # PST, the day after
    ],
)
def test_names_on_both_sides_of_dst_changes(name: str, expected: datetime) -> None:
    parsed = parse_video_name(name)
    assert parsed is not None and parsed.utc == expected


def test_name_in_the_repeated_and_the_skipped_hour() -> None:
    repeated = parse_video_name("VID_20261101_013000.mp4")  # 01:30 happens twice on 2026-11-01
    assert repeated is not None and repeated.ambiguous
    assert repeated.candidates == (utc(2026, 11, 1, 8, 30, 0), utc(2026, 11, 1, 9, 30, 0))
    skipped = parse_video_name("VID_20260308_023000.mp4")  # 02:30 doesn't exist on 2026-03-08
    assert skipped is not None and skipped.nonexistent
    decision = decide_start(skipped, VideoMeta(None, 60.0, "pyav"))
    assert any("does not exist" in w for w in decision.warnings)


def test_other_time_zone() -> None:
    parsed = parse_video_name("VID_20260701_120000.mp4", tz="Europe/Berlin")
    assert parsed is not None and parsed.utc == utc(2026, 7, 1, 10, 0, 0)  # CEST = UTC+2


# ----------------------------------------------------------------------------------- creation_time


def test_pyav_reads_creation_time_and_duration(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "VID_20261011_143022.mp4"
    with av.open(str(path), "w") as container:
        container.metadata["creation_time"] = "2026-10-11T21:30:24.000000Z"
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for _ in range(45):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    meta = read_video_meta(path)
    assert meta.reader == "pyav"
    assert meta.creation_time == utc(2026, 10, 11, 21, 30, 24)
    assert meta.duration_s == pytest.approx(1.5, abs=0.05)


class _FakeContainer:
    """Just what `_read_meta_pyav` reads from a PyAV container."""

    def __init__(self, metadata: dict[str, str], stream_metadata: dict[str, str], duration_s: float) -> None:
        video = SimpleNamespace(metadata=stream_metadata, duration=int(duration_s * 1000), time_base=0.001)
        self.metadata = metadata
        self.streams = SimpleNamespace(video=[video])
        self.duration = int(duration_s * 1_000_000)

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_mocked_pyav_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    fake = _FakeContainer({}, {"creation_time": "2026-10-11T21:40:22.000000Z"}, 600.4)  # only on the stream
    monkeypatch.setattr(av, "open", lambda *a, **k: fake)
    meta = read_video_meta(tmp_path / "x.mp4")
    assert meta.creation_time == utc(2026, 10, 11, 21, 40, 22)
    assert meta.duration_s == pytest.approx(600.4)
    zero = _FakeContainer({"creation_time": "1904-01-01T00:00:00.000000Z"}, {}, 10.0)  # header left at 0
    monkeypatch.setattr(av, "open", lambda *a, **k: zero)
    assert read_video_meta(tmp_path / "x.mp4").creation_time is None


def test_unreadable_video_gives_no_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A.shutil, "which", lambda _name: None)  # no ffprobe either
    path = tmp_path / "VID_20261011_143022.mp4"
    path.write_bytes(b"not a video")
    assert read_video_meta(path) == VideoMeta(None, None, "none")


NAME = parse_video_name("VID_20261011_143022.mp4")  # 21:30:22 UTC


def test_creation_time_at_the_end_of_the_recording() -> None:
    meta = VideoMeta(utc(2026, 10, 11, 21, 40, 22), 600.4, "pyav")  # name + 600 s: Android stamps the end
    d = decide_start(NAME, meta)
    assert d.creation_role == "end" and "name + duration" in d.creation_role_reason
    assert d.source == "name"
    assert d.start == utc(2026, 10, 11, 21, 30, 22) + timedelta(seconds=0.5)  # middle of the truncated second
    assert d.creation_start == utc(2026, 10, 11, 21, 40, 22) + timedelta(seconds=0.5 - 600.4)
    assert d.disagreement_s == pytest.approx(-0.4)
    assert not d.warnings


def test_creation_time_at_the_start_of_the_recording() -> None:
    d = decide_start(NAME, VideoMeta(utc(2026, 10, 11, 21, 30, 23), 600.4, "pyav"))
    assert d.creation_role == "start"
    assert d.disagreement_s == pytest.approx(1.0)
    assert not d.warnings


def test_sources_that_disagree_are_flagged() -> None:
    d = decide_start(NAME, VideoMeta(utc(2026, 10, 11, 21, 40, 30), 600.4, "pyav"))  # 7.6 s past name + duration
    assert d.creation_role == "end" and "neither" in d.creation_role_reason
    assert d.disagreement_s == pytest.approx(7.6)
    assert any("disagree by +7.6 s" in w for w in d.warnings)
    assert d.source == "name"  # the name still wins by default


def test_start_source_choice_and_fallbacks() -> None:
    meta = VideoMeta(utc(2026, 10, 11, 21, 40, 22), 600.0, "pyav")
    assert decide_start(NAME, meta, prefer="creation").source == "creation"
    only_creation = decide_start(None, meta)  # renamed file: creation_time assumed to be the end
    assert only_creation.source == "creation" and only_creation.creation_role == "end"
    assert only_creation.start == utc(2026, 10, 11, 21, 30, 22, 500000)
    as_start = decide_start(None, meta, creation_is="start")
    assert as_start.start == utc(2026, 10, 11, 21, 40, 22, 500000)
    manual = decide_start(NAME, meta, manual=parse_manual_start("2026-10-11T14:30:21.25"))  # Pacific local time
    assert manual.source == "manual" and manual.start == utc(2026, 10, 11, 21, 30, 21, 250000)
    with pytest.raises(AlignError):
        decide_start(None, VideoMeta(None, 10.0, "none"))
    with pytest.raises(AlignError):
        decide_start(None, meta, prefer="name")


def test_creation_time_picks_the_hour_of_an_ambiguous_name() -> None:
    name = parse_video_name("VID_20261101_013000.mp4")
    d = decide_start(name, VideoMeta(utc(2026, 11, 1, 9, 31, 0), 60.0, "pyav"))  # ends 60 s after 01:30 PST
    assert d.name_start == utc(2026, 11, 1, 9, 30, 0, 500000)
    assert d.creation_role == "end"


# --------------------------------------------------------------------------------------------- GPX files


def straight_track(
    speeds: list[float], start: datetime, first_offset_s: float = -3.0, with_speed: bool = True
) -> list[TrackPoint]:
    """1 Hz fixes driving north at the given speeds (m/s)."""
    points, here = [], ORIGIN
    for i, speed in enumerate(speeds):
        t = start + timedelta(seconds=first_offset_s + i)
        points.append(TrackPoint(t, here[0], here[1], speed if with_speed else None, 0.0 if with_speed else None))
        here = destination(here, 0.0, speed)
    return points


GPX11_EXT = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="GPSLogger 131 - http://gpslogger.mendhak.com/" xmlns="http://www.topografix.com/GPX/1/1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<metadata><time>2026-10-11T21:30:10Z</time></metadata>
<wpt lat="37.7641" lon="-122.4353"><time>2026-10-11T21:30:12Z</time><name>parked</name></wpt>
<trk><trkseg>
{points}
</trkseg></trk></gpx>
"""


def gpx11(points: list[TrackPoint], speed_in: str = "extensions") -> str:
    rows = []
    for p in points:
        stamp = p.time.strftime("%Y-%m-%dT%H:%M:%SZ")
        extra = ""
        if speed_in == "extensions" and p.speed_mps is not None:
            extra = f"<extensions><speed>{p.speed_mps}</speed><course>{p.course_deg}</course></extensions>"
        rows.append(f'<trkpt lat="{p.lat}" lon="{p.lon}"><ele>12.0</ele><time>{stamp}</time>{extra}</trkpt>')
    return GPX11_EXT.format(points="\n".join(rows))


SPEEDS = [0.0] * 6 + [1.0, 3.0, 5.0, 7.0] + [8.0] * 10


def test_gpx10_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "drive.gpx"
    points = straight_track(SPEEDS, TRUE_START)
    src.write_text(write_gpx([points], points[0].time, "drive", "test", "drive"), encoding="utf-8")
    before = src.read_bytes()
    out = tmp_path / "drive.aligned.gpx"
    write_aligned_gpx(src, out, TRUE_START, "test")
    assert src.read_bytes() == before  # the original is never modified
    replay = GpxReplay(out)
    fix = replay.fix_at(5.0)  # point 8 (fixes every second from 3 s before the first frame)
    assert fix is not None and fix.speed_mps == pytest.approx(SPEEDS[8])  # recorded speed, not from positions
    assert fix.heading == pytest.approx(0.0)
    first = replay.fix_at(-3.0)  # the first fix
    assert first is not None and haversine_m(first.latlon, ORIGIN) < 0.01


def test_gpx11_speed_in_extensions_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "gpslogger.gpx"
    points = straight_track(SPEEDS, TRUE_START)
    src.write_text(gpx11(points), encoding="utf-8")
    track = read_gpx(src)
    assert track.version == "1.1" and track.speed_origin == "extensions"
    assert [p.speed_mps for p in track.points] == SPEEDS
    out = tmp_path / "gpslogger.aligned.gpx"
    write_aligned_gpx(src, out, TRUE_START, "test")
    text = out.read_text(encoding="utf-8")
    assert 'version="1.0"' in text and "<speed>" in text and "<name>parked</name>" in text
    replay = GpxReplay(out)
    fix = replay.fix_at(6.3)  # 21:30:29.0: point 10
    assert fix is not None and fix.speed_mps == pytest.approx(8.0) and fix.heading == pytest.approx(0.0)
    assert replay._speed[7] == pytest.approx(SPEEDS[7])  # every point's speed made it


def test_gpx11_without_speed(tmp_path: Path) -> None:
    src = tmp_path / "nospeed.gpx"
    points = straight_track(SPEEDS, TRUE_START, with_speed=False)
    src.write_text(gpx11(points, speed_in="none"), encoding="utf-8")
    assert read_gpx(src).speed_origin == "none"
    out = tmp_path / "nospeed.aligned.gpx"
    write_aligned_gpx(src, out, TRUE_START, "test")
    fix = GpxReplay(out).fix_at(8.8)  # between points 11 and 12 (8 m/s): speed from positions
    assert fix is not None and fix.speed_mps == pytest.approx(8.0, abs=0.05)


def test_existing_video_start_is_replaced(tmp_path: Path) -> None:
    src = tmp_path / "old.gpx"
    points = straight_track(SPEEDS, TRUE_START)
    src.write_text(write_gpx([points], TRUE_START - timedelta(seconds=30), "old", "t", "old"), encoding="utf-8")
    out = tmp_path / "new.gpx"
    write_aligned_gpx(src, out, TRUE_START, "test")
    text = out.read_text(encoding="utf-8")
    assert text.count(f"<name>{VIDEO_START}</name>") == 1
    assert GpxReplay(out).fix_at(0.0) is not None
    assert GpxReplay(out)._t[0] == pytest.approx(-3.0)  # the first fix is 3 s before the first frame
    with pytest.raises(AlignError):
        write_aligned_gpx(src, src, TRUE_START, "test")  # never overwrites its input


# ------------------------------------------------------------------------------- the refinement


STOPPED_AFTER = 0.5
VIDEO_LAG = 0.4  # ego motion's edges are ~0.4 s late (PROGRESS.md: ±0.5 s, usually late)
GPS_LAG = 0.6  # the phone's speed lags 0.5-1 s


def speed_profile(t: float, stops: list[tuple[float, float]], cruise: float = 9.0) -> float:
    """True speed at video time t: cruising, braking at 2.5 m/s² into each stop, pulling away at 2 m/s²."""
    for arrive, depart in stops:
        if arrive <= t <= depart:
            return 0.0
    limits = [cruise]
    before = [d for a, d in stops if d < t]
    after = [a for a, d in stops if a > t]
    if before:
        limits.append(2.0 * (t - max(before)))
    if after:
        limits.append(2.5 * (min(after) - t))
    return max(0.0, min(limits))


def simulate(
    stops: list[tuple[float, float]],
    length_s: float = 60.0,
    blip_at: float | None = None,
    seed: int = 1,
    parked_at_start: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[TrackPoint]]:
    """Video stopped flags (30 fps) and a 1 Hz GPS track for a drive starting at TRUE_START."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, length_s, 1 / 30)
    stopped = np.zeros(len(t), dtype=bool)
    for arrive, depart in stops:
        on = max(arrive, 0.0) + STOPPED_AFTER + VIDEO_LAG
        if parked_at_start and arrive <= 0:
            on = 0.55  # the flag turns on as soon as the estimator has history
        stopped |= (t >= on) & (t < depart + VIDEO_LAG)
    if blip_at is not None:
        stopped &= ~((t >= blip_at) & (t < blip_at + 0.2))  # a passing truck: "moving" for 0.2 s
    points, here = [], ORIGIN
    gps_t = np.arange(-8.4, length_s + 5, 1.0)
    for i, g in enumerate(gps_t):
        v = speed_profile(g - GPS_LAG, stops)
        noisy = max(0.0, v + rng.normal(0, 0.15)) if v > 0 else float(rng.choice([0.0, 0.0, 0.2]))
        points.append(TrackPoint(TRUE_START + timedelta(seconds=float(g)), here[0], here[1], noisy, 0.0))
        if i + 1 < len(gps_t):
            step = np.mean([speed_profile(g + f, stops) for f in np.linspace(0, 1, 11)])
            here = destination(here, 0.0, float(step) + rng.normal(0, 0.05))
    return t, stopped, points


def track_of(points: list[TrackPoint]) -> GpxTrack:
    return GpxTrack((tuple(points),), "1.0", "fields")


@pytest.mark.parametrize("error_s", [-6.0, -3.0, 2.0, 5.0, 0.4])
def test_refinement_recovers_the_true_start(error_s: float) -> None:
    t, stopped, points = simulate([(18.0, 31.0)])
    nominal = TRUE_START + timedelta(seconds=error_s)
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), nominal)
    assert r.applied and r.confidence == "high", r.notes
    assert r.shift_s is not None and abs(error_s + r.shift_s) < 0.5
    assert r.video_pull_away_s == pytest.approx(31.0 + VIDEO_LAG, abs=0.05)


def test_a_brief_false_moving_blip_inside_the_stop() -> None:
    t, stopped, points = simulate([(18.0, 31.0)], blip_at=24.0)
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), TRUE_START + timedelta(seconds=3))
    assert r.applied and r.shift_s is not None and abs(3 + r.shift_s) < 0.5
    assert r.video_pull_away_s == pytest.approx(31.4, abs=0.05)  # not the blip at 24 s


def test_no_stop_no_refinement() -> None:
    t, stopped, points = simulate([])
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), TRUE_START + timedelta(seconds=2))
    assert not r.applied and r.confidence == "none"
    assert any("no stop" in n for n in r.notes)


def test_two_stops_the_first_is_used() -> None:
    t, stopped, points = simulate([(12.0, 20.0), (34.0, 44.0)], length_s=70.0)
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), TRUE_START - timedelta(seconds=4))
    assert r.applied and r.shift_s is not None and abs(r.shift_s - 4) < 0.5
    assert r.video_pull_away_s == pytest.approx(20.4, abs=0.05)
    later = [c for c in r.checks if c.residual_s is not None]
    assert later and all(abs(c.residual_s or 0) < 1.0 for c in later)  # the second stop agrees too


def test_parked_at_the_start_is_only_medium() -> None:
    t, stopped, points = simulate([(-20.0, 9.0)])  # already stopped when both recordings began
    nominal = TRUE_START + timedelta(seconds=2.5)
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), nominal)
    assert not r.applied and r.confidence == "medium"
    assert r.candidate_shift_s is not None and abs(2.5 + r.candidate_shift_s) < 0.5
    assert any("not applied" in n for n in r.notes)
    medium = RefineSettings(min_confidence="medium")
    applied = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), nominal, medium)
    assert applied.applied and applied.shift_s == pytest.approx(r.candidate_shift_s)


def test_pull_away_outside_the_search_window() -> None:
    t, stopped, points = simulate([(18.0, 31.0)])
    r = refine_from_frames(t, stopped, STOPPED_AFTER, track_of(points), TRUE_START + timedelta(seconds=14))
    assert not r.applied  # the true match needs -14 s, beyond ±10 s


def test_speed_from_positions_when_the_file_has_none() -> None:
    t, stopped, points = simulate([(18.0, 31.0)], seed=3)
    rng = np.random.default_rng(5)
    noisy = [
        replace(
            p,
            speed_mps=None,
            course_deg=None,
            lat=p.lat + rng.normal(0, 0.4) / 111_320,
            lon=p.lon + rng.normal(0, 0.4) / (111_320 * math.cos(math.radians(p.lat))),
        )
        for p in points
    ]
    r = refine_from_frames(t, stopped, STOPPED_AFTER, GpxTrack((tuple(noisy),), "1.1", "none"), TRUE_START)
    assert r.speed_source == "positions"
    assert r.applied and r.shift_s is not None and abs(r.shift_s) < 0.75


# ------------------------------------------------------------------------------------------ the whole job


def test_align_end_to_end(tmp_path: Path) -> None:
    t, stopped, points = simulate([(18.0, 31.0)])
    gpx = tmp_path / "20261011.gpx"
    gpx.write_text(gpx11(points), encoding="utf-8")
    video = tmp_path / "VID_20261011_143022.mp4"  # name: 21:30:22 UTC (+0.5) vs the true 21:30:22.7
    meta = VideoMeta(TRUE_START + timedelta(seconds=60.3), 60.0, "pyav")  # creation_time = the end

    def frames(duration_s: float | None) -> tuple[list[Any], Path | None]:
        assert duration_s == 60.0
        return [SimpleNamespace(t=float(a), ego_stopped=bool(b)) for a, b in zip(t, stopped, strict=True)], None

    result = A.align(video, gpx, frames=frames, meta_reader=lambda _p: meta)
    assert result.out == tmp_path / "20261011.aligned.gpx"
    assert result.decision.source == "name" and result.decision.creation_role == "end"
    assert result.refinement is not None and result.refinement.applied
    assert abs((result.video_start - TRUE_START).total_seconds()) < 0.5
    replay = GpxReplay(result.out)
    assert replay.fix_at(0.0) is not None
    summary = A.format_summary(result, A.DEFAULT_TZ, tmp_path)
    assert "confidence high" in summary
    assert "--gpx 20261011.aligned.gpx --set video.file.hfov_deg=null" in summary
    assert "WARNING" not in summary


def test_align_without_refinement_warns_about_coverage(tmp_path: Path) -> None:
    points = straight_track(SPEEDS, TRUE_START - timedelta(hours=2))  # the GPX is from another drive
    gpx = tmp_path / "other.gpx"
    gpx.write_text(write_gpx([points], points[0].time, "x", "x", "x"), encoding="utf-8")
    meta = VideoMeta(None, 60.0, "none")
    result = A.align(tmp_path / "VID_20261011_143022.mp4", gpx, meta_reader=lambda _p: meta)
    assert result.refinement is None
    summary = A.format_summary(result, A.DEFAULT_TZ, tmp_path)
    assert "NOT RUN (--no-refine)" in summary
    assert "does not cover the video" in summary


def test_cli_timestamps_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import align_gpx

    monkeypatch.setattr(A.shutil, "which", lambda _name: None)
    video = tmp_path / "VID_20261011_143022.mp4"
    video.write_bytes(b"not a video")  # unreadable: no creation_time, the name still gives the start
    points = straight_track(SPEEDS, TRUE_START)
    gpx = tmp_path / "drive.gpx"
    gpx.write_text(gpx11(points), encoding="utf-8")
    assert align_gpx.main([str(video), str(gpx), "--no-refine"]) == 0
    out = capsys.readouterr().out
    assert "from the file name" in out and "video_start  2026-10-11 21:30:22.500 UTC" in out
    assert GpxReplay(tmp_path / "drive.aligned.gpx").fix_at(1.0) is not None
    assert align_gpx.main([str(video), str(gpx), "--no-refine", "--out", str(gpx)]) == 1  # would overwrite
