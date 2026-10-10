"""BDD100K info files → GPX (HANDOFF.md 5A), and GPX replay aligned on the `video_start` waypoint.

Synthetic info dicts mirror the real format (checked on real files: `locations` at ~1 Hz with Unix-ms
timestamps, -1 for an unknown speed/course, `startTime`/`endTime` of the video). One test reads a real entry
when `data/bdd100k/bdd100k_info.zip` is present.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from nav.bdd_info import BddInfoError, parse_info, read_info, to_gpx
from nav.geo import destination, haversine_m
from nav.gps import GpxReplay

ROOT = Path(__file__).resolve().parents[1]
START = (37.7641, -122.4353)
T0 = 1_490_145_619_393  # the video's first frame (ms)


def info(
    first_offset_s: float = -0.4,
    n: int = 41,
    speed: float = 10.0,
    course: float = 0.0,
    duration_s: float = 40.0,
    **extra: Any,
) -> dict[str, Any]:
    """Driving due `course` at `speed` m/s, a fix every second from video start + `first_offset_s`."""
    locations = []
    for k in range(n):
        dt = first_offset_s + k
        lat, lon = destination(START, course, speed * dt)
        t_ms = round(T0 + dt * 1000)
        locations.append(
            {"timestamp": t_ms, "latitude": lat, "longitude": lon, "speed": speed, "course": course, "accuracy": 10.0}
        )
    return {
        "startTime": T0,
        "endTime": T0 + round(duration_s * 1000),
        "timelapse": False,
        "locations": locations,
        **extra,
    }


def replay_of(data: dict[str, Any], tmp_path: Path, **kwargs: Any) -> GpxReplay:
    path = tmp_path / "clip.gpx"
    path.write_text(to_gpx(parse_info(data, "clip", **kwargs)), encoding="utf-8")
    return GpxReplay(path)


def test_the_track_is_aligned_on_the_video_start(tmp_path: Path) -> None:
    # START is where we are at the video's first frame; the first fix is 0.4 s (4 m) earlier. Aligned on the first
    # fix instead, the whole track would be 4 m (0.4 s) behind the video.
    replay = replay_of(info(first_offset_s=-0.4), tmp_path)
    fix = replay.fix_at(0.0)
    assert fix is not None and haversine_m(START, fix.latlon) < 0.3
    first = replay.fix_at(-0.4)
    assert first is not None and haversine_m(START, first.latlon) == pytest.approx(4.0, abs=0.3)
    later = replay.fix_at(12.5)
    assert later is not None and haversine_m(START, later.latlon) == pytest.approx(125.0, abs=0.5)


def test_a_first_fix_after_the_video_start_leaves_the_start_without_gps(tmp_path: Path) -> None:
    replay = replay_of(info(first_offset_s=0.7), tmp_path)
    assert replay.fix_at(0.3) is None  # no GPS yet (not the first fix pretended to be at t=0)
    fix = replay.fix_at(0.7)
    assert fix is not None and haversine_m(START, fix.latlon) == pytest.approx(7.0, abs=0.3)


def test_speed_and_course_are_kept_and_due_north_is_a_heading(tmp_path: Path) -> None:
    fix = replay_of(info(course=0.0, speed=12.5), tmp_path).fix_at(5.0)
    assert fix is not None and fix.speed_mps == pytest.approx(12.5) and fix.heading == pytest.approx(0.0)


def test_speed_is_interpolated_between_fixes(tmp_path: Path) -> None:
    # Braking from 10 to 6 m/s between two fixes: halfway, 8 m/s (holding the earlier fix's speed lagged up to 1 s
    # behind the video; the phone's own speed already lags ~0.5-1 s, measured on the dev clips' stops).
    data = info(first_offset_s=0.0)
    data["locations"][6]["speed"] = 6.0
    fix = replay_of(data, tmp_path).fix_at(5.5)
    assert fix is not None and fix.speed_mps == pytest.approx(8.0)


def test_unknown_speed_and_course_come_from_the_positions(tmp_path: Path) -> None:
    data = info(course=90.0, speed=8.0)
    for loc in data["locations"]:
        loc["speed"], loc["course"] = -1.0, -1.0
    track = parse_info(data, "clip")
    assert all(f.speed_mps is None and f.course_deg is None for s in track.segments for f in s)
    fix = replay_of(data, tmp_path).fix_at(10.0)
    assert fix is not None and fix.speed_mps == pytest.approx(8.0, abs=0.1)
    assert fix.heading == pytest.approx(90.0, abs=1)


def test_bad_fixes_are_dropped_and_counted() -> None:
    data = info()
    locs = data["locations"]
    locs[3]["accuracy"] = 165.0  # inaccurate
    locs[5]["timestamp"] = locs[4]["timestamp"]  # repeated timestamp
    locs[7]["latitude"], locs[7]["longitude"] = 0.0, 0.0  # no position
    locs[9]["latitude"] = None
    locs.append(dict(locs[-1], timestamp=locs[-1]["timestamp"] + 60_000))  # a minute after the video ended
    track = parse_info(data, "clip")
    assert track.dropped == {"inaccurate": 1, "not after the previous fix": 1, "no position": 2, "outside the video": 1}
    assert track.fixes == 41 - 4


def test_a_gap_splits_the_track_and_is_reported_as_no_fix(tmp_path: Path) -> None:
    data = info()
    del data["locations"][10:18]  # fixes at 8.6 s, then 17.6 s: a 9 s gap
    track = parse_info(data, "clip", gap_s=5.0)
    assert [len(s) for s in track.segments] == [10, 23]
    replay = replay_of(data, tmp_path)
    assert replay.fix_at(8.0) is not None and replay.fix_at(18.0) is not None
    assert replay.fix_at(12.0) is None  # no straight line through the gap
    edge = replay.fix_at(8.6)  # exactly the last fix before the gap
    assert edge is not None and haversine_m(START, edge.latlon) == pytest.approx(86.0, abs=0.5)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (info(timelapse=True), "time-lapse"),
        (info(n=0), "fewer than two"),
        (info(n=1), "fewer than two"),
        ({**info(), "startTime": None}, "startTime"),
    ],
)
def test_unusable_info_files_are_refused(data: dict[str, Any], message: str) -> None:
    with pytest.raises(BddInfoError, match=message):
        parse_info(data, "clip")


def test_info_is_read_from_either_split_without_unpacking(tmp_path: Path) -> None:
    archive = tmp_path / "bdd100k_info.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("bdd100k/info/100k/train/aaaaaaaa-11111111.json", json.dumps(info(speed=3.0)))
        z.writestr("bdd100k/info/100k/val/bbbbbbbb-22222222.json", json.dumps(info(speed=4.0)))
    assert read_info(archive, "aaaaaaaa-11111111")["locations"][0]["speed"] == 3.0
    assert read_info(archive, "bbbbbbbb-22222222")["locations"][0]["speed"] == 4.0
    with pytest.raises(BddInfoError, match="no info file"):
        read_info(archive, "cccccccc-33333333")


def test_gpx_without_a_video_start_waypoint_still_uses_the_offset(tmp_path: Path) -> None:
    text = to_gpx(parse_info(info(first_offset_s=0.0), "clip"))
    start = text.index("<wpt")
    plain = text[:start] + text[text.index("</wpt>") + len("</wpt>") :]  # a GPX from any other logger
    path = tmp_path / "plain.gpx"
    path.write_text(plain, encoding="utf-8")
    fix = GpxReplay(path, offset_s=2.0).fix_at(3.0)  # video 3 s = GPX 5 s
    assert fix is not None and haversine_m(START, fix.latlon) == pytest.approx(50.0, abs=0.5)


ZIP = ROOT / "data" / "bdd100k" / "bdd100k_info.zip"


@pytest.mark.skipif(not ZIP.is_file(), reason="bdd100k_info.zip not downloaded (HANDOFF.md 5A)")
def test_a_real_dev_clip_converts(tmp_path: Path) -> None:
    # SF red-light clip: drives up, waits at the light from ~21 s (ego-motion labels), light turns green ~39 s.
    track = parse_info(read_info(ZIP, "b97b9ee8-612c36ef"), "b97b9ee8-612c36ef")
    assert track.fixes >= 38 and len(track.segments) == 1 and abs(track.first_fix_after_start_s) < 1.0
    path = tmp_path / "real.gpx"
    path.write_text(to_gpx(track), encoding="utf-8")
    replay = GpxReplay(path)
    waiting = replay.fix_at(30.0)
    assert waiting is not None and waiting.speed_mps is not None and waiting.speed_mps < 0.5
