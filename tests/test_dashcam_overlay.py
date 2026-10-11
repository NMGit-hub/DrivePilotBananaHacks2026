"""The Mercedes dashcam overlay reader (nav/dashcam_overlay.py): parsing, clip times and track smoothing."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from nav.dashcam_overlay import (
    GLYPHS_PATH,
    Glyphs,
    TrackSample,
    clip_start,
    parse_left,
    parse_right,
    smooth_track,
    video_start_utc,
)


def test_left_block() -> None:
    assert parse_left("2026-09-14  07:54:20   27.4 mph  ") == (datetime(2026, 9, 14, 7, 54, 20), 27.4)
    assert parse_left("2026 09-19  11:41:52   0.0 mph   ") == (datetime(2026, 9, 19, 11, 41, 52), 0.0)  # dim dash
    assert parse_left("2026-09-14  07:54:2?   27.4 mph") is None  # an unreadable digit: skipped, never guessed
    assert parse_left("2026-13-14  07:54:20   27.4 mph") is None


def test_right_block_signs_come_from_the_hemisphere() -> None:
    lat, lon = parse_right("47°36'35\"N -122°-7'-16\"W") or (0.0, 0.0)
    assert math.isclose(lat, 47 + 36 / 60 + 35 / 3600) and math.isclose(lon, -(122 + 7 / 60 + 16 / 3600))
    assert parse_right("47°34'44\"N  122° 24' 29\"W") == parse_right("47°34'44\"N -122°-24'-29\"W")  # dim minus
    assert parse_right("47°61'44\"N -122°-24'-29\"W") is None  # 61 minutes: a misread
    assert parse_right("47°36'35\"N -122°-7'-16\"") is None


def test_clip_times() -> None:
    start = clip_start("dm_20260914_075350.mp4")
    assert start is not None and start.hour == 7 and start.utcoffset() is not None
    utc = video_start_utc("dm_20260914_075350")
    assert utc is not None and utc.hour == 14  # PDT = UTC-7
    assert clip_start("VID_20261011_143022.mp4") is None


def test_smoothing_recovers_a_straight_drive_from_arc_second_steps() -> None:
    """Driving north at 12 m/s: the overlay's latitude moves in 1" (31 m) steps; smoothed, the track is within a
    few metres of the truth."""
    lat0, lon0 = 47.6, -122.12
    samples, truth = [], []
    for k in range(120):
        north = 12.0 * k
        true_lat = lat0 + north / 111_320.0
        quant = math.floor(true_lat * 3600) / 3600  # whole arc-seconds
        samples.append(TrackSample(float(k), 12.0, quant, lon0))
        truth.append(true_lat)
    smoothed = smooth_track(samples)
    err = np.abs((np.array([s.lat for s in smoothed]) - np.array(truth)) * 111_320.0)[10:-10]
    raw = np.abs((np.array([s.lat for s in samples]) - np.array(truth)) * 111_320.0)[10:-10]
    assert err.max() < raw.max() and float(np.median(err)) < 16.0  # (truncation biases by half a step: ~15 m)


def test_saved_glyphs_cover_the_overlay() -> None:
    glyphs = Glyphs.load(GLYPHS_PATH)
    assert set("0123456789-:.'\"°NWmph") <= set(glyphs.chars)
    assert glyphs.patches.shape[1:] == (22, 15)
