"""V2-M0 (HANDOFF.md 5B): simulated GPS drives along a route, their GPX, and the navigation check through the real
handler. Everything runs on the hand-made route in tests/nav_fixture.py (saved Mapbox routes stay out of git)."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from core.config import NavConfig
from eval.nav_check import check_drive, report_lines
from nav.geo import Polyline, angle_diff, destination, haversine_m
from nav.gps import VIDEO_START, GpxReplay, TrackPoint, write_gpx
from nav.mapbox import Place
from nav.route import parse_directions
from nav.sim_drive import SimParams, gauss_markov, maneuver_speed, simulate
from tests.nav_fixture import START, build

RESPONSE, LINE = build()
ROUTE = parse_directions(RESPONSE, "Redmond Town Center")
PLACE = Place("Redmond Town Center", "7525 166th Ave NE, Redmond, WA", LINE.points[-1], "poi")
P = SimParams()
CFG = NavConfig()
ROOT = Path(__file__).resolve().parents[1]
LEFT_TURN, RAMP, ROUNDABOUT, RING_EXIT, ARRIVE = (ROUTE.steps[i] for i in range(1, 6))


def true_speeds(drive: Any) -> np.ndarray:
    return np.array([f.true_speed_mps for f in drive.fixes])


# ------------------------------------------------------------------------------------------ speeds


def test_speed_profile_is_bounded_and_within_the_acceleration_limits() -> None:
    drive = simulate(ROUTE, stops_before=[1], seed=1)
    v = true_speeds(drive)
    assert v.min() >= 0.0 and v.max() <= P.max_cruise_mps + 1e-9
    dv = np.diff(v)  # fixes are 1 s apart
    assert dv.max() <= P.accel_mps2 + 1e-6 and dv.min() >= -P.decel_mps2 - 1e-6
    t, s, speed = drive.knots  # the exact profile: v^2 changes by at most 2 a ds between knots
    moved = np.diff(s) > 0
    rate = np.diff(speed**2)[moved] / (2.0 * np.diff(s)[moved])
    assert rate.max() <= P.accel_mps2 + 1e-6 and rate.min() >= -P.decel_mps2 - 1e-6
    assert np.all(np.diff(t) > 0)


def test_the_car_slows_for_each_maneuver_and_speeds_up_after() -> None:
    drive = simulate(ROUTE, seed=1)
    reached = {m.step.index: m for m in drive.maneuvers}
    cruise = 13.4 * P.cruise_factor  # the fixture's steps average 13.4 m/s (30 mph): 34.5 mph cruise
    assert reached[1].speed_mps <= P.turn_mps + 1e-6  # left turn
    assert reached[2].speed_mps == pytest.approx(min(cruise, P.ramp_mps))  # exit 14: a ramp is no slower here
    assert reached[3].speed_mps <= P.roundabout_mps + 1e-6 and reached[4].speed_mps <= P.roundabout_mps + 1e-6
    before, after = drive.time_at(LEFT_TURN.start_m - 150.0), drive.time_at(LEFT_TURN.start_m + 200.0)
    assert drive.speed_at(before) == pytest.approx(cruise) and drive.speed_at(after) == pytest.approx(cruise)
    # Braking from 15.4 to 7.5 m/s at 2.5 m/s² takes 36 m, ending where the turn's slow stretch starts (6 m before).
    braking = [drive.speed_at(drive.time_at(LEFT_TURN.start_m - d)) for d in (35.0, 25.0, 15.0, 8.0)]
    assert all(a > b for a, b in itertools.pairwise(braking)) and braking[0] < cruise
    slow_ramps = simulate(ROUTE, SimParams(ramp_mps=10.0), seed=1)
    assert next(m for m in slow_ramps.maneuvers if m.step is RAMP).speed_mps == pytest.approx(10.0)
    assert maneuver_speed(ARRIVE, P) == math.inf and maneuver_speed(ROUTE.steps[0], P) == math.inf


def test_the_car_stops_where_asked() -> None:
    drive = simulate(ROUTE, stops_before=[1, 2], seed=1)
    for step in (LEFT_TURN, RAMP):
        standing = [f for f in drive.fixes if f.true_speed_mps == 0.0 and abs(f.along_m - step.start_m) < 10.0]
        assert len(standing) >= P.stop_s - 1  # 20 s at the stop line, one fix a second
        assert all(step.start_m - f.along_m == pytest.approx(P.stop_line_m) for f in standing)
    assert [(s.step.index, round(s.t_end - s.t_start)) for s in drive.stops] == [(1, 20), (2, 20)]
    plain = simulate(ROUTE, seed=1)
    middle = [f for f in plain.fixes if 50.0 < f.along_m < LINE.length_m - 50.0]
    assert min(f.true_speed_mps for f in middle) > 3.0  # no standing still unless asked
    assert plain.duration_s == pytest.approx(drive.duration_s - 2 * P.stop_s, abs=12.0)  # plus braking/starting


def test_fixes_come_once_a_second_from_a_standing_start_to_a_standing_end() -> None:
    drive = simulate(ROUTE, seed=1)
    t = np.array([f.t for f in drive.fixes])
    assert t[0] == 0.0 and np.all(np.diff(t) == 1.0)
    assert all(f.true_speed_mps == 0.0 and f.along_m == 0.0 for f in drive.fixes[: int(P.start_wait_s)])
    assert all(f.along_m == pytest.approx(LINE.length_m) for f in drive.fixes[-int(P.end_wait_s) :])


def test_the_distance_driven_is_the_route_length() -> None:
    drive = simulate(ROUTE, seed=3)
    assert drive.distance_m == pytest.approx(LINE.length_m) and drive.fixes[-1].along_m == pytest.approx(LINE.length_m)
    chords = sum(haversine_m(a.true_latlon, b.true_latlon) for a, b in itertools.pairwise(drive.fixes))
    assert chords == pytest.approx(LINE.length_m, rel=0.005)  # 1 s chords cut the corners a little
    assert max(LINE.snap(f.true_latlon).offset_m for f in drive.fixes) < 0.5  # the truth is on the route


# ------------------------------------------------------------------------------------------- noise


def test_gauss_markov_noise_has_the_asked_spread_and_memory() -> None:
    x = gauss_markov(40_000, sigma=3.0, tau_s=30.0, dt=1.0, rng=np.random.default_rng(5))
    assert abs(x.mean()) < 0.3 and x.std() == pytest.approx(3.0, rel=0.08)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag1 == pytest.approx(math.exp(-1.0 / 30.0), abs=0.01)  # a slow walk, not white noise
    assert np.corrcoef(x[:-60], x[60:])[0, 1] == pytest.approx(math.exp(-2.0), abs=0.05)  # mean-reverting
    white = gauss_markov(40_000, sigma=3.0, tau_s=0.0, dt=1.0, rng=np.random.default_rng(5))
    assert abs(np.corrcoef(white[:-1], white[1:])[0, 1]) < 0.02


def test_the_position_error_is_a_few_metres_and_wanders_slowly() -> None:
    errors, steps = [], []
    for seed in range(8):
        drive = simulate(ROUTE, seed=seed)
        e = np.array([haversine_m(f.latlon, f.true_latlon) for f in drive.fixes])
        errors.append(e)
        east_north = np.array(
            [
                (haversine_m(f.true_latlon, (f.true_lat, f.lon)) * np.sign(f.lon - f.true_lon),
                 haversine_m(f.true_latlon, (f.lat, f.true_lon)) * np.sign(f.lat - f.true_lat))
                for f in drive.fixes
            ]
        )  # fmt: skip
        steps.append(np.linalg.norm(np.diff(east_north, axis=0), axis=1))
    rms = math.sqrt(float(np.mean(np.concatenate(errors) ** 2)))
    assert rms == pytest.approx(P.noise_m * math.sqrt(2.0), rel=0.3)  # ~4.2 m horizontal
    assert max(e.max() for e in errors) < 5.0 * P.noise_m
    assert float(np.median(np.concatenate(steps))) < 0.5 * P.noise_m  # fix to fix, the error barely moves


def test_reported_speed_and_course_look_like_a_phone() -> None:
    drive = simulate(ROUTE, seed=2)
    for f in drive.fixes:
        assert f.speed_mps >= 0.0 and abs(f.speed_mps - f.true_speed_mps) < 5.0 * P.speed_noise_mps
        if f.true_speed_mps < P.course_min_speed_mps:
            assert f.course_deg is None  # standing still: no course
        else:
            assert f.course_deg is not None and angle_diff(f.course_deg, f.true_heading) < 5.0 * P.course_noise_deg


def test_the_same_seed_gives_the_same_drive() -> None:
    a, b = simulate(ROUTE, stops_before=[1], seed=11), simulate(ROUTE, stops_before=[1], seed=11)
    assert a.fixes == b.fixes and a.to_gpx() == b.to_gpx()
    assert simulate(ROUTE, stops_before=[1], seed=12).fixes != a.fixes


# -------------------------------------------------------------------------------------- missed turn


def test_a_missed_turn_goes_straight_on_and_leaves_the_route() -> None:
    drive = simulate(ROUTE, missed_turn=1, seed=1)
    assert drive.missed is LEFT_TURN and drive.on_route_m == pytest.approx(LEFT_TURN.start_m)
    passed = next(m for m in drive.maneuvers if m.missed)
    assert passed.speed_mps > 2 * P.turn_mps  # no slowing for a turn we don't take
    after = [f for f in drive.fixes if f.along_m > drive.on_route_m + 1.0]
    assert all(angle_diff(f.true_heading, 90.0) < 1.0 for f in after)  # straight on, east, as we came in
    offsets = [LINE.snap(f.true_latlon).offset_m for f in after]
    far = [o for o in offsets if o > CFG.off_route_m]
    assert len(far) >= CFG.off_route_s + 1 and offsets[-1] >= P.missed_turn_clear_m  # long enough to reroute
    assert drive.fixes[-1].along_m == pytest.approx(drive.distance_m, abs=20.0)  # the track ends out there
    assert drive.distance_m - drive.on_route_m >= P.missed_turn_m
    assert drive.time_off_route(CFG.off_route_m) is not None and simulate(ROUTE).time_off_route(50.0) is None


def test_impossible_drives_are_refused() -> None:
    with pytest.raises(ValueError, match="missed turn"):
        simulate(ROUTE, missed_turn=0)
    with pytest.raises(ValueError, match="missed turn"):
        simulate(ROUTE, missed_turn=len(ROUTE.steps) - 1)
    with pytest.raises(ValueError, match="after the missed turn"):
        simulate(ROUTE, stops_before=[3], missed_turn=2)
    with pytest.raises(ValueError, match="stop"):
        simulate(ROUTE, stops_before=[5])
    straight = copy.deepcopy(RESPONSE)  # a "new name" step: straight on is the route itself
    line = Polyline([START, destination(START, 90.0, 1000.0), destination(START, 90.0, 4000.0)])
    straight["routes"][0]["geometry"]["coordinates"] = [[p[1], p[0]] for p in line.points]
    straight["routes"][0]["legs"][0]["steps"] = [
        {"maneuver": {"type": "depart", "location": [START[1], START[0]]}, "distance": 1000.0, "duration": 70.0},
        {
            "maneuver": {
                "type": "new name",
                "modifier": "straight",
                "location": [line.points[1][1], line.points[1][0]],
            },
            "distance": 3000.0,
            "duration": 200.0,
        },
        {"maneuver": {"type": "arrive", "location": [line.points[2][1], line.points[2][0]]}, "distance": 0.0},
    ]
    with pytest.raises(ValueError, match="nothing to miss"):
        simulate(parse_directions(straight, "X"), missed_turn=1)


# ---------------------------------------------------------------------------------------------- GPX


def test_the_gpx_replays_on_the_video_clock(tmp_path: Path) -> None:
    drive = simulate(ROUTE, stops_before=[1], seed=4)
    start = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    text = drive.to_gpx(start)
    assert 'version="1.0"' in text and f"<name>{VIDEO_START}</name>" in text
    path = tmp_path / "sim.gpx"
    path.write_text(text, encoding="utf-8")
    replay = GpxReplay(path)
    assert replay.duration_s == pytest.approx(drive.duration_s)
    assert replay.fix_at(-0.5) is None and replay.fix_at(drive.duration_s + 0.5) is None
    for f in drive.fixes[:: max(1, len(drive.fixes) // 40)]:
        got = replay.fix_at(f.t)  # video t = drive t: the first fix is the video's first frame
        assert got is not None and haversine_m(got.latlon, f.latlon) < 0.05
        assert got.speed_mps == pytest.approx(f.speed_mps, abs=0.006)
        if f.course_deg is not None:
            assert got.heading == pytest.approx(f.course_deg, abs=0.06)
    import gpxpy

    parsed = gpxpy.parse(text)
    assert [(w.name, w.time) for w in parsed.waypoints] == [(VIDEO_START, start)]
    assert parsed.tracks[0].segments[0].points[0].time == start


def test_write_gpx_needs_a_point() -> None:
    with pytest.raises(ValueError):
        write_gpx([[]], datetime(2026, 1, 1, tzinfo=UTC), "x", "x", "x")
    point = TrackPoint(datetime(2026, 1, 1, tzinfo=UTC), 47.6, -122.2, 3.0, None)
    assert "<speed>3.0</speed>" in write_gpx([[point]], point.time, "x", "x", "x")


# ------------------------------------------------------------------------------------ the nav check


def test_the_nav_check_passes_a_drive_along_the_fixture_route() -> None:
    result = check_drive(simulate(ROUTE, stops_before=[1], seed=1), RESPONSE, PLACE, CFG)
    assert result.ok, "\n".join(report_lines(result))
    kinds = [r.kind for r in result.alerts]
    assert kinds == ["nav_lanes", "nav_lanes", "nav_lanes", "nav_roundabout", "nav_not_this_exit", "nav_arrived"]
    lanes = [(r.step.index, r.target_m) for r in result.alerts if r.kind == "nav_lanes" and r.step is not None]
    assert lanes == [(1, 402.0), (2, 1609.0), (2, 402.0)]  # the configured distances (config.yaml)
    for r in result.alerts[:4]:  # lane reminders and the roundabout: within 10% of their distance
        assert r.true_m is not None and r.target_m is not None and abs(r.true_m - r.target_m) <= 0.1 * r.target_m
    assert result.reroute_requests == [] and result.alerts[-1].true_m is not None
    assert 0.0 <= result.alerts[-1].true_m <= CFG.arrival_radius_m + 15.0


def test_the_nav_check_times_the_reroute_after_a_missed_turn() -> None:
    result = check_drive(simulate(ROUTE, missed_turn=1, seed=1), RESPONSE, PLACE, CFG)
    assert result.ok, "\n".join(report_lines(result))
    assert result.left_route_t is not None and result.reroute_t is not None
    assert result.reroute_t - result.left_route_t == pytest.approx(CFG.off_route_s, abs=2.0)
    assert result.off_route_t == result.reroute_t
    assert "nav_arrived" not in [r.kind for r in result.alerts]


def test_the_nav_check_holds_over_seeds() -> None:
    for seed in range(5):
        for drive in (simulate(ROUTE, stops_before=[1, 2], seed=seed), simulate(ROUTE, missed_turn=2, seed=seed)):
            result = check_drive(drive, RESPONSE, PLACE, CFG)
            assert result.ok, "\n".join(report_lines(result))


def test_the_nav_check_catches_a_missing_lane_reminder() -> None:
    served = copy.deepcopy(RESPONSE)  # the handler's route has no lane data for exit 14; the drive's does
    for intersection in served["routes"][0]["legs"][0]["steps"][2]["intersections"]:
        intersection.pop("lanes", None)
    result = check_drive(simulate(ROUTE, seed=1), served, PLACE, CFG)
    failed = [c.name for c in result.checks if not c.ok]
    assert not result.ok and failed == ["lanes 1609 m before step 2", "lanes 402 m before step 2"]


def test_a_reroute_on_a_good_drive_fails_the_check() -> None:
    jumpy = NavConfig(off_route_m=2.0, off_route_s=0.0)  # every bit of GPS noise is "off the route"
    result = check_drive(simulate(ROUTE, seed=1), RESPONSE, PLACE, jumpy)
    assert not result.ok and any(c.name == "no reroute on a drive along the route" and not c.ok for c in result.checks)


# ------------------------------------------------------------------------------------------ scripts


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # (dataclasses look their module up)
    spec.loader.exec_module(module)
    return module


def saved_route(tmp_path: Path) -> Path:
    path = tmp_path / "redmond_town_center.json"
    place = {"name": PLACE.name, "address": PLACE.address, "location": list(PLACE.location)}
    path.write_text(json.dumps({"place": place, "directions": RESPONSE}), encoding="utf-8")
    return path


def test_route_to_gpx_writes_a_replayable_track(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sim.gpx"
    args = ["--route", str(saved_route(tmp_path)), "--out", str(out), "--stop-before", "1,2", "--seed", "3"]
    assert load_script("route_to_gpx").main([*args, "--missed-turn", "2", "--noise-m", "4"]) == 0
    printed = capsys.readouterr().out
    assert "Redmond Town Center: " in printed and "MISSED" in printed and "stop 20 s" in printed
    replay = GpxReplay(out)
    assert replay.fix_at(0.0) is not None and replay.duration_s > 100.0
    assert load_script("route_to_gpx").main([*args[:4], "--missed-turn", "9"]) == 2  # no such step


def test_sim_nav_check_picks_stops_and_a_turn_to_miss(tmp_path: Path) -> None:
    script = load_script("sim_nav_check")
    saved = script.load_saved(saved_route(tmp_path))
    assert script.stop_steps(saved.route) == [1]  # the left turn (the others are a ramp and a roundabout)
    assert script.missed_candidates(saved.route) == [1]
    results = script.run_route(saved, CFG, P, 7, None)
    assert [r.drive.missed.index if r.drive.missed else None for r in results] == [None, 1]
    assert all(r.ok for r in results) and "reroute asked" in script.describe(results[1])
