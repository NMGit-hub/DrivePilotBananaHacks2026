"""M6 navigation: geometry, route parsing, progress, off-route/reroute, lane math, roundabouts, answers, GPS."""

from __future__ import annotations

import itertools
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from core.config import NavConfig
from nav.geo import Polyline, bearing_deg, destination, haversine_m
from nav.gps import GpsFix, GpxReplay, PhoneGps
from nav.handler import NO_GPS, NO_ROUTE, NavHandler
from nav.mapbox import FixtureClient, MapboxClient, NavError, Place, pick_place
from nav.phrasing import describe_lanes, lane_change, ordinal, spoken_distance
from nav.progress import RouteTracker
from nav.route import Lane, parse_directions
from state.world import empty_snapshot
from tests.nav_fixture import START, build, track
from voice.router import route as ask

RESPONSE, LINE = build()
ROUTE = parse_directions(RESPONSE, "Redmond Town Center")
PLACE = Place("Redmond Town Center", "7525 166th Ave NE, Redmond, WA", LINE.points[-1], "poi")


def lanes(pattern: str) -> list[Lane]:
    return [Lane(("x",), c == "V") for c in pattern]


# --------------------------------------------------------------------------------------------- geometry


def test_geometry_basics() -> None:
    assert haversine_m((47.0, -122.0), (48.0, -122.0)) == pytest.approx(111_195, rel=1e-3)
    assert bearing_deg((47.0, -122.0), (47.1, -122.0)) == pytest.approx(0.0, abs=0.01)
    there = destination((47.0, -122.0), 37.0, 1500.0)
    assert haversine_m((47.0, -122.0), there) == pytest.approx(1500.0, abs=0.5)
    assert bearing_deg((47.0, -122.0), there) == pytest.approx(37.0, abs=0.05)


def test_snapping_onto_a_polyline() -> None:
    line = Polyline([(47.0, -122.0), destination((47.0, -122.0), 90.0, 1000.0)])
    fix = destination(destination((47.0, -122.0), 90.0, 400.0), 0.0, 30.0)  # 400 m along, 30 m to the side
    snap = line.snap(fix)
    assert snap.along_m == pytest.approx(400.0, abs=0.5) and snap.offset_m == pytest.approx(30.0, abs=0.5)


def test_snapping_prefers_where_we_were_on_a_route_that_doubles_back() -> None:
    out_and_back = [(47.0, -122.0), destination((47.0, -122.0), 90.0, 1000.0), (47.0001, -122.0)]
    line = Polyline(out_and_back)
    fix = destination((47.0, -122.0), 90.0, 300.0)
    assert line.snap(fix, near_m=300.0, window_m=200.0).along_m < 1000.0
    assert line.snap(fix, near_m=1700.0, window_m=200.0).along_m > 1000.0


# ------------------------------------------------------------------------------------------ the route


def test_route_parsing() -> None:
    steps = ROUTE.steps
    assert [s.maneuver for s in steps] == ["depart", "turn", "off ramp", "roundabout", "exit roundabout", "arrive"]
    assert [round(s.start_m, -1) for s in steps] == [0, 800, 3200, 5200, 5230, 5530]
    assert steps[1].needs_lanes and [lane.valid for lane in steps[1].lanes] == [True, False, False]
    assert steps[2].exit_ref == "14" and steps[2].destinations == "SR 520 East: Redmond"
    assert steps[3].is_roundabout and steps[3].roundabout_exit == 2
    assert [round(m) for m in steps[3].roundabout_exits_m] == [5215]  # one exit passed before ours
    assert ROUTE.next_maneuver(900.0) is steps[2] and ROUTE.current_step(900.0) is steps[1]


def test_bad_directions_responses() -> None:
    from nav.route import RouteError

    for bad in (
        {"code": "NoRoute", "message": "no route"},
        {"code": "Ok", "routes": []},
        {"code": "Ok", "routes": [{"geometry": {"coordinates": []}, "legs": []}]},
    ):
        with pytest.raises(RouteError):
            parse_directions(bad)


# ------------------------------------------------------------------------------------------- phrasing


@pytest.mark.parametrize(
    ("metres", "said"),
    [
        (20, "50 feet"),
        (100, "350 feet"),
        (250, "800 feet"),
        (402, "a quarter mile"),
        (800, "half a mile"),
        (1200, "three quarters of a mile"),
        (1609, "a mile"),
        (2300, "1.4 miles"),
        (25000, "16 miles"),
    ],
)
def test_spoken_distances(metres: float, said: str) -> None:
    assert spoken_distance(metres) == said


@pytest.mark.parametrize(
    ("pattern", "said"),
    [
        ("Vxx", "the left lane"),
        ("VVxx", "the two left lanes"),
        ("xxVV", "the two right lanes"),
        ("xxV", "the right lane"),
        ("xVx", "the middle lane"),
        ("xVVx", "the two middle lanes"),
        ("VxV", "lanes 1 and 3 from the left"),
        ("VVV", None),
        ("xxx", None),
    ],
)
def test_lane_descriptions(pattern: str, said: str | None) -> None:
    assert describe_lanes(lanes(pattern)) == said


@pytest.mark.parametrize(
    ("pattern", "ours", "said"),
    [
        ("VVxx", 1, "stay in this lane"),
        ("VVxx", 3, "move one lane left"),
        ("xxVV", 1, "move two lanes right"),
        ("xVx", 3, "move one lane left"),
    ],
)
def test_lane_math(pattern: str, ours: int, said: str) -> None:
    assert lane_change(lanes(pattern), ours) == said


def test_ordinals() -> None:
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
        "22nd",
    ]


# ------------------------------------------------------------------------------------------- progress


def fixes(**kwargs: object) -> list[GpsFix]:
    return [GpsFix(t, p[0], p[1], 15.0, heading) for t, p, heading in track(LINE, **kwargs)]  # type: ignore[arg-type]


def at(metres: float) -> GpsFix:
    """One fix at a distance along the route, at the time a car at 15 m/s from the start would get there."""
    return fixes(start_m=metres, until_m=metres, t0=metres / 15.0)[0]


def test_progress_along_the_route() -> None:
    tracker = RouteTracker(ROUTE, NavConfig())
    progress = [tracker.update(f) for f in fixes(until_m=1500.0)]
    assert all(b.along_m >= a.along_m for a, b in itertools.pairwise(progress))
    p = progress[-1]
    assert p.next_step is ROUTE.steps[2] and p.to_next_m == pytest.approx(3200 - p.along_m, abs=2)
    assert p.remaining_m == pytest.approx(LINE.length_m - p.along_m, abs=1) and not p.off_route


def test_off_route_only_after_50_m_for_5_s() -> None:
    tracker = RouteTracker(ROUTE, NavConfig())
    states = [tracker.update(f) for f in fixes(until_m=2000.0, off_route=(40.0, 52.0, 80.0))]
    flags = [s.off_route for s in states]
    assert not any(flags[:44]) and flags[46] and not flags[-1]  # 5 s after leaving; back on route afterwards


def test_brief_gps_glitch_is_not_off_route() -> None:
    tracker = RouteTracker(ROUTE, NavConfig())
    assert not any(s.off_route for s in (tracker.update(f) for f in fixes(until_m=2000.0, off_route=(40, 43, 80))))


def test_driving_the_wrong_way_is_off_route() -> None:
    tracker = RouteTracker(ROUTE, NavConfig())
    backwards = [GpsFix(t, f.lat, f.lon, 15.0, (f.heading or 0) + 180.0) for t, f in enumerate(fixes(until_m=600.0))]
    assert [tracker.update(f) for f in backwards][-1].off_route


# --------------------------------------------------------------------------------- handler + alerts


class Gps:
    def __init__(self) -> None:
        self.fix: GpsFix | None = GpsFix(0.0, START[0], START[1], 0.0, 90.0)

    def __call__(self) -> GpsFix | None:
        return self.fix


def handler(lane_reader=None) -> tuple[NavHandler, Gps, list[tuple[float, str, str]]]:  # type: ignore[no-untyped-def]
    gps = Gps()
    said: list[tuple[float, str, str]] = []
    client = FixtureClient({"redmond town center": PLACE}, {"redmond town center": RESPONSE})
    nav = NavHandler(
        NavConfig(), client, gps, lambda kind, text: said.append((gps.fix.t if gps.fix else 0, kind, text)), lane_reader
    )
    return nav, gps, said


def drive(nav: NavHandler, gps: Gps, fix_list: list[GpsFix]) -> None:
    for fix in fix_list:
        gps.fix = fix
        nav.on_fix(fix)


def test_destination_by_voice() -> None:
    nav, _, _ = handler()
    text = nav.answer(ask("Hey DrivePilot, I'm going to Redmond Town Center"), empty_snapshot())
    assert text.startswith("Okay, Redmond Town Center: 7 minutes, 3.4 miles.")
    assert "First, turn left onto 148th Avenue Northeast in half a mile." in text


def test_destination_needs_gps_and_a_place() -> None:
    nav, gps, _ = handler()
    gps.fix = None
    assert nav.set_destination("Redmond Town Center") == NO_GPS
    gps.fix = GpsFix(0.0, START[0], START[1])
    assert nav.set_destination("the moon") == "Sorry, I couldn't find the moon."
    assert nav.answer(ask("What's my next turn?"), empty_snapshot()) == NO_ROUTE


def test_guidance_along_the_whole_route() -> None:
    nav, gps, said = handler()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, fixes())
    texts = [(round(t), text) for t, _, text in said]
    kinds = [kind for _, kind, _ in said]
    assert kinds == ["nav_lanes", "nav_lanes", "nav_lanes", "nav_roundabout", "nav_not_this_exit", "nav_arrived"]
    assert texts[0][1] == "In a quarter mile, use the left lane for 148th Avenue Northeast."
    assert texts[1][1] == "In a mile, use the two right lanes for SR 520."
    assert texts[2][1] == "In a quarter mile, use the two right lanes for SR 520."
    assert texts[3][1] == "At the roundabout, take the 2nd exit onto Main Street."
    assert texts[4][1] == "Not this one, next exit."
    assert texts[5][1] == "You've arrived at Redmond Town Center."
    # the 1-mile reminder came about a mile before exit 14 (3200 m along the route, at 15 m/s)
    assert (3200 - 1609) / 15 - 1 <= texts[1][0] <= (3200 - 1609) / 15 + 1


def test_off_route_reroutes_once_and_says_so() -> None:
    nav, gps, said = handler()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, fixes(until_m=1300.0, off_route=(40.0, 60.0, 90.0)))  # ask soon after: "did I miss it?"
    deadline = time.perf_counter() + 10.0
    while not any(k == "nav_rerouted" for _, k, _ in said) and time.perf_counter() < deadline:
        time.sleep(0.02)
    rerouted = [text for _, kind, text in said if kind == "nav_rerouted"]
    assert len(rerouted) == 1 and rerouted[0].startswith("You're off the route, so I've rerouted.")
    assert nav.answer(ask("Did I miss my turn?"), empty_snapshot()).startswith("Yes, so I've rerouted.")


def test_route_questions() -> None:
    nav, gps, _ = handler()
    nav.set_destination("Redmond Town Center")
    gps.fix = at(1000.0)
    snap = empty_snapshot()
    assert nav.answer(ask("What's my next turn?"), snap) == "Take exit 14 onto SR 520 East toward Redmond in 1.4 miles."
    assert (
        nav.answer(ask("Is my exit this one or the next?"), snap)
        == "Not this one. Exit 14 toward Redmond is in 1.4 miles."
    )
    assert nav.answer(ask("Did I miss my turn?"), snap).startswith("No, you're on the route.")
    assert nav.answer(ask("How long until we get there?"), snap) == "About 6 minutes, 2.8 miles to go."
    assert (
        nav.answer(ask("Where am I going?"), snap)
        == "We're going to Redmond Town Center, 7525 166th Ave NE, Redmond, WA."
    )
    gps.fix = at(3000.0)
    assert nav.answer(ask("Which exit?"), snap) == "This one: exit 14 toward Redmond, in 700 feet."
    assert nav.answer(ask("Cancel the navigation"), snap) == "Okay, navigation is off."


def test_lane_questions_with_and_without_a_lane_count() -> None:
    nav, gps, _ = handler()
    nav.set_destination("Redmond Town Center")
    gps.fix = at(1500.0)
    text = nav.answer(ask("Which lane for 520 east?"), empty_snapshot())
    assert text == "For SR 520 in a mile, use the two right lanes."
    nav.lane_reader = lambda: (4, 1)
    assert nav.answer(ask("Am I in the right lane?"), empty_snapshot()).endswith(
        "You're in lane 1 from the left, so move two lanes right."
    )
    nav.lane_reader = lambda: (3, 1)  # the camera saw 3 lanes, the route says 4: don't guess
    assert "I can't tell exactly which lane you're in" in nav.answer(ask("Am I in the right lane?"), empty_snapshot())


def test_model_context_lines() -> None:
    nav, gps, _ = handler()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, fixes(start_m=2500.0, until_m=2600.0))
    lines = nav.extra_context()
    assert lines[0] == "Speed: 34 mph."
    assert lines[1].startswith("Navigation: next, take exit 14 onto SR 520 East toward Redmond in")
    assert lines[2] == "For that maneuver, use the two right lanes (left to right, 4 lanes)."


# ----------------------------------------------------------------------------------------------- GPS


def test_gpx_replay_interpolates_on_the_video_clock(tmp_path: Path) -> None:
    import gpxpy.gpx

    gpx = gpxpy.gpx.GPX()
    segment = gpxpy.gpx.GPXTrackSegment()
    start = datetime(2026, 10, 11, 9, 0, 0)
    for i in range(11):
        point = destination(START, 90.0, 10.0 * i)
        segment.points.append(gpxpy.gpx.GPXTrackPoint(point[0], point[1], time=start + timedelta(seconds=i)))
    track_ = gpxpy.gpx.GPXTrack()
    track_.segments.append(segment)
    gpx.tracks.append(track_)
    path = tmp_path / "drive.gpx"
    path.write_text(gpx.to_xml(), encoding="utf-8")
    replay = GpxReplay(path, offset_s=2.0)  # the GPX starts 2 s before the video
    fix = replay.fix_at(3.5)  # video 3.5 s = GPX 5.5 s
    assert fix is not None
    assert haversine_m(START, fix.latlon) == pytest.approx(55.0, abs=0.5)
    assert fix.speed_mps == pytest.approx(10.0, abs=0.1) and fix.heading == pytest.approx(90.0, abs=0.5)
    assert replay.fix_at(-3.0) is None and replay.fix_at(9.0) is None


def test_phone_gps_goes_stale() -> None:
    now = [100.0]
    phone = PhoneGps(stale_s=3.0, clock=lambda: now[0])
    assert phone.fix_at(0.0) is None
    phone.push(47.6, -122.2, speed_mps=12.0)
    assert phone.fix_at(0.0) is not None
    now[0] += 3.5
    assert phone.fix_at(0.0) is None


# ------------------------------------------------------------------------------------------ the client


def test_destination_choice_weighs_name_and_distance() -> None:
    near = (47.544, -122.017)
    candidates = [
        Place("Nomadic", "Lake Havasu City, Arizona", (34.47, -114.33), "poi"),
        Place("Space Needle Loop", "Seattle, Washington 98109", (47.62, -122.35), "street"),
    ]
    assert pick_place("Space Needle", candidates, near).name == "Space Needle Loop"
    town = [
        Place("Town Center", "Redmond", (47.672, -122.118), "neighborhood"),
        Place("Redmond Town Center", "7525 166th Ave NE", (47.671, -122.118), "poi"),
        Place("The French Bakery @Redmond Town Center", "7330 164th Ave NE", (47.670, -122.12), "poi"),
    ]
    assert pick_place("Redmond Town Center", town, near).name == "Redmond Town Center"


def test_mapbox_client_requests_and_errors() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "searchbox" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "properties": {
                                "name": "Redmond Town Center",
                                "full_address": "7525 166th Ave NE",
                                "feature_type": "poi",
                            },
                            "geometry": {"coordinates": [-122.118, 47.671]},
                        }
                    ]
                },
            )
        if "geocode" in request.url.path:
            return httpx.Response(200, json={"features": []})
        return httpx.Response(200, json=RESPONSE)

    client = MapboxClient("pk.test-token", transport=httpx.MockTransport(respond))
    place = client.geocode("Redmond Town Center", near=START)
    route = client.route(START, place, heading=92.0)
    assert place.name == "Redmond Town Center" and len(route.steps) == 6
    params = dict(seen[-1].url.params)
    assert params["roundabout_exits"] == "true" and params["steps"] == "true" and params["bearings"] == "92,60;"
    assert seen[0].url.params["proximity"] == f"{START[1]:.6f},{START[0]:.6f}"

    rejected = MapboxClient("pk.secret-token", transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(NavError) as error:
        rejected.geocode("anywhere")
    assert "token was rejected" in str(error.value) and "secret-token" not in str(error.value)


def test_fixture_client_from_files(tmp_path: Path) -> None:
    (tmp_path / "redmond_town_center.json").write_text(
        json.dumps({"place": {"name": "Redmond Town Center", "location": list(PLACE.location)}, "directions": RESPONSE})
    )
    client = FixtureClient.from_files(tmp_path)
    place = client.geocode("redmond town center")
    assert len(client.route(START, place).steps) == 6
    assert math.isclose(place.location[0], PLACE.location[0])
