"""M6 review regressions: each test reproduces a finding of the independent navigation review (all confirmed)."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from core.config import NavConfig
from nav.geo import LatLon, Polyline, destination
from nav.gps import GpsFix
from nav.handler import GPS_LOST, NO_HOME, NavHandler
from nav.mapbox import FixtureClient, MapboxClient, NavError, Place
from nav.progress import RouteTracker
from nav.route import Route, parse_directions
from state.world import empty_snapshot
from tests.nav_fixture import START, build, track
from voice.router import route as ask

RESPONSE, LINE = build()
PLACE = Place("Redmond Town Center", "7525 166th Ave NE, Redmond, WA", LINE.points[-1], "poi")
OTHER = Place("Bellevue Square", "575 Bellevue Square, Bellevue, WA", LINE.points[-1], "poi")
SNAP = empty_snapshot()
SPEED = 15.0


class Gps:
    def __init__(self, fix: GpsFix | None = None) -> None:
        self.fix: GpsFix | None = fix or GpsFix(0.0, START[0], START[1], 0.0, 90.0)

    def __call__(self) -> GpsFix | None:
        return self.fix


Said = list[tuple[float, str, str]]


def make(
    response: dict[str, Any] = RESPONSE, client: Any = None, cfg: NavConfig | None = None, gps: Gps | None = None
) -> tuple[NavHandler, Gps, Said]:
    gps = gps or Gps()
    said: Said = []
    client = client or FixtureClient({"redmond town center": PLACE}, {"redmond town center": response})

    def on_alert(kind: str, text: str) -> None:
        said.append((gps.fix.t if gps.fix else 0.0, kind, text))

    return NavHandler(cfg or NavConfig(), client, gps, on_alert), gps, said


def at(metres: float, t: float | None = None, side: float = 0.0, line: Polyline = LINE) -> GpsFix:
    """A fix on the route (or `side` metres to its right) at the time a car at 15 m/s would get there."""
    point, heading = line.point_at(metres), line.bearing_at(metres)
    if side:
        point = destination(point, (heading + 90.0) % 360.0, side)
    return GpsFix(metres / SPEED if t is None else t, point[0], point[1], SPEED, heading)


def drive(nav: NavHandler, gps: Gps, fixes: list[GpsFix]) -> None:
    for fix in fixes:
        gps.fix = fix
        nav.on_fix(fix)


def along(start_m: float, end_m: float, **kwargs: Any) -> list[GpsFix]:
    return [GpsFix(t, p[0], p[1], SPEED, h) for t, p, h in track(LINE, start_m=start_m, until_m=end_m, **kwargs)]


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(0.01)
    return predicate()


def kinds(said: Said) -> list[str]:
    return [kind for _, kind, _ in said]


def without_lanes(step_index: int) -> dict[str, Any]:
    response = copy.deepcopy(RESPONSE)
    for intersection in response["routes"][0]["legs"][0]["steps"][step_index]["intersections"]:
        intersection.pop("lanes", None)
    return response


# ------------------------------------------------------------------------- 1. missed turn, asked at once


def test_missed_turn_is_noticed_when_asked_right_after_it() -> None:
    nav, gps, said = make()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, 795.0))
    corner = LINE.point_at(800.0)
    t0 = gps.fix.t if gps.fix else 0.0
    answers = []
    for k in range(1, 4):  # straight on (east) past the left turn, 15 m a second
        p = destination(corner, 90.0, 15.0 * k)
        drive(nav, gps, [GpsFix(t0 + k, p[0], p[1], SPEED, 90.0)])
        answers.append(nav.answer(ask("Did I miss my turn?"), SNAP))
    assert answers[0].startswith("No, you're on the route.")  # 15 m: within GPS noise of the corner
    assert answers[1] == "It looks like you've left the route. I'm finding a new one now."
    assert wait_for(lambda: "nav_rerouted" in kinds(said))
    assert kinds(said).count("nav_rerouted") == 1  # asking again doesn't start a second reroute


# ------------------------------------------------------------------------------------- 2. lane answers


def test_lane_question_without_lane_data_says_to_check_the_signs() -> None:
    nav, gps, _ = make(without_lanes(2))  # Mapbox often has no lanes on ramps
    nav.set_destination("Redmond Town Center")
    gps.fix = at(1500.0)
    text = nav.answer(ask("Which lane for 520 east?"), SNAP)
    assert text.startswith("I don't have lane data for that.") and "exit 14" in text and "overhead signs" in text


def test_lane_answer_is_about_the_next_maneuver() -> None:
    nav, gps, _ = make(without_lanes(1))  # the left turn has no lane data; exit 14 (later) has
    nav.set_destination("Redmond Town Center")
    gps.fix = at(500.0)
    text = nav.answer(ask("Am I in the right lane?"), SNAP)
    assert "turn left onto 148th Avenue Northeast" in text and "SR 520" not in text


def test_right_in_am_i_in_the_right_lane_is_not_a_road_name() -> None:
    response = copy.deepcopy(RESPONSE)
    ramp = response["routes"][0]["legs"][0]["steps"][2]
    ramp["maneuver"]["instruction"] = "Take exit 14 on the right toward Redmond."
    nav, gps, _ = make(response)
    nav.set_destination("Redmond Town Center")
    gps.fix = at(100.0)
    expected = "For 148th Avenue Northeast in half a mile, use the left lane."
    assert nav.answer(ask("Am I in the right lane?"), SNAP) == expected


def test_lane_question_naming_a_later_road_still_works() -> None:
    nav, gps, _ = make()
    nav.set_destination("Redmond Town Center")
    gps.fix = at(100.0)
    assert nav.answer(ask("Which lane for 520 east?"), SNAP) == "For SR 520 in 1.9 miles, use the two right lanes."
    assert nav.answer(ask("Which lane for my exit?"), SNAP).startswith("For SR 520")


# ------------------------------------------------------------------------------ 3. how far to my turn


def test_how_far_to_my_turn_is_the_next_turn_not_the_destination() -> None:
    nav, gps, _ = make()
    nav.set_destination("Redmond Town Center")
    gps.fix = at(1000.0)
    assert nav.answer(ask("How far to my turn?"), SNAP) == "In 1.4 miles: take exit 14 onto SR 520 East toward Redmond."
    assert nav.answer(ask("How far to my exit?"), SNAP) == "Not this one. Exit 14 toward Redmond is in 1.4 miles."
    assert nav.answer(ask("How long until we get there?"), SNAP).endswith("2.8 miles to go.")


# -------------------------------------------------------------------------------- 4. arrive phrasing


def test_destination_ahead_is_not_announced_as_already_reached() -> None:
    nav, gps, _ = make()
    nav.set_destination("Redmond Town Center")
    gps.fix = at(LINE.length_m - 230.0)
    text = nav.answer(ask("What's my next turn?"), SNAP)
    assert text.startswith("You'll reach your destination in") and "arrived" not in text
    assert any("you'll reach your destination" in line for line in nav.extra_context())
    assert nav.answer(ask("How far to my turn?"), SNAP).startswith("Your destination is in")


# ---------------------------------------------------------------- 5. reroute storms, arrival, ending


def test_starting_in_a_parking_lot_does_not_reroute() -> None:
    lot = destination(START, 0.0, 100.0)  # 100 m from the road the route starts on
    nav, gps, said = make(gps=Gps(GpsFix(0.0, lot[0], lot[1], 0.0, None)))
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, [GpsFix(float(t), lot[0], lot[1], 0.0, None) for t in range(1, 91)])
    time.sleep(0.2)
    assert said == []
    t = gps.fix.t if gps.fix else 0.0
    drive(nav, gps, along(0.0, 700.0, t0=t + 10))  # out of the lot and onto the route
    time.sleep(0.2)
    assert "nav_rerouted" not in kinds(said) and "nav_reroute_failed" not in kinds(said)


def test_leaving_the_lot_the_wrong_way_still_reroutes() -> None:
    lot = destination(START, 0.0, 100.0)
    nav, gps, said = make(gps=Gps(GpsFix(0.0, lot[0], lot[1], 0.0, None)))
    nav.set_destination("Redmond Town Center")
    away = [destination(lot, 0.0, 15.0 * k) for k in range(1, 30)]  # north, ever further from the route
    drive(nav, gps, [GpsFix(float(k), p[0], p[1], SPEED, 0.0) for k, p in enumerate(away, start=1)])
    assert wait_for(lambda: "nav_rerouted" in kinds(said))


def test_turning_into_the_destination_lot_is_arrival() -> None:
    nav, gps, said = make()
    nav.set_destination("Redmond Town Center")
    end = LINE.length_m
    drive(nav, gps, along(0.0, end - 60.0))
    t = gps.fix.t if gps.fix else 0.0
    corner = LINE.point_at(end - 60.0)
    lot = [destination(corner, 180.0, min(8.0 * k, 120.0)) for k in range(1, 31)]
    drive(nav, gps, [GpsFix(t + k, p[0], p[1], 3.0, 180.0) for k, p in enumerate(lot, start=1)])
    time.sleep(0.2)
    assert kinds(said)[-1] == "nav_arrived" and "nav_rerouted" not in kinds(said)


def test_navigation_ends_on_arrival() -> None:
    nav, gps, said = make()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, LINE.length_m))
    t = gps.fix.t if gps.fix else 0.0
    end = LINE.points[-1]
    away = [destination(end, 180.0, 10.0 * k) for k in range(1, 60)]
    drive(nav, gps, [GpsFix(t + k, p[0], p[1], 10.0, 180.0) for k, p in enumerate(away, start=1)])
    time.sleep(0.2)
    assert kinds(said)[-1] == "nav_arrived" and kinds(said).count("nav_arrived") == 1
    assert nav.route is None
    assert nav.answer(ask("What's my next turn?"), SNAP) == "You've arrived at Redmond Town Center."
    assert nav.answer(ask("Where am I going?"), SNAP) == "You've arrived at Redmond Town Center."


# ------------------------------------------------------------------- 6. self-crossing route, noisy fix


def _cloverleaf() -> tuple[Route, Polyline, Callable[[float, float], LatLon]]:
    origin = (47.6, -122.2)

    def xy(x: float, y: float) -> LatLon:  # metres east/north of the origin
        p = destination(origin, 90.0 if x >= 0 else 270.0, abs(x))
        return destination(p, 0.0 if y >= 0 else 180.0, abs(y))

    pts = [xy(x, 0.0) for x in range(0, 701, 25)]  # east 700 m
    for a in range(90, -181, -10):  # a 270° loop ramp, radius 50 m, ending at (650, -50) heading north
        pts.append(xy(700 + 50 * math.cos(math.radians(a)), -50 + 50 * math.sin(math.radians(a))))
    pts += [xy(650.0, y) for y in range(-25, 501, 25)]  # north, crossing the first leg at (650, 0)
    line = Polyline(pts)

    def ll(p: LatLon) -> list[float]:
        return [p[1], p[0]]

    ramp = line.point_at(700.0)
    response = {
        "code": "Ok",
        "routes": [
            {
                "distance": line.length_m,
                "duration": line.length_m / 13.4,
                "geometry": {"coordinates": [ll(p) for p in pts]},
                "legs": [
                    {
                        "steps": [
                            {
                                "maneuver": {"type": "depart", "location": ll(pts[0]), "instruction": "Drive east."},
                                "distance": 700.0,
                                "name": "A Street",
                            },
                            {
                                "maneuver": {"type": "off ramp", "location": ll(ramp), "instruction": "Take the ramp."},
                                "distance": line.length_m - 700.0,
                                "name": "B Street",
                            },
                            {
                                "maneuver": {"type": "arrive", "location": ll(pts[-1]), "instruction": "Arrive."},
                                "distance": 0.0,
                                "name": "B Street",
                            },
                        ]
                    }
                ],
            }
        ],
    }
    return parse_directions(response, "B"), line, xy


def test_a_noisy_fix_where_the_route_crosses_itself_does_not_jump_ahead() -> None:
    route, line, xy = _cloverleaf()
    tracker = RouteTracker(route, NavConfig())
    for t, m in enumerate(range(0, 641, 20)):
        tracker.update(at(float(m), float(t), line=line))
    crossing = xy(650.0, 2.0)  # 2 m off the first leg, right where the ramp later crosses it
    progress = tracker.update(GpsFix(33.0, crossing[0], crossing[1], SPEED, 90.0))
    assert progress.along_m < 700.0  # still on the first leg, not ~990 m along on the far side of the loop
    for t, m in enumerate(range(660, 691, 15), start=34):
        progress = tracker.update(at(float(m), float(t), line=line))
        assert abs(progress.along_m - m) < 20.0 and not progress.off_route
        assert progress.next_step is not None and progress.next_step.maneuver == "off ramp"  # not skipped


def test_a_real_gps_jump_is_followed_after_a_second_fix_agrees() -> None:
    tracker = RouteTracker(parse_directions(RESPONSE, "X"), NavConfig())
    tracker.update(at(0.0, 0.0))
    tracker.update(at(15.0, 1.0))
    first = tracker.update(at(3000.0, 2.0))  # one fix can't move us 3 km in a second...
    second = tracker.update(at(3015.0, 3.0))  # ...but two that agree can (a GPS restart)
    assert first.along_m < 100.0 and abs(second.along_m - 3015.0) < 20.0


# -------------------------------------------------------------------------------- 7. reroute races


class ScriptedClient(FixtureClient):
    """A fixture client whose reroutes can be held, failed, or made to crash."""

    def __init__(self, places: dict[str, Place] | None = None) -> None:
        places = places or {"redmond town center": PLACE, "bellevue square": OTHER}
        super().__init__(places, {name: RESPONSE for name in places})
        self.calls = 0
        self.fail_from_call: int | None = None
        self.error: Exception = NavError("mapbox: ReadTimeout")
        self.hold_call: int | None = None
        self.held = threading.Event()
        self.release = threading.Event()
        self.on_geocode: Callable[[str], None] | None = None

    def geocode(self, query: str, near: LatLon | None = None) -> Place:
        if self.on_geocode is not None:
            self.on_geocode(query)
        return super().geocode(query, near)

    def route(self, origin: LatLon, destination: Place, heading: float | None = None) -> Route:
        self.calls += 1
        call = self.calls
        if call == self.hold_call:
            self.held.set()
            self.release.wait(5.0)
        if self.fail_from_call is not None and call >= self.fail_from_call:
            raise self.error
        route = super().route(origin, destination, heading)
        route.destination = f"call {call}"
        return route


def go_off_route_until_held(nav: NavHandler, gps: Gps, client: ScriptedClient) -> None:
    for fix in along(0.0, 1500.0, off_route=(40.0, 80.0, 90.0)):
        drive(nav, gps, [fix])
        if client.held.is_set():
            return
    raise AssertionError("no reroute started")


def test_a_reroute_that_fails_after_cancel_says_nothing() -> None:
    client = ScriptedClient()
    client.hold_call, client.fail_from_call = 2, 2
    nav, gps, said = make(client=client)
    nav.set_destination("Redmond Town Center")
    go_off_route_until_held(nav, gps, client)
    assert nav.answer(ask("Cancel the navigation"), SNAP) == "Okay, navigation is off."
    client.release.set()
    time.sleep(0.3)
    assert "nav_reroute_failed" not in kinds(said)


def test_a_new_destination_forgets_the_old_reroute() -> None:
    nav, gps, said = make(client=ScriptedClient())
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, 1000.0, off_route=(40.0, 60.0, 90.0)))
    assert wait_for(lambda: "nav_rerouted" in kinds(said))
    t = gps.fix.t if gps.fix else 0.0
    gps.fix = GpsFix(t, START[0], START[1], 0.0, 90.0)
    assert nav.answer(ask("I'm going to Bellevue Square"), SNAP).startswith("Okay, Bellevue Square")
    gps.fix = at(100.0, t + 7.0)
    assert nav.answer(ask("Did I miss my turn?"), SNAP).startswith("No, you're on the route.")


def test_a_reroute_finishing_while_a_new_destination_is_set_is_dropped() -> None:
    client = ScriptedClient()
    client.hold_call = 2
    nav, gps, said = make(client=client)
    nav.set_destination("Redmond Town Center")
    go_off_route_until_held(nav, gps, client)

    def finish_old_reroute(query: str) -> None:  # the old reroute completes during the new geocode
        if "bellevue" in query.lower():
            client.release.set()
            time.sleep(0.3)

    client.on_geocode = finish_old_reroute
    answer = nav.answer(ask("Take me to Bellevue Square"), SNAP)
    time.sleep(0.2)
    assert answer.startswith("Okay, Bellevue Square")
    assert nav.place == OTHER and nav.route is not None and nav.route.destination == "call 3"
    assert "nav_rerouted" not in kinds(said)


def test_a_stale_reroute_to_the_same_place_is_dropped() -> None:
    client = ScriptedClient({"redmond town center": PLACE})  # geocode returns the very same Place object
    client.hold_call = 2
    nav, gps, said = make(client=client)
    nav.set_destination("Redmond Town Center")
    go_off_route_until_held(nav, gps, client)
    nav.cancel()
    nav.set_destination("Redmond Town Center")  # route call 3
    client.release.set()
    time.sleep(0.3)
    assert nav.route is not None and nav.route.destination == "call 3"
    assert "nav_rerouted" not in kinds(said)


def test_reroute_failures_are_said_once_and_retried_less_often() -> None:
    client = ScriptedClient()
    client.fail_from_call = 2
    nav, gps, said = make(client=client)
    nav.set_destination("Redmond Town Center")
    for fix in along(0.0, 3500.0, off_route=(40.0, 230.0, 120.0)):  # off the route for over 3 minutes
        drive(nav, gps, [fix])
        time.sleep(0.003)  # let each reroute thread finish before the next fix
    time.sleep(0.2)
    assert kinds(said).count("nav_reroute_failed") == 1
    assert 3 <= client.calls - 1 <= 5  # waits of 20, 40, 80, 120 s: not one attempt every 20 s (9)


def test_an_unexpected_reroute_error_is_spoken_not_lost() -> None:
    client = ScriptedClient()
    client.fail_from_call, client.error = 2, ValueError("cannot convert float NaN to integer")
    nav, gps, said = make(client=client)
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, 1000.0, off_route=(40.0, 60.0, 90.0)))
    assert wait_for(lambda: "nav_reroute_failed" in kinds(said))


# ------------------------------------------------------------------------------------- 8. stale GPS


def test_route_answers_say_when_gps_is_lost() -> None:
    nav, gps, _ = make()
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, 1000.0))
    gps.fix = None  # phone page closed, tunnel
    for question in ("Which exit?", "Did I miss my turn?", "How far to my turn?", "Which lane for 520 east?"):
        assert nav.answer(ask(question), SNAP).startswith(GPS_LOST), question
    assert nav.extra_context() == ["Navigation: the GPS signal is lost."]
    assert nav.answer(ask("Where am I going?"), SNAP).startswith("We're going to Redmond Town Center")


# -------------------------------------------------------------------------- 9. errors from the service


def test_unusable_routes_become_nav_errors() -> None:
    broken = copy.deepcopy(RESPONSE)
    broken["routes"][0]["geometry"] = {}
    client = FixtureClient({"redmond town center": PLACE}, {"redmond town center": broken})
    with pytest.raises(NavError):
        client.route(START, PLACE)

    sloppy = copy.deepcopy(RESPONSE)  # nulls and missing fields Mapbox can send
    sloppy["routes"][0]["legs"][0]["steps"][1]["distance"] = None
    del sloppy["routes"][0]["legs"][0]["steps"][3]["intersections"][1]["location"]
    route = parse_directions(sloppy, "X")
    assert route.steps[1].distance_m == 0.0 and route.steps[3].roundabout_exits_m == ()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "Ok", "routes": [{"geometry": {"coordinates": [[1, 2], "x"]}}]})

    mapbox = MapboxClient("pk.test-token", transport=httpx.MockTransport(respond))
    with pytest.raises(NavError):
        mapbox.route(START, PLACE)


def test_a_nan_heading_is_not_sent() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RESPONSE)

    mapbox = MapboxClient("pk.test-token", transport=httpx.MockTransport(respond))
    mapbox.route(START, PLACE, heading=float("nan"))  # the phone's heading while standing still
    assert "bearings" not in seen[-1].url.params


# ------------------------------------------------------------------------------ 10. roundabout exits


def _roundabout(exit_number: int, keep_intersections: int) -> dict[str, Any]:
    response = copy.deepcopy(RESPONSE)
    ring = response["routes"][0]["legs"][0]["steps"][3]
    ring["maneuver"]["exit"] = exit_number
    ring["maneuver"]["instruction"] = f"Enter the roundabout and take exit {exit_number} onto Main Street."
    ring["intersections"] = ring["intersections"][:keep_intersections]
    return response


def test_roundabout_first_exit_while_on_the_ring() -> None:
    nav, gps, said = make(_roundabout(1, 1))
    nav.set_destination("Redmond Town Center")
    ring_start = parse_directions(_roundabout(1, 1), "X").steps[3].start_m
    drive(nav, gps, along(0.0, ring_start + 5.0))
    assert nav.answer(ask("Which exit?"), SNAP) == "Take the next exit onto Main Street."
    assert "nav_not_this_exit" not in kinds(said)


def test_no_exit_counting_when_our_count_disagrees_with_mapbox() -> None:
    nav, gps, said = make(_roundabout(3, 2))  # Mapbox says exit 3, but we only see one exit before ours
    nav.set_destination("Redmond Town Center")
    drive(nav, gps, along(0.0, 5000.0))
    assert nav.answer(ask("Which exit?"), SNAP) == "At the roundabout in 700 feet, take the 3rd exit onto Main Street."
    drive(nav, gps, along(5000.0, LINE.length_m, t0=(gps.fix.t if gps.fix else 0.0) + 1.0))
    assert "nav_roundabout" in kinds(said) and "nav_not_this_exit" not in kinds(said)


# --------------------------------------------------------------------------------------- 11. router


def test_take_me_home_needs_a_home_address() -> None:
    nav, _, _ = make()
    assert nav.answer(ask("Take me home"), SNAP) == NO_HOME
    nav, _, _ = make(cfg=NavConfig(home="Redmond Town Center"))
    assert nav.answer(ask("Take me home"), SNAP).startswith("Okay, Redmond Town Center")


def test_stop_navigating() -> None:
    nav, _, _ = make()
    nav.set_destination("Redmond Town Center")
    assert nav.answer(ask("Stop navigating"), SNAP) == "Okay, navigation is off."
