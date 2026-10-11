"""Alerts DrivePilot says without being asked (`alerts.auto`): the watches, the switches, the alert engine's
handling of the new world events, turn-by-turn prompts, and the page's endpoints for them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from answers.alerts import LIGHT_GREEN, AlertEngine
from answers.auto_alerts import (
    CAR_AHEAD_MOVING,
    FIXED_TEXTS,
    LIGHT_GREEN_AUTO,
    RIGHT_LANE,
    STOP_SIGN_AHEAD,
    AutoSwitches,
    SpeedingWatch,
    WrongLaneWatch,
    speeding_text,
    wrong_lane_text,
)
from answers.local import Watch
from answers.speech import Priority, TextSpeaker, Utterance
from app import DrivePilotApp, apply_cli, build_parser
from capture.file_source import FileSource
from core.config import AlertsConfig, AutoAlertsConfig, NavConfig, load_config
from nav.gps import PhoneGps
from nav.handler import NavHandler
from nav.mapbox import FixtureClient, Place
from perception.detector import Track
from state.heads_up import CarAheadWatch, StopSignWatch
from state.signs import SignsSnapshot
from state.world import WorldEvent
from tests.nav_fixture import START
from tests.test_app import Actions, FakeDetector, wait_for
from tests.test_nav import PLACE, RESPONSE, Gps, drive, fixes
from ui.hub import UiHub
from ui.server import create_server

CFG = AutoAlertsConfig()


# ------------------------------------------------------------------------------------------ car ahead


Lead = tuple[int, float, float | None]  # (track id, box height, stopped since)
Frames = list[tuple[float, bool, Lead | None]]


def run_car_ahead(frames: Frames, red_at: set[float] | None = None) -> list[float]:
    """(t, we're stopped, lead) per frame -> the times the alert fired (we're stopped since t=0). `red_at`: times
    our light reads red."""
    watch = CarAheadWatch(CFG)
    red = red_at or set()
    return [
        t
        for t, stopped, lead in frames
        if watch.update(t, stopped, 0.0 if stopped else None, True, lead, red_light=t in red)
    ]


def frames_of(
    heights: list[float], dt: float = 0.1, track: int = 7, since: float | None = 0.0, moves_at: float | None = None
) -> Frames:
    """The car ahead's box height per frame; it has been stopped in front of us since `since` (None: moving), until
    the vehicle-motion tracker sees it move at `moves_at`."""
    out: Frames = []
    for i, h in enumerate(heights):
        t = round(i * dt, 2)
        out.append((t, True, (track, h, None if moves_at is not None and t >= moves_at else since)))
    return out


PULL_AWAY = [200.0] * 30 + [200.0 - 10 * k for k in range(1, 15)] + [60.0] * 30  # pulls away from 3.0 s


def test_the_car_ahead_pulling_away_is_announced_once_after_the_delay() -> None:
    # Stopped behind it from t=0 (height 200 px); at 3.0 s it starts pulling away (shrinks 10 px every 0.1 s), and
    # the motion tracker sees it moving from 3.4 s.
    fired = run_car_ahead(frames_of(PULL_AWAY, moves_at=3.4))
    # 15 % smaller (170 px) at 3.3 s, plus the 1.5 s delay
    assert fired == [pytest.approx(3.3 + CFG.car_ahead_delay_s, abs=0.11)]


def test_a_shrinking_box_the_motion_tracker_calls_standing_is_not_a_car_pulling_away() -> None:
    # (a car creeping a little closer to the line and stopping again, a box that tightened on the car)
    assert run_car_ahead(frames_of(PULL_AWAY)) == []


def test_a_red_light_on_the_approach_or_a_blind_frame_still_keeps_it_quiet() -> None:
    # Review findings: the light read red while we drove up behind a truck, which hides it once we stop; and a
    # frame where ego motion can't tell must not wipe the red seen during the stop.
    watch = CarAheadWatch(CFG)
    fired = []
    for i in range(120):
        t = round(i * 0.1, 2)
        stopped = t >= 2.0
        lead = (7, PULL_AWAY[i - 20] if stopped and i - 20 < len(PULL_AWAY) else 200.0, 2.0 if t < 5.4 else None)
        if watch.update(t, stopped, 2.0 if stopped else None, True, lead if stopped else None, red_light=t < 1.5):
            fired.append(t)
    assert fired == []
    watch = CarAheadWatch(CFG)
    for i, (t, _, lead) in enumerate(frames_of(PULL_AWAY, moves_at=3.4)):
        confident = i != 20  # one blind frame at 2.0 s
        if watch.update(t, True, 0.0, confident, lead, red_light=t < 1.0):
            fired.append(t)
    assert fired == []


def test_box_jitter_is_not_a_car_pulling_away() -> None:
    heights = [200.0, 196.0, 203.0, 190.0, 200.0, 198.0] * 15
    assert run_car_ahead(frames_of(heights)) == []


def test_a_short_wait_or_moving_off_ourselves_says_nothing() -> None:
    # it pulls away 0.5 s after we stopped: we hadn't been waiting behind it (a rolling stop in traffic)
    assert run_car_ahead(frames_of([200.0] * 5 + [150.0] * 40)) == []
    # we move off right after it does
    frames = frames_of([200.0] * 30 + [150.0] * 5) + [(3.6 + 0.1 * k, False, (7, 140.0, None)) for k in range(30)]
    assert run_car_ahead(frames) == []


def test_a_car_that_drives_out_of_view_after_pulling_away_still_counts() -> None:
    frames = frames_of([200.0] * 30 + [160.0] * 3, moves_at=3.1) + [(3.3 + 0.1 * k, True, None) for k in range(1, 30)]
    assert len(run_car_ahead(frames)) == 1


def test_cross_traffic_driving_off_ahead_of_us_is_not_the_car_ahead() -> None:
    # Seen on the SF replay: first at a red light, a car crossed in front, turned up the street and drove off. It
    # was never stopped in front of us: not "the car ahead".
    assert run_car_ahead(frames_of(PULL_AWAY, since=None)) == []


def test_never_at_a_red_light() -> None:
    # Stopped behind a car at a red light; it pulls away (turning right on red?) while the light still reads red,
    # and the reading drops out afterwards: nothing is said at this stop.
    frames = frames_of(PULL_AWAY, moves_at=3.4)
    assert run_car_ahead(frames, red_at={t for t, _, _ in frames if t < 2.0}) == []


def test_a_different_car_ahead_starts_over() -> None:
    # a cut-in: another (closer) car becomes the lead: its box is no reason to say ours is moving
    frames = frames_of([200.0] * 30) + frames_of([120.0] * 30, track=9)
    frames = [(round(i * 0.1, 2), s, lead) for i, (_, s, lead) in enumerate(frames)]
    assert run_car_ahead(frames) == []


# ------------------------------------------------------------------------------------------ stop signs


SIGN_RED, SIGN_BACK = (30, 20, 200), (150, 150, 150)  # BGR: a stop sign's face, and its grey back


def sign(track: int, x: float, height: float, width: float | None = None) -> Track:
    w = height if width is None else width
    return Track(track, "stop sign", 0.9, (x - w / 2, 300.0, x + w / 2, 300.0 + height))


def street_with(*signs: tuple[Track, tuple[int, int, int]]) -> np.ndarray:
    """A 1920x1080 frame with each sign's box painted in its color."""
    image = np.full((1080, 1920, 3), 90, np.uint8)
    for track, color in signs:
        x1, y1, x2, y2 = (round(v) for v in track.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def seen(*tracks: Track, color: tuple[int, int, int] = SIGN_RED) -> tuple[list[Track], np.ndarray]:
    return list(tracks), street_with(*((t, color) for t in tracks))


def looks(watch: StopSignWatch, t: float, moving: bool, speed: float | None, *view: Any) -> list[int | None]:
    """The same view for `stop_sign_frames` frames in a row (one frame alone is never enough)."""
    return [watch.update(t + k / 30, moving, speed, *view) for k in range(CFG.stop_sign_frames)]


def test_a_stop_sign_ahead_on_our_side_is_announced_once() -> None:
    watch = StopSignWatch(CFG)
    assert looks(watch, 0.0, True, 10.0, *seen(sign(1, 1500.0, 15.0)))[-1] is None  # far off (small)
    assert looks(watch, 0.5, True, 10.0, *seen(sign(1, 1500.0, 40.0))) == [None] * (CFG.stop_sign_frames - 1) + [1]
    assert looks(watch, 1.0, True, 10.0, *seen(sign(1, 1550.0, 60.0)))[-1] is None  # same sign
    assert looks(watch, 2.0, True, 10.0, *seen(sign(2, 1600.0, 40.0)))[-1] is None  # the corner's other sign: soon
    assert looks(watch, 30.0, True, 10.0, *seen(sign(3, 1600.0, 40.0)))[-1] == 3  # the next intersection


def test_stop_signs_are_not_announced_when_slow_stopped_or_across_the_road() -> None:
    watch = StopSignWatch(CFG)
    assert looks(watch, 0.0, True, 2.0, *seen(sign(1, 1500.0, 40.0)))[-1] is None  # creeping up to it
    assert looks(watch, 1.0, False, None, *seen(sign(2, 1500.0, 40.0)))[-1] is None  # stopped, no GPS
    assert looks(watch, 2.0, True, 10.0, *seen(sign(3, 300.0, 40.0)))[-1] is None  # far left: not ours
    assert looks(watch, 3.0, True, None, *seen(sign(4, 1500.0, 40.0)))[-1] == 4  # no GPS: ego motion says moving


def test_the_back_of_a_stop_sign_a_sliver_or_a_single_frame_is_not_announced() -> None:
    # Seen on the SF replay: YOLO found the grey back of the cross street's sign (30 x 60 px with its pole).
    watch = StopSignWatch(CFG)
    assert looks(watch, 0.0, True, 10.0, *seen(sign(1, 1500.0, 40.0), color=SIGN_BACK))[-1] is None
    assert looks(watch, 1.0, True, 10.0, *seen(sign(2, 1500.0, 60.0, width=28.0)))[-1] is None  # tall and thin
    orange = (40, 120, 200)  # BGR: a grey back in warm evening light is orange-ish, not red
    assert looks(watch, 2.0, True, 10.0, *seen(sign(5, 1500.0, 40.0), color=orange))[-1] is None
    # a red face seen once, then missed: starts over
    assert watch.update(3.0, True, 10.0, *seen(sign(3, 1500.0, 40.0))) is None
    assert watch.update(3.03, True, 10.0, [], street_with()) is None
    assert looks(watch, 3.1, True, 10.0, *seen(sign(3, 1500.0, 40.0)))[-1] == 3  # about square, red


# ------------------------------------------------------------------------------------------ wrong lane


def test_the_wrong_lane_is_said_after_the_hold_within_range_then_once_more_and_confirmed() -> None:
    watch = WrongLaneWatch(CFG)
    wrong = {"lane": "left", "moves": 1}
    assert watch.update(0.0, wrong, 3, 1500.0) is None  # too far from the turn
    assert watch.update(10.0, wrong, 3, 780.0) is not None  # in range, and in the wrong lane for 10 s already
    watch = WrongLaneWatch(CFG)
    assert watch.update(0.0, wrong, 3, 700.0) is None
    assert watch.update(1.0, wrong, 3, 690.0) is None  # changing lanes takes a moment
    assert watch.update(2.0, wrong, 3, 680.0) == "You're in the wrong lane for your turn. Move one lane left."
    assert watch.update(3.0, wrong, 3, 600.0) is None  # once
    assert watch.update(20.0, wrong, 3, 240.0) == "You're still in the wrong lane. Move one lane left."
    assert watch.update(21.0, wrong, 3, 200.0) is None
    assert watch.update(22.0, {"lane": "ours"}, 3, 180.0) is None  # one reading mid lane change isn't enough
    assert watch.update(22.5, wrong, 3, 175.0) is None
    assert watch.update(23.0, {"lane": "ours"}, 3, 170.0) is None
    assert watch.update(25.0, {"lane": "ours"}, 3, 150.0) == RIGHT_LANE  # in it for the hold time
    assert watch.update(26.0, {"lane": "ours"}, 3, 140.0) is None


def test_a_reroute_makes_new_maneuvers() -> None:
    watch = WrongLaneWatch(CFG)
    wrong = {"lane": "left", "moves": 1}
    assert [watch.update(t, wrong, ("route A", 3), 600.0) for t in (0.0, 2.0)][-1] is not None
    # rerouted: the new route's step 3 is another maneuver, warned about on its own
    assert [watch.update(t, wrong, ("route B", 3), 600.0) for t in (3.0, 5.0)][-1] is not None


def test_a_lane_change_in_progress_or_an_unknown_lane_says_nothing() -> None:
    watch = WrongLaneWatch(CFG)
    wrong = {"lane": "right", "moves": 2}
    assert watch.update(0.0, wrong, 1, 500.0) is None
    assert watch.update(1.0, None, 1, 490.0) is None  # lane detection lost the lines: the hold starts over
    assert watch.update(2.5, wrong, 1, 480.0) is None
    assert watch.update(4.5, wrong, 1, 470.0) == wrong_lane_text("right", 2)
    assert "Move two lanes right." in wrong_lane_text("right", 2)


def test_each_maneuver_gets_its_own_warning_and_good_lanes_need_no_confirmation() -> None:
    watch = WrongLaneWatch(CFG)
    assert watch.update(0.0, {"lane": "ours"}, 1, 500.0) is None  # never wrong: no "That's the right lane."
    assert watch.update(1.0, {"lane": "left", "moves": 1}, 2, 500.0) is None
    assert watch.update(3.0, {"lane": "left", "moves": 1}, 2, 450.0) is not None
    assert watch.update(4.0, {"lane": "left", "moves": 1}, 3, 700.0) is None  # the next maneuver: new hold
    assert watch.update(6.5, {"lane": "left", "moves": 1}, 3, 650.0) is not None


# ------------------------------------------------------------------------------------------- speeding


MPH = 1 / 2.23694


def test_speeding_over_the_margin_for_the_hold_is_said_once_per_limit() -> None:
    watch = SpeedingWatch(CFG)
    said = [watch.update(t, 42 * MPH, 35, 0.0) for t in range(0, 10)]  # 7 over a 35 limit
    assert said.count(speeding_text(35)) == 1 and said.index(speeding_text(35)) == round(CFG.speeding_hold_s)
    assert watch.update(10.0, 39 * MPH, 35, 0.0) is None  # within the margin: fine
    assert [watch.update(t, 45 * MPH, 35, 0.0) for t in range(11, 20)].count(speeding_text(35)) == 0  # said already
    assert [watch.update(t, 45 * MPH, 25, 20.0) for t in range(20, 30)].count(speeding_text(25)) == 1  # a new limit


def test_speeding_is_said_again_only_after_slowing_down_for_a_while() -> None:
    watch = SpeedingWatch(CFG)
    assert speeding_text(30) in [watch.update(t, 40 * MPH, 30, 0.0) for t in range(6)]
    assert all(watch.update(t, 28 * MPH, 30, 0.0) is None for t in range(6, 6 + round(CFG.speeding_repeat_s) + 1))
    assert speeding_text(30) in [watch.update(t, 40 * MPH, 30, 0.0) for t in range(200, 206)]


def test_a_limit_is_forgotten_after_turning_onto_another_road() -> None:
    # Review finding: a 25 limit read on a side street, then a turn onto a 45 mph road with no sign yet.
    watch = SpeedingWatch(CFG)
    assert all(watch.update(t, 24 * MPH, 25, 0.0, heading=90.0) is None for t in range(5))
    assert all(watch.update(5 + t, 40 * MPH, 25, 0.0, heading=0.0) is None for t in range(20))  # turned north
    assert speeding_text(35) in [watch.update(30 + t, 45 * MPH, 35, 30.0, heading=0.0) for t in range(6)]


def test_a_bend_in_the_road_does_not_forget_the_limit() -> None:
    watch = SpeedingWatch(CFG)
    said = [watch.update(t, 40 * MPH, 30, 0.0, heading=90.0 + (70.0 if 3 <= t < 5 else 0.0)) for t in range(12)]
    assert speeding_text(30) in said  # 2 s at 70 degrees off: a bend, not another road


def test_no_speed_no_limit_or_an_old_limit_says_nothing() -> None:
    watch = SpeedingWatch(CFG)
    assert all(watch.update(t, None, 30, 0.0) is None for t in range(10))
    assert all(watch.update(t, 50 * MPH, None, None) is None for t in range(10))
    old = CFG.speeding_limit_max_age_s
    assert all(watch.update(old + 1 + t, 50 * MPH, 30, 0.0) is None for t in range(10))


def test_fixed_texts_are_prerendered() -> None:
    assert {LIGHT_GREEN_AUTO, CAR_AHEAD_MOVING, STOP_SIGN_AHEAD, RIGHT_LANE, speeding_text(35)} <= set(FIXED_TEXTS)
    assert wrong_lane_text("left", 1) in FIXED_TEXTS and wrong_lane_text("right", 2, again=True) in FIXED_TEXTS


# ----------------------------------------------------------------------------------------- alert engine


def engine() -> tuple[AlertEngine, list[Utterance]]:
    said: list[Utterance] = []
    return AlertEngine(AlertsConfig(), said.append), said


def test_the_light_turning_green_is_said_unasked_unless_switched_off_or_quiet() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("light_green", 10.0, {"ego_stopped": True}))
    assert [(u.text, u.priority) for u in said] == [(LIGHT_GREEN_AUTO, Priority.INFO)]
    alerts.on_world_event(WorldEvent("drove_off", 15.0, {"moving_since": 12.0}))
    alerts.switches.set("light_green", False)
    alerts.on_world_event(WorldEvent("light_green", 60.0, {"ego_stopped": True}))
    assert len(said) == 1
    alerts.switches.set("light_green", True)
    alerts.set_quiet(True)
    alerts.on_world_event(WorldEvent("drove_off", 65.0, {"moving_since": 62.0}))
    alerts.on_world_event(WorldEvent("light_green", 120.0, {"ego_stopped": True}))
    assert len(said) == 1  # quiet mode: only what was asked for


def test_an_asked_for_green_still_gets_the_full_sentence_even_when_switched_off() -> None:
    alerts, said = engine()
    alerts.switches.set("light_green", False)
    alerts.arm(Watch.LIGHT, 5.0)
    alerts.on_world_event(WorldEvent("light_green", 10.0, {"ego_stopped": True}))
    assert [u.text for u in said] == [LIGHT_GREEN]


def test_car_ahead_and_stop_sign_events_follow_their_switches() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("car_ahead_moving", 5.0, {"track": 3}))
    alerts.on_world_event(WorldEvent("stop_sign_ahead", 30.0, {"track": 8}))
    assert [u.text for u in said] == [CAR_AHEAD_MOVING, STOP_SIGN_AHEAD]
    assert not said[1].drops_answers  # a stop sign ahead doesn't make an answer in progress wrong
    alerts.switches.set("car_ahead", False)
    alerts.switches.set("stop_sign", False)
    alerts.on_world_event(WorldEvent("car_ahead_moving", 60.0, {"track": 4}))
    alerts.on_world_event(WorldEvent("stop_sign_ahead", 90.0, {"track": 9}))
    assert len(said) == 2


def test_the_car_ahead_moving_right_after_the_green_is_not_said_twice() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("light_green", 10.0, {"ego_stopped": True}))
    alerts.on_world_event(WorldEvent("car_ahead_moving", 12.0, {"track": 3}))
    assert [u.text for u in said] == [LIGHT_GREEN_AUTO]


def test_switches() -> None:
    switches = AutoSwitches.from_config(CFG.model_copy(update={"speeding": False}))
    assert switches.as_dict()["speeding"] is False and switches.enabled("turns")
    switches.set("speeding", True)
    assert switches.enabled("speeding")
    with pytest.raises(ValueError):
        switches.set("fireworks", True)


# ------------------------------------------------------------------------------------------ turn-by-turn


def nav_handler(prompts: dict[str, bool]) -> tuple[NavHandler, Gps, list[tuple[float, str, str]]]:
    gps = Gps()
    said: list[tuple[float, str, str]] = []
    client = FixtureClient({"redmond town center": PLACE}, {"redmond town center": RESPONSE})

    def on_alert(kind: str, text: str) -> None:
        said.append((gps.fix.t if gps.fix else 0.0, kind, text))

    nav = NavHandler(NavConfig(), client, gps, on_alert, turn_prompts=lambda: prompts["on"])
    return nav, gps, said


def test_turn_by_turn_prompts_along_the_route() -> None:
    nav, gps, said = nav_handler({"on": True})
    answer = nav.set_destination("Redmond Town Center")
    assert "First, turn left onto 148th Avenue Northeast in half a mile." in answer
    drive(nav, gps, fixes())
    turns = [(round(t), text) for t, kind, text in said if kind == "nav_turn"]
    # The first turn's half-mile prompt was in the answer; the roundabout has its own announcement.
    assert [text for _, text in turns] == [
        "In 450 feet, turn left onto 148th Avenue Northeast.",
        "In half a mile, take exit 14 onto SR 520 East toward Redmond.",
        "In 450 feet, take exit 14 onto SR 520 East toward Redmond.",
        "In 500 feet, you'll reach your destination.",
    ]
    assert (800 - 150) / 15 - 1 <= turns[0][0] <= (800 - 75) / 15 + 1  # within 150 m of the turn at 15 m/s
    assert sum(1 for _, kind, _ in said if kind == "nav_roundabout") == 1


def test_a_maneuver_right_after_another_still_gets_its_prompt() -> None:
    # Review finding: a maneuver that becomes next under 75 m away (turn right, then left 60 m later), or a GPS gap
    # over the 75-150 m window, was never announced. The last prompt now goes down to 20 m.
    nav, gps, said = nav_handler({"on": True})
    nav.set_destination("Redmond Town Center")
    track = fixes()
    skip = [f for f in track if not 640.0 <= f.t * 15.0 <= 760.0]  # no fix from 160 m to 40 m before the turn
    drive(nav, gps, skip)
    first = next(text for _, kind, text in said if kind == "nav_turn")
    assert first.endswith("turn left onto 148th Avenue Northeast.") and first.startswith("In 1")


def test_turn_by_turn_can_be_switched_off_while_driving() -> None:
    switch = {"on": False}
    nav, gps, said = nav_handler(switch)
    nav.set_destination("Redmond Town Center")
    all_fixes = fixes()
    drive(nav, gps, all_fixes[:150])  # ~2250 m: past the first turn, before exit 14's prompts
    assert not any(kind == "nav_turn" for _, kind, _ in said)
    switch["on"] = True
    drive(nav, gps, all_fixes[150:])
    assert sum(1 for _, kind, _ in said if kind == "nav_turn") == 3


# ------------------------------------------------------------------------------------------ the page


class PageActions(Actions):
    def __init__(self) -> None:
        super().__init__()
        self.modes: list[str] = []
        self.switched: list[tuple[str, bool]] = []
        self.destinations: list[tuple[str, dict[str, Any] | None]] = []
        self.cancelled = 0

    def set_wake_mode(self, mode: str) -> bool:
        self.modes.append(mode)
        return True

    def set_alert(self, name: str, on: bool) -> None:
        if name not in ("light_green", "turns"):
            raise ValueError(name)
        self.switched.append((name, on))

    def suggest_places(self, query: str) -> list[dict[str, Any]]:
        return [{"name": f"{query} One", "address": "1 Main St", "lat": 47.6, "lon": -122.2, "kind": "poi"}]

    def set_destination(self, query: str, place: dict[str, Any] | None) -> bool:
        self.destinations.append((query, place))
        return True

    def cancel_route(self) -> bool:
        self.cancelled += 1
        return True


def test_page_endpoints_for_destination_wake_mode_and_alerts() -> None:
    actions = PageActions()
    with TestClient(create_server(UiHub(), actions), base_url="http://localhost") as client:
        assert client.post("/api/places", json={"q": "cos"}).json()["places"][0]["name"] == "cos One"
        assert client.post("/api/places", json={"q": "c"}).json() == {"places": []}  # too short to search
        picked = {"name": "Costco", "address": "1 Main St", "lat": 47.6, "lon": -122.2, "kind": "poi"}
        assert client.post("/api/destination", json={"query": "Costco", "place": picked}).json() == {"ok": True}
        assert client.post("/api/destination", json={"query": "the library"}).json() == {"ok": True}
        assert client.post("/api/destination", json={"query": ""}).status_code == 422
        bad_place = {"query": "x", "place": {"name": "y", "lat": 99}}
        assert client.post("/api/destination", json=bad_place).status_code == 422
        assert client.post("/api/route", json={"action": "cancel"}).json() == {"ok": True}
        assert client.post("/api/wake", json={"mode": "hey"}).json() == {"ok": True}
        assert client.post("/api/wake", json={"mode": "hello"}).status_code == 422
        assert client.post("/api/alerts", json={"name": "turns", "on": False}).json() == {"ok": True}
        assert client.post("/api/alerts", json={"name": "fireworks", "on": True}).status_code == 422
        evil = {"Origin": "http://evil.example"}
        assert client.post("/api/destination", json={"query": "x"}, headers=evil).status_code == 403
        assert client.post("/api/places", json={"q": "home"}, headers=evil).status_code == 403
    assert actions.destinations == [("Costco", picked), ("the library", None)]
    assert actions.modes == ["hey"] and actions.switched == [("turns", False)] and actions.cancelled == 1


# ------------------------------------------------------------------------------------------- the app


@pytest.fixture(scope="module")
def drive_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3 s synthetic video (the app needs a source; nothing in it matters here)."""
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("auto") / "drive.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(90):
        writer.write(np.full((180, 320, 3), i, np.uint8))
    writer.release()
    return path


def app_with_nav(clip_path: Path) -> tuple[DrivePilotApp, PhoneGps, TextSpeaker]:
    phone = PhoneGps()
    speaker = TextSpeaker()
    client = FixtureClient({"redmond town center": PLACE}, {"redmond town center": RESPONSE})
    app = DrivePilotApp(
        load_config(),
        lambda: FileSource(clip_path),
        FakeDetector(),  # type: ignore[arg-type]
        speaker,
        routing=client,
        gps=phone,
    )
    return app, phone, speaker


def test_the_page_sets_and_ends_a_destination(drive_clip: Path) -> None:
    app, phone, speaker = app_with_nav(drive_clip)
    assert app.suggest_places("Redmond")[0]["name"] == "Redmond Town Center"
    assert app._route_status() is None
    assert app.set_destination("Redmond Town Center")  # no GPS yet: says so
    assert wait_for(lambda: any("GPS position" in t for t in speaker.texts()))
    phone.push(START[0], START[1], 0.0, 90.0, 5.0)
    picked = Place(PLACE.name, PLACE.address, PLACE.location, "poi")
    assert app.set_destination("Redmond Town Center", picked)
    assert wait_for(lambda: app.nav is not None and app.nav.route is not None)
    assert wait_for(lambda: any(t.startswith("Okay, Redmond Town Center") for t in speaker.texts()))
    status = app._route_status()
    assert status is not None and status["name"] == "Redmond Town Center" and status["miles"] == pytest.approx(3.4, 0.1)
    assert [q.text for q in app.questions.values()][-1] == "Take me to Redmond Town Center"
    assert app.cancel_route() and app._route_status() is None
    assert speaker.texts()[-1] == "Okay, navigation is off."


def test_navigation_off_and_wake_mode_without_voice(drive_clip: Path) -> None:
    app = DrivePilotApp(load_config(), lambda: FileSource(drive_clip), FakeDetector(), TextSpeaker())  # type: ignore[arg-type]
    assert app._route_status() == {"off": True} and not app.set_destination("anywhere") and not app.cancel_route()
    assert app.suggest_places("Costco") == [] and app.set_wake_mode("hey") is False
    with pytest.raises(ValueError):
        app.set_wake_mode("hello")


def test_the_wake_mode_and_alert_switches_reach_the_pipeline_engine_and_navigation(drive_clip: Path) -> None:
    app, _, _ = app_with_nav(drive_clip)

    class Voice:
        mode = "hey_drivepilot"

        def set_mode(self, mode: str) -> None:
            self.mode = mode

    app.attach_voice(Voice(), None)
    assert app.set_wake_mode("none") and app.voice.mode == "none"
    assert app.assistant.alerts.switches is app.switches
    app.set_alert("turns", False)
    assert app.nav is not None and app.nav.turn_prompts is not None and not app.nav.turn_prompts()
    app.set_alert("light_green", False)
    assert not app.assistant.alerts.switches.enabled("light_green")


def test_speeding_through_the_app_respects_the_switch_and_quiet_mode(drive_clip: Path, monkeypatch: Any) -> None:
    app, phone, speaker = app_with_nav(drive_clip)
    now = {"t": 0.0}
    monkeypatch.setattr(app, "media_time", lambda: now["t"])
    monkeypatch.setattr(app.signs, "snapshot", lambda: SignsSnapshot(now["t"], 30, 0.0, None, None, ()))

    def drive_for(seconds: int) -> None:
        for _ in range(seconds * 2):
            now["t"] += 0.5
            phone.push(START[0], START[1], 45 * MPH, 90.0, 5.0)
            app._auto_alerts()

    drive_for(6)
    assert speaker.texts().count(speeding_text(30)) == 1
    app.set_alert("speeding", False)
    monkeypatch.setattr(app.signs, "snapshot", lambda: SignsSnapshot(now["t"], 25, now["t"], None, None, ()))
    drive_for(6)
    assert speeding_text(25) not in speaker.texts()  # switched off
    app.set_alert("speeding", True)
    app.assistant.alerts.set_quiet(True)
    monkeypatch.setattr(app.signs, "snapshot", lambda: SignsSnapshot(now["t"], 20, now["t"], None, None, ()))
    drive_for(6)
    assert speeding_text(20) not in speaker.texts()  # quiet mode


def test_cli_wake_and_destination() -> None:
    args = build_parser().parse_args(["--camera", "--wake", "hey", "--destination", "Costco"])
    cfg = apply_cli(load_config(), args)
    assert cfg.voice.wake_mode == "hey" and args.destination == "Costco"
    assert apply_cli(load_config(), build_parser().parse_args([])).voice.wake_mode == "hey_drivepilot"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--wake", "hello"])


def test_mapbox_place_search_merges_both_services_best_match_first() -> None:
    import httpx

    from nav.mapbox import MapboxClient, NavError

    def feature(name: str, address: str, lon: float, lat: float, kind: str = "poi") -> dict[str, Any]:
        props = {"name": name, "full_address": address, "feature_type": kind}
        return {"properties": props, "geometry": {"coordinates": [lon, lat]}}

    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "searchbox" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "features": [
                        feature("Costco Wholesale", "Far Away, AZ", -112.0, 33.4),
                        feature("Costco", "Kirkland, WA", -122.19, 47.68),
                    ]
                },
            )
        return httpx.Response(
            200, json={"features": [feature("Costco", "Kirkland, WA", -122.1901, 47.6801, "address")]}
        )

    client = MapboxClient("pk.test-token", transport=httpx.MockTransport(respond))
    places = client.search("costco", near=START, limit=5)
    assert [(p.name, p.address) for p in places] == [("Costco", "Kirkland, WA"), ("Costco Wholesale", "Far Away, AZ")]
    assert seen[1].url.params["autocomplete"] == "true" and seen[0].url.params["limit"] == "5"
    assert client.search("c") == [] and len(seen) == 2  # one letter: no request
    down = MapboxClient("pk.test-token", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(NavError):
        down.search("costco")


def test_suggestions_rank_a_half_typed_name() -> None:
    from nav.mapbox import rank_places

    near = START
    cost_plus = Place("Cost Plus World Market", "Bellevue, WA", (47.62, -122.2), "poi")
    costco = Place("Costco", "Kirkland, WA", (47.62, -122.2), "poi")
    assert [p.name for p in rank_places("costc", [cost_plus, costco], near)] == ["Costco", "Cost Plus World Market"]


def test_navigation_suggestions_survive_a_search_failure() -> None:
    from nav.mapbox import NavError

    class Down(FixtureClient):
        def search(self, query: str, near: Any = None, limit: int = 5) -> list[Place]:
            raise NavError("mapbox: HTTP 500")

    nav = NavHandler(NavConfig(), Down({}, {}), Gps())
    assert nav.suggest("costco") == []


def test_the_world_raises_car_ahead_and_stop_sign_events() -> None:
    from state.world import WorldState
    from tests.test_world_state import H, W, result

    events: list[WorldEvent] = []
    world = WorldState(load_config(), on_event=events.append)
    # Driving (no GPS: ego motion says moving) up to a stop sign on the right that grows as we close in.
    for i in range(30):
        size = 20.0 + i * 2.0
        stop = Track(5, "stop sign", 0.9, (1000.0, 200.0, 1000.0 + size, 200.0 + size))
        image = np.full((H, W, 3), 100, np.uint8)
        cv2.rectangle(image, (1000, 200), (round(1000 + size), round(200 + size)), SIGN_RED, -1)
        world.on_perception(result(i, [stop], False, None, image=image))
    # Then stopped behind a car (the lead: centered, big) that pulls away at 4 s (its box shrinks).
    for i in range(30, 300):
        t = i / 30
        shrink = 0.0 if t < 4.0 else min(250.0, (t - 4.0) * 60.0)  # drives off, receding for ~4 s
        box = (470.0 + shrink / 2, 350.0 + shrink / 3, 810.0 - shrink / 2, 640.0 - shrink / 3)
        world.on_perception(result(i, [Track(9, "car", 0.9, box)], True, 1.0))
    kinds = [e.kind for e in events]
    assert kinds.count("stop_sign_ahead") == 1 and kinds.count("car_ahead_moving") == 1
    moved = next(e for e in events if e.kind == "car_ahead_moving")
    # (a steady, gentle recede: the motion tracker calls it moving after ~2 s, then the 1.5 s delay)
    assert 4.0 + CFG.car_ahead_delay_s <= moved.t <= 4.0 + 4.0
