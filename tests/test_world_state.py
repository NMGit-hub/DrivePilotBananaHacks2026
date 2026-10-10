"""M2 world state: traffic lights, per-vehicle motion, approach zones, colors and the WorldState glue."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from core.config import IntersectionConfig, TrafficLightConfig, VehicleMotionConfig, load_config
from core.types import Frame
from perception.detector import Track
from perception.ego_motion import NOT_MEASURED, EgoMotion
from perception.pipeline import PerceptionResult
from state.approach import classify_approach, is_clipped
from state.colors import color_name
from state.stop_queue import Approach
from state.traffic_light import LightColor, TrafficLightTracker, classify_light
from state.vehicle_motion import VehicleMotionTracker
from state.world import WorldEvent, WorldState

W, H = 1280, 720
TL = TrafficLightConfig()
LAMP_HSV = {"red": (2, 220), "yellow": (25, 220), "green": (88, 210)}  # (hue, saturation) of the lamp's rim


def draw_light(
    image: np.ndarray, x: int, y: int, lit: str | None, housing_bgr: tuple[int, int, int] = (30, 30, 30), lamp: int = 12
) -> tuple[float, float, float, float]:
    """Draw an upright 3-lamp housing at (x, y); returns its box. `lit` = red/yellow/green/None."""
    w, h = lamp * 2 + 8, lamp * 6 + 16
    cv2.rectangle(image, (x, y), (x + w, y + h), housing_bgr, -1)
    for i, color in enumerate(("red", "yellow", "green")):
        center = (x + w // 2, y + 6 + lamp + i * (2 * lamp + 2))
        if color == lit:
            hue, sat = LAMP_HSV[color]
            rim = cv2.cvtColor(np.uint8([[[hue, sat, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.circle(image, center, lamp, rim, -1)
            cv2.circle(image, center, lamp // 2, (245, 250, 255), -1)  # overexposed core
        else:
            cv2.circle(image, center, lamp, (45, 45, 45), -1)
    return float(x), float(y), float(x + w), float(y + h)


def scene(lit: str | None = "red", **kw: object) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    image = np.full((H, W, 3), (120, 110, 100), np.uint8)
    box = draw_light(image, 620, 150, lit, **kw)  # type: ignore[arg-type]
    return image, box


def light_track(box: tuple[float, float, float, float], track_id: int = 1, conf: float = 0.6) -> Track:
    return Track(track_id, "traffic light", conf, box)


# ------------------------------------------------------------------------------------- light classifier


@pytest.mark.parametrize("lit", ["red", "yellow", "green"])
def test_classifies_the_lit_lamp(lit: str) -> None:
    image, box = scene(lit)
    assert classify_light(image, box, TL).color.value == lit


def test_unlit_housing_is_unknown() -> None:
    image, box = scene(None)
    assert classify_light(image, box, TL).color is LightColor.UNKNOWN


def test_yellow_painted_housing_in_sun_is_not_a_yellow_light() -> None:
    image, box = scene(None, housing_bgr=(40, 200, 235))  # bright yellow paint, no lamp lit
    assert classify_light(image, box, TL).color is LightColor.UNKNOWN


def test_red_lamp_in_a_yellow_housing_is_red() -> None:
    image, box = scene("red", housing_bgr=(40, 190, 225))
    assert classify_light(image, box, TL).color is LightColor.RED


def test_glint_on_sunlit_yellow_paint_is_not_a_yellow_light() -> None:
    image = np.full((H, W, 3), (120, 110, 100), np.uint8)
    cv2.rectangle(image, (600, 150), (640, 250), (40, 190, 225), -1)  # yellow housing in the sun, lamps dark
    cv2.circle(image, (620, 200), 4, (250, 252, 255), -1)  # sun glint mid-housing, where a yellow lamp would be
    assert classify_light(image, (600.0, 150.0, 641.0, 251.0), TL).color is LightColor.UNKNOWN


def test_bright_sky_behind_the_box_is_not_a_lamp() -> None:
    image, box = scene(None)
    x1, _, x2, y2 = (int(v) for v in box)
    image[y2 - 15 : y2, x1:x2] = (250, 220, 170)  # bright sky showing at the bottom edge of the box
    assert classify_light(image, box, TL).color is LightColor.UNKNOWN


def test_lamp_in_the_wrong_position_is_rejected() -> None:
    image, box = scene(None)
    x1, y1, x2, _ = (int(v) for v in box)
    cx, cy = (x1 + x2) // 2, y1 + 18  # a green-colored spot at the *top* (a reflection, not a green lamp)
    rim = cv2.cvtColor(np.uint8([[[88, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.circle(image, (cx, cy), 12, rim, -1)
    cv2.circle(image, (cx, cy), 6, (245, 250, 255), -1)
    assert classify_light(image, box, TL).color is LightColor.UNKNOWN


# ---------------------------------------------------------------------------------------- light tracker


def run_lights(
    tracker: TrafficLightTracker, frames: list[tuple[np.ndarray, list[Track], bool]], t0: float = 0.0, fps: float = 30.0
) -> list[tuple[float, LightColor, bool, bool]]:
    out = []
    for i, (image, tracks, stopped) in enumerate(frames):
        reading, green = tracker.update(t0 + i / fps, image, tracks, ego_stopped=stopped)
        out.append((t0 + i / fps, reading.color, reading.confident, green))
    return out


def test_turned_green_fires_once_even_when_yolo_loses_the_green_light() -> None:
    red, box = scene("red")
    green, _ = scene("green")
    tracker = TrafficLightTracker(TL)
    frames = [(red, [light_track(box)], True)] * 45  # stopped at red, detected
    frames += [(green, [], True)] * 30  # turns green; YOLO no longer detects it (small distant light)
    results = run_lights(tracker, frames)
    events = [t for t, _, _, g in results if g]
    assert len(events) == 1
    assert events[0] == pytest.approx(45 / 30 + (TL.green_confirm_frames - 1) / 30, abs=1e-6)
    assert results[-1][1] is LightColor.GREEN and results[-1][2]


def test_no_green_event_while_driving_up_to_a_green_light() -> None:
    green, box = scene("green")
    results = run_lights(TrafficLightTracker(TL), [(green, [light_track(box)], False)] * 40)
    assert not any(g for *_, g in results)
    assert results[-1][1] is LightColor.GREEN  # still reads green when asked


def test_cross_street_light_does_not_mask_our_green() -> None:
    image_red = np.full((H, W, 3), (120, 110, 100), np.uint8)
    ours = draw_light(image_red, 600, 150, "red")
    cross = draw_light(image_red, 760, 140, "red", lamp=9)  # side street light, smaller
    image_green = image_red.copy()
    draw_light(image_green, 600, 150, "green")
    tracks = [light_track(ours, 1, 0.6), light_track(cross, 2, 0.3)]
    frames = [(image_red, tracks, True)] * 45 + [(image_green, tracks, True)] * 30
    results = run_lights(TrafficLightTracker(TL), frames)
    assert sum(g for *_, g in results) == 1


def test_moving_clears_anchors_and_a_new_stop_can_announce_again() -> None:
    red, box = scene("red")
    green, _ = scene("green")
    tracker = TrafficLightTracker(TL)
    first = [(red, [light_track(box)], True)] * 40 + [(green, [], True)] * 20
    drive = [(green, [], False)] * 30  # we drive off: anchors dropped
    second = [(red, [light_track(box, 7)], True)] * 40 + [(green, [light_track(box, 7)], True)] * 20
    results = run_lights(tracker, first + drive + second)
    assert sum(g for *_, g in results) == 2
    assert not any(g for _, _, _, g in results[60:90])


def test_single_misread_frame_does_not_flip_the_reading() -> None:
    red, box = scene("red")
    green, _ = scene("green")
    frames = (
        [(red, [light_track(box)], False)] * 20
        + [(green, [light_track(box)], False)]
        + [(red, [light_track(box)], False)] * 5
    )
    results = run_lights(TrafficLightTracker(TL), frames)
    assert all(color is LightColor.RED for _, color, _, _ in results[15:])


# --------------------------------------------------------------------------------------- vehicle motion


def car(track_id: int, x: float, y: float = 420.0, w: float = 220.0, h: float = 110.0) -> Track:
    return Track(track_id, "car", 0.9, (x, y - h, x + w, y))


def test_parked_then_departing_car() -> None:
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    states = {}
    for i in range(45):  # 1.5 s still
        states = tracker.update(i / 30, [car(1, 100)], None, ego_stopped=True)
    s = states[1]
    assert s.stopped and s.stopped_since == pytest.approx(0.0, abs=0.05) and not s.arrival_known
    for i in range(45, 75):  # drives off at ~1 box height per 0.3 s
        states = tracker.update(i / 30, [car(1, 100 + (i - 44) * 12)], None, ego_stopped=True)
    s = states[1]
    assert not s.stopped and s.seen_moving and s.moved_from_stop > 1.0


def test_car_that_arrives_and_stops_has_a_known_arrival() -> None:
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    states = {}
    for i in range(90):
        x = 100 + min(i, 30) * 10  # moves for 1 s, then stops
        states = tracker.update(i / 30, [car(1, x)], None, ego_stopped=True)
    s = states[1]
    assert s.stopped and s.arrival_known
    assert s.stopped_since == pytest.approx(1.0, abs=0.3)


def test_ego_motion_is_compensated_with_the_homography() -> None:
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    shift = np.array([[1.0, 0.0, -6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # scene slides 6 px/frame (we turn)
    states = {}
    for i in range(60):  # the car is still in the world, so on screen it slides with the scene
        states = tracker.update(i / 30, [car(1, 600 - 6 * i)], shift, ego_stopped=False)
    assert states[1].stopped
    tracker.reset()
    for i in range(60):  # same on-screen motion but the scene doesn't move: the car itself moves
        states = tracker.update(i / 30, [car(1, 600 - 6 * i)], np.eye(3), ego_stopped=False)
    assert not states[1].stopped


def test_unknown_ego_motion_clears_history() -> None:
    tracker = VehicleMotionTracker(VehicleMotionConfig())
    for i in range(30):
        tracker.update(i / 30, [car(1, 100)], None, ego_stopped=True)
    states = tracker.update(1.0, [car(1, 100)], None, ego_stopped=False)  # moving, no homography
    assert states[1].speed is None


# ------------------------------------------------------------------------------------- approach / color


ZONES = IntersectionConfig()


@pytest.mark.parametrize(
    ("box", "cls", "expected"),
    [
        ((20, 300, 300, 400), "car", Approach.LEFT),  # side-on, left
        ((1000, 300, 1270, 400), "car", Approach.RIGHT),  # side-on, right
        ((80, 300, 200, 400), "car", None),  # rear view at the left edge: parked along our street
        ((580, 280, 680, 340), "car", Approach.AHEAD),  # across the intersection, head-on
        ((470, 350, 810, 640), "car", "lead"),  # big and centered: the car in front of us
        ((600, 100, 640, 115), "car", None),  # too small: beyond the intersection
        ((20, 300, 300, 400), "person", None),
    ],
)
def test_approach_zones(box: tuple[float, ...], cls: str, expected: object) -> None:
    track = Track(1, cls, 0.9, tuple(float(v) for v in box))  # type: ignore[arg-type]
    assert classify_approach(track, W, H, ZONES) == expected


def test_clipped_boxes() -> None:
    assert is_clipped(Track(1, "car", 0.9, (0.0, 300.0, 200.0, 400.0)), W, H)
    assert not is_clipped(Track(1, "car", 0.9, (50.0, 300.0, 200.0, 400.0)), W, H)


CAR_BOX = (500.0, 300.0, 800.0, 450.0)


def street(
    car_bgr: tuple[int, int, int], cast: tuple[float, float, float] = (1.0, 1.0, 1.0), exposure: float = 1.0
) -> np.ndarray:
    """A neutral street scene (road, sky, buildings) with one car, then a color cast and exposure applied."""
    image = np.empty((H, W, 3), np.float32)
    image[: H // 3] = (200, 200, 200)  # sky
    image[H // 3 : 2 * H // 3] = (90, 95, 100)  # buildings
    image[2 * H // 3 :] = (70, 70, 70)  # road
    x1, y1, x2, y2 = (int(v) for v in CAR_BOX)
    image[y1:y2, x1:x2] = car_bgr
    return np.clip(image * np.array(cast) * exposure, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("bgr", "name"),
    [
        ((235, 235, 235), "white"),
        ((25, 25, 25), "black"),
        ((150, 150, 150), "silver"),
        ((90, 90, 90), "gray"),
        ((40, 40, 190), "red"),
        ((180, 70, 30), "blue"),
        ((50, 150, 50), "green"),
        ((150, 175, 200), "white"),
    ],
)
def test_color_names(bgr: tuple[int, int, int], name: str) -> None:
    assert color_name(street(bgr), CAR_BOX) == name  # the last one: pale champagne paint is not "orange"


def test_blue_dawn_cast_and_dim_exposure_are_normalized() -> None:
    dawn = ((1.35, 1.0, 0.75), 0.5)  # blue cast, under-exposed
    assert color_name(street((130, 130, 130), *dawn), CAR_BOX) == "silver"  # not "blue"
    assert color_name(street((25, 25, 25), *dawn), CAR_BOX) == "black"
    assert color_name(street((235, 235, 235), *dawn), CAR_BOX) == "white"  # not "gray"


def test_tiny_crop_has_no_color() -> None:
    assert color_name(street((235, 235, 235)), (500.0, 300.0, 505.0, 304.0)) == ""


# ------------------------------------------------------------------------------------------ WorldState


def ego(t: float, stopped: bool, since: float | None) -> EgoMotion:
    return EgoMotion(
        t=t,
        short=NOT_MEASURED,
        long=NOT_MEASURED,
        value=0.0 if stopped else 10.0,
        stopped=stopped,
        stopped_since=since,
        confident=True,
        homography=None,
        elapsed_ms=0.0,
    )


def result(
    i: int, tracks: list[Track], stopped: bool, since: float | None, image: np.ndarray | None = None, segment: int = 0
) -> PerceptionResult:
    t = i / 30
    frame = Frame(
        image=image if image is not None else np.full((H, W, 3), 100, np.uint8),
        t=t,
        index=i,
        captured_at=0.0,
        segment=segment,
    )
    return PerceptionResult(frame=frame, tracks=tracks, ego=ego(t, stopped, since), timings_ms={}, new_segment=(i == 0))


def test_world_runs_a_4_way_stop_end_to_end() -> None:
    events: list[WorldEvent] = []
    world = WorldState(load_config(), on_event=events.append)
    sign = Track(50, "stop sign", 0.8, (900.0, 200.0, 960.0, 260.0))
    for i in range(240):  # 8 s
        t = i / 30
        ego_stopped = t >= 2.0
        tracks = [sign] if t < 2.5 else []
        left_x = 40.0 if t < 6.0 else 40.0 + (t - 6.0) * 600  # waiting on the left, drives off at 6 s
        if left_x < 1100:
            tracks.append(Track(3, "car", 0.9, (left_x, 300.0, left_x + 260.0, 400.0)))
        world.on_perception(result(i, tracks, ego_stopped, 2.0 if ego_stopped else None))
    kinds = [e.kind for e in events]
    assert "arrived_at_stop" in kinds and "vehicle_departed" in kinds
    assert kinds.count("your_turn") == 1
    your_turn = next(e for e in events if e.kind == "your_turn")
    assert 6.9 <= your_turn.t <= 7.4  # after the car kept driving for 1 s (a creep to the line is not a departure)
    snap = world.snapshot()
    assert snap is not None and snap.at_stop_sign and snap.queue.your_turn


def test_lead_vehicle_means_we_are_not_at_the_line() -> None:
    world = WorldState(load_config())
    sign = Track(50, "stop sign", 0.8, (900.0, 200.0, 960.0, 260.0))
    lead = Track(9, "car", 0.9, (470.0, 350.0, 810.0, 640.0))
    for i in range(60):
        world.on_perception(result(i, [sign, lead], True, 0.0))
    snap = world.snapshot()
    assert snap is not None and snap.lead_vehicle is not None and not snap.at_stop_sign
    assert not snap.queue.active


def test_world_announces_green_light() -> None:
    events: list[WorldEvent] = []
    world = WorldState(load_config(), on_event=events.append)
    red, box = scene("red")
    green, _ = scene("green")
    for i in range(60):
        world.on_perception(result(i, [light_track(box)], True, 0.0, image=red))
    for i in range(60, 90):
        world.on_perception(result(i, [], True, 0.0, image=green))
    assert [e.kind for e in events].count("light_green") == 1
    snap = world.snapshot()
    assert snap is not None and snap.light.color is LightColor.GREEN


def test_recent_frames_are_spaced_and_newest_last() -> None:
    world = WorldState(load_config())
    for i in range(90):
        world.on_perception(result(i, [], False, None))
    frames = world.recent_frames(count=3, spacing_s=0.5)
    assert len(frames) == 3 and frames[-1].index == 89
    assert all(b.t - a.t >= 0.5 - 1e-9 for a, b in itertools.pairwise(frames))
