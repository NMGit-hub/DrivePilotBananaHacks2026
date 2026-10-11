"""The V2 page's overlay packets (ui/overlay.py) and the frame + overlay video messages (ui/hub.py, ui/server.py)."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.types import Frame
from nav.progress import Progress
from nav.route import Lane, Step
from perception.detector import Track
from state.parking import EMPTY as EMPTY_PARKING
from state.parking import ParkedCar, ParkingSnapshot, Space
from state.stop_queue import Approach, QueueEntry, QueueSnapshot
from state.traffic_light import LightColor, LightReading
from state.world import VehicleView, WorldSnapshot
from tests.test_app import Actions, wait_for
from ui.hub import UiHub, frame_message, parse_frame_message
from ui.overlay import AMBER, GRAY, GREEN, RED, WHITE, frame_packet, hud_packet, ordinal
from ui.server import create_server

W, H = 1280, 720


def vehicle(track_id: int, box: tuple[float, float, float, float], **kw: object) -> VehicleView:
    fields: dict[str, object] = dict(
        cls="car", color="white", approach=None, is_lead=False, stopped=False, stopped_since=None, clipped=False
    )
    fields.update(kw)
    return VehicleView(track_id=track_id, box=box, **fields)  # type: ignore[arg-type]


def snapshot(
    vehicles: tuple[VehicleView, ...] = (),
    queue: QueueSnapshot | None = None,
    light: LightReading | None = None,
    parking: ParkingSnapshot = EMPTY_PARKING,
    t: float = 10.0,
) -> WorldSnapshot:
    return WorldSnapshot(
        t=t,
        frame=None,
        ego_stopped=True,
        ego_stopped_since=5.0,
        ego_motion_confident=True,
        stop_sign_visible=True,
        stop_sign_last_seen=t,
        at_stop_sign=True,
        lead_vehicle=None,
        vehicles=vehicles,
        queue=queue
        or QueueSnapshot(
            t=t, active=False, order=(), departed=(), ego_position=None, tie_groups=(), deadlock=False, your_turn=False
        ),
        light=light or LightReading(t, LightColor.UNKNOWN, False, 0, ()),
        caveats=(),
        parking=parking,
    )


def items(packet: dict[str, object], kind: str) -> list[dict[str, object]]:
    return [i for i in packet["items"] if i["kind"] == kind]  # type: ignore[attr-defined,index]


def test_boxes_are_normalized_to_the_frame_and_keep_their_ids() -> None:
    snap = snapshot((vehicle(7, (128.0, 72.0, 640.0, 360.0)), vehicle(8, (-20.0, 700.0, 1300.0, 900.0))))
    packet = frame_packet(snap, [], W, H)
    boxes = {i["id"]: i["box"] for i in items(packet, "vehicle")}
    assert boxes[7] == [0.1, 0.1, 0.5, 0.5]
    assert 8 not in boxes  # a wide box at the bottom edge is our own hood, not a car
    snap = snapshot((vehicle(9, (-20.0, 600.0, 300.0, 900.0)),))
    assert items(frame_packet(snap, [], W, H), "vehicle")[0]["box"] == [0.0, 0.8333, 0.2344, 1.0]  # clamped


def test_the_hood_threshold_follows_a_masked_frame_bottom() -> None:
    """The car dashcam's text bar is blacked out, so its hood box ends ~96 % down: perception.hood_bottom 0.93."""
    snap = snapshot((vehicle(8, (300.0, 600.0, 1000.0, 0.96 * H)),))
    assert [i["id"] for i in items(frame_packet(snap, [], W, H), "vehicle")] == [8]  # default 0.97: drawn
    assert items(frame_packet(snap, [], W, H, hood_bottom=0.93), "vehicle") == []


def test_four_way_badges_say_the_order_the_next_car_is_amber_and_you_turn_green() -> None:
    order = (
        QueueEntry("track:3", Approach.LEFT, arrived=4.0, label="white SUV", track_id=3),
        QueueEntry("track:4", Approach.AHEAD, arrived=5.0, label="gray truck", track_id=4),
        QueueEntry("ego", Approach.US, arrived=6.0),
    )
    queue = QueueSnapshot(10.0, True, order, (), ego_position=3, tie_groups=(), deadlock=False, your_turn=False)
    cars = (vehicle(3, (100, 300, 300, 400)), vehicle(4, (600, 300, 700, 380)), vehicle(5, (900, 300, 1000, 380)))
    packet = frame_packet(snapshot(cars, queue), [], W, H)
    by_id = {i["id"]: i for i in items(packet, "vehicle")}
    assert by_id[3]["badge"] == "1st" and by_id[3]["color"] == AMBER and by_id[3]["timer"] == 6.0
    assert by_id[4]["badge"] == "2nd" and by_id[4]["color"] == WHITE and by_id[4]["label"] == "white car"
    assert "badge" not in by_id[5] and by_id[5]["color"] == WHITE
    assert packet["you"] == {"text": "YOU: 3rd", "color": WHITE}
    mine = QueueSnapshot(12.0, True, (order[2],), order[:2], 1, (), False, True)
    assert frame_packet(snapshot(cars, mine, t=12.0), [], W, H)["you"] == {"text": "YOU: your turn", "color": GREEN}


def test_people_are_amber_lights_show_their_color_and_parked_cars_are_gray() -> None:
    tracks = [Track(20, "person", 0.9, (500, 300, 540, 420)), Track(21, "bicycle", 0.8, (700, 320, 760, 420))]
    light = LightReading(10.0, LightColor.RED, True, 2, ((600, 50, 620, 100), (900, 60, 915, 95)))
    car = ParkedCar(5, "car", "right", 3.5, 9.0, (900, 300, 1000, 380), 1.0)
    parking = ParkingSnapshot(10.0, "lot", 360.0, 2.0, {"left": (), "right": (car,)}, {}, (), True, 0.0)
    packet = frame_packet(snapshot((vehicle(5, car.box),), light=light, parking=parking), tracks, W, H)
    people = items(packet, "person")
    assert [(p["label"], p["color"]) for p in people] == [("pedestrian", AMBER), ("cyclist", AMBER)]
    lights = items(packet, "light")
    assert lights[0]["color"] == RED and lights[0]["label"] == "RED" and lights[1]["color"] == GRAY
    assert items(packet, "vehicle")[0]["color"] == GRAY
    unsure = LightReading(10.0, LightColor.GREEN, False, 1, ((600, 50, 620, 100),))
    assert items(frame_packet(snapshot(light=unsure), [], W, H), "light")[0]["label"] == "GREEN?"


def test_open_spots_are_drawn_only_while_the_parking_watch_is_on() -> None:
    near = ParkedCar(1, "car", "right", 3.5, 6.0, (900, 300, 1000, 380), 1.0)
    far = ParkedCar(2, "car", "right", 3.5, 11.4, (800, 320, 860, 370), 1.0)
    space = Space(("right", 1, 2), "right", near, far, 1, 7.9, (850, 280, 900, 360), 1, 2.7, 3.0, 1.2, True, 30.0)
    parking = ParkingSnapshot(10.0, "lot", 360.0, 2.0, {"left": (), "right": (near, far)}, {}, (space,), True, 0.0)
    assert items(frame_packet(snapshot(parking=parking), [], W, H), "spot") == []
    spots = items(frame_packet(snapshot(parking=parking), [], W, H, parking_active=True), "spot")
    assert len(spots) == 1 and spots[0]["color"] == GREEN and spots[0]["label"] == "Open spot"


def test_the_highlighted_track_pulses() -> None:
    packet = frame_packet(snapshot((vehicle(3, (100, 300, 300, 400)), vehicle(4, (600, 300, 700, 380)))), [], W, H,
                          highlight=4)  # fmt: skip
    assert [i.get("pulse", False) for i in items(packet, "vehicle")] == [False, True]


def step(**kw: object) -> Step:
    fields: dict[str, object] = dict(
        index=1, maneuver="turn", modifier="left", instruction="Turn left onto Main Street", name="Main Street",
        ref=None, destinations=None, exit_ref=None, roundabout_exit=None, start_m=500.0, distance_m=300.0,
        duration_s=30.0, location=(47.6, -122.2),
    )  # fmt: skip
    fields.update(kw)
    return Step(**fields)  # type: ignore[arg-type]


def progress(current: Step, nxt: Step | None, along: float) -> Progress:
    to_next = None if nxt is None else nxt.start_m - along
    return Progress(along, along, 2.0, current, nxt, to_next, 1000.0, 100.0, False, 10.0, False)


def test_hud_has_the_banner_with_lanes_the_speed_and_the_roundabout() -> None:
    assert hud_packet(None, 10.0) == {"speed_mph": 22, "speed_limit": None, "school_limit": None}
    assert hud_packet(None, None, 35)["speed_limit"] == 35
    lanes = (Lane(("left",), True), Lane(("straight",), False), Lane(("straight", "right"), False))
    turn = step(lanes=lanes)
    hud = hud_packet(progress(step(index=0, maneuver="depart"), turn, 100.0), None)
    assert hud["speed_mph"] is None and hud["banner"]["distance"].startswith("in ")
    assert [lane["valid"] for lane in hud["banner"]["lanes"]] == [True, False, False]
    ring = step(maneuver="roundabout", roundabout_exit=3, roundabout_exits_m=(520.0, 540.0), name="Elm Street")
    assert hud_packet(progress(step(index=0), ring, 300.0), None)["roundabout"] == {"exit": 3, "passed": 0}
    inside = hud_packet(progress(ring, step(index=2, start_m=800.0), 530.0), None)
    assert inside["roundabout"] == {"exit": 3, "passed": 1}


def test_ordinals() -> None:
    assert [ordinal(n) for n in (1, 2, 3, 4, 11)] == ["1st", "2nd", "3rd", "4th", "11th"]


def test_frame_messages_round_trip() -> None:
    header, jpeg = parse_frame_message(frame_message({"t": 1.5, "overlay": {"items": []}}, b"\xff\xd8jpeg"))
    assert header == {"t": 1.5, "overlay": {"items": []}} and jpeg == b"\xff\xd8jpeg"


def test_the_video_socket_sends_each_frame_with_its_own_overlay() -> None:
    hub = UiHub(width=320, fps=1000.0).start()
    image = np.full((360, 640, 3), 90, np.uint8)
    overlay = {"items": [{"kind": "vehicle", "id": 1, "box": [0.1, 0.1, 0.2, 0.2], "color": "white"}], "you": None}
    hub.offer_frame(Frame(image=image, t=3.25, index=7, captured_at=time.perf_counter()), overlay)
    assert wait_for(lambda: hub.latest_message()[1] is not None)
    try:
        with (
            TestClient(create_server(hub, Actions()), base_url="http://localhost") as client,
            client.websocket_connect("ws://localhost/ws/video") as socket,
        ):
            header, jpeg = parse_frame_message(socket.receive_bytes())
            assert header["t"] == 3.25 and header["w"] == 640 and header["h"] == 360
            assert header["overlay"] == overlay and abs(header["captured"] - time.time()) < 5.0
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape[1] == 320
            threading.Thread(
                target=hub.offer_frame, args=(Frame(image=image, t=3.5, index=8, captured_at=time.perf_counter()),)
            ).start()
            header, _ = parse_frame_message(socket.receive_bytes())
            assert header["t"] == 3.5 and header["overlay"] is None
            assert "v2.html" not in client.get("/").text and "/ws/video" in client.get("/").text
            assert "/video.mjpg" in client.get("/v1").text
    finally:
        hub.stop()


def test_other_sites_cannot_watch_the_video_socket() -> None:
    from starlette.testclient import WebSocketDenialResponse
    from starlette.websockets import WebSocketDisconnect

    hub = UiHub()
    with (
        TestClient(create_server(hub, Actions()), base_url="http://localhost") as client,
        pytest.raises((WebSocketDenialResponse, WebSocketDisconnect)),
        client.websocket_connect("ws://localhost/ws/video", headers={"Origin": "http://evil.example"}) as socket,
    ):
        socket.receive_bytes()


def test_the_plain_v1_page_is_served_when_overlays_are_off() -> None:
    with TestClient(create_server(UiHub(), Actions(), overlays=False), base_url="http://localhost") as client:
        assert "/video.mjpg" in client.get("/").text and "/ws/video" in client.get("/v2").text


def test_someone_in_our_path_is_red_while_we_move() -> None:
    from state.pedestrians import PedestrianSnapshot, PersonView

    def person(track_id: int, in_path: bool, kind: str = "pedestrian") -> PersonView:
        return PersonView(track_id, kind, (500, 300, 540, 420), 0.2, 9.0, 0.2, None, 3.0, in_path, False, 3, False)

    tracks = [Track(20, "person", 0.9, (500, 300, 540, 420)), Track(21, "person", 0.9, (900, 300, 940, 420))]
    tracks.append(Track(22, "bicycle", 0.8, (700, 320, 760, 420)))
    for moving, expected in ((True, RED), (False, AMBER)):
        peds = PedestrianSnapshot(10.0, None, moving, None if moving else "stopped", False,
                                  (person(20, True), person(21, False), person(22, False, "bicycle")))  # fmt: skip
        snap = replace(snapshot(), pedestrians=peds)
        colors = {i["id"]: (i["color"], i["label"]) for i in items(frame_packet(snap, tracks, W, H), "person")}
        assert colors[20] == (expected, "pedestrian") and colors[21] == (AMBER, "pedestrian")
        assert colors[22] == (GRAY, "parked bicycle")
