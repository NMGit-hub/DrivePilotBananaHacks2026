"""V2-M4: scene memory ("what was that sign?"), numbered objects and the hidden tags of model answers (highlight,
thumbnail), which must never be spoken."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pytest

from answers.assistant import Assistant
from answers.scene import SceneHandler, draw_marks, pick_marks, strip_tags
from answers.speech import TextSpeaker
from answers.vision import MockVision, VisionRequest
from core.config import AppConfig, MemoryConfig, load_config
from core.types import Frame
from perception.detector import Track
from state.memory import SceneMemory
from state.stop_queue import EGO_KEY, Approach, QueueEntry, QueueSnapshot
from state.traffic_light import LightColor, LightReading
from state.world import VehicleView, WorldSnapshot, empty_snapshot
from voice.router import RouteKind, route

SATURDAY = datetime(2026, 10, 10, 17, 40)


def cfg(*overrides: str) -> AppConfig:
    return load_config(overrides=list(overrides), env_file=None, environ={})


def frame(t: float, w: int = 1280, h: int = 720, value: int = 90) -> Frame:
    return Frame(image=np.full((h, w, 3), value, np.uint8), t=t, index=round(t * 30), captured_at=0.0)


def car(track_id: int, box: tuple[float, float, float, float]) -> VehicleView:
    return VehicleView(track_id, "car", "white", None, False, True, 1.0, box, False)


# ----------------------------------------------------------------------------------------- memory


def test_memory_keeps_a_frame_a_second_for_a_minute_downscaled() -> None:
    memory = SceneMemory(MemoryConfig())
    for i in range(90 * 10):  # 90 s at 10 fps
        memory.add(frame(i / 10), [])
    kept = memory.frames(window_s=120.0, count=100)
    assert 59 <= len(kept) <= 61 and kept[-1].t == pytest.approx(89.9, abs=1.0)
    assert all(f.image.shape[1] == 960 for f in kept)
    assert kept[-1].t - kept[0].t <= 60.0
    spread = memory.frames(window_s=15.0, count=5)
    assert len(spread) == 5 and spread[-1].t > 88.0 and spread[0].t >= 74.0
    assert all(b.t > a.t for a, b in itertools.pairwise(spread))


def test_memory_keeps_the_biggest_whole_view_of_each_sign_and_forgets_on_a_seek() -> None:
    memory = SceneMemory(MemoryConfig())
    for k, size in enumerate((14, 30, 46, 60)):  # the sign grows as we approach, then is cut off by the edge
        box = (1000.0, 200.0, 1000.0 + size, 200.0 + size) if size < 60 else (1240.0, 200.0, 1280.0, 260.0)
        memory.add(frame(10.0 + k), [Track(7, "stop sign", 0.9, box)])
    memory.add(frame(15.0), [Track(8, "stop sign", 0.9, (100.0, 300.0, 108.0, 308.0))])  # too small to read
    signs = memory.signs(window_s=60.0, count=4, now=15.0)
    assert len(signs) == 1 and signs[0].track_id == 7 and signs[0].t == 12.0 and signs[0].side == "right"
    assert signs[0].image.shape[0] == 192
    memory.add(frame(3.0), [])  # time went backwards: another moment
    assert memory.signs(window_s=60.0, count=4, now=3.0) == [] and len(memory.frames(60.0, 10)) == 1


def test_a_detector_reading_is_kept_with_the_sign() -> None:
    memory = SceneMemory(MemoryConfig())
    memory.add_sign(frame(5.0), 3, "speed_limit", (600.0, 200.0, 640.0, 250.0), text="35 mph")
    memory.add_sign(frame(5.5), 3, "speed_limit", (590.0, 190.0, 650.0, 260.0))
    (sign,) = memory.signs(30.0, 4, now=6.0)
    assert sign.text == "35 mph" and sign.t == 5.5


# ---------------------------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What was that sign?", "memory"),
        ("Hey DrivePilot, what did that sign say?", "memory"),
        ("what was the sign we just passed", "memory"),
        ("What did we just pass?", "memory"),
        ("what was the speed limit back there", "memory"),
        ("did you see that sign?", "memory"),
        ("which store did we just pass?", "memory"),
        ("what does that sign say?", "scene"),  # in view now
        ("what's the speed limit?", "scene"),
        ("can I park here?", "scene"),
    ],
)
def test_memory_questions_are_routed_to_the_memory_answer(question: str, intent: str) -> None:
    r = route(question)
    assert r.kind is RouteKind.MODEL and r.intent == intent


def test_what_was_that_still_repeats_the_last_message() -> None:
    assert route("what was that?").intent == "repeat"


# ------------------------------------------------------------------------------------------- tags


@pytest.mark.parametrize(
    ("raw", "spoken", "tags"),
    [
        ("The white SUV on your left goes first. [ref 3]", "The white SUV on your left goes first.", [("ref", 3)]),
        ("[ref 2]", "", [("ref", 2)]),
        ("It said speed limit 35 [image 2].", "It said speed limit 35.", [("image", 2)]),
        ("Yes (REF #12)", "Yes", [("ref", 12)]),
        ("The car (marked 4) is parked.", "The car is parked.", []),
        ("The truck tagged 5 is blocking the lane.", "The truck is blocking the lane.", []),
        ("No tags here.", "No tags here.", []),
    ],
)
def test_tags_and_mark_numbers_are_taken_out_of_what_is_spoken(raw: str, spoken: str, tags: list[Any]) -> None:
    assert strip_tags(raw) == (spoken, tags)


def test_objects_are_numbered_nearest_first_and_drawn() -> None:
    snap = replace(
        empty_snapshot(10.0),
        vehicles=(car(1, (100, 300, 200, 360)), car(2, (500, 250, 900, 500))),
        light=LightReading(10.0, LightColor.RED, True, 1, ((600, 40, 620, 90),)),
    )
    marks = pick_marks(snap, [Track(9, "person", 0.9, (1000, 300, 1040, 420)), Track(4, "car", 0.9, (0, 0, 9, 9))], 3)
    assert [(m.number, m.kind, m.track_id) for m in marks] == [(1, "vehicle", 2), (2, "vehicle", 1), (3, "person", 9)]
    image = np.full((720, 1280, 3), 90, np.uint8)
    drawn = draw_marks(image, marks)
    assert (drawn != image).any() and (image == 90).all()  # (a copy)
    assert len(pick_marks(snap, [], 12)) == 3  # lights come last


def scene(model: MockVision, c: AppConfig, snap: WorldSnapshot, tracks: list[Track], **kw: Any) -> SceneHandler:
    latest = snap.frame
    assert latest is not None
    return SceneHandler(
        c,
        model,
        lambda count, spacing: [frame(latest.t - 1.4), latest],
        now=lambda: SATURDAY,
        tracks=lambda: tracks,
        **kw,
    )


def test_a_model_answer_names_the_object_it_means_and_the_tag_is_never_spoken() -> None:
    c = cfg()
    latest = frame(20.0)
    snap = replace(
        empty_snapshot(20.0), frame=latest, vehicles=(car(11, (100, 300, 300, 420)), car(12, (700, 320, 760, 380)))
    )
    seen: list[int] = []
    model = MockVision("That's the white car on your left, waiting at its stop line. [ref 1]", chunk_chars=4)
    handler = scene(model, c, snap, [], on_highlight=seen.append)
    spoken = list(handler.answer(route("what is that car doing?"), snap))
    assert spoken == ["That's the white car on your left, waiting at its stop line."]
    assert seen == [11]  # number 1 = the biggest (nearest) vehicle
    request: VisionRequest = model.requests[0]
    assert "[ref 3]" in request.system  # the model was told how to tag
    now = request.images[-1]
    assert now.label == "Now"
    plain = cv2.imencode(".jpg", latest.image, [cv2.IMWRITE_JPEG_QUALITY, c.vision.jpeg_quality])[1].tobytes()
    assert now.jpeg != plain  # the marks are drawn on what the model sees


def test_no_marks_when_the_latest_frame_is_not_the_snapshots_own() -> None:
    c = cfg()
    snap = replace(empty_snapshot(20.0), frame=frame(20.0), vehicles=(car(11, (100, 300, 300, 420)),))
    model = MockVision("A car. [ref 1]")
    seen: list[int] = []
    handler = SceneHandler(c, model, lambda n, s: [frame(20.03)], now=lambda: SATURDAY, on_highlight=seen.append)
    assert list(handler.answer(route("what is that car doing?"), snap)) == ["A car."]
    assert seen == [] and "[ref 3]" not in model.requests[0].system


def test_marking_can_be_switched_off() -> None:
    c = cfg("vision.mark_objects=false")
    snap = replace(empty_snapshot(20.0), frame=frame(20.0), vehicles=(car(11, (100, 300, 300, 420)),))
    model = MockVision("A car.")
    list(scene(model, c, snap, []).answer(route("what is that car doing?"), snap))
    assert "[ref 3]" not in model.requests[0].system


def test_a_memory_question_gets_the_remembered_frames_and_signs_and_shows_the_frame_it_used() -> None:
    c = cfg()
    memory = SceneMemory(c.memory)
    for i in range(200):  # 20 s at 10 fps; a stop sign seen around 12 s
        t = i / 10
        tracks = [Track(5, "stop sign", 0.9, (900.0, 250.0, 940.0, 290.0))] if 110 <= i <= 125 else []
        memory.add(frame(t, value=40 + i % 50), tracks)
    latest = frame(20.0)
    snap = replace(empty_snapshot(20.0), frame=latest)
    shown: list[Frame] = []

    def reply(request: VisionRequest) -> str:
        return "It was a stop sign, on the right about 8 seconds ago. [image 3]"

    model = MockVision(reply, chunk_chars=6)
    handler = scene(model, c, snap, [], memory=memory, on_image=shown.append)
    spoken = list(handler.answer(route("what was that sign we just passed?"), snap))
    assert spoken == ["It was a stop sign, on the right about 8 seconds ago."]
    request = model.requests[0]
    labels = [image.label for image in request.images]
    assert labels[0].startswith("Image 1: close-up of a stop sign we saw") and "right" in labels[0]
    assert [label.split(":")[0] for label in labels] == [f"Image {k}" for k in range(1, len(labels) + 1)]
    assert labels[-1].endswith("now") and len(labels) == 1 + c.memory.frames + 1
    assert "[image 2]" in request.system
    assert len(shown) == 1 and shown[0].t < 20.0  # image 3 = the second remembered frame
    assert shown[0].image.shape[1] == 960


def test_a_memory_question_without_memory_is_an_ordinary_scene_question() -> None:
    c = cfg()
    snap = replace(empty_snapshot(20.0), frame=frame(20.0))
    model = MockVision("I can't tell from here.")
    spoken = list(scene(model, c, snap, [], memory=SceneMemory(c.memory)).answer(route("what was that sign?"), snap))
    assert spoken == ["I can't tell from here."] and model.requests[0].images[-1].label == "Now"


# --------------------------------------------------------------------------------- local highlight


def test_a_right_of_way_answer_highlights_the_car_that_goes_next() -> None:
    c = cfg()
    order = (
        QueueEntry("track:3", Approach.LEFT, arrived=4.0, label="white SUV", track_id=3, last_seen=10.0),
        QueueEntry(EGO_KEY, Approach.US, arrived=6.0),
    )
    queue = QueueSnapshot(10.0, True, order, (), ego_position=2, tie_groups=(), deadlock=False, your_turn=False)
    snap = replace(empty_snapshot(10.0), frame=frame(10.0), queue=queue, ego_stopped=True, ego_motion_confident=True)
    assistant = Assistant(c, lambda: snap, TextSpeaker())
    seen: list[int] = []
    assistant.on_highlight = seen.append
    reply = assistant.ask("who got here first?")
    assert "white SUV" in str(reply.text) and seen == [3]


# ----------------------------------------------------------------------------------- which object


def _marks() -> dict[int, Any]:
    from answers.scene import Mark

    return {
        1: Mark(1, "vehicle", 101, (900, 140, 1230, 710), "bus", "yellow"),
        2: Mark(2, "vehicle", 102, (285, 400, 570, 610), "car", "gray"),
        3: Mark(3, "vehicle", 103, (0, 440, 100, 580), "car", "black"),
        12: Mark(12, "person", 112, (795, 385, 818, 437), "person", ""),
        13: Mark(13, "light", None, (600, 40, 620, 90), "traffic light", ""),
    }


@pytest.mark.parametrize(
    ("answer", "tagged", "expected"),
    [
        ("There's a school bus stopped on the right with its stop arm out.", [12], 1),  # wrong tag: the bus
        ("The school bus ahead has its red lights on.", [1], 1),
        ("That's a TD Bank branch on the right side of the street.", [2], None),  # not a marked object
        ("There's construction work on the right side.", [], None),
        ("Yes, the person is walking by the bus.", [], 12),  # no tag: the only person
        ("The gray car ahead is waiting.", [3], 2),  # the tag is a car, but the black one: color decides
        ("The black car on your left is parked.", [], 3),
        ("A silver car on your left is parked.", [2], 2),  # (our "gray" is the model's "silver")
        ("The car is turning.", [], 2),  # two cars, but the nearest is clearly bigger
        ("The light facing us is red.", [], None),  # lights have no track to pulse
        ('The sign says "LAST LOCAL TRUCK ROUTE".', [2], None),  # quoted text names nothing; signs aren't marked
        ("That white sign on the right says no trucks.", [2], None),
    ],
)
def test_the_highlight_is_the_object_the_answer_names(answer: str, tagged: list[int], expected: int | None) -> None:
    from answers.scene import resolve_highlight

    assert resolve_highlight(answer, tagged, _marks(), 1280) == expected


def test_a_tag_on_a_far_off_person_gives_way_to_the_one_crossing_in_front() -> None:
    from answers.scene import Mark, resolve_highlight

    marks = {
        1: Mark(1, "person", 201, (986, 136, 1164, 652), "person", ""),  # crossing close in front, frame right
        7: Mark(7, "person", 207, (334, 350, 352, 384), "person", ""),  # far away by the church, on the left
    }
    assert resolve_highlight("Yes, a person is crossing in front of us.", [7], marks, 1280) == 1
    assert resolve_highlight("A person is standing on your left by the church.", [7], marks, 1280) == 7
