"""Regression tests for the independent review of V2-M1 (parking fixes), V2-M2 (overlays) and V2-M4 (memory,
highlight), 2026-10-10. Each test failed on the reviewed code."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from answers.assistant import Assistant
from answers.scene import NO_VOUCH, Mark, SceneHandler, resolve_highlight, strip_tags
from answers.speech import TextSpeaker
from answers.vision import MockVision
from core.config import AppConfig, ParkingConfig, load_config
from core.types import Frame
from eval.parking_eval import ClipLabels, LabeledSpot, match_report
from state.memory import SceneMemory
from state.stop_queue import EGO_KEY, Approach, QueueEntry, QueueSnapshot
from state.world import VehicleView, WorldSnapshot, empty_snapshot
from tests.test_parking import Lot, row
from ui.overlay import AMBER, frame_packet
from voice.router import route

SATURDAY = datetime(2026, 10, 10, 17, 40)


def cfg() -> AppConfig:
    return load_config(overrides=[], env_file=None, environ={})


def frame(t: float, w: int = 1280, h: int = 720) -> Frame:
    return Frame(image=np.full((h, w, 3), 90, np.uint8), t=t, index=round(t * 30), captured_at=0.0)


def car(track_id: int, box: tuple[float, float, float, float]) -> VehicleView:
    return VehicleView(track_id, "car", "white", None, False, True, 1.0, box, False)


def answer(
    question: str, reply: str, chunk: int = 7, vehicles: tuple[VehicleView, ...] = ()
) -> tuple[list[str], list[int]]:
    latest = frame(20.0)
    snap = replace(empty_snapshot(20.0), frame=latest, vehicles=vehicles or (car(11, (100, 300, 300, 420)),))
    seen: list[int] = []
    handler = SceneHandler(
        cfg(), MockVision(reply, chunk_chars=chunk), lambda n, s: [frame(18.6), latest], now=lambda: SATURDAY,
        tracks=lambda: [], on_highlight=seen.append,
    )  # fmt: skip
    return list(handler.answer(route(question), snap)), seen


# ---------------------------------------------------------------- 1. a tag between sentences hid a bare "Go."


@pytest.mark.parametrize("chunk", [1, 4, 7, 1000])
@pytest.mark.parametrize("permission", ["Go.", "Merge now.", "Yes, go."])
def test_a_tag_between_sentences_does_not_let_a_permission_through(chunk: int, permission: str) -> None:
    spoken, _ = answer(
        "is the truck on the right stopping?", f"The truck on your right is stopping. [ref 1] {permission}", chunk
    )
    assert spoken == ["The truck on your right is stopping.", NO_VOUCH]


def test_a_short_lead_in_joined_to_a_bare_go_is_still_caught() -> None:
    spoken, _ = answer("what's the light doing?", "Sure thing. Go. The light is green.", 1000)
    assert NO_VOUCH in spoken and not any(s.startswith("Sure thing. Go") for s in spoken)


# ---------------------------------------------------------------- 2. deadlock / moving pulsed one car


def queue_snapshot(deadlock: bool, stopped: bool = True) -> WorldSnapshot:
    order = (
        QueueEntry("track:3", Approach.LEFT, arrived=4.0, label="white SUV", track_id=3, last_seen=10.0),
        QueueEntry("track:4", Approach.AHEAD, arrived=4.2, label="gray truck", track_id=4, last_seen=10.0),
        QueueEntry(EGO_KEY, Approach.US, arrived=4.1),
    )
    queue = QueueSnapshot(10.0, True, order, (), 3, (), deadlock, False)
    return replace(
        empty_snapshot(10.0), frame=frame(10.0), queue=queue, ego_stopped=stopped, ego_motion_confident=True,
        vehicles=(car(3, (100, 300, 300, 400)), car(4, (600, 300, 700, 380))),
    )  # fmt: skip


def highlights_for(question: str, snap: WorldSnapshot) -> list[int]:
    assistant = Assistant(cfg(), lambda: snap, TextSpeaker())
    seen: list[int] = []
    assistant.on_highlight = seen.append
    assistant.ask(question)
    return seen


def test_no_car_is_pointed_at_in_a_deadlock_or_while_moving() -> None:
    assert highlights_for("who goes first?", queue_snapshot(deadlock=True)) == []
    assert highlights_for("can I go?", queue_snapshot(deadlock=False, stopped=False)) == []
    assert highlights_for("who goes first?", queue_snapshot(deadlock=False)) == [3]
    packet = frame_packet(queue_snapshot(deadlock=True), [], 1280, 720)
    assert not any(i.get("color") == AMBER for i in packet["items"] if i["kind"] == "vehicle")


# ---------------------------------------------------------------- 3. a replaced answer still pulsed its car


def test_an_answer_replaced_by_a_safety_net_points_at_nothing() -> None:
    spoken, seen = answer(
        "what is the white car doing?", "Go ahead, the white car on your left has stopped for you. [ref 1]"
    )
    assert spoken == [NO_VOUCH] and seen == []


# ---------------------------------------------------------------- 4. / 5. what the tag stripping keeps and removes


@pytest.mark.parametrize(
    "text",
    [
        "The speed limit sign is marked 35.",
        "The exit is labeled 25 on the sign.",
        "The lane is marked as 3 on the overhead sign.",
        "The sign reads [EXIT ONLY].",
        "It says [No parking 8am-6pm] on the sign.",
        "You're in lane number 2 of 3.",
        "Take exit 15 for the airport.",
        "Refer to the sign on the right.",
    ],
)
def test_sign_text_and_real_numbers_are_kept(text: str) -> None:
    assert strip_tags(text) == (text, [])


@pytest.mark.parametrize(
    ("raw", "spoken", "tags"),
    [
        ("In Image 2, the sign said speed limit 35.", "The sign said speed limit 35.", []),
        ("The sign in image 3 said no left turn.", "The sign said no left turn.", []),
        ("The car number 1 on your left is parked.", "The car on your left is parked.", []),
        ("Car #1 is parked on your left.", "Car is parked on your left.", []),
        ("The car (1) is parked.", "The car is parked.", []),
        ("That's the parked car. (ref: 1)", "That's the parked car.", [("ref", 1)]),
        ("That's the parked car. Ref 1.", "That's the parked car.", [("ref", 1)]),
        ("That's the parked car [ref 1", "That's the parked car", [("ref", 1)]),  # cut off at max_tokens
        ("Both cars are parked. [ref 1, ref 2]", "Both cars are parked.", [("ref", 1), ("ref", 2)]),
    ],
)
def test_mark_and_image_numbers_are_never_spoken(raw: str, spoken: str, tags: list[tuple[str, int]]) -> None:
    assert strip_tags(raw) == (spoken, tags)


# ---------------------------------------------------------------- 6. / 10. memory thumbnails


def test_a_memory_answer_from_a_sign_close_up_shows_it_after_the_answer() -> None:
    c = cfg()
    memory = SceneMemory(c.memory)
    from perception.detector import Track

    for i in range(150):
        tracks = [Track(5, "stop sign", 0.9, (900.0, 250.0, 940.0, 290.0))] if 50 <= i <= 60 else []
        memory.add(frame(i / 10), tracks)
    latest = frame(15.0)
    snap = replace(empty_snapshot(15.0), frame=latest)
    events: list[str] = []
    handler = SceneHandler(
        c, MockVision("It was a stop sign on the right. [image 1]"), lambda n, s: [latest], now=lambda: SATURDAY,
        memory=memory, on_image=lambda f: events.append(f"image {f.image.shape[0]}"),
    )  # fmt: skip
    for sentence in handler.answer(route("what was that sign?"), snap):
        events.append(f"say {sentence}")
    assert events == ["say It was a stop sign on the right.", "image 192"]  # the close-up, after the answer


# ---------------------------------------------------------------- 7. sparse rows measured a gap as the pitch


@pytest.mark.xfail(
    strict=True,
    reason="known limit (PROGRESS.md): fixing it (tighter pitch window, lower quartile, harmonic voting) added false "
    "spots on the real BDD100K lot clips, so the measured-better estimator stays",
)
@pytest.mark.parametrize("empty", [{2, 4, 6, 8, 10, 12}, {3, 5, 7, 10, 12, 14}])
def test_a_row_with_many_single_gaps_finds_them_all(empty: set[int]) -> None:
    lot = Lot(row("right", range(18), empty=empty) + row("left", range(18), first_id=100), cfg=ParkingConfig())
    lot.drive(18.0)
    right = [e for e in lot.spots() if e.space is not None and e.space.side == "right"]
    assert len(right) == len(empty)


# ---------------------------------------------------------------- 8. the hood / a car beside us beat the tag


def test_our_hood_is_not_numbered_and_a_car_beside_us_does_not_override_the_tag() -> None:
    from answers.scene import pick_marks

    hood = car(1, (40.0, 640.0, 1240.0, 719.0))
    lead = car(2, (600.0, 350.0, 680.0, 410.0))
    snap = replace(empty_snapshot(10.0), vehicles=(hood, lead))
    marks = pick_marks(snap, [], 12, 1280, 720)
    assert [m.track_id for m in marks] == [2]
    beside = {
        1: Mark(1, "vehicle", 31, (900.0, 300.0, 1279.0, 700.0), "car", "white"),  # parked right beside us
        2: Mark(2, "vehicle", 32, (600.0, 350.0, 680.0, 410.0), "car", "white"),
    }
    assert resolve_highlight("The car ahead is braking.", [2], beside, 1280) == 2


# ---------------------------------------------------------------- 11. scoring windows


def test_a_spot_that_never_leaves_the_frame_matches_only_for_a_while() -> None:
    spot = LabeledSpot("right", 5.0, None, "none", "clear", 1, "car")
    clip = ClipLabels("c", "lot_aisles", True, ((0.0, 60.0),), None, (spot,), (), ())
    assert match_report(10.0, "right", clip, 60.0) == (spot, None)
    assert match_report(50.0, "right", clip, 60.0) == (None, None)


# ---------------------------------------------------------------- risk: memory routes


@pytest.mark.parametrize(
    "question",
    [
        "if I go past the school what is the speed limit?",
        "is the car back there still coming?",
        "did you see the light turn green?",
        "did you see that guy crossing?",
    ],
)
def test_questions_about_now_are_not_memory_questions(question: str) -> None:
    assert route(question).intent == "scene"
