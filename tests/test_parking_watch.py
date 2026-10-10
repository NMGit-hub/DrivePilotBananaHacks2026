"""The parking watch (answers/parking.py), the router's parking intents and the assistant's wiring.

Spaces and events are made by hand; the vision model is a fake that answers the spot check and the lot check.
The user's rules are in answers/parking.py's docstring; each test names the one it checks.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from answers import parking as P
from answers.assistant import CANCELLED, Assistant
from answers.speech import Priority, TextSpeaker, Utterance
from answers.vision import VisionError, VisionModel, VisionRequest
from core.config import ParkingConfig, load_config
from core.types import Frame
from state.parking import EMPTY, ParkedCar, ParkingEvent, Space
from state.world import WorldEvent, WorldSnapshot, empty_snapshot
from voice.router import RouteKind, parse_yes_no, route

FRAME = Frame(image=np.zeros((720, 1280, 3), np.uint8), t=0.0, index=0, captured_at=0.0)


class FakeModel(VisionModel):
    """Answers the spot check with `spot` and the lot check with `place`; `gate` holds spot checks back."""

    name = "fake"

    def __init__(self, spot: dict[str, Any] | None = None, place: str = "lot") -> None:
        self.spot = (
            spot
            if spot is not None
            else {"open": True, "restriction": "none", "before": "white SUV", "confidence": "high"}
        )
        self.place = place
        self.fail = False
        self.gate: threading.Event | None = None
        self.requests: list[VisionRequest] = []

    def stream(self, request: VisionRequest) -> Iterator[str]:  # pragma: no cover - not used
        raise VisionError("not used")

    def complete_json(
        self, request: VisionRequest, first_token_s: float = 4.0, total_s: float = 10.0
    ) -> dict[str, Any]:
        self.requests.append(request)
        if request.system == P.LOT_SYSTEM:
            return {"place": self.place, "confidence": "high"}
        if self.gate is not None:
            self.gate.wait(5.0)
        if self.fail:
            raise VisionError("timeout")
        return dict(self.spot)


def car(track_id: int, side: str, z: float) -> ParkedCar:
    return ParkedCar(track_id, "car", side, 3.5 if side == "right" else -3.5, z, (600, 300, 700, 400), 0.0)


def space(side: str = "right", along: float = 20.0, near: int = 6, far: int = 8, stalls: int = 1) -> Space:
    return Space(
        key=(side, near, far),
        side=side,
        near=car(near, side, 8.0),
        far=car(far, side, 13.0),
        stalls=stalls,
        z=9.85,
        region=(500.0, 300.0, 560.0, 380.0),
        cars_before=1,
        pitch=2.7,
        seen_since=0.0,
        visible_depth=1.2,
        confirmed=True,
        along=along,
    )


class World:
    """The snapshot the watch reads: how far we have driven, the parked rows, `lot_like`."""

    def __init__(self) -> None:
        self.along = 0.0
        self.lot_like = True
        self.rows: dict[str, tuple[ParkedCar, ...]] = {"left": (), "right": ()}

    def snapshot(self) -> WorldSnapshot:
        parking = dataclasses.replace(EMPTY, along=self.along, lot_like=self.lot_like, rows=self.rows)
        return dataclasses.replace(empty_snapshot(), frame=FRAME, parking=parking)


class Rig:
    def __init__(self, model: FakeModel | None = None, cfg: ParkingConfig | None = None, quiet: bool = False):
        self.world = World()
        self.said: list[Utterance] = []
        self.modes: list[str] = []
        self.replies: list[float] = []
        self.now = 0.0
        self.model = model if model is not None else FakeModel()
        self.watch = P.ParkingWatch(
            cfg or ParkingConfig(),
            load_config().vision,
            self.model,
            self.said.append,
            self.world.snapshot,
            self.modes.append,
            lambda: self.now,
            expect_reply=self.replies.append,
            quiet=lambda: quiet,
        )

    def event(self, kind: str, sp: Space | None = None, pc: ParkedCar | None = None) -> None:
        pe = ParkingEvent(kind, self.now, space=sp, car=pc)
        self.watch.on_world_event(WorldEvent(f"parking_{kind}", self.now, {}, payload=(pe, FRAME)))
        self.drain()

    def drain(self) -> None:
        self.watch._pool.submit(lambda: None).result(timeout=10)

    def texts(self) -> list[str]:
        return [u.text for u in self.said]

    def wait_for(self, condition: Callable[[], bool], seconds: float = 3.0) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not condition():
            time.sleep(0.05)


def found(rig: Rig, sp: Space) -> None:
    rig.event("candidate", sp)
    rig.event("spot", sp)


# ------------------------------------------------------------------------------------------- spots


def test_a_checked_spot_is_announced_with_cars_ahead_and_the_car_before_it() -> None:
    rig = Rig()
    rig.world.rows["right"] = (car(5, "right", 4.0), car(6, "right", 8.0), car(8, "right", 13.0))
    assert rig.watch.arm() == P.ARMED and rig.modes == ["lot"]
    found(rig, space(along=9.85))  # its near edge is 9.85 m ahead: the cars at 4 m and 8 m are before it
    assert rig.texts() == ["Open spot on your right, two cars ahead, just after the white SUV."]
    assert rig.said[0].priority is Priority.REQUESTED  # asked for: kept in quiet mode
    assert len(rig.model.requests) == 1  # one model check per space


@pytest.mark.parametrize(
    ("restriction", "said"),
    [
        ("handicap", []),  # the user: skip handicap, reserved, fire lane, no parking
        ("reserved", []),
        ("fire_lane", []),
        ("no_parking", []),
        ("ev", ["Open EV charging spot on your right, just after the white SUV."]),  # ...but name EV spots
        ("compact", ["Open compact spot on your right, just after the white SUV."]),
    ],
)
def test_restricted_spots_follow_the_users_rules(restriction: str, said: list[str]) -> None:
    rig = Rig(FakeModel({"open": True, "restriction": restriction, "before": "white SUV", "confidence": "high"}))
    rig.watch.arm()
    found(rig, space())
    assert rig.texts() == said


def test_the_model_saying_taken_or_unsure_means_no_announcement() -> None:
    for verdict in (
        {"open": False, "restriction": "none", "before": None, "confidence": "high"},
        {"open": True, "restriction": "none", "before": None, "confidence": "low"},
    ):
        rig = Rig(FakeModel(verdict))
        rig.watch.arm()
        found(rig, space())
        assert rig.texts() == []


def test_a_spot_the_model_could_not_check_is_announced_hedged() -> None:
    model = FakeModel()
    model.fail = True
    rig = Rig(model)
    rig.watch.arm()
    found(rig, space())
    assert rig.texts() == ["Looks like an open spot on your right."]


def test_a_spot_confirmed_before_its_check_returns_is_announced_when_it_does() -> None:
    model = FakeModel()
    model.gate = threading.Event()
    rig = Rig(model)
    rig.watch.arm()
    sp = space()
    pe = ParkingEvent("candidate", 0.0, space=sp)
    rig.watch.on_world_event(WorldEvent("parking_candidate", 0.0, {}, payload=(pe, FRAME)))
    pe = ParkingEvent("spot", 0.0, space=sp)
    rig.watch.on_world_event(WorldEvent("parking_spot", 0.0, {}, payload=(pe, FRAME)))
    assert rig.texts() == []  # the check is still running
    model.gate.set()
    rig.drain()
    assert rig.texts() == ["Open spot on your right, just after the white SUV."]


def test_the_same_place_found_again_is_not_repeated() -> None:
    rig = Rig()
    rig.watch.arm()
    found(rig, space(along=20.0, near=6, far=8))
    found(rig, space(along=20.4, near=16, far=18))  # re-detected under new track ids
    assert len(rig.texts()) == 1


def test_only_the_nearest_spot_is_spoken_and_the_next_once_it_is_passed() -> None:
    rig = Rig()
    rig.watch.arm()
    found(rig, space(along=20.0, near=6, far=8))
    found(rig, space(along=35.0, near=12, far=14, side="left"))
    assert len(rig.texts()) == 1  # the user: speak again only if we pass it, it's taken, or a closer one opens
    rig.world.along = 21.0  # we drove past the first one (without parking)
    rig.wait_for(lambda: len(rig.texts()) == 2)
    assert rig.texts()[1].startswith("Open spot on your left")


def test_a_closer_spot_is_spoken_at_once() -> None:
    rig = Rig()
    rig.watch.arm()
    found(rig, space(along=30.0, near=6, far=8))
    found(rig, space(along=22.0, near=3, far=5, side="left"))
    assert [t.split(",")[0] for t in rig.texts()] == ["Open spot on your right", "Open spot on your left"]


def test_a_taken_spot_is_said_and_the_next_one_follows() -> None:
    rig = Rig()
    rig.watch.arm()
    first, second = space(along=20.0, near=6, far=8), space(along=35.0, near=12, far=14, side="left")
    found(rig, first)
    found(rig, second)
    rig.event("taken", first)
    assert rig.texts()[1:] == [P.TAKEN, "Open spot on your left, just after the white SUV."]


def test_a_car_pulling_out_is_announced() -> None:
    rig = Rig()
    rig.world.rows["right"] = (car(5, "right", 4.0), car(6, "right", 8.0), car(9, "right", 15.0))
    rig.watch.arm()
    rig.event("pulling_out", pc=car(9, "right", 15.0))
    assert rig.texts() == ["A car is pulling out on your right, three cars ahead."]


def test_nothing_is_said_without_a_watch() -> None:
    rig = Rig()
    found(rig, space())
    rig.event("pulling_out", pc=car(9, "right", 15.0))
    assert rig.texts() == [] and rig.model.requests == []


# -------------------------------------------------------------------------------------- the end


def test_parking_ends_the_watch_silently() -> None:
    rig = Rig()
    rig.watch.arm()
    rig.event("parked")
    assert not rig.watch.active and rig.texts() == []


def test_the_watch_ends_after_the_timeout_and_says_so() -> None:
    rig = Rig(cfg=ParkingConfig(watch_timeout_s=60.0))
    rig.watch.arm()
    rig.now = 61.0
    rig.wait_for(lambda: not rig.watch.active)
    assert not rig.watch.active and rig.texts() == [P.TIMED_OUT]


def test_stop_looking() -> None:
    rig = Rig()
    assert rig.watch.cancel() == P.NOT_LOOKING
    rig.watch.arm()
    assert rig.watch.arm() == P.ALREADY
    assert rig.watch.cancel() == P.STOPPED and not rig.watch.active


# ------------------------------------------------------------------------------------ the offer


def test_a_lot_gets_one_offer_and_a_plain_yes_starts_the_watch() -> None:
    rig = Rig()
    rig.event("lot")
    assert rig.texts() == [P.OFFER] and rig.said[0].priority is Priority.INFO  # (muted in quiet mode)
    assert rig.replies == [ParkingConfig().reply_window_s]  # listen for a plain yes/no
    assert rig.watch.awaiting_reply()
    rig.event("lot")  # still the same lot: no second offer
    assert rig.texts() == [P.OFFER]
    assert rig.watch.on_reply(True) == P.ARMED and rig.watch.active
    assert not rig.watch.awaiting_reply()


def test_no_offer_in_quiet_mode_or_when_the_model_says_its_a_street() -> None:
    quiet = Rig(quiet=True)
    quiet.event("lot")
    street = Rig(FakeModel(place="street"))
    street.event("lot")
    assert quiet.texts() == [] and street.texts() == []


def test_declining_the_offer() -> None:
    rig = Rig()
    rig.event("lot")
    assert rig.watch.on_reply(False) == P.DECLINED and not rig.watch.active


def test_the_offer_is_only_open_for_a_while() -> None:
    rig = Rig()
    rig.event("lot")
    rig.now = ParkingConfig().reply_window_s + 5.0
    assert not rig.watch.awaiting_reply()


def test_street_or_lot_mode_comes_from_the_model_when_asked_off_a_lot() -> None:
    rig = Rig(FakeModel(place="street"))
    rig.world.lot_like = False
    rig.watch.arm()
    rig.drain()
    assert rig.modes == ["street"]
    asked = Rig()
    assert asked.watch.arm(street=True) == P.ARMED_STREET and asked.modes == ["street"]


# ------------------------------------------------------------------------------ router, assistant


def test_plain_yes_and_no() -> None:
    assert [parse_yes_no(t) for t in ("Yes.", "yeah sure", "No thanks.", "nope", "what's that sign?")] == [
        True,
        True,
        False,
        False,
        None,
    ]


def make_assistant() -> tuple[Assistant, Rig, TextSpeaker]:
    rig = Rig()
    speaker = TextSpeaker()
    assistant = Assistant(load_config(), rig.world.snapshot, speaker, parking=rig.watch)
    return assistant, rig, speaker


def test_the_assistant_starts_and_stops_the_watch() -> None:
    assistant, rig, _ = make_assistant()
    assert route("find me a parking spot").kind is RouteKind.PARKING
    assert assistant.ask("Hey DrivePilot, find me a parking spot.").text == P.ARMED and rig.watch.active
    assert assistant.ask("stop looking").text == P.STOPPED and not rig.watch.active
    assistant.ask("help me find a spot")
    assert assistant.ask("never mind").text == CANCELLED and not rig.watch.active


def test_a_yes_to_the_offer_reaches_the_watch_typed_spoken_or_without_the_wake_word() -> None:
    assistant, rig, _ = make_assistant()
    assert not assistant.takes_reply("yes")  # no offer pending
    rig.event("lot")
    assert assistant.takes_reply("Yes please.") and not assistant.takes_reply("look at that dog")
    assert assistant.ask("Yes please.").text == P.ARMED and rig.watch.active
