"""M3 review regressions: answers and alerts must never vouch for traffic, flip yes/no, or break a promise."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

from answers import local
from answers.alerts import LIGHT_GREEN, YOUR_TURN
from answers.assistant import Assistant
from answers.local import Watch
from answers.speech import Priority, TextSpeaker
from core.config import load_config
from state.stop_queue import Approach, QueueEntry
from state.traffic_light import LightColor
from state.world import WorldEvent, WorldSnapshot
from tests.test_answers import A, FakeHandler, L, assistant, engine, light, queue, snap
from voice.router import Route, RouteKind


def test_should_wait_is_answered_as_asked() -> None:
    waiting = snap(queue(12.0, [(2, A, 10.0, "gray truck")]))
    answer = local.should_wait(waiting, "should i wait")
    assert answer is not None and answer.text.startswith("Yes, wait, the gray truck across from you is next")
    ours = snap(queue(12.0, [(1, L, 8.0, "white car")], departed=(1,)))
    answer = local.should_wait(ours, "should i wait")
    assert answer is not None and not answer.text.startswith("Yes") and "check the intersection" in answer.text.lower()
    red = local.should_wait(snap(light=light(LightColor.RED)), "should i keep waiting")
    assert red is not None and red.text.startswith("Yes, the light's red") and red.watches == (Watch.LIGHT,)
    walker = local.should_wait(ours, "should i wait for the pedestrian")
    assert walker is not None and walker.text == local.PEDESTRIAN_YIELD


@pytest.mark.parametrize(
    "question",
    [
        "Is it safe to pull out?",
        "Is it safe to turn left?",
        "Is it clear to go?",
        "Good to go?",
        "Is it OK for me to go?",
    ],
)
def test_is_it_safe_never_gets_a_yes(question: str) -> None:
    a, _ = assistant(snap(queue(12.0, [])), model=FakeHandler("<model>"))  # alone at a stop (maybe a 2-way stop)
    reply = a.ask(question)
    assert reply.text is not None and not reply.text.lower().startswith("yes")
    if reply.source == "local":
        assert "can't judge" in reply.text and "check" in reply.text.lower()


@pytest.mark.parametrize(
    "question",
    [
        "Can I go?",
        "Should I go?",
        "Can I still make it?",
        "Should I stop?",
        "Can I go through the yellow?",
        "Should I wait?",
        "Will I make the light?",
    ],
)
def test_stop_or_go_questions_are_refused_while_moving(question: str) -> None:
    a, _ = assistant(
        snap(ego_stopped=False, ego_stopped_since=None, light=light(LightColor.YELLOW)), model=FakeHandler("<model>")
    )
    reply = a.ask(question)
    assert reply.source == "local" and "can't judge that safely while we're moving" in (reply.text or "")
    assert a.alerts.armed() == set()


@pytest.mark.parametrize("question", ["Can I merge now?", "Is it safe to go?", "Is the left lane clear?", "Can I go?"])
def test_unsure_if_stopped_is_refused(question: str) -> None:
    a, _ = assistant(snap(queue(12.0, []), ego_motion_confident=False), model=FakeHandler("yes the lane is clear"))
    assert a.ask(question).text == local.UNSURE_REFUSAL


def test_tell_me_when_i_can_turn_is_not_a_promise_about_traffic() -> None:
    a, _ = assistant(snap(light=light(LightColor.RED)))
    text = a.ask("Tell me when I can turn left").text or ""
    assert "judge the traffic yourself" in text and a.alerts.armed() == {Watch.LIGHT}
    b, _ = assistant(snap(light=light(LightColor.GREEN)))
    assert b.ask("Tell me when it's safe to merge").text == local.GAP_WATCH_STOPPED and b.alerts.armed() == set()
    c, _ = assistant(snap(ego_stopped=False))
    assert c.ask("Tell me when the lane is clear").text == local.GAP_WATCH_MOVING


def test_navigation_phrasings_never_arm_a_watch() -> None:
    a, speaker = assistant(snap(ego_stopped=False, t=100.0))
    for question in ("Tell me where to go", "Tell me about the next turn", "Tell me when to turn"):
        assert a.ask(question).source == "nav"
    assert a.alerts.armed() == set()
    a.on_world_event(WorldEvent("light_green", 120.0))
    assert LIGHT_GREEN not in speaker.texts()


def test_requested_youre_up_at_the_next_intersection_is_spoken() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("your_turn", 100.0))  # stop A
    alerts.on_world_event(WorldEvent("left_intersection", 101.5))
    assert alerts.arm(Watch.TURN, 108.0) == "armed"  # stop B: "can I go?" -> "I'll tell you"
    alerts.on_world_event(WorldEvent("your_turn", 112.0))
    assert [u.text for u in said] == [YOUR_TURN, YOUR_TURN] and said[1].priority is Priority.REQUESTED


def test_a_watch_is_only_used_up_when_its_alert_is_spoken() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("your_turn", 100.0))
    alerts.arm(Watch.TURN, 105.0)
    alerts.on_world_event(WorldEvent("your_turn", 106.0))  # within the cooldown: not spoken...
    assert len(said) == 1 and alerts.armed() == {Watch.TURN}  # ...so the request stays


def test_never_mind_silences_youre_up_for_this_stop_only() -> None:
    a, speaker = assistant(snap(queue(12.0, [(2, A, 10.0, "gray truck")])))
    a.ask("Can I go?")
    assert a.ask("Never mind").text == "Okay, cancelled."
    a.on_world_event(WorldEvent("your_turn", 25.0))
    assert YOUR_TURN not in speaker.texts()
    a.on_world_event(WorldEvent("left_intersection", 27.0))
    a.on_world_event(WorldEvent("your_turn", 60.0))  # the next stop
    assert speaker.texts()[-1] == YOUR_TURN


def test_youre_up_that_fires_while_answering_is_not_promised_again() -> None:
    """The event lands between reading the snapshot and arming the watch (perception thread)."""
    waiting = snap(queue(12.0, [(2, A, 10.0, "gray truck")]))
    holder: dict[str, Assistant] = {}

    def read_snapshot() -> WorldSnapshot:
        holder["a"].on_world_event(WorldEvent("your_turn", 20.0))
        return waiting

    a = Assistant(load_config(), read_snapshot, TextSpeaker())
    holder["a"] = a
    a.alerts.set_quiet(True)  # the unrequested alert is muted...
    assert a.ask("Can I go?").text == YOUR_TURN  # ...but the answer says it, instead of "Not yet ... I'll tell you"


def test_race_with_newer_state_answers_from_the_newer_state() -> None:
    before = snap(queue(12.0, [(2, A, 10.0, "gray truck")]), t=20.0)
    after = snap(queue(12.0, [(2, A, 10.0, "gray truck")], departed=(2,), t=21.0), t=21.0)
    calls = iter([before, after, after])
    a = Assistant(load_config(), lambda: next(calls), TextSpeaker())
    a.on_world_event(WorldEvent("your_turn", 20.5))
    assert a.ask("Can I go?").text == "It's your turn now. Check the intersection before you go."


def test_watch_asked_while_driving_expires_if_we_never_stop() -> None:
    alerts, said = engine(approach_timeout_s=45.0)
    alerts.arm(Watch.LIGHT, 10.0, moving=True)  # approaching light A, which turned green before we got there
    alerts.on_world_event(WorldEvent("light_green", 250.0))  # light B, minutes later
    assert said == []
    alerts.arm(Watch.LIGHT, 300.0, moving=True)
    alerts.on_world_event(WorldEvent("ego_stopped", 310.0))
    alerts.on_world_event(WorldEvent("light_green", 380.0))  # we did stop at this light: announced
    assert [u.text for u in said] == [LIGHT_GREEN]


def test_a_request_made_before_a_seek_is_ignored() -> None:
    alerts, _ = engine()
    generation = alerts.generation
    alerts.reset()  # segment_reset
    assert alerts.arm(Watch.LIGHT, 10.0, generation=generation) == "stale" and alerts.armed() == set()


def test_repeating_an_old_alert_says_it_was_earlier() -> None:
    holder = {"s": snap(queue(12.0, [(2, A, 10.0, "gray truck")]))}
    a = Assistant(load_config(), lambda: holder["s"], TextSpeaker())
    a.on_world_event(WorldEvent("your_turn", 20.0))
    holder["s"] = snap(ego_stopped=False, t=60.0)
    assert a.ask("Repeat that").text == "Earlier I said: " + YOUR_TURN


def test_dropped_car_ahead_is_never_you_are_first() -> None:
    lost = QueueEntry(
        key="track:1", approach=Approach.LEFT, arrived=8.0, label="white car", drop_reason="lost from view"
    )
    s = snap(replace(queue(12.0, []), dropped=(lost,)))
    for handler in (local.who_first, local.position, local.can_i_go):
        text = handler(s).text
        assert text.startswith("Nobody I can see is ahead of you.") and "I lost sight of the white car" in text


def test_a_failing_handler_gives_a_spoken_fallback() -> None:
    class Broken:
        def answer(self, route: Route, snapshot: WorldSnapshot, cancel: object = None) -> str:
            raise RuntimeError("network down")

    a, speaker = assistant(snap(), model=Broken())
    reply = a.ask("What does that sign say?")
    assert reply.source == "error" and speaker.texts() == [reply.text]


def test_gap_like_questions_that_reach_the_model_are_refused_while_moving() -> None:
    model = FakeHandler("Sure, go for it.")
    a, _ = assistant(snap(ego_stopped=False), model=model)
    reply = a.ask("Would merging now be dumb?")
    assert reply.route.kind is RouteKind.MODEL and reply.text == local.GAP_REFUSAL and not model.routes


def test_model_answers_stream_and_stop_cuts_them_off() -> None:
    closed: list[bool] = []
    holder: dict[str, Assistant] = {}

    class Streaming:
        def answer(self, route: Route, snapshot: WorldSnapshot, cancel: object = None) -> Iterator[str]:
            try:
                yield "The sign says no parking."
                holder["a"].ask("Stop")  # the driver says "stop" while the answer is still coming
                yield "Tow-away zone after 4 PM."
            finally:
                closed.append(True)

    a, speaker = assistant(snap(), model=Streaming())
    holder["a"] = a
    reply = a.ask("What does that sign say?")
    assert reply.text == "The sign says no parking." and speaker.texts() == ["The sign says no parking."]
    assert reply.first_ms is not None and closed == [True]
