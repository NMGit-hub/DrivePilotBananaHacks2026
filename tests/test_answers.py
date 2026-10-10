"""M3: local answers (right of way, lights, refusals), the alert engine and the assistant (controls, routing)."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from answers import local
from answers.alerts import LIGHT_GREEN, YOUR_TURN, AlertEngine
from answers.assistant import NO_MODEL, NO_NAV, QUIET_ON, STARTING, Assistant
from answers.local import Watch
from answers.speech import Priority, TextSpeaker, Utterance
from core.config import AlertsConfig, load_config
from state.stop_queue import Approach, EgoStop, QueueSnapshot, StopQueue, WaitingVehicle
from state.traffic_light import LightColor, LightReading
from state.world import VehicleView, WorldEvent, WorldSnapshot, empty_snapshot
from voice.router import Route

L, A, R = Approach.LEFT, Approach.AHEAD, Approach.RIGHT
BOXES = {L: (50.0, 300.0, 250.0, 380.0), A: (560.0, 280.0, 640.0, 330.0), R: (1000.0, 300.0, 1200.0, 380.0)}


def queue(
    ego_arrived: float | None,
    cars: list[tuple[int, Approach, float, str]],
    t: float = 20.0,
    departed: tuple[int, ...] = (),
    unknown: tuple[int, ...] = (),
) -> QueueSnapshot:
    """A queue where we stopped at `ego_arrived` and `cars` = (track id, approach, stop time, label)."""
    q = StopQueue()
    ego = EgoStop(at_stop=ego_arrived is not None, stopped_since=ego_arrived, moving_for_s=0.0)
    q.update(ego_arrived or 0.0, ego, [])

    def vehicles(gone: bool) -> list[WaitingVehicle]:
        return [
            WaitingVehicle(
                tid,
                appr,
                not (gone and tid in departed),
                None if gone and tid in departed else stop_t,
                2.0 if gone and tid in departed else 0.0,
                BOXES[appr],
                label=label,
                arrival_known=tid not in unknown,
                moving_for=1.5 if gone and tid in departed else 0.0,
            )
            for tid, appr, stop_t, label in cars
        ]

    q.update(t - 1.0, ego, vehicles(False))
    if departed:
        q.update(t, ego, vehicles(True))
    return q.snapshot()


def snap(q: QueueSnapshot | None = None, t: float = 20.0, **kw: object) -> WorldSnapshot:
    base = empty_snapshot(t)
    fields: dict[str, object] = {"ego_stopped": True, "ego_stopped_since": 10.0, "ego_motion_confident": True}
    if q is not None:
        fields["queue"] = q
        fields["at_stop_sign"] = q.active
        fields["stop_sign_last_seen"] = 9.0
    fields.update(kw)
    return replace(base, **fields)  # type: ignore[arg-type]


def light(color: LightColor, confident: bool = True) -> LightReading:
    return LightReading(t=20.0, color=color, confident=confident, n_lights=1, boxes=((600.0, 100.0, 620.0, 150.0),))


def assert_never_bare_go(text: str) -> None:
    """PLAN.md safety rule: anything that lets us go also says to check first."""
    lowered = text.lower()
    permits = re.search(r"\b(yes|your turn|you're first|it's green|light's green|already green|go first)\b", lowered)
    if permits and "not yet" not in lowered:
        assert "check" in lowered, text


# ------------------------------------------------------------------------------------- right of way


def test_who_first_lists_the_cars_ahead_and_our_place() -> None:
    q = queue(12.0, [(1, L, 8.0, "white car"), (2, A, 10.0, "gray truck")])
    answer = local.who_first(snap(q))
    assert answer.text == "The white car on your left, then the gray truck across from you. You're third."


def test_tie_with_the_car_on_our_right_is_explained() -> None:
    q = queue(12.0, [(1, R, 12.4, "black car")])
    text = local.who_first(snap(q)).text
    assert text.startswith("You and the black car on your right stopped together. The car on the right goes first.")
    assert text.endswith("You're second.")


def test_tie_with_the_car_on_our_left_we_go_first_but_check() -> None:
    q = queue(12.0, [(1, L, 12.3, "red car")])
    text = local.who_first(snap(q)).text
    assert "You're on their right, so you go first" in text
    assert_never_bare_go(text)


def test_alone_at_the_stop() -> None:
    text = local.who_first(snap(queue(12.0, []))).text
    assert text == "Nobody else is waiting. You're first. Check before you go."


def test_car_already_waiting_is_described_as_such() -> None:
    q = queue(12.0, [(1, L, 12.5, "gray car")], unknown=(1,))
    assert "already waiting when you got here" in local.who_first(snap(q)).text


def test_can_i_go_not_yet_names_who_is_next_and_arms_the_turn_alert() -> None:
    q = queue(12.0, [(2, A, 10.0, "gray truck")])
    answer = local.can_i_go(snap(q))
    assert answer.text == "Not yet, the gray truck across from you is next. I'll tell you."
    assert answer.watches == (Watch.TURN,)


def test_can_i_go_with_two_ahead() -> None:
    q = queue(12.0, [(1, L, 8.0, "white car"), (2, A, 10.0, "gray truck")])
    text = local.can_i_go(snap(q)).text
    assert text.startswith("Not yet, 2 cars go before you: the white car on your left, then the gray truck")


def test_can_i_go_after_the_others_left() -> None:
    q = queue(12.0, [(1, L, 8.0, "white car")], departed=(1,))
    assert q.your_turn
    answer = local.can_i_go(snap(q))
    assert answer.text == "It's your turn now. Check the intersection before you go." and not answer.watches
    assert_never_bare_go(answer.text)


def test_four_way_deadlock_has_no_clear_right_of_way() -> None:
    q = queue(12.0, [(1, L, 12.1, "car"), (2, A, 12.2, "car"), (3, R, 12.3, "car")])
    assert q.deadlock
    for handler in (local.who_first, local.can_i_go, local.position):
        text = handler(snap(q)).text
        assert "nobody has a clear right of way" in text
        assert_never_bare_go(text)


def test_position() -> None:
    q = queue(12.0, [(1, L, 8.0, "white car"), (2, A, 10.0, "gray truck")])
    assert local.position(snap(q)).text == "You're third. 2 cars go before you."
    assert local.position(snap(queue(12.0, [(1, A, 9.0, "car")]))).text == "You're second. One car goes before you."


def test_caveats_say_what_cannot_be_seen() -> None:
    q = queue(12.0, [(1, R, 9.0, "white car")])
    s = snap(q, caveats=("narrow_view", "clipped:right"), people_ahead=1)
    text = local.who_first(s).text
    assert "The car on your right is partly out of view." in text
    assert "I can't see the sides of the intersection well." in text
    assert "Watch for the pedestrian ahead." in text


def test_not_at_a_stop() -> None:
    assert local.can_i_go(snap(ego_stopped=False)).text == local.MOVING_REFUSAL
    assert local.who_first(snap(ego_stopped=False)).text.startswith("We're moving.")
    lead = VehicleView(5, "car", "silver", None, True, True, 3.0, (400.0, 300.0, 800.0, 600.0), False)
    behind = local.can_i_go(snap(lead_vehicle=lead, stop_sign_last_seen=9.0)).text
    assert behind == "You're behind the silver car. Wait for it to go, then pull up to the stop line."
    assert "can't tell whose turn" in local.who_first(snap()).text


# --------------------------------------------------------------------------------------------- lights


def test_can_i_go_at_a_red_light_arms_the_light_watch() -> None:
    answer = local.can_i_go(snap(light=light(LightColor.RED)))
    assert answer.text == "Not yet, the light's red. I'll tell you when it turns green."
    assert answer.watches == (Watch.LIGHT,)


def test_can_i_go_at_a_green_light_says_check() -> None:
    text = local.can_i_go(snap(light=light(LightColor.GREEN))).text
    assert text == "The light's green. Check the intersection before you go."
    assert_never_bare_go(text)


def test_who_first_at_a_light_says_the_light_decides() -> None:
    text = local.who_first(snap(light=light(LightColor.RED))).text
    assert text == "There's no stop sign here; the light decides, and it's red."


def test_unsure_light_is_not_used_for_can_i_go() -> None:
    assert "can't tell whose turn" in local.can_i_go(snap(light=light(LightColor.GREEN, confident=False))).text


@pytest.mark.parametrize(
    ("reading", "expected", "watches"),
    [
        (light(LightColor.RED), "Okay, I'll tell you when it turns green.", (Watch.LIGHT,)),
        (light(LightColor.GREEN), "It's already green. Check the intersection before you go.", ()),
        (light(LightColor.UNKNOWN, False), "I can't read the light clearly right now", (Watch.LIGHT,)),
    ],
)
def test_light_watch(reading: LightReading, expected: str, watches: tuple[Watch, ...]) -> None:
    answer = local.light_watch(snap(light=reading))
    assert answer.text.startswith(expected) and answer.watches == watches


def test_go_watch_picks_the_queue_or_the_light() -> None:
    assert local.go_watch(snap(queue(12.0, [(1, A, 9.0, "car")]))).watches == (Watch.TURN,)
    assert local.go_watch(snap(light=light(LightColor.RED))).watches == (Watch.LIGHT,)
    assert set(local.go_watch(snap(ego_stopped=False)).watches) == {Watch.TURN, Watch.LIGHT}


# ------------------------------------------------------------------------------------------- refusals


def test_gap_questions_are_refused_while_moving() -> None:
    assert local.gap_check(snap(ego_stopped=False), "can i merge now") == local.LocalAnswer(local.GAP_REFUSAL)


def test_gap_questions_when_stopped_go_to_the_model_except_is_it_safe_to_go_at_a_stop() -> None:
    assert local.gap_check(snap(), "is the left lane clear") is None
    q = queue(12.0, [(2, A, 10.0, "gray truck")])
    answer = local.gap_check(snap(q), "is it safe to go")
    assert answer is not None and answer.text.startswith("Not yet")


# --------------------------------------------------------------------------------------- alert engine


def engine(**cfg: float) -> tuple[AlertEngine, list[Utterance]]:
    said: list[Utterance] = []
    return AlertEngine(AlertsConfig(**cfg), said.append), said


def test_your_turn_alert_is_spoken_once() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("your_turn", 30.0))
    alerts.on_world_event(WorldEvent("your_turn", 31.0))  # a flicker: within the cooldown
    assert [u.text for u in said] == [YOUR_TURN]
    assert said[0].kind == "alert" and said[0].priority is Priority.INFO and said[0].clip == "your_turn"
    alerts.on_world_event(WorldEvent("your_turn", 60.0))  # a new intersection later: spoken again
    assert len(said) == 2


def test_quiet_mode_keeps_only_requested_alerts() -> None:
    alerts, said = engine()
    alerts.set_quiet(True)
    alerts.on_world_event(WorldEvent("your_turn", 30.0))
    assert said == []
    alerts.arm(Watch.TURN, 40.0)  # "can I go?" → "I'll tell you"
    alerts.on_world_event(WorldEvent("your_turn", 50.0))
    assert [u.priority for u in said] == [Priority.REQUESTED]


def test_light_green_only_when_asked_and_only_once() -> None:
    alerts, said = engine()
    alerts.on_world_event(WorldEvent("light_green", 30.0))
    assert said == []  # nobody asked
    alerts.arm(Watch.LIGHT, 40.0)
    alerts.on_world_event(WorldEvent("light_green", 50.0))
    alerts.on_world_event(WorldEvent("light_green", 80.0))  # the watch was used up
    assert [u.text for u in said] == [LIGHT_GREEN] and said[0].priority is Priority.REQUESTED
    assert alerts.armed() == set()


def test_watches_expire() -> None:
    alerts, said = engine(watch_timeout_s=60.0)
    alerts.arm(Watch.LIGHT, 0.0)
    alerts.on_world_event(WorldEvent("light_green", 61.0))
    assert said == []


def test_driving_off_drops_requests_about_that_place_only() -> None:
    alerts, _ = engine()
    alerts.arm(Watch.LIGHT, 10.0)  # asked while waiting
    alerts.on_world_event(WorldEvent("drove_off", 35.0, {"moving_since": 30.0}))
    assert alerts.armed() == set()
    alerts.arm(Watch.LIGHT, 33.0)  # asked while driving toward the next light
    alerts.on_world_event(WorldEvent("drove_off", 40.0, {"moving_since": 30.0}))
    assert alerts.armed() == {Watch.LIGHT}


def test_leaving_the_intersection_drops_the_turn_watch() -> None:
    alerts, _ = engine()
    alerts.arm(Watch.TURN, 10.0)
    alerts.arm(Watch.LIGHT, 10.0)
    alerts.on_world_event(WorldEvent("left_intersection", 20.0))
    assert alerts.armed() == {Watch.LIGHT}


def test_seeking_back_drops_watches() -> None:
    alerts, said = engine()
    alerts.arm(Watch.LIGHT, 50.0)
    alerts.on_world_event(WorldEvent("light_green", 10.0))  # replay jumped back before the request
    assert said == [] and alerts.armed() == set()


# ------------------------------------------------------------------------------------------ assistant


class FakeHandler:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.routes: list[Route] = []

    def answer(self, route: Route, snapshot: WorldSnapshot, cancel: object = None) -> str:
        self.routes.append(route)
        return self.reply


def assistant(s: WorldSnapshot | None, **kw: object) -> tuple[Assistant, TextSpeaker]:
    speaker = TextSpeaker()
    return Assistant(load_config(), lambda: s, speaker, **kw), speaker  # type: ignore[arg-type]


def test_right_of_way_is_answered_locally_and_fast() -> None:
    a, speaker = assistant(snap(queue(12.0, [(1, L, 8.0, "white car")])))
    reply = a.ask("Hey DrivePilot, who got here first?")
    assert reply.source == "local" and reply.text == "The white car on your left. You're second."
    assert speaker.texts() == [reply.text] and speaker.said[0].priority is Priority.ANSWER
    assert reply.elapsed_ms < 20.0


def test_can_i_go_then_the_turn_alert_then_repeat() -> None:
    a, speaker = assistant(snap(queue(12.0, [(2, A, 10.0, "gray truck")])))
    a.ask("Can I go?")
    a.on_world_event(WorldEvent("your_turn", 20.0))
    assert speaker.said[-1].text == YOUR_TURN and speaker.said[-1].priority is Priority.REQUESTED
    a.ask("Repeat that.")
    # "repeat that" stops an answer playing, but an alert playing is left alone
    assert speaker.texts()[-1] == YOUR_TURN and speaker.answer_stops == 1 and speaker.stops == 0


def test_quiet_and_normal_mode() -> None:
    a, speaker = assistant(snap())
    assert a.ask("Quiet mode").text == QUIET_ON and a.alerts.quiet
    a.on_world_event(WorldEvent("your_turn", 30.0))
    assert speaker.texts() == [QUIET_ON]  # the unrequested alert stayed silent
    a.ask("normal mode")
    a.on_world_event(WorldEvent("your_turn", 60.0))
    assert speaker.texts()[-1] == YOUR_TURN


def test_stop_and_never_mind() -> None:
    a, speaker = assistant(snap(light=light(LightColor.RED)))
    a.ask("Tell me when it's green")
    assert a.alerts.armed() == {Watch.LIGHT}
    reply = a.ask("Stop.")
    assert reply.text is None and speaker.stops == 1
    assert a.ask("Never mind").text == "Okay, cancelled." and a.alerts.armed() == set()
    assert a.ask("Never mind").text == "Okay."


def test_tell_me_when_its_green_end_to_end() -> None:
    a, speaker = assistant(snap(light=light(LightColor.RED)))
    assert a.ask("Tell me when the light turns green").text == "Okay, I'll tell you when it turns green."
    a.on_world_event(WorldEvent("light_green", 30.0))
    assert speaker.texts()[-1] == LIGHT_GREEN


def test_scene_and_navigation_questions_go_to_their_handlers() -> None:
    model, nav = FakeHandler("That sign says no parking."), FakeHandler("The next one, in 1.2 miles.")
    a, _ = assistant(snap(), model=model, nav=nav)
    reply = a.ask("Hey DrivePilot, what did that sign say?")
    assert reply.source == "model" and reply.text == "That sign says no parking."
    assert model.routes[0].question == "what did that sign say?"
    assert a.ask("Is my exit this one or the next?").text == "The next one, in 1.2 miles."
    assert a.ask("I'm going to Redmond Town Center").source == "nav" and nav.routes[-1].place == "redmond town center"
    assert a.ask("Which lane for 520 east?").source == "nav"


def test_missing_handlers_say_so() -> None:
    a, _ = assistant(snap())
    assert a.ask("What's the speed limit?").text == NO_MODEL
    assert a.ask("Did I miss my turn?").text == NO_NAV


def test_gap_check_refused_while_moving_and_described_when_stopped() -> None:
    model = FakeHandler("A car is approaching in the left lane.")
    a, _ = assistant(snap(ego_stopped=False), model=model)
    assert a.ask("Can I merge now?").text == local.GAP_REFUSAL and not model.routes
    b, _ = assistant(snap(), model=model)
    assert b.ask("Is the left lane clear?").source == "model"


def test_before_the_first_frame() -> None:
    a, _ = assistant(None)
    assert a.ask("Can I go?").text == STARTING
    assert a.ask("quiet mode").text == QUIET_ON  # controls work anyway


def test_repeat_with_nothing_said() -> None:
    a, _ = assistant(snap())
    assert a.ask("repeat that").text == "I haven't said anything yet."


def test_segment_reset_clears_alert_state() -> None:
    a, _ = assistant(snap())
    a.alerts.arm(Watch.LIGHT, 10.0)
    a.on_world_event(WorldEvent("segment_reset", 0.0))
    assert a.alerts.armed() == set()
