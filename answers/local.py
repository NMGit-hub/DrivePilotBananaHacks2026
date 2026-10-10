"""Instant local answers from the world state: right of way, "tell me when it's green", refusals.

Phrasing follows PLAN.md ("The white SUV on the left, then the pickup across from you. You're third.") and its
safety rules:
- never a bare "go": anything that lets the driver go says to check first, and never starts with "Yes" (a
  "yes" to "is it safe to pull out?" would vouch for traffic we can't judge, e.g. at a 2-way stop where cross
  traffic never stops);
- say what can't be seen (cars out of view, a narrow view, cars we lost track of, pedestrians);
- refuse split-second and stop-or-go judgments while moving, or when we're not sure we are stopped.
Pure functions of a `WorldSnapshot`: no I/O, microseconds per call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from state.stop_queue import EGO_KEY, RIGHT_OF, Approach, QueueEntry, QueueSnapshot
from state.traffic_light import LightColor
from state.world import WorldSnapshot

ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
WHERE = {Approach.LEFT: "on your left", Approach.AHEAD: "across from you", Approach.RIGHT: "on your right"}
CHECK = "check the intersection before you go"
MOVING_REFUSAL = "I can't judge that safely while we're moving. Check around you."
UNSURE_REFUSAL = "I'm not sure we're fully stopped, so I can't judge that. Check around you."
GAP_REFUSAL = "I can't judge that safely while we're moving. Check your mirrors and blind spot."
SCHOOL_BUS_RULE = (
    "No. Never pass a school bus that's stopped with its red lights flashing or its stop arm out. "
    "Stop and wait until it moves on."
)
GAP_WATCH_MOVING = "I can't judge gaps in traffic safely. Check your mirrors and blind spot."
GAP_WATCH_STOPPED = "I can't judge gaps in moving traffic. Check both ways yourself before you go."
PEDESTRIAN_YIELD = "Yes, let them cross first, then check again before you go."
PEDESTRIANS = re.compile(
    r"\b(?:pedestrians?|person|people|walkers?|kids?|child|children|cyclists?|bikes?|bikers?|"
    r"someone crossing|him|her|them|that guy|that lady)\b"
)
GO_WORDS = re.compile(r"\b(?:to go|go now|to cross|to turn|to pull out|to proceed|to drive|if i go|for me to go)\b")


class Watch(StrEnum):
    """Something the user asked to be told about later (armed in the alert engine)."""

    TURN = "turn"  # "you're up" at a 4-way stop
    LIGHT = "light"  # the light turned green


@dataclass(frozen=True, slots=True)
class LocalAnswer:
    """A spoken answer, plus watches to arm."""

    text: str
    watches: tuple[Watch, ...] = ()


def ordinal(n: int) -> str:
    """1 -> "first"."""
    return ORDINALS[n - 1] if 1 <= n <= len(ORDINALS) else f"number {n}"


def describe(entry: QueueEntry) -> str:
    """ "the white car on your left"."""
    label = entry.label if entry.label and entry.label != "car" else "car"
    return f"the {label} {WHERE.get(entry.approach, '')}".strip()


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def caveats(snapshot: WorldSnapshot) -> list[str]:
    """What the answer should admit it can't see."""
    out = []
    clipped = [c.split(":", 1)[1] for c in snapshot.caveats if c.startswith("clipped:")]
    for side in dict.fromkeys(clipped):
        where = WHERE.get(Approach(side), side)
        out.append(f"The car {where} is partly out of view.")
    if "narrow_view" in snapshot.caveats:
        out.append("I can't see the sides of the intersection well.")
    if "ego_motion_unknown" in snapshot.caveats:
        out.append("I'm not sure we're fully stopped.")
    for entry in snapshot.queue.dropped:
        if entry.drop_reason == "probably parked":
            out.append(f"{_sentence(describe(entry))} hasn't moved; it may be parked.")
        else:
            out.append(f"I lost sight of {describe(entry)}.")
    if snapshot.people_ahead:
        out.append(
            "Watch for the pedestrian ahead." if snapshot.people_ahead == 1 else "Watch for the pedestrians ahead."
        )
    return out


def _with_caveats(text: str, snapshot: WorldSnapshot) -> str:
    return " ".join([text, *caveats(snapshot)])


def ahead_of_us(queue: QueueSnapshot) -> list[QueueEntry]:
    """The waiting cars that go before us, in order."""
    out = []
    for entry in queue.order:
        if entry.is_ego:
            break
        out.append(entry)
    return out


def _tie_note(queue: QueueSnapshot) -> str | None:
    """Explain a tie that involves us (PLAN.md: "You and the car on your right stopped together...")."""
    group = next((g for g in queue.tie_groups if EGO_KEY in g), None)
    if group is None or queue.deadlock:
        return None
    others = [e for e in queue.order if e.key in group and not e.is_ego]
    if len(others) != 1:
        return None  # 3-way ties: the order sentence says it all
    other = others[0]
    if other.approach is RIGHT_OF[Approach.US]:
        return f"You and {describe(other)} stopped together. The car on the right goes first."
    if RIGHT_OF[other.approach] is Approach.US:
        return f"You and {describe(other)} stopped together. You're on their right, so you go first."
    return None


def _list(entries: list[QueueEntry]) -> str:
    names = [describe(e) + ("" if e.arrival_known else ", already waiting when you got here") for e in entries]
    return ", then ".join(names)


def _motion_refusal(snapshot: WorldSnapshot) -> str | None:
    """Stop-or-go questions need us to be stopped, and to be sure of it."""
    if not snapshot.ego_stopped:
        return MOVING_REFUSAL
    if not snapshot.ego_motion_confident:
        return UNSURE_REFUSAL
    return None


def _first_text(queue: QueueSnapshot) -> str:
    """Nobody ahead of us: but say so carefully if a car ahead was only lost from view."""
    if queue.dropped:
        return "Nobody I can see is ahead of you."
    if queue.departed:
        return "It's your turn now."
    return "Nobody else is waiting at the stop, so it's your turn."


def _not_yet(queue: QueueSnapshot, prefix: str = "Not yet") -> LocalAnswer:
    ahead = ahead_of_us(queue)
    if len(ahead) == 1:
        text = f"{prefix}, {describe(ahead[0])} is next. I'll tell you."
    else:
        text = f"{prefix}, {len(ahead)} cars go before you: {_list(ahead)}. I'll tell you."
    tie = _tie_note(queue)
    return LocalAnswer(f"{tie} {text}" if tie else text, (Watch.TURN,))


# -------------------------------------------------------------------------------------- right of way


def who_first(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Who got here first?" """
    queue = snapshot.queue
    if not queue.active:
        return LocalAnswer(_not_at_a_stop(snapshot))
    if queue.deadlock:
        return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
    ahead = ahead_of_us(queue)
    behind = [e for e in queue.order if not e.is_ego][len(ahead) :]
    if not ahead:
        if queue.dropped:
            text = "Nobody I can see is ahead of you."
        else:
            text = "You got here first." if behind else "Nobody else is waiting. You're first."
        if behind:
            text += f" Then {_list(behind)}."
        tie = _tie_note(queue)
        if tie:
            text = f"{tie} {text}"
        return LocalAnswer(_with_caveats(f"{text} Check before you go.", snapshot))
    text = f"{_sentence(_list(ahead))}. You're {ordinal(len(ahead) + 1)}."
    tie = _tie_note(queue)
    if tie:
        text = f"{tie} {text}"
    return LocalAnswer(_with_caveats(text, snapshot))


def can_i_go(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Can I go?" — at a stop sign from the queue, at a light from the light state; refused while moving."""
    refusal = _motion_refusal(snapshot)
    if refusal is not None:
        return LocalAnswer(refusal)
    queue = snapshot.queue
    if queue.active:
        if queue.deadlock:
            return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
        if queue.your_turn:
            return LocalAnswer(_with_caveats(f"{_first_text(queue)} {_sentence(CHECK)}.", snapshot))
        answer = _not_yet(queue)
        return LocalAnswer(_with_caveats(answer.text, snapshot), answer.watches)
    light = _light_answer(snapshot)
    if light is not None:
        return light
    return LocalAnswer(_not_at_a_stop(snapshot))


def should_wait(snapshot: WorldSnapshot, question: str) -> LocalAnswer | None:
    """ "Should I wait?" — answered as asked ("yes, wait" is safe to say; "no, go" is not). None: the vision
    model describes the thing to wait for ("should I wait for that truck?") once we are stopped."""
    if PEDESTRIANS.search(question):
        return LocalAnswer(PEDESTRIAN_YIELD)
    refusal = _motion_refusal(snapshot)
    if re.search(r"\bwait for\b", question):
        return LocalAnswer(refusal) if refusal is not None else None
    if refusal is not None:
        return LocalAnswer(refusal)
    queue = snapshot.queue
    if queue.active:
        if queue.deadlock:
            return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
        if queue.your_turn:
            lead = "Nobody I can see is ahead of you" if queue.dropped else "By arrival order it's your turn now"
            return LocalAnswer(_with_caveats(f"{lead}. {_sentence(CHECK)}.", snapshot))
        answer = _not_yet(queue, prefix="Yes, wait")
        return LocalAnswer(_with_caveats(answer.text, snapshot), answer.watches)
    light = snapshot.light
    if light.confident and light.color in (LightColor.RED, LightColor.YELLOW):
        return LocalAnswer(f"Yes, the light's {light.color.value}. I'll tell you when it turns green.", (Watch.LIGHT,))
    if light.confident and light.color is LightColor.GREEN:
        return _green(snapshot)
    return LocalAnswer(_not_at_a_stop(snapshot))


def position(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "How many cars are ahead of me?" """
    queue = snapshot.queue
    if not queue.active:
        return LocalAnswer(_not_at_a_stop(snapshot))
    if queue.deadlock:
        return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
    ahead = ahead_of_us(queue)
    if not ahead:
        first = "Nobody I can see is ahead of you." if queue.dropped else "You're first."
        return LocalAnswer(_with_caveats(f"{first} {_sentence(CHECK)}.", snapshot))
    cars = "One car goes" if len(ahead) == 1 else f"{len(ahead)} cars go"
    return LocalAnswer(_with_caveats(f"You're {ordinal(len(ahead) + 1)}. {cars} before you.", snapshot))


def _deadlock_text() -> str:
    return (
        "Everyone stopped at the same time, so nobody has a clear right of way. "
        "Make eye contact and take turns; check before you go."
    )


def _not_at_a_stop(snapshot: WorldSnapshot) -> str:
    if not snapshot.ego_stopped:
        return "We're moving. I track whose turn it is once we stop at a stop sign."
    if snapshot.lead_vehicle is not None:
        lead = snapshot.lead_vehicle.label or "car"
        if snapshot.stop_sign_last_seen is not None:
            return f"You're behind the {lead}. Wait for it to go, then pull up to the stop line."
        return f"You're stopped behind the {lead}."
    if snapshot.behind_lead and snapshot.stop_sign_last_seen is not None:
        return "Pull up to the stop line first; I'll keep track of the order from there."
    if snapshot.light.confident and snapshot.light.color is not LightColor.UNKNOWN:
        return f"There's no stop sign here; the light decides, and it's {snapshot.light.color.value}."
    return "I don't see a stop sign or a traffic light here, so I can't tell whose turn it is. Check around you."


# ------------------------------------------------------------------------------------------- lights


def _green(snapshot: WorldSnapshot) -> LocalAnswer:
    if snapshot.lead_vehicle is not None and snapshot.ego_stopped:
        return LocalAnswer(f"The light's green; wait for the {snapshot.lead_vehicle.label or 'car'} ahead to move.")
    return LocalAnswer(_with_caveats(f"The light's green. {_sentence(CHECK)}.", snapshot))


def _light_answer(snapshot: WorldSnapshot) -> LocalAnswer | None:
    """ "Can I go?" at a traffic light, while stopped (None if no light is read confidently)."""
    light = snapshot.light
    if not light.confident:
        return None
    if light.color in (LightColor.RED, LightColor.YELLOW):
        text = f"Not yet, the light's {light.color.value}. I'll tell you when it turns green."
        return LocalAnswer(text, (Watch.LIGHT,))
    if light.color is LightColor.GREEN:
        return _green(snapshot)
    return None


def light_watch(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Tell me when it's green." """
    light = snapshot.light
    if light.confident and light.color is LightColor.GREEN:
        return LocalAnswer(f"It's already green. {_sentence(CHECK)}.")
    if light.confident and light.color in (LightColor.RED, LightColor.YELLOW):
        return LocalAnswer("Okay, I'll tell you when it turns green.", (Watch.LIGHT,))
    return LocalAnswer(
        "I can't read the light clearly right now, but I'll watch it and tell you when it turns green.", (Watch.LIGHT,)
    )


def go_watch(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Tell me when I can go": the 4-way-stop turn, or the light."""
    queue = snapshot.queue
    if queue.active:
        if queue.your_turn:
            return LocalAnswer(_with_caveats(f"{_first_text(queue)} {_sentence(CHECK)}.", snapshot))
        return LocalAnswer("Okay, I'll tell you when it's your turn.", (Watch.TURN,))
    light = snapshot.light
    if light.confident and light.color is LightColor.GREEN:
        return LocalAnswer(f"The light's already green. {_sentence(CHECK)}.")
    if light.confident and light.color in (LightColor.RED, LightColor.YELLOW):
        return LocalAnswer("Okay, I'll tell you when the light turns green.", (Watch.LIGHT,))
    return LocalAnswer(
        "Okay. I'll tell you when it's your turn at the stop, or when the light turns green.", (Watch.TURN, Watch.LIGHT)
    )


# ------------------------------------------------------------------------------------- gap judgments


def gap_check(snapshot: WorldSnapshot, question: str) -> LocalAnswer | None:
    """Split-second judgments: refused unless we are surely stopped. Stopped at a stop or light, "is it safe to
    go/turn?" gets the right-of-way facts plus "I can't judge moving traffic" (never a "yes"). Otherwise None:
    the vision model describes what it sees (it never says "go")."""
    if not snapshot.ego_stopped:
        return LocalAnswer(GAP_REFUSAL)
    if not snapshot.ego_motion_confident:
        return LocalAnswer(UNSURE_REFUSAL)
    queue, light = snapshot.queue, snapshot.light
    if GO_WORDS.search(question) and (queue.active or (light.confident and light.color is not LightColor.UNKNOWN)):
        if queue.active:
            if queue.deadlock:
                return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
            if queue.your_turn:
                lead = "Nobody I can see is ahead of you" if queue.dropped else "By arrival order it's your turn"
                return LocalAnswer(
                    _with_caveats(f"{lead}, but I can't judge moving traffic. Check both ways before you go.", snapshot)
                )
            answer = _not_yet(queue)
            return LocalAnswer(_with_caveats(answer.text, snapshot), answer.watches)
        if light.color in (LightColor.RED, LightColor.YELLOW):
            return LocalAnswer(
                f"Not yet, the light's {light.color.value}. I'll tell you when it turns green.", (Watch.LIGHT,)
            )
        return LocalAnswer(
            _with_caveats("The light's green, but I can't judge oncoming traffic. Check before you go.", snapshot)
        )
    return None


def gap_watch(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Tell me when it's safe to merge / when I can turn left": we can't watch for gaps; offer the light."""
    light = snapshot.light
    stopped = snapshot.ego_stopped and snapshot.ego_motion_confident
    if stopped and light.confident and light.color in (LightColor.RED, LightColor.YELLOW):
        return LocalAnswer(
            "I'll tell you when the light turns green, but you'll need to judge the traffic yourself.", (Watch.LIGHT,)
        )
    return LocalAnswer(GAP_WATCH_STOPPED if stopped else GAP_WATCH_MOVING)
