"""Instant local answers from the world state: right of way, "tell me when it's green", refusals.

Phrasing follows PLAN.md ("The white SUV on the left, then the pickup across from you. You're third.") and its
safety rules: never a bare "go" (always "check ... before you go"), say what can't be seen, and refuse
split-second judgments while moving. Pure functions of a `WorldSnapshot`: no I/O, microseconds per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from state.stop_queue import EGO_KEY, RIGHT_OF, Approach, QueueEntry, QueueSnapshot
from state.traffic_light import LightColor
from state.world import WorldSnapshot

ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
WHERE = {Approach.LEFT: "on your left", Approach.AHEAD: "across from you", Approach.RIGHT: "on your right"}
CHECK = "check the intersection before you go"


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
    if snapshot.people_ahead:
        out.append(
            "Watch for the pedestrian ahead." if snapshot.people_ahead == 1 else "Watch for the pedestrians ahead."
        )
    return out


def _with_caveats(text: str, snapshot: WorldSnapshot) -> str:
    return " ".join([text, *caveats(snapshot)])


def _ahead_of_us(queue: QueueSnapshot) -> list[QueueEntry]:
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


# -------------------------------------------------------------------------------------- right of way


def who_first(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Who got here first?" """
    queue = snapshot.queue
    if not queue.active:
        return LocalAnswer(_not_at_a_stop(snapshot))
    if queue.deadlock:
        return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
    ahead = _ahead_of_us(queue)
    behind = [e for e in queue.order if not e.is_ego][len(ahead) :]
    if not ahead:
        text = "You got here first." if behind else "Nobody else is waiting. You're first."
        if behind:
            text += f" Then {_list(behind)}."
        tie = _tie_note(queue)
        if tie:
            text = f"{tie} {text}"
        return LocalAnswer(_with_caveats(f"{text} Check before you go.", snapshot))
    position = len(ahead) + 1
    text = f"{_sentence(_list(ahead))}. You're {ordinal(position)}."
    tie = _tie_note(queue)
    if tie:
        text = f"{tie} {text}"
    return LocalAnswer(_with_caveats(text, snapshot))


def can_i_go(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "Can I go?" — at a stop sign from the queue, at a light from the light state."""
    queue = snapshot.queue
    if queue.active:
        if queue.deadlock:
            return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
        if queue.your_turn:
            went = len(queue.departed)
            lead = "Yes, it's your turn" if went else "Yes, nobody else is waiting"
            return LocalAnswer(_with_caveats(f"{lead}. {_sentence(CHECK)}.", snapshot))
        ahead = _ahead_of_us(queue)
        if len(ahead) == 1:
            text = f"Not yet, {describe(ahead[0])} is next. I'll tell you."
        else:
            text = f"Not yet, {len(ahead)} cars go before you: {_list(ahead)}. I'll tell you."
        tie = _tie_note(queue)
        if tie:
            text = f"{tie} {text}"
        return LocalAnswer(_with_caveats(text, snapshot), (Watch.TURN,))
    light = _light_answer(snapshot)
    if light is not None:
        return light
    return LocalAnswer(_not_at_a_stop(snapshot))


def position(snapshot: WorldSnapshot) -> LocalAnswer:
    """ "How many cars are ahead of me?" """
    queue = snapshot.queue
    if not queue.active:
        return LocalAnswer(_not_at_a_stop(snapshot))
    if queue.deadlock:
        return LocalAnswer(_with_caveats(_deadlock_text(), snapshot))
    ahead = _ahead_of_us(queue)
    if not ahead:
        return LocalAnswer(_with_caveats(f"You're first. {_sentence(CHECK)}.", snapshot))
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
    if snapshot.light.confident and snapshot.light.color is not LightColor.UNKNOWN:
        return f"There's no stop sign here; the light decides, and it's {snapshot.light.color.value}."
    return "I don't see a stop sign or a traffic light here, so I can't tell whose turn it is. Check around you."


# ------------------------------------------------------------------------------------------- lights


def _light_answer(snapshot: WorldSnapshot) -> LocalAnswer | None:
    """ "Can I go?" at a traffic light (None if no light is read confidently)."""
    light = snapshot.light
    if not light.confident:
        return None
    if light.color in (LightColor.RED, LightColor.YELLOW):
        color = "red" if light.color is LightColor.RED else "yellow"
        return LocalAnswer(f"Not yet, the light's {color}. I'll tell you when it turns green.", (Watch.LIGHT,))
    if light.color is LightColor.GREEN:
        if snapshot.lead_vehicle is not None and snapshot.ego_stopped:
            return LocalAnswer(f"The light's green; wait for the {snapshot.lead_vehicle.label or 'car'} ahead to move.")
        return LocalAnswer(_with_caveats(f"The light's green. {_sentence(CHECK)}.", snapshot))
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
            return LocalAnswer(_with_caveats(f"It's your turn now. {_sentence(CHECK)}.", snapshot))
        return LocalAnswer("Okay, I'll tell you when it's your turn.", (Watch.TURN,))
    light = snapshot.light
    if light.confident and light.color is LightColor.GREEN:
        return LocalAnswer(f"The light's already green. {_sentence(CHECK)}.")
    if light.confident and light.color in (LightColor.RED, LightColor.YELLOW):
        return LocalAnswer("Okay, I'll tell you when the light turns green.", (Watch.LIGHT,))
    return LocalAnswer(
        "Okay. I'll tell you when it's your turn at the stop, or when the light turns green.", (Watch.TURN, Watch.LIGHT)
    )


# ----------------------------------------------------------------------------------------- refusals


GAP_REFUSAL = "I can't judge that safely while we're moving. Check your mirrors and blind spot."


def gap_check(snapshot: WorldSnapshot, question: str) -> LocalAnswer | None:
    """Split-second judgments: refused while moving. When stopped, None (the vision model describes the scene),
    except "is it safe to go?" at a stop sign or light, which is a right-of-way question."""
    if not snapshot.ego_stopped:
        return LocalAnswer(GAP_REFUSAL)
    if snapshot.queue.active or (snapshot.light.confident and snapshot.light.color is not LightColor.UNKNOWN):
        go_words = ("to go", "go now", "to cross", "to turn", "to pull out")
        if any(word in question for word in go_words):
            return can_i_go(snapshot)
    return None
