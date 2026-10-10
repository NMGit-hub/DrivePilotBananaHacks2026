"""How navigation is said out loud: distances in US units, lanes, maneuvers.

Distances: under ~0.2 mile in feet (rounded to 50/100 ft), then "a quarter mile", "half a mile", "three
quarters of a mile", "a mile", then "1.4 miles" / "12 miles". Lanes are described the way a passenger would:
"the two left lanes", "the right lane", "the middle lane", "lanes 2 and 3 from the left". Lane math: given the
valid lanes (left to right) and our lane counted from the left, "move one lane left" / "stay in this lane".
"""

from __future__ import annotations

from collections.abc import Sequence

from nav.route import Lane, Step

FEET_PER_M = 3.28084
M_PER_MILE = 1609.344
_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_ORDINAL = {1: "st", 2: "nd", 3: "rd"}


def spoken_distance(metres: float) -> str:
    """ "500 feet", "a quarter mile", "1.4 miles"."""
    miles = metres / M_PER_MILE
    if miles < 0.19:
        feet = metres * FEET_PER_M
        step = 50 if feet < 500 else 100
        return f"{max(step, round(feet / step) * step)} feet"
    if miles < 0.375:
        return "a quarter mile"
    if miles < 0.625:
        return "half a mile"
    if miles < 0.875:
        return "three quarters of a mile"
    if miles < 1.15:
        return "a mile"
    if miles < 10:
        return f"{miles:.1f} miles"
    return f"{round(miles)} miles"


def in_distance(metres: float) -> str:
    """ "in 500 feet" / "in a quarter mile" / "now" (very close)."""
    return "now" if metres < 30 else f"in {spoken_distance(metres)}"


def ordinal(n: int) -> str:
    """1 -> "1st", 12 -> "12th", 22 -> "22nd"."""
    if 10 < n % 100 < 14:
        return f"{n}th"
    return f"{n}{_ORDINAL.get(n % 10, 'th')}"


def describe_lanes(lanes: Sequence[Lane]) -> str | None:
    """The valid lanes as a passenger would say them (None if all or none are valid)."""
    valid = [i for i, lane in enumerate(lanes) if lane.valid]
    n = len(lanes)
    if not valid or len(valid) == n:
        return None
    count = len(valid)
    contiguous = valid == list(range(valid[0], valid[-1] + 1))
    word = _NUMBERS.get(count, str(count))
    if contiguous and valid[0] == 0:
        return "the left lane" if count == 1 else f"the {word} left lanes"
    if contiguous and valid[-1] == n - 1:
        return "the right lane" if count == 1 else f"the {word} right lanes"
    if contiguous and count == 1 and n % 2 == 1 and valid[0] == n // 2:
        return "the middle lane"
    if contiguous and count == 2 and n == 4 and valid == [1, 2]:
        return "the two middle lanes"
    positions = [str(i + 1) for i in valid]
    joined = positions[0] if count == 1 else ", ".join(positions[:-1]) + f" and {positions[-1]}"
    return f"lane {joined} from the left" if count == 1 else f"lanes {joined} from the left"


def lane_change(lanes: Sequence[Lane], our_lane: int) -> str:
    """What to do from our lane (1 = leftmost): "stay in this lane" / "move one lane left"."""
    valid = [i + 1 for i, lane in enumerate(lanes) if lane.valid]
    if not valid or our_lane in valid:
        return "stay in this lane"
    target = min(valid, key=lambda v: abs(v - our_lane))
    moves = abs(target - our_lane)
    side = "left" if target < our_lane else "right"
    return f"move {_NUMBERS.get(moves, str(moves))} lane{'s' if moves > 1 else ''} {side}"


def road_name(step: Step) -> str:
    """The road a maneuver leads onto, as said: "SR 520 East", "148th Avenue Northeast"."""
    if step.ref and step.name and step.ref not in step.name:
        return f"{step.name} ({step.ref})" if step.maneuver not in ("off ramp", "on ramp", "fork") else step.ref
    return step.name or step.ref or ""


def maneuver_phrase(step: Step) -> str:
    """A short spoken instruction for a step (Mapbox's text, with roundabout exits made explicit)."""
    if step.is_roundabout and step.roundabout_exit:
        onto = f" onto {step.name}" if step.name else ""
        return f"at the roundabout, take the {ordinal(step.roundabout_exit)} exit{onto}"
    if step.maneuver == "arrive":  # Mapbox says "You have arrived..." for a destination still ahead
        side = f" on the {step.modifier}" if step.modifier in ("left", "right") else ""
        return f"you'll reach your destination{side}"
    text = step.instruction.rstrip(".")
    return text[:1].lower() + text[1:] if text else "continue"
