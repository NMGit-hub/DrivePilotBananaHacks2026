"""The US sign taxonomy of the fine-tuned sign detector, and the mapping from dataset labels to it.

Shared by the training scripts (`scripts/signs/`) and the runtime detector (`perception/signs.py`), so a class
index means the same thing everywhere. Kinds follow the MUTCD: R2-1 "SPEED LIMIT", S5-1 "SCHOOL SPEED LIMIT",
R1-1 stop, R1-2 yield, R5-1 do not enter, R5-1a wrong way, R10-11 no turn on red, R6-1/R6-2 one way, R3-2/R3-1/R3-4
no left/right/U turn, W11-2 pedestrian crossing, S1-1 school zone.
"""

from __future__ import annotations

import re

# Detector class order (index = YOLO class id). Never reorder: trained weights depend on it.
KINDS: tuple[str, ...] = (
    "speed_limit",
    "school_speed_limit",
    "stop",
    "yield",
    "do_not_enter",
    "wrong_way",
    "no_turn_on_red",
    "one_way",
    "no_left_turn",
    "no_right_turn",
    "no_u_turn",
    "pedestrian_crossing",
    "school_zone",
)
KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(KINDS)}

# Kinds whose sign carries a number (mph) read by the second-stage value classifier.
VALUE_KINDS: frozenset[str] = frozenset({"speed_limit", "school_speed_limit"})

# Posted US limits are multiples of 5; 5-85 mph covers parking lots to Texas toll roads. The value classifier
# has one class per value (plus a reject class), so a 75 is never forced into "70".
SPEED_VALUES: tuple[int, ...] = tuple(range(5, 90, 5))
VALUE_REJECT = "none"  # classifier class for crops that are not a readable speed-limit sign

_NUMBER = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


def parse_speed_value(text: str | int | None) -> int | None:
    """The mph value in a label or reading ("35", "mph35", "speedLimit35", "SPEED LIMIT 35 MPH", 35).

    Returns None unless exactly one plausible posted limit (a multiple of 5 in SPEED_VALUES) is present: a reading
    with two numbers ("25 35") or an implausible one ("37", "120") is not trusted.
    """
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, int):
        return text if text in SPEED_VALUES else None
    numbers = [int(m) for m in _NUMBER.findall(str(text))]
    if len(numbers) != 1:
        return None
    return numbers[0] if numbers[0] in SPEED_VALUES else None


def _norm(label: str) -> str:
    """Lowercase alphanumerics only: "speedLimit35", "Speed Limit 35" and "speed-limit-35" all match."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


# Normalized label patterns → kind. Order matters: the first match wins (school speed limit before speed limit,
# "no turn on red" before the turn restrictions). Checked against LISA, Roboflow exports and our crop labels.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^schoolspeedlimit\d*$"), "school_speed_limit"),
    (re.compile(r"^(?:speedlimit|maximumspeedlimit|mph)(?:\d+|urdbl)?$"), "speed_limit"),  # LISA: Urdbl = unreadable
    (re.compile(r"^(?:stop|stopsign|regulatorystop)$"), "stop"),
    (re.compile(r"^(?:yield|yieldsign)$"), "yield"),
    (re.compile(r"^(?:donotenter|noentry)$"), "do_not_enter"),
    (re.compile(r"^wrongway$"), "wrong_way"),
    (re.compile(r"^noturnonred$"), "no_turn_on_red"),
    (re.compile(r"^(?:oneway|onewayleft|onewayright)$"), "one_way"),
    (re.compile(r"^(?:noleftturn|noturnleft|donotturnl)$"), "no_left_turn"),
    (re.compile(r"^(?:norightturn|noturnright|donotturnr)$"), "no_right_turn"),
    (re.compile(r"^(?:nouturn|donotuturn)$"), "no_u_turn"),
    (re.compile(r"^(?:pedestriancrossing|pedestriancrosswalk|pedcrossing)$"), "pedestrian_crossing"),
    (re.compile(r"^(?:school|schoolzone|schoolcrossing)$"), "school_zone"),
)


def kind_from_label(label: str) -> tuple[str, int | None] | None:
    """Map a dataset label to (kind, mph value or None); None for signs outside the taxonomy.

    A speed-limit label without a parsable value maps to ("speed_limit", None): the box still trains the
    detector, only the value classifier skips it. Labels for look-alikes that must not become speed limits
    ("rampSpeedAdvisory35", "truckSpeedLimit55", "zoneAhead25", "minimumSpeed40") return None.
    """
    norm = _norm(label)
    for pattern, kind in _RULES:
        if pattern.match(norm):
            value = parse_speed_value(norm) if kind in VALUE_KINDS else None
            return kind, value
    return None
