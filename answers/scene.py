"""Vision-model answers about the scene (PLAN.md: signs, parking, lights, road awareness, lane counts).

`SceneHandler.answer` builds one request: the latest frame at full resolution plus 1-2 earlier frames (smaller,
or full-size and further back for "what did that sign say?"), close-ups of up to three traffic lights for light
questions, and a short text context: local day and time (time-limited signs), whether we are stopped, what the
tracker sees (cars by side, the right-of-way verdict, the light, pedestrians) and, from M6, speed and the next
navigation step. The model is told to answer in 1-2 spoken sentences, left/right from the driver's seat, to say
what it can't see, and never to vouch for traffic.

V2-M4: the tracked objects on the latest frame are numbered (thin boxes with small tags) so the model can end
its answer with a hidden "[ref N]" naming the object it talks about: the page then pulses that object's box. A
memory question ("what was that sign?") gets the last seconds of remembered frames plus close-ups of the signs
seen (state/memory.py), numbered "Image k"; an answer ending in "[image k]" shows that frame as a thumbnail. Tags
(and any stray mention of a mark number) are stripped before anything is spoken.

The reply streams back sentence by sentence (speech starts on the first). Safety nets on every sentence:
- anything that gives permission ("go ahead", "you're good to go", "you can merge now", a bare "Go.") is
  replaced by "I can't judge moving traffic, so check both ways yourself before you go.";
- for questions about going, turning, merging or waiting, any vouching for traffic ("it's clear", "nothing's
  coming") is replaced the same way, and the answer always ends with that sentence.
Deadlines and "stop" cancel the provider request itself (`guarded_stream`); spoken fallbacks cover failures.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

import cv2
import numpy as np

from answers.local import SCHOOL_BUS_RULE, ahead_of_us, describe
from answers.sentences import sentences
from answers.vision import (
    ImagePart,
    VisionCancelled,
    VisionError,
    VisionModel,
    VisionRequest,
    VisionTimeout,
    guarded_stream,
)
from core.config import AppConfig
from core.logging_setup import get_logger
from core.types import Frame
from perception.detector import HOOD_BOTTOM, Track, is_own_hood
from state.memory import SceneMemory
from state.signs import box_iou
from state.stop_queue import Approach
from state.traffic_light import LightColor
from state.world import WorldSnapshot
from voice.router import Route, RouteKind

log = get_logger(__name__)

SYSTEM_PROMPT = """You are DrivePilot, a voice co-pilot riding in a car. You see the road through a dashcam on the \
windshield, looking forward. The driver asks you questions out loud while driving, and your answer is read aloud.

Rules:
- Answer in one or two short sentences of plain spoken English, about 30 words at most. No lists, markdown, or emojis.
- Left and right are from the driver's seat (image left is the driver's left).
- You cannot judge moving traffic or gaps. Never say or imply that it is safe or clear to go, turn, merge, pull \
out, cross or change lanes, and never say "go", "go ahead", "you're good to go", "you can go" or "you're clear". \
Describe what you see, and when the question is about going, turning or merging, end with "check both ways \
yourself".
- If you can't see something clearly (too far, glare, blur, out of view), say so instead of guessing.
- Read signs exactly. Quote only words you can actually read; never fill in text you can't make out, and never \
name a sign (like "no turn on red") unless you can read those words on it. Electronic message boards are usually \
too small to read: say so. For time-limited rules (parking, turn restrictions, school zones) use the day and \
time given.
- A school bus stopped with its red lights flashing or its STOP arm out is loading or unloading children: \
traffic must stop and wait until it moves on. Never suggest passing or going around it.
- To count lanes, count the lanes in your direction between the solid edge lines, using the dashed lane lines \
and the cars in them; say which one you are in, counted from the left. If the edges aren't visible, say so.
- The context comes from an automatic tracker and can be wrong: if it disagrees with what you see, trust what \
you see. It reads only the main color of the light facing us; it cannot read arrows. Don't guess why we are \
stopped (traffic, a light, a sign) unless you can see it.
- Don't mention images, frames, cameras or the tracker; speak as if you are looking at the road.
- For parking, look at what decides it: markings on the stall itself (a wheelchair symbol or blue paint means \
disabled permit only; EV, reserved, compact), curb paint, hydrants, driveways, crosswalks, bus stops, meters and \
parking signs. Signs about traffic (STOP, DO NOT ENTER, ONE WAY) don't decide parking. Answer about the spot the \
driver asks about. An ordinary stall or curb with none of those restrictions in view is normally fine: say so \
plainly first, then to check any posted signs. Only if you can't see the spot itself, say what you can see and to \
check the posted signs.
- Rules that depend on the place (turning on red, parking, school zones) follow the Location line when there \
is one. In New York City, turning right on red is illegal unless a sign allows it.
- Use US units (mph, feet, miles)."""

MARKS_PROMPT = (
    'Objects in the "Now" image are numbered (thin boxes with small number tags). When your answer is about a '
    "specific object there (a car, truck, bus, person, cyclist, sign or light), you must end your answer with "
    "that object's tag, like [ref 3]: the driver's screen then points at it. Write only the tag: never say the "
    "numbers, and never mention marks, tags or boxes."
)
MARKS_REMINDER = "(If your answer is about one numbered object, end with its tag, e.g. [ref 2].)"
MEMORY_PROMPT = (
    "The driver is asking about something we already passed: look at the earlier images (numbered Image 1, "
    "Image 2, ...; close-ups of signs we passed come first). End your answer with the tag of the image that "
    "shows it best, like [image 2]: the driver's screen then shows that image. Write only the tag: never say "
    "image numbers."
)
MEMORY_REMINDER = "(End with the tag of the image that shows it best, e.g. [image 2].)"
# Hidden tags: "[ref 3]", "(ref: 3)", "[image 2]", "[ref 1, ref 2]", a bare "Ref 1." and one cut off at the end of
# the reply ("... [ref 1").
_TAG_NUM = r"\d{1,3}(?:\s*(?:,|and|&)\s*(?:(?:ref|image|img)s?\s*[:#]?\s*)?\d{1,3})*"
TAG = re.compile(
    rf"\s*[\[(]\s*(ref|image|img)s?\s*[:#]?\s*({_TAG_NUM})\s*(?:[\])]|$)|\s*\b(ref)\s*[:#]?\s*({_TAG_NUM})\b\.?",
    re.IGNORECASE,
)
# Mark or image numbers said out loud despite the prompt: "Car #1", "the car (1)", "object 1", "in image 2", ...
# (not "marked 35", "labeled 25" or "[EXIT ONLY]": those are sign text).
STRAY_MARK = re.compile(
    r"\s*,?\s*\b(?:in|from|on)\s+(?:the\s+)?(?:image|img|frame|picture)\s*#?\d{1,3}\b,?"
    r"|\s*\b(?:in|at|inside)\s+(?:the\s+)?(?:box|mark|tag)\s*#?\d{1,3}\b"
    r"|\s*\((?:marked |box |tag |object |mark |#)?\s*#?\d{1,3}\)"
    r"|\s*#\d{1,3}\b"
    r"|\s*\[\s*#?\d{1,3}\s*\]",
    re.IGNORECASE,
)
STRAY_OBJECT = re.compile(r"\b(?:the\s+)?(?:object|box|mark|image|img|frame)\s*#?\d{1,3}\b", re.IGNORECASE)
# "the car number 1" (a mark number after an object noun; "lane number 2" is a real lane)
OBJECT_NUMBER = re.compile(
    r"\b(cars?|bus(?:es)?|vans?|suvs?|trucks?|lights?|signs?|person|people|cyclists?|pedestrians?|vehicles?)"
    r"\s+(?:number|no\.|(?:marked|tagged|labeled|labelled)(?: as)?)\s*#?\d{1,3}\b",
    re.IGNORECASE,
)
MARK_COLORS = {"vehicle": (255, 220, 60), "person": (40, 190, 255), "light": (200, 200, 200), "sign": (90, 90, 255)}

NO_VOUCH = "I can't judge moving traffic, so check both ways yourself before you go."
GREEN_CHECK = "Check the intersection before you go."
# "Our light is green now", said plainly: "The light facing us is green.", "Yes, it just turned green.", "Green."
_OUR_LIGHT = r"(?:it|the light|our light|your light|the light (?:facing|in front of|ahead of) (?:us|you))"
GREEN_NOW = re.compile(
    rf"\b{_OUR_LIGHT}(?:'s| is)(?: now)? green\b|\b{_OUR_LIGHT}(?: has)?(?: now| just)? turned green\b"
    r"|^\W*(?:yes\W+)?green\b",
    re.IGNORECASE,
)
# ...with nothing that makes it another light, the future or the past, or that already reminds or warns.
NOT_OUR_GREEN_NOW = re.compile(
    r"\b(?:red|yellow|amber|not|never|about to|will|should|soon|wait|until|yet|arrows?|turn lane|left[- ]turn|"
    r"right[- ]turn|cross street|other|was|were|passed|pull over|check)\b|n't\b",
    re.IGNORECASE,
)


def says_green(spoken: Sequence[str], light: LightColor | None = None) -> bool:
    """The answer says plainly that our light is green now, and the tracker doesn't read it red or yellow: "The light
    facing us is green." (test sheet row 8) yes; "It's about to turn green.", "The left-turn arrow just turned
    green.", "It was green when we went through." and "The green sign says Main St." no (the review's cases)."""
    if light in (LightColor.RED, LightColor.YELLOW):
        return False
    text = " ".join(spoken)
    return bool(GREEN_NOW.search(text)) and not NOT_OUR_GREEN_NOW.search(text)


FALLBACK_TIMEOUT = "Sorry, I couldn't get a clear answer in time."
FALLBACK_ERROR = "Sorry, I can't reach the vision model right now."
FALLBACK_EMPTY = "Sorry, I didn't get an answer to that."
LOST_REST = "Sorry, I lost the rest of that."
LIGHT_WORDS = re.compile(r"\b(?:light|lights|signal|signals|arrow|arrows|green|red|yellow|amber)\b", re.IGNORECASE)
SIGN_WORDS = re.compile(r"\b(?:sign|signs|said|say|says|read|speed limit|limit|plate)\b", re.IGNORECASE)
# Questions about *our* moving off (the router's gap checks are covered by the route kind already).
GO_QUESTION = re.compile(
    r"\b(?:(?:can|could|should|may|do|shall) (?:i|we) (?:still )?(?:go|turn|merge|pull out|pull in|cross|proceed|pass|"
    r"change lanes?|get over|move over|make it|pull (?:forward|up|ahead)|creep|inch|roll (?:forward|up)|move up)|"
    r"(?:is it|it is|am i|are we|is that) (?:safe|clear|ok|okay|good)|"
    r"(?:safe|clear|ok|okay) to|(?:anyone|anybody|anything|someone|somebody|cars?|traffic) coming|room to|gap)\b",
    re.IGNORECASE,
)
# Questions about passing / going around something; a school bus in the answer then gets the fixed rule.
PASS_QUESTION = re.compile(
    r"\b(?:pass|passing|overtake|overtaking|go(?:ing)? around|get around|drive around|go past|get past|get by)\b",
    re.IGNORECASE,
)
SCHOOL_BUS = re.compile(r"\bschool ?bus(?:es)?\b", re.IGNORECASE)
# Permission in any sentence, whatever the question.
PERMISSION = re.compile(
    r"^\W*(?:yes\W+|ok(?:ay)?\W+|sure\W+)?go\b(?! to\b)"  # "Go.", "Yes, go." (not "Go to the next exit")
    r"|^\W*(?:yes\W+|ok(?:ay)?\W+)?(?:merge|turn|pull out|cross|pass|change lanes)(?: \w+)? now\b"  # "Merge now."
    r"|\bgo ahead\b|\bgo for it\b|\bgood to go\b|\bgo now\b|\bgo(?:,| -|—)? it'?s clear\b"
    r"|\byou(?:'re| are) (?:good|clear|all clear|safe|fine|ok|okay)\b(?! to (?:park|stop))"
    r"|\ball clear\b"
    r"|\byou (?:can|may|could|should) (?:go|merge|turn|pull out|proceed|cross|pass|change lanes|make it|get over|"
    r"pull (?:forward|up|ahead)|creep|inch|roll (?:forward|up)|move up)\b"
    r"|\b(?:safe|clear|ok|okay|fine) (?:for you )?to "
    r"(?:go|merge|turn|pull out|proceed|cross|pass|change lanes|get over)\b",
    re.IGNORECASE,
)
# Vouching for traffic: not allowed in answers to go/turn/merge/wait questions ("isn't clear" is fine).
VOUCHING = re.compile(
    r"(?<!not )(?<!n't )\b(?:clear|safe|empty)\b"
    r"|\b(?:nothing(?:'s| is) coming|no (?:one|cars?|traffic|vehicles?)(?:'s| is)? coming|nobody(?:'s| is) coming|"
    r"no traffic)\b",
    re.IGNORECASE,
)
# A condition to wait for isn't vouching: "...so wait until they clear and check both ways yourself" (test sheet row
# 45: the net replaced that whole warning about pedestrians). Only a waiting clause within one clause is exempt
# ("until/till ... clear", "wait for them to clear"): "It's clear until the light", "Once the truck passes, it's
# safe", "Pull out when it's safe" and "You have time to clear the lane" still count as vouching.
CONDITIONAL_CLEAR = re.compile(
    r"\b(?:until|till)\b[^.;:,]*?\b(?:clear|clears|cleared|safe|empty)\b"
    r"|\b(?:wait|waiting)\b[^.;:,]*?\bto (?:clear|be clear)\b",
    re.IGNORECASE,
)


def vouches(sentence: str) -> bool:
    """A sentence that says or implies the way is clear (for go questions), conditions to wait for left out."""
    return bool(VOUCHING.search(CONDITIONAL_CLEAR.sub(" ", sentence)))


WHERE = {Approach.LEFT: "on the left", Approach.AHEAD: "across the intersection", Approach.RIGHT: "on the right"}


def spoken_time(now: datetime) -> str:
    """ "Saturday 5:40 PM"."""
    hour = now.hour % 12 or 12
    return f"{now:%A} {hour}:{now:%M} {'AM' if now.hour < 12 else 'PM'}"


def encode_jpeg(image: np.ndarray, max_width: int, quality: int) -> bytes:
    """Downscale to `max_width` (never up) and JPEG-encode."""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def light_crop(image: np.ndarray, box: tuple[float, float, float, float], height: int = 256) -> np.ndarray | None:
    """A close-up of a traffic light: the box padded by half its size, upscaled to `height` pixels."""
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.5, (y2 - y1) * 0.5
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(w_img, int(x2 + pad_x)), min(h_img, int(y2 + pad_y))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    scale = height / crop.shape[0]
    return cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)


def sign_closeups(memory: SceneMemory, now: float, window_s: float, count: int) -> list[ImagePart]:
    """Close-ups of the signs the detectors saw in the last `window_s` (their best view so far), for a sign
    question: a sign 25 px tall in the frame can't be read, its close-up can (the test sheet's DO NOT ENTER signs).
    Labeled with what the detector took it for, as a hint, never as the answer."""
    parts: list[ImagePart] = []
    picked: list[tuple[float, tuple[float, float, float, float]]] = []
    for sign in memory.signs(window_s, count * 3, now):
        # One sign can be remembered twice (as the COCO detector's "stop sign" and the sign detector's "stop", or
        # as two sightings): a box overlapping one already picked at about the same time is the same sign.
        if any(abs(sign.t - t) < 1.0 and box_iou(sign.box, box) > 0.3 for t, box in picked):
            continue
        if len(parts) == count:
            break
        picked.append((sign.t, sign.box))
        where = "ahead" if sign.side == "ahead" else f"on the {sign.side}"
        read = f", read as {sign.text}" if sign.text else ""
        label = (
            f"Close-up of a sign {where}, {max(0.0, now - sign.t):.0f} s ago "
            f"(the sign detector took it for: {sign.kind.replace('_', ' ')}{read})"
        )
        parts.append(ImagePart(encode_jpeg(sign.image, 512, 90), label))
    return parts


@dataclass(frozen=True, slots=True)
class Mark:
    """A numbered object on the image sent to the model."""

    number: int
    kind: str  # "vehicle" | "person" | "light" | "sign"
    track_id: int | None  # (lights have no stable track id)
    box: tuple[float, float, float, float]
    cls: str = ""  # the detector's class ("car", "bus", "person", "traffic light", ...)
    color: str = ""  # a vehicle's color name, if known


def pick_marks(
    snapshot: WorldSnapshot,
    tracks: Sequence[Track],
    limit: int,
    width: int = 0,
    height: int = 0,
    hood_bottom: float = HOOD_BOTTOM,
) -> list[Mark]:
    """The objects worth numbering on the latest frame: vehicles (not our own hood), people and stop signs
    (nearest = largest first), then the lights; at most `limit`."""
    Obj = tuple[float, str, int | None, tuple[float, float, float, float], str, str]
    objects: list[Obj] = []
    for v in snapshot.vehicles:
        if width and height and is_own_hood(v.box, width, height, hood_bottom):
            continue
        area = (v.box[2] - v.box[0]) * (v.box[3] - v.box[1])
        objects.append((area, "vehicle", v.track_id, v.box, v.cls, v.color))
    for tr in tracks:
        kind = {"person": "person", "bicycle": "person", "stop sign": "sign"}.get(tr.cls)
        if kind is not None:
            objects.append((tr.width * tr.height, kind, tr.track_id, tr.box, tr.cls, ""))
    objects.sort(key=lambda o: -o[0])
    picked = objects[: max(0, limit)]
    for box in snapshot.light.boxes:
        if len(picked) >= limit:
            break
        picked.append((0.0, "light", None, box, "traffic light", ""))
    return [
        Mark(i, kind, track_id, box, cls, color)
        for i, (_, kind, track_id, box, cls, color) in enumerate(picked, start=1)
    ]


# What an answer can call each detector class (the first one it names is what it is about).
_ANY_VEHICLE = {"car", "truck", "bus", "motorcycle"}
OBJECT_NOUNS: list[tuple[re.Pattern[str], set[str]]] = [
    (re.compile(r"\b(?:school ?)?bus(?:es)?\b", re.IGNORECASE), {"bus"}),
    (re.compile(r"\b(?:trucks?|pickups?|semis?|lorry|lorries)\b", re.IGNORECASE), {"truck", "car"}),
    (re.compile(r"\b(?:motorcycles?|motorbikes?|scooters?|mopeds?)\b", re.IGNORECASE), {"motorcycle"}),
    (re.compile(r"\b(?:cyclists?|bicyclists?|bikers?|bikes?|bicycles?)\b", re.IGNORECASE), {"bicycle", "person"}),
    (
        re.compile(
            r"\b(?:person|pedestrians?|people|man|woman|men|women|guy|lady|kids?|child|children|someone|somebody|"
            r"walkers?|joggers?)\b",
            re.IGNORECASE,
        ),
        {"person", "bicycle"},
    ),
    (
        re.compile(
            r"\b(?:cars?|sedans?|suvs?|vans?|minivans?|hatchbacks?|coupes?|wagons?|jeeps?|taxis?|cabs?)\b",
            re.IGNORECASE,
        ),
        {"car", "truck"},
    ),
    (re.compile(r"\bvehicles?\b", re.IGNORECASE), _ANY_VEHICLE),
    (re.compile(r"\b(?:stop )?signs?\b|\bboards?\b", re.IGNORECASE), {"stop sign"}),  # (only stop signs are marked)
    (re.compile(r"\b(?:traffic )?(?:lights?|signals?)\b", re.IGNORECASE), {"traffic light"}),
]
COLOR_WORDS = re.compile(
    r"\b(white|black|gray|grey|silver|red|blue|green|yellow|orange|brown|beige|gold|tan|purple|maroon)\b", re.I
)
# Only "on your left / right" says where in the frame: "ahead" is anywhere up the road, and someone "crossing in
# front of us" close by can be at the frame's edge.
SIDE_WORDS = re.compile(r"\b(?:on|to) (?:your|the|our) (left|right)\b", re.IGNORECASE)
TAG_MIN_SHARE = 0.25  # a tagged object this much smaller than the biggest that fits the answer gives way to it


def _named_object(answer: str) -> tuple[set[str], re.Match[str]] | None:
    """The classes the answer's first-named object can be, and where it is named (quoted sign text, like "LAST
    LOCAL TRUCK ROUTE", names nothing)."""
    unquoted = re.sub(r"\"[^\"]*\"|“[^”]*”", lambda m: " " * len(m.group(0)), answer)
    best: tuple[int, set[str], re.Match[str]] | None = None
    for pattern, classes in OBJECT_NOUNS:
        match = pattern.search(unquoted)
        if match is not None and (best is None or match.start() < best[0]):
            best = (match.start(), classes, match)
    return None if best is None else (best[1], best[2])


_SAME_COLOR = {"grey": "gray", "silver": "gray", "maroon": "red", "tan": "beige", "gold": "beige", "brown": "beige"}


def _color(name: str) -> str:
    return _SAME_COLOR.get(name.lower(), name.lower())


def _on_side(box: tuple[float, float, float, float], width: int, side: str) -> bool:
    """Whether a box is on the side of the frame the answer says."""
    center = (box[0] + box[2]) / 2 / max(width, 1)
    return center < 0.4 if side == "left" else center > 0.6


def _at_side_edge(box: tuple[float, float, float, float], width: int, margin: float = 6.0) -> bool:
    """Cut off by the frame's side: a car right beside us (parked, passing), rarely what an answer is about."""
    return box[0] <= margin or box[2] >= width - margin


def _area(mark: Mark) -> float:
    return (mark.box[2] - mark.box[0]) * (mark.box[3] - mark.box[1])


def resolve_highlight(answer: str, tagged: Sequence[int], marks: dict[int, Mark], width: int) -> int | None:
    """Which numbered object the answer is about, or None. The model's tag is kept only if it fits what the answer
    names (a "school bus" answer tagged with a person is wrong); otherwise the one object matching the named class,
    color and side wins (the largest, if it is clearly the nearest). An answer that names no object (a building,
    construction) highlights nothing."""
    named = _named_object(answer)
    if named is None:
        return None
    classes, match = named
    sentence = re.split(r"(?<=[.!?])\s+", answer[match.start() :], maxsplit=1)[0]
    colors = COLOR_WORDS.findall(answer[max(0, match.start() - 24) : match.start()])
    color = _color(colors[-1]) if colors else None
    side_match = SIDE_WORDS.search(sentence)
    side = side_match.group(1).lower() if side_match is not None else None

    def fits(mark: Mark, use_color: bool) -> bool:
        if mark.cls not in classes or mark.track_id is None:
            return False
        if use_color and color is not None and mark.color and _color(mark.color) != color:
            return False
        return side is None or _on_side(mark.box, width, side)

    candidates = [m for m in marks.values() if fits(m, use_color=True)]
    if not candidates and color is not None:  # (our color names can differ from the model's)
        candidates = [m for m in marks.values() if fits(m, use_color=False)]
    biggest = max(candidates, key=_area) if candidates else None
    for number in tagged:  # the model's own pick, if it agrees with what the answer says about it...
        mark = marks.get(number)
        if mark is not None and fits(mark, use_color=True):
            if (
                biggest is not None
                and _area(mark) < TAG_MIN_SHARE * _area(biggest)
                and color is None
                and side is None
                and not _at_side_edge(biggest.box, width)
            ):
                return biggest.number  # ...unless a far bigger one in full view fits just as well (the nearest)
            return number
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].number
    area = sorted((_area(m), m.number) for m in candidates)
    return area[-1][1] if area[-1][0] >= 2.0 * area[-2][0] else None  # the clearly nearest one, else unsure


def draw_marks(image: np.ndarray, marks: Sequence[Mark]) -> np.ndarray:
    """A copy of the image with each mark's box and number tag drawn (what the model sees)."""
    out = image.copy()
    w = out.shape[1]
    thick = max(1, w // 640)
    scale = max(0.45, w / 1800)
    for mark in marks:
        x1, y1, x2, y2 = (round(v) for v in mark.box)
        color = MARK_COLORS.get(mark.kind, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = str(mark.number)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick))
        tx, ty = max(0, x1), max(th + base + 2, y1)
        cv2.rectangle(out, (tx, ty - th - base - 2), (min(w - 1, tx + tw + 4), ty), color, -1)
        cv2.putText(out, label, (tx + 2, ty - base - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, thick))
    return out


def strip_tags(sentence: str) -> tuple[str, list[tuple[str, int]]]:
    """The sentence without its hidden tags ("[ref 3]", "[image 2]") or stray mark numbers, and the tags found."""
    tags: list[tuple[str, int]] = []
    for kind_a, nums_a, kind_b, nums_b in TAG.findall(sentence):
        kind = "image" if (kind_a or kind_b).lower() in ("image", "img") else "ref"
        tags += [(kind, int(n)) for n in re.findall(r"\d{1,3}", nums_a or nums_b)]
    sentence = re.sub(r",\s*(\[[^\]]*\])\s*,", r" \1", sentence)  # "The sign, [ref 3], says" -> "The sign says"
    text = STRAY_OBJECT.sub("that one", OBJECT_NUMBER.sub(r"\1", STRAY_MARK.sub("", TAG.sub("", sentence))))
    text = re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"\s{2,}", " ", text))
    # Punctuation around a removed tag: "30 mph, [ref 3]." -> "30 mph."; "The sign, [ref 3], says" -> "The sign, says"
    text = re.sub(r"([,;:])[,;:]+", r"\1", re.sub(r"[,;:]+(?=[.!?])", "", text)).strip(" ,")
    if text and text[0].islower() and sentence.lstrip()[:1].isupper():  # ("In Image 2, the sign..." -> "The sign...")
        text = text[0].upper() + text[1:]
    return text, tags


@dataclass(slots=True)
class RequestPlan:
    """One request and what its numbers mean (to act on the answer's tags)."""

    request: VisionRequest
    marks: dict[int, Mark] = field(default_factory=dict)
    images: dict[int, Frame] = field(default_factory=dict)  # memory: image number -> frame
    marked: np.ndarray | None = None  # the "Now" image with its numbered objects (what the model saw; debugging)
    width: int = 0  # of the frame the marks are on


def _side(box: tuple[float, float, float, float], width: int) -> str:
    center = (box[0] + box[2]) / 2 / max(width, 1)
    return "on the left" if center < 0.4 else "on the right" if center > 0.6 else "in the middle"


def describe_world(snapshot: WorldSnapshot) -> list[str]:
    """What the tracker knows, as short context lines for the prompt (verdicts, not raw data)."""
    lines = []
    if not snapshot.ego_motion_confident:
        lines.append("Unsure whether we are moving.")
    elif snapshot.ego_stopped:
        since = snapshot.ego_stopped_since
        lines.append(f"We are stopped (for {snapshot.t - since:.0f} s)." if since is not None else "We are stopped.")
    else:
        lines.append("We are moving.")
    if snapshot.stop_sign_visible:
        lines.append("A stop sign is in view.")
    elif snapshot.stop_sign_last_seen is not None and snapshot.t - snapshot.stop_sign_last_seen < 15:
        lines.append(f"A stop sign was seen {snapshot.t - snapshot.stop_sign_last_seen:.0f} s ago.")

    # Whether other cars are moving is only reliable while we stand still (our own motion is cancelled only
    # approximately); otherwise just say where they are and let the model judge from the frames.
    def motion(stopped: bool) -> str:
        if not (snapshot.ego_stopped and snapshot.ego_motion_confident):
            return ""
        return " (stopped)" if stopped else " (moving)"

    cars = []
    if snapshot.lead_vehicle is not None:
        lead = snapshot.lead_vehicle
        cars.append(f"{lead.label} directly ahead of us{motion(lead.stopped)}")
    for v in snapshot.vehicles:
        if v.approach is not None and len(cars) < 6:
            cars.append(f"{v.label} {WHERE[v.approach]}{motion(v.stopped)}")
    if cars:
        lines.append("Vehicles near the intersection: " + "; ".join(cars) + ".")
    queue = snapshot.queue
    if queue.active:
        lines.append("We are at a stop sign (the tracker can't tell whether it is an all-way stop).")
        if queue.deadlock:
            lines.append("Right of way: several cars stopped together; nobody has a clear right of way.")
        elif queue.your_turn:
            lines.append("Right of way: by arrival order it is our turn; moving traffic is not judged.")
        else:
            ahead = ahead_of_us(queue)
            names = ", then ".join(describe(e) for e in ahead)
            lines.append(f"Right of way: not our turn yet; {len(ahead)} car(s) go before us: {names}.")
        for entry in queue.dropped:
            lines.append(f"Lost track of {describe(entry)}; it may still be there.")
    light = snapshot.light
    if light.color is not LightColor.UNKNOWN:
        lines.append(f"Traffic light facing us looks {light.color.value}{'' if light.confident else ' (unsure)'}.")
    if snapshot.people_ahead:
        lines.append(f"{snapshot.people_ahead} pedestrian{'s' if snapshot.people_ahead > 1 else ''} ahead.")
    return lines


# The reading model (vision.reading) answers questions that read or describe the scene; the main model keeps lights,
# lanes, memory and go questions. Measured 2026-10-11 (eval/test_sheet.csv + a held-out sheet): Claude Haiku 5.5
# read more signs and parking rules and described scenes better, while Claude Haiku 4.5 was better at counting lanes
# and at "what did we just pass?", and Haiku 5.5 once called a red light yellow.
_LANES = re.compile(r"\blanes?\b", re.IGNORECASE)
LIGHT_NAMED = re.compile(r"\b(?:light|lights|signal|signals|arrow|arrows)\b", re.IGNORECASE)
_COLORS = re.compile(r"\b(?:green|red|yellow|amber)\b", re.IGNORECASE)


def uses_reading_model(route: Route) -> bool:
    """True for questions the reading model answers (see above)."""
    text = route.text
    if route.intent == "memory" or is_go_question(route) or LIGHT_NAMED.search(text) or _LANES.search(text):
        return False
    return not (_COLORS.search(text) and not SIGN_WORDS.search(text))  # "is it green yet?" is about the light


def is_go_question(route: Route) -> bool:
    """Questions where vouching for traffic is dangerous: gap checks, "should I wait", go/turn/merge wording."""
    return route.kind is RouteKind.GAP_CHECK or route.intent == "should_wait" or bool(GO_QUESTION.search(route.text))


class SceneHandler:
    """Answers scene questions with the vision model (the assistant's `SceneAnswerer`)."""

    def __init__(
        self,
        cfg: AppConfig,
        model: VisionModel,
        frames: Callable[[int, float], list[Frame]],
        extra_context: Callable[[], list[str]] | None = None,
        now: Callable[[], datetime] = datetime.now,
        tracks: Callable[[], Sequence[Track]] | None = None,
        memory: SceneMemory | None = None,
        on_highlight: Callable[[int], None] | None = None,
        on_image: Callable[[Frame], None] | None = None,
        reading_model: VisionModel | None = None,
    ) -> None:
        """`tracks`: the latest frame's tracks (people, signs to number); `memory`: for "what was that?";
        `on_highlight(track_id)` / `on_image(frame)`: what an answer's hidden tag points at (the page shows it);
        `reading_model`: answers sign, parking and scene-description questions (`uses_reading_model`)."""
        self.cfg = cfg
        self.model = model
        self.reading_model = reading_model
        self._frames = frames
        self._extra = extra_context
        self._now = now
        self._tracks = tracks
        self.memory = memory
        self.on_highlight = on_highlight
        self.on_image = on_image
        self.last_plan: RequestPlan | None = None  # (debugging, the test sheet)

    # ---------------------------------------------------------------------------------- prompt

    def build_request(self, question: str, snapshot: WorldSnapshot, memory: bool = False) -> VisionRequest:
        """The images and text for one question."""
        return self.plan(question, snapshot, memory).request

    def plan(self, question: str, snapshot: WorldSnapshot, memory: bool = False) -> RequestPlan:
        """The request for one question, with what its object and image numbers mean."""
        if memory and self.memory is not None:
            planned = self._memory_plan(question, snapshot)
            if planned is not None:
                return planned
        vcfg = self.cfg.vision
        about_sign = bool(SIGN_WORDS.search(question))
        spacing = vcfg.sign_frame_spacing_s if about_sign else vcfg.frame_spacing_s
        older_width = vcfg.latest_width if about_sign else vcfg.older_width  # signs just passed: keep them legible
        frames = self._frames(vcfg.frames, spacing)
        if not frames and snapshot.frame is not None:
            frames = [snapshot.frame]
        images: list[ImagePart] = []
        latest = frames[-1] if frames else None
        for frame in frames[:-1]:
            age = (latest.t - frame.t) if latest is not None else 0.0
            images.append(ImagePart(encode_jpeg(frame.image, older_width, vcfg.jpeg_quality), f"{age:.1f} s ago"))
        marks: list[Mark] = []
        if latest is not None:
            now_image = latest.image
            if vcfg.mark_objects and latest is snapshot.frame:  # (the boxes are from this very frame)
                tracks = self._tracks() if self._tracks is not None else ()
                marks = pick_marks(
                    snapshot, tracks, vcfg.max_marks, latest.width, latest.height, self.cfg.perception.hood_bottom
                )
                if marks:
                    now_image = draw_marks(now_image, marks)
            images.append(ImagePart(encode_jpeg(now_image, vcfg.latest_width, vcfg.jpeg_quality), "Now"))
            if LIGHT_WORDS.search(question):
                for box in snapshot.light.boxes[:3]:
                    crop = light_crop(latest.image, box)
                    if crop is not None:
                        label = f"Close-up of the traffic light {_side(box, latest.image.shape[1])}"
                        images.append(ImagePart(encode_jpeg(crop, 512, 92), label))
            if about_sign and self.memory is not None and vcfg.sign_closeups > 0:
                images += sign_closeups(self.memory, latest.t, vcfg.sign_closeup_window_s, vcfg.sign_closeups)
        when = vcfg.time_override or spoken_time(self._now())
        context = [f"Local time: {when}.", *describe_world(snapshot)]
        if self._extra is not None:
            context += self._extra()
        if latest is None:
            context.append("No camera image is available right now.")
        text = "Context:\n" + "\n".join(f"- {line}" for line in context) + f"\n\nDriver's question: {question}"
        if marks:
            text += "\n" + MARKS_REMINDER
        request = VisionRequest(
            system=SYSTEM_PROMPT + ("\n- " + MARKS_PROMPT if marks else ""),
            text=text,
            images=tuple(images),
            max_tokens=self._max_tokens(),
            temperature=vcfg.temperature,
        )
        return RequestPlan(
            request,
            marks={m.number: m for m in marks},
            marked=now_image if marks else None,
            width=latest.width if latest is not None else 0,
        )

    def _memory_plan(self, question: str, snapshot: WorldSnapshot) -> RequestPlan | None:
        """Sign close-ups and the last seconds of frames, numbered, plus the latest frame (None: nothing kept)."""
        assert self.memory is not None
        vcfg, mcfg = self.cfg.vision, self.cfg.memory
        latest = snapshot.frame
        now = latest.t if latest is not None else snapshot.t
        frames = [
            f for f in self.memory.frames(mcfg.window_s, mcfg.frames, now=now) if latest is None or f.t < latest.t
        ]
        signs = self.memory.signs(mcfg.sign_window_s, mcfg.signs, now)
        if not frames and not signs:
            return None
        # A sign we passed: its frames as kept (memory.width), not shrunk to older_width (at 640 px a night exit sign's
        # text was invented: test sheet row 35).
        width = mcfg.width if SIGN_WORDS.search(question) else vcfg.older_width
        images: list[ImagePart] = []
        numbered: dict[int, Frame] = {}
        number = 0
        for sign in signs:
            number += 1
            what = sign.kind.replace("_", " ")
            read = f", read as {sign.text}" if sign.text else ""
            label = f"Image {number}: close-up of a {what} we saw {now - sign.t:.0f} s ago, {sign.side}{read}"
            images.append(ImagePart(encode_jpeg(sign.image, 512, 90), label))
            numbered[number] = Frame(image=sign.image, t=sign.t, index=0, captured_at=0.0)  # (its thumbnail)
        for frame in frames:
            number += 1
            numbered[number] = frame
            label = f"Image {number}: {now - frame.t:.0f} s ago"
            images.append(ImagePart(encode_jpeg(frame.image, width, vcfg.jpeg_quality), label))
        if latest is not None:
            number += 1
            numbered[number] = latest
            images.append(ImagePart(encode_jpeg(latest.image, width, vcfg.jpeg_quality), f"Image {number}: now"))
        when = vcfg.time_override or spoken_time(self._now())
        context = [f"Local time: {when}.", *describe_world(snapshot)]
        if self._extra is not None:
            context += self._extra()
        text = "Context:\n" + "\n".join(f"- {line}" for line in context) + f"\n\nDriver's question: {question}"
        text += "\n" + MEMORY_REMINDER
        request = VisionRequest(
            system=SYSTEM_PROMPT + "\n- " + MEMORY_PROMPT,
            text=text,
            images=tuple(images),
            max_tokens=self._max_tokens(),
            temperature=vcfg.temperature,
        )
        return RequestPlan(request, images=numbered)

    def _max_tokens(self) -> int:
        by_provider = {"claude": self.cfg.vision.claude.max_tokens, "gemini": self.cfg.vision.gemini.max_tokens}
        return by_provider.get(self.model.name, 300)

    # ---------------------------------------------------------------------------------- answer

    def answer(self, route: Route, snapshot: WorldSnapshot, cancel: threading.Event | None = None) -> Iterator[str]:
        """Spoken sentences, as they arrive (stops early if `cancel` is set)."""
        planned = self.plan(route.question or route.text, snapshot, memory=route.intent == "memory")
        self.last_plan = planned
        vcfg = self.cfg.vision
        model, request = self.model, planned.request
        if self.reading_model is not None and uses_reading_model(route):
            model, request = self.reading_model, replace(request, temperature=vcfg.reading.temperature)
        log.info("vision model for this question", model=getattr(model, "model", model.name))
        pieces = guarded_stream(model, request, vcfg.first_token_timeout_s, vcfg.total_timeout_s, cancel)
        passing = bool(PASS_QUESTION.search(route.text))
        # "It's green" gets "check the intersection" while we wait at the light (not a memory answer about the past)
        green_check = route.intent != "memory" and snapshot.ego_stopped
        return self._safe(
            sentences(pieces),
            strict=is_go_question(route),
            passing=passing,
            planned=planned,
            green_check=snapshot.light.color if green_check else None,
        )

    def _show_image(self, number: int, planned: RequestPlan) -> None:
        """A memory answer's "[image k]": show that remembered frame."""
        try:
            if number in planned.images and self.on_image is not None:
                self.on_image(planned.images[number])
        except Exception:
            log.exception("showing an answer's image failed", number=number)

    def _highlight(self, answer: str, tagged: list[int], planned: RequestPlan) -> None:
        """Once the answer is complete: pulse the object it is about (if we can tell which)."""
        if not planned.marks or self.on_highlight is None:
            return
        try:
            number = resolve_highlight(answer, tagged, planned.marks, planned.width)
            log.info("answer highlight", tagged=tagged, chosen=number)
            if number is not None:
                track_id = planned.marks[number].track_id
                if track_id is not None:
                    self.on_highlight(track_id)
        except Exception:
            log.exception("highlight failed")

    def _safe(
        self,
        stream: Iterator[str],
        strict: bool,
        passing: bool = False,
        planned: RequestPlan | None = None,
        green_check: LightColor | None = None,
    ) -> Iterator[str]:
        """Safety nets on every sentence, hidden tags taken out, and spoken fallbacks for failures (`green_check`: the
        tracker's light while we're stopped; an answer that says our light is green then ends with a reminder to check
        the intersection, unless the tracker reads red or yellow)."""
        said = False
        no_vouch_said = False
        replaced = False  # a safety net changed the answer: it no longer points at what the model named
        image: int | None = None
        spoken: list[str] = []
        tagged: list[int] = []
        try:
            for raw in stream:
                sentence, tags = strip_tags(raw)
                for kind, number in tags:
                    if kind == "ref":
                        tagged.append(number)
                    elif image is None:
                        image = number
                if not sentence or not re.search(r"\w", sentence):
                    continue
                # Every piece on its own: a short sentence joined to the next, or one left over where a tag was
                # taken out, must still meet the start-anchored nets ("... stopping. Go.").
                pieces = [p for p in re.split(r"(?<=[.!?])\s+", sentence) if p]
                if passing and SCHOOL_BUS.search(sentence):
                    # "Should I go around them?" with a school bus in view: the rule, not the model's judgment.
                    log.warning("school bus in a passing answer: replaced by the rule", sentence=sentence)
                    yield SCHOOL_BUS_RULE
                    return
                if any(PERMISSION.search(p) or (strict and vouches(p)) for p in pieces):
                    log.warning("replaced a sentence that vouched for traffic", sentence=sentence, strict=strict)
                    replaced = True
                    if no_vouch_said:
                        continue
                    sentence, no_vouch_said = NO_VOUCH, True
                else:
                    spoken.append(sentence)
                said = True
                yield sentence
        except VisionCancelled:
            return
        except VisionError as exc:
            log.warning("vision answer failed", error=str(exc), after_sentences=said)
            if said:
                yield LOST_REST
            else:
                yield FALLBACK_TIMEOUT if isinstance(exc, VisionTimeout) else FALLBACK_ERROR
            return
        if planned is not None and spoken and not replaced:
            self._highlight(" ".join(spoken), tagged, planned)
            if image is not None:
                self._show_image(image, planned)  # (after the answer's sentences: the page shows it under them)
        if not said:
            log.warning("vision answer was empty")
            yield FALLBACK_EMPTY
            return
        if strict:
            if not no_vouch_said:
                yield NO_VOUCH  # (says "check" already: no second reminder)
        elif green_check is not None and says_green(spoken, green_check):
            yield GREEN_CHECK  # like the local light answers: never a bare "it's green"
