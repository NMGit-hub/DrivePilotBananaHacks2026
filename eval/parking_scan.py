"""A periodic vision-model scan for open parking spots (V2-M1's alternative to the geometry tracker).

V2_BUILD_PLAN.md (V2-M1): "a periodic model scan (every ~2 s while the watch is on) instead of, or besides, the
geometry. The variant that scores best on the labeled clips ships." This module is that variant's logic, kept in
plain functions and classes so the app can use it later (scripts/eval_parking_scan.py measures it offline):
- **prompts** (`PROMPTS`, one per version; each version caches its raw replies in its own folder), built into a
  `VisionRequest` by `ScanPrompt.request`;
- **parsing** the model's JSON into `ScanReport`s (side, cars ahead, the vehicle right before the spot, restriction,
  confidence), robust to malformed replies (`parse_scan`);
- **deciding** what the app would announce (`SpotAnnouncer`): drop low-confidence and restricted reports (the
  user's rules: skip handicap, reserved, fire lane, no parking, loading; announce EV-only and compact as such),
  optionally require several agreeing scans, and dedupe the same spot across scans by side, cars ahead (which
  shrinks as we drive past parked cars) and elapsed time (and, optionally, the colour of the car before it);
- **scoring** announcements against the hand labels with `eval.parking_eval.match_report`, exactly as the geometry
  tracker's spot events are scored (`score_scan`, `summarize_scan`).

Announcements correspond to the tracker's "spot" events (one per distinct spot found); which of them the watch
actually speaks (the nearest; again only when it is passed, taken or a closer one opens) is answers/parking.py's job.

Measured 2026-10-10 (Claude Haiku 4.5, 28 usable clips, 30 targets, 13.5 min of lot driving): far from the bar
(>= 70 % recall, <= 1 false spot per 10 min). Best v3 settings: every 2 s, confidence high: 47 % recall at 26.6
false / 10 min; every 1 s, high + 2 agreeing scans: 33 % at 11.1. Prompt versions (dev subset of 11 clips): v1 a
list of spots (reports one in almost every frame), v2 each row listed stall by stall, v3 + what borders our lane
and stricter "empty", v3e + a frame 0.7 s earlier (worse), v4 + enlarged side close-ups (no better, 2x tokens),
v5 + a visible row end after trailing empty stalls (fewer false alarms, much lower recall). Claude Sonnet 4.6 with
v3: 20 % recall at 1.9 false / 10 min on the dev subset (precise but rarely says "empty").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import cv2
import numpy as np

from answers.parking import RESTRICTIONS, SKIP
from answers.scene import encode_jpeg
from answers.vision import ImagePart, VisionRequest
from core.config import VisionConfig
from eval.parking_eval import ClipLabels, LabeledSpot, LookAlike, match_report

CONFIDENCES = ("low", "medium", "high")
SIDES = ("left", "right")
_NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
MAX_BEFORE_CHARS = 40

# ------------------------------------------------------------------------------------------------ prompts

_RESTRICTION_ENUM = '"none"|"handicap"|"ev"|"reserved"|"compact"|"fire_lane"|"no_parking"|"loading"|"unknown"'

SYSTEM_V1 = (
    "You look at a dashcam image from a car driving slowly through a parking lot while the driver looks for an "
    "empty parking space. Left and right are from the driver's seat (image left is the driver's left). Report "
    "only stalls you can clearly see are completely empty: missing a space is acceptable, but calling a space "
    'open when it is not is a serious error. Say "low" confidence when unsure.'
)
TEXT_V1 = (
    "Look only at the first row of parking stalls directly beside our driving lane on each side (the row on our "
    "left and the row on our right, at most about six stalls ahead; not rows further away, across another aisle, "
    "or behind the first row). Is there a completely empty marked stall in either of these rows that we could pull "
    "into? It must be a real parking stall, not a driving lane, cross aisle, walkway, crosswalk, island, driveway "
    "or the open end of a row, and nothing may be in it: no car of any size, motorcycle, scooter, cart, cone, "
    "barrier or person. Don't report a stall you can't see into (hidden behind a nearer car). For the nearest "
    "open stall on each side give: side; cars_ahead = how many parked vehicles in that row are between us and "
    'the stall; before = the parked vehicle right before it (nearer to us), e.g. "white SUV", or null; stalls = '
    "how many empty stalls there are side by side; restriction by paint or sign: handicap (blue paint, wheelchair "
    "symbol), ev (EV charging), reserved, compact, fire_lane, no_parking, loading, none, or unknown; confidence. "
    "in_lot is true if we are driving inside a parking lot or garage (rows of marked stalls), false on a street. "
    'Reply as {"in_lot": true|false, "spots": [{"side": "left"|"right", "cars_ahead": <integer>, "before": '
    '"<color and type>" or null, "stalls": <integer>, "restriction": ' + _RESTRICTION_ENUM + ', "confidence": '
    '"high"|"medium"|"low"}]}, with "spots": [] when there is no open stall.'
)

# v2: the model lists each row stall by stall (nearest first) instead of picking a spot; we find the gap ourselves.
SYSTEM_V2 = (
    "You look at a dashcam image from a car driving slowly through a parking lot while the driver looks for an "
    "empty parking space. Left and right are from the driver's seat (image left is the driver's left). Most of the "
    "time every stall beside us is taken, or you can't see well enough to tell: then list no empty stall. Calling "
    "a stall empty when it is not is a serious error; missing one is acceptable."
)
TEXT_V2 = (
    "List the first row of parking stalls directly beside our driving lane on each side, nearest to us first, up "
    "to about six stalls ahead. Only the row right next to our lane: not rows further away, across another aisle, "
    'or behind the first row. Write each stall as the vehicle in it ("white SUV", "gray sedan", "motorcycle"), or '
    '"empty" for a marked stall you can clearly see is completely empty: its painted lines and bare pavement '
    "visible, no car of any size, motorcycle, cart, cone or person in it. A stall you can't see into because a "
    'nearer car hides it is not empty: stop the list there. Write "empty:handicap", "empty:ev", "empty:reserved", '
    '"empty:compact", "empty:fire_lane", "empty:no_parking" or "empty:loading" when paint or a sign marks it '
    "(handicap: blue paint or a wheelchair symbol). Driving lanes, cross aisles, walkways, crosswalks, islands, "
    "driveways and the end of a row are not stalls: end the list there. With no row of stalls right beside us on "
    "a side (a curb, a building, a street, open pavement without stall lines) give an empty list. in_lot is true "
    "if we are driving inside a parking lot or garage, false on a street. Reply as "
    '{"in_lot": true|false, "left": {"stalls": [...], "confidence": "high"|"medium"|"low"}, "right": {"stalls": '
    '[...], "confidence": "high"|"medium"|"low"}}, where confidence is how sure you are of that row\'s list.'
)

# v3: first say what borders our lane on each side (only a row of stalls is listed), stricter "empty", and the
# look-alikes that fooled v2 (hatched access aisles, the stall in front of a car we face, rows behind a curb,
# anything once we are parked).
SYSTEM_V3 = SYSTEM_V2
TEXT_V3 = (
    'For each side, first say what borders our driving lane right next to us on that side: "stalls" (a row of '
    'marked parking stalls opening onto our lane), "curb" (a curb, island, planter, sidewalk, grass or building), '
    '"lane" (another driving lane, a cross aisle or a street) or "unclear" (glare, blur, too dark). Only for '
    '"stalls", list that row\'s stalls nearest to us first, up to about six stalls ahead: each as the vehicle in '
    'it ("white SUV", "gray sedan", "motorcycle"), or "empty" for a stall you can clearly see is completely '
    "empty: both its painted lines and the bare pavement between them visible, nothing of any size in it (car, "
    "motorcycle, cart, cone, person). Stop the list at the first stall you can't see into (hidden behind a nearer "
    'car), at a cross aisle, crosswalk or island, or at the end of the row. Mark restricted empty stalls "empty:'
    'handicap" (blue paint or a wheelchair symbol), "empty:ev", "empty:reserved", "empty:compact", "empty:'
    'fire_lane", "empty:no_parking" or "empty:loading". These are never empty stalls: hatched (striped) areas '
    "beside handicap stalls, the space in front of a car we are facing, gaps in a row behind a curb or island or "
    "across another aisle, and anything while we are parked or pulling into a stall (then list nothing). Reply as "
    '{"in_lot": true|false, "left": {"beside": "stalls"|"curb"|"lane"|"unclear", "stalls": [...], "confidence": '
    '"high"|"medium"|"low"}, "right": {...the same...}}; in_lot is true inside a parking lot or garage, false on a '
    "street; confidence is how sure you are of that side."
)
MOTION_NOTE = (
    " The first image is {s:.1f} s earlier (smaller), for motion only: as we drive, the cars of the row right "
    'beside us slide across the image fastest. Answer about the "Now" image.'
)
# v5: v3 plus what ends each listed row ("end"): empty stalls at the end of a listing must be bounded by a row end
# the model names (island, cross aisle, curb), not pavement running out of view (v2/v3's commonest false alarm);
# "facing" for a row we face head-on or a stall we are in; stricter EV vs handicap.
TEXT_V5 = (
    'For each side, first say what borders our driving lane right next to us on that side: "stalls" (a row of '
    'marked parking stalls opening onto our lane, which we drive along), "facing" (we face that row head-on, or '
    'we are parked or pulling into a stall), "curb" (a curb, island, planter, sidewalk, grass or building), '
    '"lane" (another driving lane, a cross aisle or a street) or "unclear" (glare, blur, too dark). Only for '
    '"stalls", list that row\'s stalls nearest to us first, up to about five stalls ahead: each as the vehicle in '
    'it ("white SUV", "gray sedan", "motorcycle"), or "empty" for a stall you can clearly see is completely '
    "empty: both its painted lines and the bare pavement between them visible, nothing of any size in it (car, "
    'motorcycle, cart, cone, person). Then say what ends your list (end): "car" (the next stall is hidden behind '
    'a parked car), "out_of_view" (the row continues too far or too obliquely to judge), "island", "cross_aisle", '
    '"row_end" or "curb". Empty stalls need a visible boundary after them: the next parked vehicle in the list, or '
    "an island, cross aisle or row end you can see; where the pavement just continues out of view, list nothing "
    'more. Mark restricted empty stalls "empty:handicap" (blue paint or a wheelchair symbol), "empty:ev" (only with '
    'an EV charger or EV marking), "empty:reserved", "empty:compact", "empty:fire_lane", "empty:no_parking" or '
    '"empty:loading". Never empty stalls: hatched (striped) areas beside handicap stalls, the space in front of a '
    "car we face, and gaps in a row behind a curb or island or across another aisle. Reply as "
    '{"in_lot": true|false, "left": {"beside": "stalls"|"facing"|"curb"|"lane"|"unclear", "stalls": [...], "end": '
    '"car"|"out_of_view"|"island"|"cross_aisle"|"row_end"|"curb", "confidence": "high"|"medium"|"low"}, "right": '
    "{...the same...}}; in_lot is true inside a parking lot or garage, false on a street; confidence is how sure "
    "you are of that side."
)
ROW_ENDS = ("island", "cross_aisle", "row_end", "curb")  # visible boundaries after an open run of empty stalls

# v4: v3 plus enlarged close-ups of the left and right parts of the frame (more pixels per stall).
CROPS_NOTE = (
    ' The two close-ups are the left and right parts of the same "Now" image, enlarged: use them to check each '
    "stall you list."
)
CROP_X = 0.55  # each close-up spans this share of the width (they overlap in the middle)...
CROP_Y = (0.2, 0.95)  # ...and this band of the height (sky and hood cut off)
CROP_SCALE = 1.5


def side_crops(image: np.ndarray, scale: float = CROP_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """Enlarged close-ups of the left and right parts of a frame (the rows beside our lane)."""
    h, w = image.shape[:2]
    y0, y1 = round(CROP_Y[0] * h), round(CROP_Y[1] * h)
    cut = round(CROP_X * w)
    left, right = image[y0:y1, :cut], image[y0:y1, w - cut :]
    size = (round(cut * scale), round((y1 - y0) * scale))
    return (
        cv2.resize(left, size, interpolation=cv2.INTER_CUBIC),
        cv2.resize(right, size, interpolation=cv2.INTER_CUBIC),
    )


@dataclass(frozen=True, slots=True)
class ScanPrompt:
    """One prompt version: the system and question text, the reply format, and what images go with it."""

    version: str
    system: str
    text: str
    kind: str = "spots"  # reply format: "spots" (a list of open stalls) | "rows" (each row listed stall by stall)
    earlier_s: float | None = None  # also send the frame this long before (motion context), or None
    crops: bool = False  # also send enlarged close-ups of the left and right parts of the frame
    max_tokens: int = 220

    def request(self, now: np.ndarray, earlier: np.ndarray | None, vision: VisionConfig) -> VisionRequest:
        """The request for one scan: the frame (full width), plus the earlier frame (smaller) and the side close-ups
        if this version uses them."""
        images = [ImagePart(encode_jpeg(now, vision.latest_width, vision.jpeg_quality), "Now")]
        if self.earlier_s is not None and earlier is not None:
            label = f"{self.earlier_s:.1f} s earlier"
            images.insert(0, ImagePart(encode_jpeg(earlier, vision.older_width, vision.jpeg_quality), label))
        if self.crops:
            left, right = side_crops(now)
            for crop, label in ((left, "Close-up of the left side"), (right, "Close-up of the right side")):
                images.append(ImagePart(encode_jpeg(crop, crop.shape[1], vision.jpeg_quality), label))
        return VisionRequest(
            system=self.system,
            text=self.text,
            images=tuple(images),
            max_tokens=self.max_tokens,
            temperature=vision.temperature,
        )


PROMPTS: dict[str, ScanPrompt] = {
    "v1": ScanPrompt("v1", SYSTEM_V1, TEXT_V1),
    "v2": ScanPrompt("v2", SYSTEM_V2, TEXT_V2, kind="rows", max_tokens=260),
    "v3": ScanPrompt("v3", SYSTEM_V3, TEXT_V3, kind="rows", max_tokens=300),
    "v3e": ScanPrompt(
        "v3e", SYSTEM_V3, TEXT_V3 + MOTION_NOTE.format(s=0.7), kind="rows", earlier_s=0.7, max_tokens=300
    ),
    "v4": ScanPrompt("v4", SYSTEM_V3, TEXT_V3 + CROPS_NOTE, kind="rows", crops=True, max_tokens=300),
    "v5": ScanPrompt("v5", SYSTEM_V3, TEXT_V5, kind="rows", max_tokens=320),
}

# ------------------------------------------------------------------------------------------------ parsing


@dataclass(frozen=True, slots=True)
class ScanReport:
    """One open spot the model reported on one scan."""

    t: float  # video time of the scanned frame
    side: str
    cars_ahead: int | None  # parked vehicles between us and the spot in its row (None: not given)
    before: str | None  # the parked vehicle right before it, e.g. "white SUV"
    restriction: str
    confidence: str
    stalls: int = 1
    closed: bool | None = None  # (row listings) a parked vehicle is listed after it: a gap, not the row's open end

    @property
    def restricted(self) -> bool:
        """A spot the watch skips (handicap, reserved, fire lane, no parking, loading)."""
        return self.restriction in SKIP


@dataclass(frozen=True, slots=True)
class ScanResult:
    """The parsed reply of one scan."""

    t: float
    in_lot: bool | None
    reports: tuple[ScanReport, ...]
    latency_s: float | None = None  # how long the model took (the announcement comes this much after t)


def _int_or_none(value: Any, lo: int = 0, hi: int = 50) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and lo <= value <= hi:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return _int_or_none(int(value.strip()), lo, hi)
    return None


def _before(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_BEFORE_CHARS or text.lower() in ("null", "none", "unknown", "n/a"):
        return None
    return text


_RESTRICTION_ALIASES = {
    "disabled": "handicap",
    "accessible": "handicap",
    "wheelchair": "handicap",
    "charging": "ev",
    "electric": "ev",
    "fire": "fire_lane",
    "no_park": "no_parking",
}


def _restriction(value: Any) -> str:
    """One of RESTRICTIONS ("none" when absent; "unknown" when it can't be understood)."""
    if value is None:
        return "none"
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return "none"
    if text in RESTRICTIONS:
        return text
    for name in RESTRICTIONS:
        if name not in ("none", "unknown") and name in text:
            return name
    return next((r for alias, r in _RESTRICTION_ALIASES.items() if alias in text), "unknown")


def _side(value: Any) -> str | None:
    text = str(value).strip().lower() if isinstance(value, str) else ""
    return text if text in SIDES else None


def _confidence(value: Any) -> str | None:
    text = str(value).strip().lower() if isinstance(value, str) else ""
    return text if text in CONFIDENCES else None


def parse_spots(data: dict[str, Any], t: float) -> ScanResult:
    """A ScanResult from a "spots" reply: {"in_lot": bool, "spots": [{side, cars_ahead, before, stalls, restriction,
    confidence}]}. Entries without a valid side or confidence are dropped; the rest are cleaned up."""
    in_lot = data.get("in_lot")
    spots = data.get("spots")
    reports: list[ScanReport] = []
    for item in spots if isinstance(spots, list) else []:
        if not isinstance(item, dict):
            continue
        side, confidence = _side(item.get("side")), _confidence(item.get("confidence"))
        if side is None or confidence is None:
            continue
        reports.append(
            ScanReport(
                t=t,
                side=side,
                cars_ahead=_int_or_none(item.get("cars_ahead")),
                before=_before(item.get("before")),
                restriction=_restriction(item.get("restriction", "none")),
                confidence=confidence,
                stalls=_int_or_none(item.get("stalls"), 1, 50) or 1,
            )
        )
    return ScanResult(t, in_lot if isinstance(in_lot, bool) else None, tuple(reports))


_EMPTY = re.compile(r"^\s*(?:empty|open|free)\b[\s:(\-]*(.*?)\)?\s*$", re.IGNORECASE)


def parse_rows(data: dict[str, Any], t: float) -> ScanResult:
    """A ScanResult from a "rows" reply: {"in_lot": bool, "left": {"stalls": [...], "confidence": ...}, "right":
    {...}}, each row listed nearest first as vehicles ("white SUV") and "empty" (or "empty:handicap", ...).

    Each run of empty stalls (with one restriction) is a report: cars_ahead = vehicles listed before it, before =
    the last vehicle before it, stalls = the run's length. A side's runs are reported up to its first unrestricted
    one (the nearest spot the watch could announce); later ones would only be announced once we get there.
    (v3+) A side whose "beside" isn't "stalls" has no row beside our lane: nothing. (v5) A run at the end of the
    listing (no vehicle after it) counts only if "end" names a visible row end (ROW_ENDS)."""
    in_lot = data.get("in_lot")
    reports: list[ScanReport] = []
    for side in SIDES:
        row = data.get(side)
        if not isinstance(row, dict):
            continue
        confidence = _confidence(row.get("confidence"))
        items = row.get("stalls")
        if confidence is None or not isinstance(items, list):
            continue
        beside = row.get("beside")
        if beside is not None and str(beside).strip().lower() != "stalls":
            continue
        end = row.get("end")
        bounded = end is None or str(end).strip().lower().replace(" ", "_") in ROW_ENDS
        runs, total = _empty_runs(items)
        for cars, before, restriction, stalls in runs:
            closed = cars < total
            if not closed and not bounded:
                break  # empty stalls running out of view: not trusted
            reports.append(ScanReport(t, side, cars, before, restriction, confidence, stalls, closed=closed))
            if restriction not in SKIP:
                break  # the nearest spot the watch could announce
    return ScanResult(t, in_lot if isinstance(in_lot, bool) else None, tuple(reports))


def _empty_runs(items: list[Any]) -> tuple[list[tuple[int, str | None, str, int]], int]:
    """(vehicles before it, the last vehicle before it, restriction, length) of each run of empty stalls with one
    restriction in a row listing (nearest first; a malformed item ends the listing), and the vehicles listed."""
    runs: list[tuple[int, str | None, str, int]] = []
    cars, before = 0, None
    restriction, length = "none", 0
    for item in items:
        if not isinstance(item, str):
            break
        empty = _EMPTY.match(item)
        if empty is None:
            if length:
                runs.append((cars, before, restriction, length))
                length = 0
            cars += 1
            before = _before(item)
            continue
        kind = _restriction(empty.group(1)) if empty.group(1).strip() else "none"
        if length and kind != restriction:
            runs.append((cars, before, restriction, length))
            length = 0
        restriction = kind
        length += 1
    if length:
        runs.append((cars, before, restriction, length))
    return runs, cars


def parse_scan(data: dict[str, Any], t: float, kind: str = "spots", latency_s: float | None = None) -> ScanResult:
    """The model's JSON reply (of a prompt of this `kind`) as a ScanResult."""
    result = parse_rows(data, t) if kind == "rows" else parse_spots(data, t)
    return ScanResult(result.t, result.in_lot, result.reports, latency_s)


# --------------------------------------------------------------------------------------------- deciding

_COLOR_GROUPS = {
    "white": "white",
    "black": "black",
    "silver": "gray",
    "gray": "gray",
    "grey": "gray",
    "red": "red",
    "maroon": "red",
    "burgundy": "red",
    "blue": "blue",
    "navy": "blue",
    "teal": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "gold": "tan",
    "beige": "tan",
    "tan": "tan",
    "brown": "tan",
}


def colors(description: str | None) -> frozenset[str]:
    """The colour groups named in a vehicle description ("silver" and "gray" are one group)."""
    if not description:
        return frozenset()
    words = re.findall(r"[a-z]+", description.lower())
    return frozenset(_COLOR_GROUPS[w] for w in words if w in _COLOR_GROUPS)


def different_cars(a: str | None, b: str | None) -> bool:
    """Both descriptions name colours and they share none (so they can't be the same car)."""
    ca, cb = colors(a), colors(b)
    return bool(ca) and bool(cb) and not (ca & cb)


@dataclass(frozen=True, slots=True)
class ScanDecision:
    """How scan reports become announcements (eval defaults; config.yaml `parking:` if this variant ships)."""

    min_confidence: str = "medium"  # reports below this are ignored
    confirm_scans: int = 1  # agreeing scans needed before a spot is announced
    track_gap_s: float = 4.5  # a report this soon after a spot's last sighting can be the same spot
    count_slack: int = 1  # the same spot's cars_ahead may grow by this much (model noise)...
    max_drop_per_s: float = 1.5  # ...or shrink by up to this many cars per second (we drive past them) + slack
    require_lot: bool = False  # ignore scans where the model says we are not in a lot
    use_before: bool = True  # differently coloured cars right before = different spots
    require_closed: bool = False  # (row listings) only gaps with a parked vehicle listed after them
    require_neighbour: bool = False  # only spots with a parked vehicle right before them (cars_ahead >= 1)

    def __post_init__(self) -> None:
        if self.min_confidence not in CONFIDENCES:
            raise ValueError(f"min_confidence must be one of {CONFIDENCES}")
        if self.confirm_scans < 1:
            raise ValueError("confirm_scans must be >= 1")

    def accepts(self, report: ScanReport) -> bool:
        """Confident enough, not a restriction the watch skips, and (if asked) a gap between parked vehicles."""
        if CONFIDENCES.index(report.confidence) < CONFIDENCES.index(self.min_confidence) or report.restricted:
            return False
        if self.require_closed and report.closed is False:
            return False
        return not (self.require_neighbour and not report.cars_ahead)

    def label(self) -> str:
        """A short name for tables."""
        flags = "".join(
            f" {name}" for name, on in (("closed", self.require_closed), ("neighbour", self.require_neighbour)) if on
        )
        return (
            f"conf>={self.min_confidence} confirm={self.confirm_scans} gap={self.track_gap_s:g} "
            f"lot={'y' if self.require_lot else 'n'} before={'y' if self.use_before else 'n'}{flags}"
        )


@dataclass(frozen=True, slots=True)
class Announcement:
    """An open spot the app would announce (one per distinct spot)."""

    t: float  # time of the scan that confirmed it (what the model saw)
    first_t: float  # its first sighting
    report: ScanReport  # the confirming report
    latency_s: float | None = None  # the confirming call's latency: it is spoken at t + latency

    @property
    def spoken_t(self) -> float:
        """When it would be spoken (the model's reply arrives `latency_s` after the frame)."""
        return self.t + (self.latency_s or 0.0)

    @property
    def text(self) -> str:
        """What the watch would say."""
        return describe_report(self.report)


@dataclass(slots=True)
class _Track:
    """One spot followed across scans."""

    side: str
    first_t: float
    last_t: float
    cars_ahead: int | None
    before: str | None
    hits: int = 1
    announced: bool = False


class SpotAnnouncer:
    """Turns a stream of scan results (in time order) into announcements, once per distinct spot."""

    def __init__(self, decision: ScanDecision | None = None) -> None:
        self.decision = decision or ScanDecision()
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        """Forget every spot (a new watch)."""
        self._tracks = []

    def update(self, result: ScanResult) -> list[Announcement]:
        """The announcements this scan triggers."""
        d = self.decision
        self._tracks = [tr for tr in self._tracks if result.t - tr.last_t <= d.track_gap_s]
        if d.require_lot and result.in_lot is False:
            return []
        reports = sorted(
            (r for r in result.reports if d.accepts(r)),
            key=lambda r: (r.cars_ahead is None, r.cars_ahead or 0),
        )
        out: list[Announcement] = []
        used: set[int] = set()
        for r in reports:
            track = self._continued(r, used)
            if track is None:
                track = _Track(r.side, r.t, r.t, r.cars_ahead, r.before)
                self._tracks.append(track)
            else:
                track.hits += 1
                track.last_t = r.t
                track.cars_ahead = r.cars_ahead if r.cars_ahead is not None else track.cars_ahead
                track.before = r.before or track.before
            used.add(id(track))
            if not track.announced and track.hits >= d.confirm_scans:
                track.announced = True
                out.append(Announcement(r.t, track.first_t, r, result.latency_s))
        return out

    def _continued(self, r: ScanReport, used: set[int]) -> _Track | None:
        """The track this report continues (the closest in cars ahead), or None for a new spot."""
        d = self.decision
        best: tuple[float, _Track] | None = None
        for tr in self._tracks:
            dt = r.t - tr.last_t
            if tr.side != r.side or id(tr) in used or dt <= 0 or dt > d.track_gap_s:
                continue
            gap = 0.0
            if tr.cars_ahead is not None and r.cars_ahead is not None:
                if r.cars_ahead > tr.cars_ahead + d.count_slack:
                    continue  # further ahead than this spot: another one
                if r.cars_ahead < tr.cars_ahead - d.count_slack - d.max_drop_per_s * dt:
                    continue
                gap = abs(r.cars_ahead - tr.cars_ahead)
            if d.use_before and different_cars(r.before, tr.before):
                continue
            if best is None or (gap, -tr.last_t) < (best[0], -best[1].last_t):
                best = (gap, tr)
        return best[1] if best is not None else None


def announce(results: Iterable[ScanResult], decision: ScanDecision | None = None) -> list[Announcement]:
    """Every announcement of a scan stream (sorted by time first)."""
    announcer = SpotAnnouncer(decision)
    out: list[Announcement] = []
    for result in sorted(results, key=lambda r: r.t):
        out += announcer.update(result)
    return out


def spoken_count(n: int) -> str:
    """'two' for 2 (digits past ten)."""
    return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else str(n)


def describe_report(report: ScanReport) -> str:
    """ "Open spot on your left, two cars ahead, just after the white SUV." (as answers/parking.describe)"""
    kind = {"ev": "EV charging spot", "compact": "compact spot"}.get(report.restriction, "spot")
    if report.stalls > 1:
        text = f"{spoken_count(report.stalls).capitalize()} open {kind}s on your {report.side}"
    else:
        text = f"Open {kind} on your {report.side}"
    if report.cars_ahead is not None and report.cars_ahead > 1:
        text += f", {spoken_count(report.cars_ahead)} cars ahead"
    if report.before:
        text += f", just after the {report.before}"
    return text + "."


# ---------------------------------------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class AnnouncementMatch:
    """One announcement and what it matched in the labels."""

    announcement: Announcement
    spot: LabeledSpot | None
    look_alike: LookAlike | None
    repeat: bool = False  # the labeled spot was already announced

    @property
    def false_alarm(self) -> bool:
        """Matches no labeled spot."""
        return self.spot is None


@dataclass(slots=True)
class ScanClipScore:
    """One clip's announcements tied to its labels."""

    clip: ClipLabels
    duration: float
    scans: int
    matches: list[AnnouncementMatch] = field(default_factory=list)
    hits: dict[int, Announcement] = field(default_factory=dict)  # target index -> its first matching announcement

    @property
    def targets(self) -> list[int]:
        """Indexes of the recall targets (clear and unrestricted, or EV / compact)."""
        return [i for i, s in enumerate(self.clip.spots) if s.target]

    @property
    def false_alarms(self) -> list[AnnouncementMatch]:
        """Announcements that hit no labeled spot."""
        return [m for m in self.matches if m.false_alarm]


def score_scan(
    clip: ClipLabels,
    duration: float,
    results: Sequence[ScanResult],
    decision: ScanDecision | None = None,
    early_s: float = 1.0,
    late_s: float = 0.5,
) -> ScanClipScore:
    """Run the announcer over a clip's scans and tie each announcement to the labels with `match_report` (the same
    tolerances as the geometry tracker's spot events, at the time of the frame the model saw)."""
    score = ScanClipScore(clip, duration, len(results))
    index = {id(s): i for i, s in enumerate(clip.spots)}
    seen: set[int] = set()
    for a in announce(results, decision):
        spot, look = match_report(a.t, a.report.side, clip, duration, early_s, late_s)
        repeat = spot is not None and id(spot) in seen
        if spot is not None:
            seen.add(id(spot))
            if spot.target:
                score.hits.setdefault(index[id(spot)], a)
        score.matches.append(AnnouncementMatch(a, spot, look, repeat))
    return score


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """Totals over clips."""

    clips: int
    scans: int
    targets: int
    found: int
    targets_with_neighbour: int
    found_with_neighbour: int
    announcements: int
    false_alarms: int
    look_alike_hits: int
    repeats: int
    restricted: int  # announcements of a labeled spot the watch must skip (handicap, reserved, ...)
    lot_minutes: float
    delays: tuple[float, ...]  # confirming frame time − t_visible, per found target
    spoken_delays: tuple[float, ...]  # ... + the call's latency (when it would be spoken)
    late: int  # found targets spoken only after they left the frame (t_pass, or the clip's end)

    @property
    def recall(self) -> float:
        """Share of the targets announced."""
        return self.found / self.targets if self.targets else float("nan")

    @property
    def recall_with_neighbour(self) -> float:
        """Recall over targets with a parked car right before them."""
        n = self.targets_with_neighbour
        return self.found_with_neighbour / n if n else float("nan")

    @property
    def recall_without_neighbour(self) -> float:
        """Recall over targets at a row start or after a gap (no car right before them)."""
        n = self.targets - self.targets_with_neighbour
        return (self.found - self.found_with_neighbour) / n if n else float("nan")

    @property
    def false_per_10min(self) -> float:
        """False announcements per 10 minutes of lot driving."""
        return 10.0 * self.false_alarms / self.lot_minutes if self.lot_minutes else float("nan")

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly."""
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        d["delays"] = list(self.delays)
        d["spoken_delays"] = list(self.spoken_delays)
        for k in ("recall", "recall_with_neighbour", "recall_without_neighbour", "false_per_10min"):
            d[k] = getattr(self, k)
        d["delay_median"] = median(self.delays)
        d["spoken_delay_median"] = median(self.spoken_delays)
        return d


def median(values: Sequence[float]) -> float:
    """The median (nan for none)."""
    return float(np.median(values)) if len(values) else float("nan")


def summarize_scan(scores: Sequence[ScanClipScore]) -> ScanSummary:
    """Totals over the scored clips."""
    targets = found = with_n = found_n = events = false = looks = repeats = late = scans = restricted = 0
    delays: list[float] = []
    spoken: list[float] = []
    lot_s = 0.0
    for s in scores:
        clip = s.clip
        lot_s += clip.lot_s
        scans += s.scans
        for i in s.targets:
            spot = clip.spots[i]
            has_n = spot.neighbor_before is not None
            targets += 1
            with_n += has_n
            a = s.hits.get(i)
            if a is None:
                continue
            found += 1
            found_n += has_n
            delays.append(a.t - spot.t_visible)
            spoken.append(a.spoken_t - spot.t_visible)
            late += a.spoken_t > (s.duration if spot.t_pass is None else spot.t_pass)
        events += len(s.matches)
        false += len(s.false_alarms)
        looks += sum(m.look_alike is not None for m in s.false_alarms)
        repeats += sum(m.repeat for m in s.matches)
        restricted += sum(m.spot is not None and m.spot.restriction in SKIP for m in s.matches)
    return ScanSummary(
        clips=len(scores),
        scans=scans,
        targets=targets,
        found=found,
        targets_with_neighbour=with_n,
        found_with_neighbour=found_n,
        announcements=events,
        false_alarms=false,
        look_alike_hits=looks,
        repeats=repeats,
        restricted=restricted,
        lot_minutes=lot_s / 60.0,
        delays=tuple(delays),
        spoken_delays=tuple(spoken),
        late=late,
    )
