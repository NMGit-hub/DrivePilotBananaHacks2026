"""Signs the fine-tuned detector sees (V2-M6): stable sightings, the current speed limit, and once-only alerts.

The detector (perception/signs.py) runs on its own thread at a few fps and reports, per frame, each sign's kind,
box, confidence and, for speed limits, a value only when it is sure. This tracker:
- links detections across frames into sightings (same kind, overlapping or nearby box, seen again within
  `link_s`), so one sign approached over several frames is one sighting;
- announces a speed limit only when two frames of the same sighting agree on its value (`agree_frames` within
  `agree_window_s`) and it differs from the limit announced last ("Speed limit is now 35."): one frame can be a
  look-alike, and a wrong limit is worse than none (V2_BUILD_PLAN.md V2-M6, "0 wrong values");
- does the same for school speed limits ("School speed limit 20.") and announces a school zone once per zone
  ("School zone ahead."): a zone often has a sign on each side of the road, and one sign can be lost and found
  again as a new sighting (its plaque detected apart, a gap of a few frames), so a confirmed school-zone sign
  starts a zone only after `zone_gap_s` without a confirmed school sign (school zone or school speed limit: both
  extend the zone; a one-frame look-alike does neither);
- keeps the current limit (for the page's badge and the vision model's context) and the recent sightings.
Detections are duck-typed (`SignLike`: kind, value, conf, box) so this module doesn't need the GPU stack.
Called from the sign thread only (`update`); `snapshot` is safe from any thread.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from core.config import SignAlertsConfig

Box = tuple[float, float, float, float]
VALUE_KINDS = ("speed_limit", "school_speed_limit")


class SignLike(Protocol):
    """What the tracker reads from a detection (perception.signs.SignDetection has these)."""

    @property
    def kind(self) -> str: ...
    @property
    def value(self) -> int | None: ...
    @property
    def conf(self) -> float: ...
    @property
    def box(self) -> Box: ...


def limit_text(kind: str, value: int) -> str:
    """The spoken alert for a newly read limit."""
    return f"School speed limit {value}." if kind == "school_speed_limit" else f"Speed limit is now {value}."


SCHOOL_ZONE_TEXT = "School zone ahead."
CROSSING_TEXT = "Pedestrian crossing ahead."


def alert_texts(values: Sequence[int]) -> list[str]:
    """Every alert this tracker can say (the app pre-renders them so they play at once)."""
    texts = [limit_text("speed_limit", v) for v in values]
    texts += [limit_text("school_speed_limit", v) for v in values if v <= 25]
    return [*texts, SCHOOL_ZONE_TEXT, CROSSING_TEXT]


@dataclass(slots=True)
class Sighting:
    """One physical sign, seen over several frames."""

    id: int
    kind: str
    first_t: float
    last_t: float
    box: Box  # latest
    conf: float  # best
    frames: int = 0  # frames it was detected in
    readings: list[tuple[float, int]] = field(default_factory=list)  # (t, value) of confident value reads
    announced: bool = False

    def agreed_value(self, t: float, frames: int, window_s: float) -> int | None:
        """A value read in at least `frames` frames within the last `window_s` (and no other value then)."""
        recent = [v for when, v in self.readings if t - when <= window_s]
        if len(recent) < frames or len(set(recent)) != 1:
            return None
        return recent[0]


@dataclass(frozen=True, slots=True)
class SignEvent:
    """Something to say about a sign."""

    kind: str  # "speed_limit" | "school_speed_limit" | "school_zone"
    t: float
    text: str
    value: int | None
    sighting: int
    box: Box


@dataclass(frozen=True, slots=True)
class SignView:
    """A recent sighting, for the page and answers."""

    id: int
    kind: str
    value: int | None
    t: float
    box: Box


@dataclass(frozen=True, slots=True)
class SignsSnapshot:
    """The current speed limit and the recent signs."""

    t: float
    limit: int | None  # the last announced speed limit (mph)
    limit_t: float | None  # when it was read
    school_limit: int | None
    school_limit_t: float | None
    recent: tuple[SignView, ...]  # newest first


EMPTY = SignsSnapshot(0.0, None, None, None, None, ())


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _near(a: Box, b: Box) -> bool:
    """Box b is where box a's sign could be a moment later (signs grow and slide outwards as we approach)."""
    if box_iou(a, b) > 0.1:
        return True
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    return (dx * dx + dy * dy) ** 0.5 <= 2.0 * size


class SignTracker:
    """Detections from the sign thread in; sightings, the current limit and alert events out."""

    def __init__(self, cfg: SignAlertsConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.reset()

    def reset(self) -> None:
        """Forget everything (a seek or a new segment)."""
        with self._lock:
            self._sightings: list[Sighting] = []
            self._limit: tuple[int, float] | None = None
            self._school: tuple[int, float] | None = None
            self._zone_t: float | None = None  # a school-zone sign of the current zone was last seen then
            self._crossing_t: float | None = None  # "Pedestrian crossing ahead." was last said then
            self._last_t: float | None = None
            self._snapshot = EMPTY
            self.last_frame: list[SignView] = []  # this frame's detections as sightings (memory close-ups)

    def update(self, t: float, detections: Sequence[SignLike]) -> list[SignEvent]:
        """One frame's detections; returns what to say."""
        cfg = self.cfg
        events: list[SignEvent] = []
        seen: list[SignView] = []
        with self._lock:
            if self._last_t is not None and t < self._last_t:  # time went backwards: another moment
                self._sightings, self._limit, self._school, self._zone_t = [], None, None, None
                self._crossing_t = None
            self._last_t = t
            live = [s for s in self._sightings if t - s.last_t <= cfg.link_s]
            for det in sorted(detections, key=lambda d: -d.conf):
                if det.conf < cfg.min_conf:
                    continue
                match = next((s for s in live if s.kind == det.kind and s.last_t < t and _near(s.box, det.box)), None)
                if match is None:
                    match = Sighting(next(self._ids), det.kind, t, t, det.box, det.conf)
                    self._sightings.append(match)
                    live.append(match)
                match.last_t, match.box, match.conf = t, det.box, max(match.conf, det.conf)
                match.frames += 1
                if det.kind in VALUE_KINDS and det.value is not None:
                    match.readings.append((t, int(det.value)))
                events += self._decide(match, t)
                agreed = match.agreed_value(t, cfg.agree_frames, cfg.agree_window_s * 4)
                seen.append(SignView(match.id, match.kind, agreed, t, det.box))
            self.last_frame = seen
            self._sightings = [s for s in self._sightings if t - s.last_t <= cfg.keep_s]
            self._snapshot = self._make_snapshot(t)
        return events

    def _decide(self, s: Sighting, t: float) -> list[SignEvent]:
        """An alert for this sighting, if it has earned one (caller holds the lock)."""
        cfg = self.cfg
        if s.kind in ("school_zone", "school_speed_limit") and s.frames >= cfg.agree_frames:
            # A confirmed school sign of either kind: are we already in a school zone? Then it extends the zone
            # (a one-frame look-alike neither starts nor extends one).
            in_zone = self._zone_t is not None and t - self._zone_t <= cfg.zone_gap_s
            self._zone_t = t
            if s.kind == "school_zone":
                if s.announced:
                    return []
                s.announced = True  # (said, or quiet because its zone was said)
                return [] if in_zone else [SignEvent("school_zone", t, SCHOOL_ZONE_TEXT, None, s.id, s.box)]
        if s.kind == "school_zone":
            return []
        if s.kind == "pedestrian_crossing":  # confirmed (a one-frame look-alike says nothing), once per sign
            if s.announced or s.frames < cfg.agree_frames or not cfg.crossing:
                return []
            s.announced = True
            if self._crossing_t is not None and t - self._crossing_t <= cfg.crossing_repeat_s:
                return []  # (one said a moment ago: a crossing at every corner isn't news)
            self._crossing_t = t
            return [SignEvent("pedestrian_crossing", t, CROSSING_TEXT, None, s.id, s.box)]
        if s.announced:
            return []
        if s.kind not in VALUE_KINDS:
            return []
        value = s.agreed_value(t, cfg.agree_frames, cfg.agree_window_s)
        if value is None:
            return []
        s.announced = True
        if s.kind == "speed_limit":
            same = self._limit is not None and self._limit[0] == value
            self._limit = (value, t)
        else:
            same = self._school is not None and self._school[0] == value and t - self._school[1] <= cfg.repeat_s
            self._school = (value, t)
        if same:
            return []  # the limit we already said (a repeat sign): keep quiet
        return [SignEvent(s.kind, t, limit_text(s.kind, value), value, s.id, s.box)]

    def _make_snapshot(self, t: float) -> SignsSnapshot:
        recent = sorted(self._sightings, key=lambda s: -s.last_t)[: self.cfg.recent]
        views = tuple(
            SignView(
                s.id,
                s.kind,
                s.agreed_value(s.last_t, self.cfg.agree_frames, self.cfg.agree_window_s * 4),
                s.last_t,
                s.box,
            )
            for s in recent
        )
        return SignsSnapshot(
            t=t,
            limit=None if self._limit is None else self._limit[0],
            limit_t=None if self._limit is None else self._limit[1],
            school_limit=None if self._school is None else self._school[0],
            school_limit_t=None if self._school is None else self._school[1],
            recent=views,
        )

    def snapshot(self) -> SignsSnapshot:
        """The latest state (any thread)."""
        with self._lock:
            return self._snapshot

    def context(self, now: float) -> list[str]:
        """Lines for the vision model's prompt: what the sign detector read recently."""
        snap = self.snapshot()
        lines = []
        if snap.limit is not None and snap.limit_t is not None:
            lines.append(
                f"The sign detector read a speed limit of {snap.limit} mph {max(0.0, now - snap.limit_t):.0f} s ago."
            )
        if snap.school_limit is not None and snap.school_limit_t is not None and now - snap.school_limit_t <= 300:
            ago = max(0.0, now - snap.school_limit_t)
            lines.append(f"It read a school speed limit of {snap.school_limit} mph {ago:.0f} s ago.")
        kinds = [v.kind.replace("_", " ") for v in snap.recent if now - v.t <= 15.0 and v.kind not in VALUE_KINDS]
        if kinds:
            lines.append("Signs it saw in the last 15 s: " + ", ".join(dict.fromkeys(kinds)) + ".")
        return lines
