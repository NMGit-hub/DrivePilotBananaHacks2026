"""Traffic-light state from YOLO "traffic light" boxes: which lamp is lit, and when it turns green.

Per light, per frame (`classify_light`): find the lit lamp (a compact, round spot much brighter than its own
housing, inside the housing) and read its color from the spot and a thin rim of saturated pixels. LED greens
look cyan on camera (hue up to 100); the overexposed core of a red LED looks yellow-white, so the rim matters.
For upright housings the lamp's position must agree with its color (red on top, green at the bottom). In
sunlit, brightly painted housings a glint on the paint can look like a lamp, so there a lamp only counts if its
color differs from the paint.

Over time, each light is followed individually (`TrafficLightTracker`):
- by track id while we move, and by a fixed *anchor* box while we are stopped. A distant light is only ~8x22 px
  at the detector's input size and YOLO often loses it after it changes (dev clip: detected while red, never
  while green); with the camera still, the light stays at the same pixels, so an anchored light is read every
  frame even when YOLO misses it. Anchor boxes are padded because YOLO boxes sometimes clip the bottom lamp.
- each light keeps its own majority vote over `stable_s`, so one misread frame doesn't flip it.
- "turned green" fires once per stop when an individual light goes from stable red/yellow to
  `green_confirm_frames` consecutive green readings while we are stopped (cross-street lights seen side-on
  change the other way and are judged separately instead of being mixed into one vote).
- the overall reading answers "what color is our light": the most prominent light (box area x detection
  confidence x lamp brightness: side-on lamps are dim). If two prominent lights disagree, it is not confident.
Glare and distance make this unreliable at times: answers say so (see `LightReading.confident`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from core.config import TrafficLightConfig
from perception.detector import Track

Box = tuple[float, float, float, float]


class LightColor(StrEnum):
    """Lit lamp color."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LightClassification:
    """One light, one frame."""

    color: LightColor
    confidence: float  # 0..1 share of the rim hue votes for the color (reduced if the lamp position disagrees)
    scores: dict[str, float]
    lamp_brightness: float = 0.0  # mean brightness (0..1) of the lit lamp: frontal lamps are brighter


@dataclass(frozen=True, slots=True)
class LightReading:
    """Smoothed state of the light facing us."""

    t: float
    color: LightColor
    confident: bool  # a clear, stable reading with no strong disagreement between lights
    n_lights: int  # lights considered this frame
    boxes: tuple[Box, ...]  # the lights considered (for crops / model questions), most prominent first


def _lamp_blob(val: np.ndarray, threshold: float) -> np.ndarray | None:
    """The lit lamp: the brightest compact, roughly round bright spot that doesn't touch the crop border.

    Rejects bright sky or background showing around the housing (touches the border / irregular) and painted
    housings in sunshine (no spot clearly brighter than the housing itself).
    """
    mask = (val >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = val.shape
    best, best_v = None, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # touches the border: background, not a lamp inside the housing
        if area < 3 or not 0.5 <= bw / max(bh, 1) <= 2.0 or area < 0.45 * bw * bh:
            continue  # not compact/round enough
        if bw > 0.9 * w:
            continue
        component = labels == i
        mean_v = float(val[component].mean())
        if mean_v > best_v:
            best, best_v = component, mean_v
    return best


def _hue_counts(hues: np.ndarray) -> dict[str, float]:
    """Votes of OpenCV hues (0..179) for each lamp color."""
    return {
        "red": float(((hues <= 9) | (hues >= 165)).sum()),
        "yellow": float(((hues >= 14) & (hues <= 34)).sum()),
        "green": float(((hues >= 45) & (hues <= 100)).sum()),
    }


def classify_light(image: np.ndarray, box: Box, cfg: TrafficLightConfig) -> LightClassification:
    """Classify the lit lamp of one traffic light (see the module docstring)."""
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return LightClassification(LightColor.UNKNOWN, 0.0, {})
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    housing_v = float(np.median(val))
    # A lit lamp clips near 255, so in sunlit paint (housing nearly as bright) the contrast rule is capped.
    bright_housing = housing_v + cfg.lamp_contrast > cfg.lamp_v_cap
    threshold = max(cfg.v_min, min(housing_v + cfg.lamp_contrast, cfg.lamp_v_cap))
    blob = _lamp_blob(val, threshold)
    if blob is None:
        return LightClassification(LightColor.UNKNOWN, 0.0, {})
    lamp_frac = float(blob.mean())
    brightness = float(val[blob].mean()) / 255.0
    if lamp_frac < cfg.min_lit_frac or lamp_frac > cfg.max_lamp_frac:
        return LightClassification(LightColor.UNKNOWN, 0.0, {"lamp_frac": lamp_frac}, brightness)

    rim = max(1, round(0.06 * min(blob.shape)))
    near = cv2.dilate(blob.astype(np.uint8), np.ones((2 * rim + 1, 2 * rim + 1), np.uint8)).astype(bool)
    saturated = (sat >= cfg.s_min_halo) & (val >= cfg.v_halo_min)
    counts = _hue_counts(hue[near & saturated])
    total = sum(counts.values())
    if total < 3:
        return LightClassification(LightColor.UNKNOWN, 0.0, {"lamp_frac": lamp_frac}, brightness)
    scores = {k: v / total for k, v in counts.items()}
    best = max(scores, key=lambda k: scores[k])
    confidence = scores[best]

    if bright_housing:
        # A highlight on sunlit paint looks like a lit lamp of the paint's color: believe only a lamp whose color
        # differs from the paint (a red lamp in a yellow housing, not "yellow" in a yellow housing).
        paint = _hue_counts(hue[~near & saturated])
        paint_color = max(paint, key=lambda k: paint[k])
        if sum(paint.values()) >= 0.25 * hue.size and paint_color == best:
            return LightClassification(LightColor.UNKNOWN, 0.0, scores, brightness)

    crop_h, crop_w = hue.shape
    if crop_h >= 1.6 * crop_w:  # upright housing: the lit lamp's position must match its color
        ys = np.nonzero(blob)[0]
        rel_y = float(ys.mean()) / crop_h
        expected = {"red": (0.0, 0.5), "yellow": (0.25, 0.75), "green": (0.45, 1.0)}[best]
        if not expected[0] <= rel_y <= expected[1]:
            confidence *= 0.3
    if confidence < cfg.min_confidence:
        return LightClassification(LightColor.UNKNOWN, confidence, scores, brightness)
    return LightClassification(LightColor(best), confidence, scores, brightness)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def _full_housing(box: Box) -> Box:
    """Extend an upright light's box down to a full 3-lamp housing (~3x as tall as wide).

    YOLO's box for a red light sometimes covers only the top lamps; an anchor must include the green lamp below.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    if y2 - y1 >= 1.2 * width:  # upright (a horizontal housing is left as is)
        y2 = max(y2, y1 + 3.0 * width)
    return x1, y1, x2, y2


def _pad(box: Box, frac: float, w: int, h: int) -> Box:
    """Grow a box by `frac` of its size on each side (YOLO boxes sometimes clip the bottom lamp)."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(w), x2 + dx), min(float(h), y2 + dy)


@dataclass(slots=True)
class _LightRecord:
    """History of one physical light."""

    box: Box
    track_id: int | None
    anchored: bool = False  # box fixed while we are stopped (read even when YOLO misses it)
    last_seen: float = 0.0  # last time YOLO detected it
    det_conf: float = 0.0
    votes: deque[tuple[float, LightColor]] = field(default_factory=deque)
    stable: LightColor = LightColor.UNKNOWN
    stable_share: float = 0.0
    last_red_t: float | None = None
    green_streak: int = 0
    prominence: float = 0.0  # area x detection confidence x lamp brightness (EMA)


class TrafficLightTracker:
    """Follows each relevant traffic light, reports the state of ours and detects the turn to green."""

    def __init__(self, cfg: TrafficLightConfig | None = None) -> None:
        self.cfg = cfg or TrafficLightConfig()
        self.reset()

    def reset(self) -> None:
        """Forget history (new segment)."""
        self._records: list[_LightRecord] = []
        self._green_reported = False  # one "turned green" per stop

    def relevant(self, tracks: list[Track], frame_w: int, frame_h: int) -> list[Track]:
        """Traffic lights that plausibly face our lane (by position and size)."""
        cfg = self.cfg
        out = []
        for track in tracks:
            if track.cls != "traffic light":
                continue
            cx, cy = track.center
            if (
                abs(cx / frame_w - 0.5) <= cfg.center_band
                and cy / frame_h <= cfg.max_center_y
                and track.height / frame_h >= cfg.min_height_frac
            ):
                out.append(track)
        return out

    def update(
        self, t: float, image: np.ndarray, tracks: list[Track], ego_stopped: bool = False
    ) -> tuple[LightReading, bool]:
        """Process one frame. Returns (reading of our light, True if a light just turned green while stopped)."""
        cfg = self.cfg
        h, w = image.shape[:2]
        if not ego_stopped:
            # The camera moves: anchors are invalid, and a new stop gets a new "turned green" announcement.
            self._records = [r for r in self._records if not r.anchored]
            self._green_reported = False

        seen: set[int] = set()
        for light in self.relevant(tracks, w, h):
            record = self._match(light)
            if record is None:
                record = _LightRecord(box=light.box, track_id=light.track_id)
                self._records.append(record)
            seen.add(id(record))
            record.last_seen, record.det_conf, record.track_id = t, light.conf, light.track_id
            if not record.anchored:
                record.box = light.box

        turned_green = False
        for record in list(self._records):
            detected_now = id(record) in seen
            if not detected_now and not record.anchored:
                if t - record.last_seen > cfg.stable_s:
                    self._records.remove(record)  # out of sight for a while (moving): forget
                continue
            if record.anchored:
                reading = classify_light(image, _pad(record.box, cfg.anchor_pad, w, h), cfg)
                if reading.color is LightColor.UNKNOWN:
                    # The box seen while red may cover only the top lamps: try the full housing below it.
                    reading = classify_light(image, _pad(_full_housing(record.box), cfg.anchor_pad, w, h), cfg)
            else:
                reading = classify_light(image, record.box, cfg)
            self._vote(record, t, reading)
            if ego_stopped and not record.anchored and record.stable in (LightColor.RED, LightColor.YELLOW):
                record.anchored = True  # remember where it is while we wait
            if (
                ego_stopped
                and not self._green_reported
                and record.last_red_t is not None
                and t - record.last_red_t <= cfg.red_memory_s
                and record.green_streak >= cfg.green_confirm_frames
            ):
                turned_green, self._green_reported = True, True
                record.stable, record.stable_share = LightColor.GREEN, 1.0

        return self._overall(t, turned_green), turned_green

    def _match(self, light: Track) -> _LightRecord | None:
        for record in self._records:
            if record.track_id is not None and record.track_id == light.track_id and not record.anchored:
                return record
        best, best_iou = None, 0.3
        for record in self._records:
            iou = _iou(record.box, light.box)
            if iou >= best_iou:
                best, best_iou = record, iou
        return best

    def _vote(self, record: _LightRecord, t: float, reading: LightClassification) -> None:
        cfg = self.cfg
        record.votes.append((t, reading.color))
        while record.votes and t - record.votes[0][0] > cfg.stable_s:
            record.votes.popleft()
        known = [c for _, c in record.votes if c is not LightColor.UNKNOWN]
        if known:
            top = max(set(known), key=known.count)
            share = known.count(top) / len(record.votes)
            if share >= 0.6:
                record.stable, record.stable_share = top, share
        elif t - record.votes[0][0] >= 0.8 * cfg.stable_s:
            record.stable, record.stable_share = LightColor.UNKNOWN, 0.0
        if record.stable in (LightColor.RED, LightColor.YELLOW):
            record.last_red_t = t
        record.green_streak = record.green_streak + 1 if reading.color is LightColor.GREEN else 0
        x1, y1, x2, y2 = record.box
        if reading.color is not LightColor.UNKNOWN:
            prominence = (x2 - x1) * (y2 - y1) * max(record.det_conf, cfg.anchor_weight) * reading.lamp_brightness
            record.prominence = prominence if record.prominence == 0 else 0.8 * record.prominence + 0.2 * prominence

    def _overall(self, t: float, turned_green: bool) -> LightReading:
        ranked = sorted(self._records, key=lambda r: r.prominence, reverse=True)
        known = [r for r in ranked if r.stable is not LightColor.UNKNOWN]
        boxes = tuple(r.box for r in ranked)
        if not known:
            return LightReading(t=t, color=LightColor.UNKNOWN, confident=False, n_lights=len(ranked), boxes=boxes)
        top = known[0]
        rivals = [r for r in known[1:] if r.stable is not top.stable and r.prominence >= 0.6 * top.prominence]
        confident = turned_green or (top.stable_share >= 0.6 and not rivals)
        return LightReading(t=t, color=top.stable, confident=confident, n_lights=len(ranked), boxes=boxes)
