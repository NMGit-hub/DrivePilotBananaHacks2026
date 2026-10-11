"""Build the sign detector and value-classifier datasets from the downloaded sources.

    .venv/Scripts/python scripts/signs/prepare_data.py det     # YOLO detection dataset -> data/signs/det/
    .venv/Scripts/python scripts/signs/prepare_data.py cls     # value-classifier crops  -> data/signs/cls/

Detector data (all at the scale the app runs the detector: frames resized to 1280 px wide, never upscaled):
- BDD100K train frames with Claude pseudo-labels (scripts/signs/label_bdd_signs.py). Boxes labeled "unclear"
  are painted over (masked) so a sign of unknown kind never trains as background.
- ctacke/speed-limit-signs train split (US dashcam, MIT): every box is a speed limit (value from its folder).
- LISA (if downloaded to data/signs/raw/lisa_yolo/): its labels mapped by perception/sign_classes.kind_from_label.
- Synthetic composites: MUTCD renders pasted into BDD frames that hold none of our signs (scripts/signs/synth.py).
Frames are cut into overlapping 640x640 tiles at native resolution (signs stay as small as the app sees them;
the app runs the detector on the whole 1280-wide frame, which a fully convolutional network allows).

Classifier data: square crops (detector-like box jitter, 15% padding, 96x96) labeled with the mph value
(v5 ... v85) or "none" (other signs, background): real BDD/ctacke/LISA speed limits plus synthetic ones for
every value.

The held-out sets are never used: BDD100K "valid" (hand-checked evaluation set), ctacke "val" (public split) and
the BDD ids of the dev clips (eval/dev_clips.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.sign_classes import KIND_INDEX, KINDS, SPEED_VALUES, VALUE_KINDS, kind_from_label  # noqa: E402

DATA = REPO_ROOT / "data" / "signs"
BDD_IMAGES = DATA / "raw" / "bdd100k_det" / "data" / "images"
CTACKE = DATA / "raw" / "ctacke" / "data"
LISA = DATA / "raw" / "lisa_yolo"
LABELS = DATA / "bdd_labels"
DET_OUT = DATA / "det"
CLS_OUT = DATA / "cls"
TILE = 640
VAL_FRAC = 0.06  # BDD train images held out (by a stable hash of the id) for val and threshold tuning
HARDNEG_DET = DATA / "hardneg_det.txt"  # mine_det_negatives.py, then curate_det_negatives.py (checked by eye)
UNBOXED = DATA / "unboxed_det.jsonl"  # scripts/signs/mine_unboxed.py: real signs BDD never boxed (masked)
WIDTH = 1280  # frames are resized to this width (the app's detector input), never upscaled
MASK_KINDS = frozenset({"unclear"})
SIGN_CLASS = 9


@dataclass
class Box:
    """One labeled box in source-image pixels.

    kind None = mask it in detector images (sign of unknown kind). cls_label overrides the value classifier's label
    for the crop (e.g. "none" for a speed limit too blurred to identify, which is masked for the detector).
    """

    xyxy: tuple[float, float, float, float]
    kind: str | None
    value: int | None = None
    cls_label: str | None = None


@dataclass
class Sample:
    """One source image with its boxes."""

    image: Path
    boxes: list[Box] = field(default_factory=list)
    source: str = ""


def stable_frac(text: str) -> float:
    """A deterministic number in [0, 1) from a string (for train/val splits that never change)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 2**32


def in_val(image: Path) -> bool:
    """Whether a BDD train image is in the held-out fraction (val / threshold tuning): never trained on."""
    return stable_frac(image.stem) < VAL_FRAC


def rel_path(image: Path) -> str:
    """Repo-relative posix path of an image (the key of the training-image records); as is outside the repo."""
    return image.relative_to(REPO_ROOT).as_posix() if image.is_relative_to(REPO_ROOT) else image.as_posix()


USED_FOR_TRAINING = {"det": DATA / "used_for_training_det.txt", "cls": DATA / "used_for_training_cls.txt"}
ORIGINAL_SYNTH = 3500  # synthetic composites of the first detector build (the run the fine-tune starts from)
ORIGINAL_NEG_RATIO = 0.3
CLS_BACKGROUNDS = 400


def touches_any(box: tuple[float, ...], others: list[tuple[float, float, float, float]]) -> bool:
    """Whether a box overlaps any of others (IoU > 0.1) or has its center inside one."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any((o[0] <= cx <= o[2] and o[1] <= cy <= o[3]) or _iou(box, o) > 0.1 for o in others)


def read_unboxed() -> dict[str, list[tuple[float, float, float, float]]]:
    """image (repo-relative) -> unboxed detections to mask in training (empty if not mined).

    Speed-kind detections are never masked: about a third of the unboxed detections are look-alikes (route
    shields, advisory plates, logos), and a number-bearing look-alike must stay a negative for the detector.
    """
    if not UNBOXED.exists():
        return {}
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for line in UNBOXED.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            if "kinds" not in r:
                raise ValueError(f"{UNBOXED.name} lacks detection kinds: re-run scripts/signs/mine_unboxed.py")
            boxes = [tuple(b) for b, k in zip(r["boxes"], r["kinds"], strict=True) if k not in VALUE_KINDS]
            if boxes:
                out[r["image"]] = boxes
    return out


def with_unboxed_masks(sample: Sample, unboxed: dict[str, list[tuple[float, float, float, float]]]) -> Sample:
    """The sample plus a masked box (kind None) for each unboxed detection that touches none of its boxes."""
    extra = [b for b in unboxed.get(rel_path(sample.image), []) if not touches_any(b, [x.xyxy for x in sample.boxes])]
    if not extra:
        return sample
    return Sample(sample.image, [*sample.boxes, *(Box(b, None) for b in extra)], sample.source)


def write_used_for_training(what: str, images: list[Path]) -> None:
    """Record every source image a dataset build put into training (evaluate.py keeps them out of the tune set)."""
    USED_FOR_TRAINING[what].write_text("\n".join(sorted({rel_path(p) for p in images})) + "\n", encoding="utf-8")


def original_build_images() -> set[str]:
    """BDD train images the first detector build and classifiers cls1-cls4 used as backgrounds, in any fraction.

    Those builds took synthetic and classifier backgrounds without checking the val fraction, so some tune-set
    frames were trained on. Reproduced exactly: the first build drew one seed per BDD positive from Random(0),
    then shuffled all negatives (no hard negatives existed), kept int(0.3 * positives) real negatives and took the
    next 3,500 as synthetic backgrounds; the classifier builds took the first 400 negatives unshuffled. Uses the
    labels as they were then (no LABEL_FIXES: a fix can turn a negative frame into a positive).
    """
    pos, neg = bdd_samples("train", fixes=False)
    rng = random.Random(0)
    for _ in pos:
        rng.randrange(2**31)
    shuffled = list(neg)
    rng.shuffle(shuffled)
    n_neg = int(ORIGINAL_NEG_RATIO * len(pos))
    used = shuffled[n_neg : n_neg + ORIGINAL_SYNTH] + neg[:CLS_BACKGROUNDS]
    return {rel_path(s.image) for s in used}


def training_images() -> set[str]:
    """Every BDD image any detector or classifier build trained on (recorded builds + the original build)."""
    out = original_build_images()
    for path in USED_FOR_TRAINING.values():
        if path.exists():
            out |= {line for line in path.read_text(encoding="utf-8").splitlines() if line}
    return out


def dev_clip_ids() -> set[str]:
    """BDD video ids of the dev clips: never trained on."""
    data = yaml.safe_load((REPO_ROOT / "eval" / "dev_clips.yaml").read_text(encoding="utf-8"))
    return {c["bdd_id"] for c in data["clips"]}


def bdd_samples(split: str, fixes: bool = True) -> tuple[list[Sample], list[Sample]]:
    """(pseudo-labeled images with at least one labeled box, labeled images with none of our signs)."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for rec in read_labels(split, fixes):
        by_image[rec["image"]].append(rec)
    if not by_image:
        return [], []
    banned = dev_clip_ids()
    signs = bdd_sign_boxes(split)
    pos: list[Sample] = []
    neg: list[Sample] = []
    for rel, recs in sorted(by_image.items()):
        image = BDD_IMAGES / rel
        if image.stem in banned:
            continue
        boxes = []
        for r in recs:
            if r["kind"] in KIND_INDEX:
                boxes.append(Box(tuple(r["box"]), r["kind"], r["value"]))
            elif r["kind"] in MASK_KINDS:
                boxes.append(Box(tuple(r["box"]), None))
        # Sign boxes too small to have been labeled may still be our kinds: masked, never background.
        labeled = {r["key"].rsplit("/", 1)[1] for r in recs}
        for k, xyxy in signs.get(rel, {}).items():
            if k not in labeled:
                boxes.append(Box(xyxy, None))
        s = Sample(image, boxes, "bdd")
        (pos if any(b.kind for b in boxes) else neg).append(s)
    return pos, neg


# Pseudo-label errors found by checking, by eye on full-resolution crops, the validation crops the value classifier
# disagreed with at high probability (2026-10-10). Only BDD-train val-fraction frames (validation / tune), never
# the held-out sets. key -> (kind, value).
LABEL_FIXES: dict[str, tuple[str, int | None]] = {
    "train/009bd04d-0f1f9967/20": ("speed_limit", 25),  # white R2-1 with a big 25 at night, called "other"
    "train/2db6f386-71e116f8/24": ("unclear", None),  # white panel "20" under two text lines: kind doubtful
}


def read_labels(split: str, fixes: bool = True) -> list[dict]:
    """Every pseudo-label of a split from data/signs/bdd_labels/<split>*.jsonl, one per crop key (first wins).

    LABEL_FIXES are applied unless fixes=False (they record the original kind under "labeler_kind").
    """
    seen: set[str] = set()
    out = []
    for path in sorted(LABELS.glob(f"{split}*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            if rec["key"] in seen:
                continue
            seen.add(rec["key"])
            if fixes and rec["key"] in LABEL_FIXES:
                kind, value = LABEL_FIXES[rec["key"]]
                rec = {**rec, "labeler_kind": rec["kind"], "kind": kind, "value": value}
            out.append(rec)
    return out


def bdd_sign_boxes(split: str) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """image path (relative, posix) -> {box index: xyxy} for every human "traffic sign" box in the metadata."""
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for meta in sorted((BDD_IMAGES / split).glob("shard_*/metadata.jsonl")):
        rel_dir = meta.parent.relative_to(BDD_IMAGES).as_posix()
        for line in meta.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            objs = row["objects"]
            boxes = {
                str(k): (x, y, x + w, y + h)
                for k, (c, (x, y, w, h)) in enumerate(zip(objs["categories"], objs["bbox"], strict=True))
                if c == SIGN_CLASS
            }
            out[f"{rel_dir}/{row['file_name']}"] = boxes
    return out


# The mph25 sequence shows S5-1 SCHOOL SPEED LIMIT 25 signs (yellow-green SCHOOL panel; checked by eye and by
# Claude on every crop), not plain R2-1 speed limits.
CTACKE_KIND = {"mph25": "school_speed_limit"}
CTACKE_CHECK = DATA / "ctacke_readable.jsonl"  # scripts/signs/verify_ctacke.py
CTACKE_CHECK_RAW = DATA / "ctacke_readable.raw.jsonl"


def ctacke_checks() -> tuple[dict[str, bool], dict[str, str]]:
    """(key -> value readable, key -> kind as re-read by Claude) from scripts/signs/verify_ctacke.py, if run."""
    readable: dict[str, bool] = {}
    kinds: dict[str, str] = {}
    if CTACKE_CHECK.exists():
        for line in CTACKE_CHECK.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            readable[r["key"]] = bool(r["readable"])
    if CTACKE_CHECK_RAW.exists():
        for line in CTACKE_CHECK_RAW.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            kinds[r["key"]] = r["kind"]
    return readable, kinds


def ctacke_samples(split: str) -> list[Sample]:
    """ctacke speed-limit frames (value from the folder name), with the re-read checks applied.

    A box whose kind Claude couldn't tell ("unclear": a dark or blurred blob) is masked for the detector and is a
    reject ("none") crop for the value classifier; a box of clear kind whose number can't be read keeps its
    detector label and is a reject crop too.
    """
    readable, kinds = ctacke_checks()
    out = []
    for lab in sorted((CTACKE / "labels" / split).glob("*/*.txt")):
        mapped = kind_from_label(lab.parent.name)
        img = CTACKE / "images" / split / lab.parent.name / (lab.stem + ".png")
        if not img.exists() or mapped is None:
            continue
        kind, value = CTACKE_KIND.get(lab.parent.name, mapped[0]), mapped[1]
        h, w = 1080, 1920  # every ctacke frame is 1920x1080 (checked); boxes are normalized
        boxes = []
        for k, line in enumerate(lab.read_text().splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            xyxy = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            key = f"ctacke/{split}/{lab.parent.name}/{lab.stem}/{k}"
            if kinds.get(key) == "unclear":
                boxes.append(Box(xyxy, None, value, cls_label="none"))
            elif key in readable and not readable[key]:
                boxes.append(Box(xyxy, kind, value, cls_label="none"))
            else:
                boxes.append(Box(xyxy, kind, value))
        if boxes:
            out.append(Sample(img, boxes, "ctacke"))
    return out


def lisa_samples(split: str) -> list[Sample]:
    """LISA in YOLO format (a Roboflow export): images/, labels/ and data.yaml with class names."""
    root = LISA / split
    yml = LISA / "data.yaml"
    if not root.exists() or not yml.exists():
        return []
    names = yaml.safe_load(yml.read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    out = []
    for lab in sorted((root / "labels").glob("*.txt")):
        img = next((p for p in (root / "images").glob(lab.stem + ".*")), None)
        if img is None:
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            mapped = kind_from_label(names[int(parts[0])])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            xyxy = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            # Signs outside our taxonomy are real signs we don't want: background (not masked).
            if mapped is not None:
                boxes.append(Box(xyxy, *mapped))
        out.append(Sample(img, boxes, "lisa"))
    return out


def restore_boxes(img: np.ndarray, original: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> None:
    """Copy the pixels of these boxes back from the original (a mask's margin must not paint a labeled sign)."""
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        a, b, c, d = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
        if c > a and d > b:
            img[b:d, a:c] = original[b:d, a:c]


def mask_box(img: np.ndarray, xyxy: tuple[float, float, float, float]) -> None:
    """Paint over a box with its surroundings' mean color (slightly enlarged), in place."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    g = 0.15 * max(x2 - x1, y2 - y1) + 2
    a, b, c, d = int(max(0, x1 - g)), int(max(0, y1 - g)), int(min(w, x2 + g)), int(min(h, y2 + g))
    if c <= a or d <= b:
        return
    ring = np.concatenate([img[b:d, a].reshape(-1, 3), img[b:d, c - 1].reshape(-1, 3)])
    img[b:d, a:c] = ring.mean(axis=0).astype(np.uint8)


def tiles_for(width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    """Top-left corners of TILE x TILE tiles covering the image with >= TILE/2 overlap (one tile if it fits).

    With half a tile of overlap, any sign narrower than TILE/2 lies wholly inside at least one tile (a 1280-wide
    frame gets x = 0, 320, 640). The vertical offset is random: frames are 720 tall, tiles 640.
    """
    if width <= TILE:
        xs = [0]
    else:
        n = math.ceil((width - TILE) / (TILE / 2)) + 1
        xs = [round(i * (width - TILE) / (n - 1)) for i in range(n)]
    ys = [0] if height <= TILE else [rng.randint(0, height - TILE)]
    return [(x, y) for x in xs for y in ys]


def write_tiles(
    sample: Sample, img: np.ndarray, out: Path, split: str, rng: random.Random, keep_empty: float
) -> Counter:
    """Cut a (resized, masked) image into tiles and write them with YOLO labels. Returns per-kind box counts."""
    counts: Counter = Counter()
    h, w = img.shape[:2]
    stem = f"{sample.source}_{sample.image.parent.name}_{sample.image.stem}"
    for n, (x0, y0) in enumerate(tiles_for(w, h, rng)):
        tile = img[y0 : y0 + TILE, x0 : x0 + TILE].copy()
        original = tile.copy()
        th, tw = tile.shape[:2]
        lines = []
        labeled: list[tuple[float, float, float, float]] = []
        for b in sample.boxes:
            x1, y1, x2, y2 = b.xyxy
            cx1, cy1, cx2, cy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            inside = (cx2 - cx1) * (cy2 - cy1) / max(1e-6, (x2 - x1) * (y2 - y1))
            if b.kind is None or inside < 0.95:
                # Unknown kind, or cut by the tile edge (it is whole in a neighbouring tile): never background.
                mask_box(tile, (cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0))
                continue
            lines.append(
                f"{KIND_INDEX[b.kind]} {(cx1 + cx2) / 2 / tw - x0 / tw:.6f} {(cy1 + cy2) / 2 / th - y0 / th:.6f} "
                f"{(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}"
            )
            labeled.append((cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0))
            counts[b.kind] += 1
        restore_boxes(tile, original, labeled)
        if not lines and rng.random() > keep_empty:
            continue
        if (th, tw) != (TILE, TILE):  # pad small images/tiles to a full tile (gray, like YOLO's letterbox)
            canvas = np.full((TILE, TILE, 3), 114, np.uint8)
            canvas[:th, :tw] = tile
            tile = canvas
            lines = [_rescale_line(line, tw / TILE, th / TILE) for line in lines]
        name = f"{stem}_{n}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return counts


def _rescale_line(line: str, sx: float, sy: float) -> str:
    """Re-normalize a YOLO label line after padding its tile to TILE x TILE."""
    c, cx, cy, bw, bh = line.split()
    return f"{c} {float(cx) * sx:.6f} {float(cy) * sy:.6f} {float(bw) * sx:.6f} {float(bh) * sy:.6f}"


def load_scaled(sample: Sample) -> tuple[np.ndarray, Sample] | None:
    """Read an image, resize it to WIDTH (never up), scale its boxes, and mask the unknown ones."""
    img = cv2.imread(str(sample.image))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, WIDTH / w)
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    boxes = [
        Box((b.xyxy[0] * s, b.xyxy[1] * s, b.xyxy[2] * s, b.xyxy[3] * s), b.kind, b.value, b.cls_label)
        for b in sample.boxes
    ]
    if any(b.kind is None for b in boxes):
        original = img.copy()
        for b in boxes:
            if b.kind is None:
                mask_box(img, b.xyxy)
        restore_boxes(img, original, [b.xyxy for b in boxes if b.kind is not None])
    return img, Sample(sample.image, boxes, sample.source)


def _det_job(args: tuple[Sample, str, int, float]) -> Counter:
    sample, split, seed, keep_empty = args
    loaded = load_scaled(sample)
    if loaded is None:
        return Counter()
    img, scaled = loaded
    return write_tiles(scaled, img, DET_OUT, split, random.Random(seed), keep_empty)


# Synthetic composites: kind weights favour what real data lacks (rare kinds, every speed value).
SYNTH_KIND_WEIGHTS = {
    "speed_limit": 0.30,
    "school_speed_limit": 0.12,
    "wrong_way": 0.08,
    "no_turn_on_red": 0.08,
    "do_not_enter": 0.06,
    "yield": 0.06,
    "no_left_turn": 0.05,
    "no_right_turn": 0.05,
    "no_u_turn": 0.05,
    "one_way": 0.04,
    "stop": 0.04,
    "pedestrian_crossing": 0.03,
    "school_zone": 0.04,
}
COMMON_VALUES = (15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)


def sample_value(kind: str, rng: np.random.Generator) -> int | None:
    """An mph value for a synthetic sign: school zones 15-25, otherwise mostly 15-70, sometimes 5-10/75-85."""
    if kind == "school_speed_limit":
        return int(rng.choice([15, 20, 20, 25, 25]))
    if kind == "speed_limit":
        return int(rng.choice(COMMON_VALUES)) if rng.random() < 0.88 else int(rng.choice([5, 10, 75, 80, 85]))
    return None


def paste_footprint(x: int, y: int, w: int, h: int, frame_h: int) -> tuple[float, float, float, float]:
    """Everything synth.composite may paint for a w x h sign at (x, y): the sign and a pole up to 3 heights long."""
    return float(x), float(y), float(x + w), float(min(frame_h, y + 4 * h))


def synth_signs(
    img: np.ndarray, boxes: list[Box], renderer: Any, rng: np.random.Generator
) -> tuple[np.ndarray, list[Box]]:
    """Paste 1-3 labeled synthetic signs and 0-2 unlabeled look-alikes into a frame (in place) and degrade it.

    A sign is pasted only where its footprint (sign + pole) overlaps no existing box, so a pasted sign is always
    labeled and never covers a labeled or masked one. Returns (degraded frame, existing + new boxes).
    """
    from scripts.signs.synth import composite, degrade, warp_sign

    h, w = img.shape[:2]
    kinds, weights = zip(*SYNTH_KIND_WEIGHTS.items(), strict=True)
    p = np.array(weights) / sum(weights)
    boxes = list(boxes)
    for _ in range(int(rng.integers(1, 4))):
        kind = str(rng.choice(kinds, p=p))
        value = sample_value(kind, rng)
        sign = renderer.any(kind, value, rng)
        height = float(np.exp(rng.uniform(np.log(12), np.log(110))))
        bgr, alpha = warp_sign(sign, height, rng)
        side = rng.random()
        if side < 0.6:
            x = int(rng.uniform(0.55, 0.97) * w - bgr.shape[1])
        elif side < 0.85:
            x = int(rng.uniform(0.03, 0.4) * w)
        else:
            x = int(rng.uniform(0.35, 0.65) * w)
        y = int(rng.uniform(0.06, 0.5) * h)
        pole = rng.random() < 0.8
        hh, ww = alpha.shape
        if any(_iou(paste_footprint(x, y, ww, hh, h), b.xyxy) > 0.0 for b in boxes):
            continue
        box = composite(img, bgr, alpha, x, y, rng, pole=pole)
        if box is not None:
            boxes.append(Box(box, kind, value))
    # Unlabeled look-alikes (route shields, advisory plates): background the detector must not call a sign.
    for _ in range(int(rng.integers(0, 3))):
        bgr, alpha = warp_sign(renderer.distractor(rng), float(np.exp(rng.uniform(np.log(14), np.log(100)))), rng)
        x = int(rng.uniform(0.03, 0.97) * w - bgr.shape[1] / 2)
        y = int(rng.uniform(0.06, 0.5) * h)
        pole = rng.random() < 0.7
        hh, ww = alpha.shape
        if any(_iou(paste_footprint(x, y, ww, hh, h), b.xyxy) > 0.0 for b in boxes):
            continue
        composite(img, bgr, alpha, x, y, rng, pole=pole)
    return degrade(img, rng, strength=0.6), boxes


def _synth_job(args: tuple[Sample, str, int]) -> Counter:
    from scripts.signs.synth import SignRenderer

    sample, split, seed = args
    rng = np.random.default_rng(seed)
    loaded = load_scaled(sample)
    if loaded is None:
        return Counter()
    img, scaled = loaded
    renderer = _renderer_cache.get("r") or _renderer_cache.setdefault("r", SignRenderer())
    img, boxes = synth_signs(img, scaled.boxes, renderer, rng)
    out = Sample(sample.image, boxes, "synth")
    return write_tiles(out, img, DET_OUT, split, random.Random(seed), keep_empty=0.0)


_renderer_cache: dict = {}


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


DetJob = tuple[Sample, str, int, float]  # (sample, split, seed, fraction of sign-free tiles kept)
SynthJob = tuple[Sample, str, int]  # (background, split, seed)


def split_of(s: Sample) -> str:
    """BDD train images in the held-out fraction go to val (and the tune set), the rest to train."""
    return "val" if in_val(s.image) else "train"


def plan_det_jobs(
    pos: list[Sample],
    neg: list[Sample],
    hard_paths: set[str],
    ctacke_train: list[Sample],
    lisa_train: list[Sample],
    lisa_val: list[Sample],
    neg_ratio: float,
    synth: int,
) -> tuple[list[DetJob], list[SynthJob]]:
    """Which frames go into the detector dataset, in which split, with which seed.

    BDD positives and real negatives are split by the stable hash (val fraction -> val). Hard negatives
    (scripts/signs/mine_det_negatives.py: frames where a trained detector fired on none of our signs) keep all
    their tiles. Synthetic composites always train, so their backgrounds come only from the training fraction.
    """
    rng = random.Random(0)
    jobs: list[DetJob] = [(s, split_of(s), rng.randrange(2**31), 0.15) for s in pos]
    hard = [s for s in neg if rel_path(s.image) in hard_paths]
    rest = [s for s in neg if rel_path(s.image) not in hard_paths]
    rng.shuffle(rest)
    n_neg = min(len(rest), int(neg_ratio * len(pos)))
    synth_bg = [s for s in rest[n_neg:] if not in_val(s.image)][:synth]
    jobs += [(s, split_of(s), rng.randrange(2**31), 0.5) for s in rest[:n_neg]]
    jobs += [(s, split_of(s), rng.randrange(2**31), 1.0) for s in hard]
    jobs += [(s, "train", rng.randrange(2**31), 0.0) for s in ctacke_train]
    jobs += [(s, "train", rng.randrange(2**31), 0.1) for s in lisa_train]
    jobs += [(s, "val", rng.randrange(2**31), 0.1) for s in lisa_val]
    synth_jobs: list[SynthJob] = [(s, "train", rng.randrange(2**31)) for s in synth_bg]
    return jobs, synth_jobs


def build_det(args: argparse.Namespace) -> None:
    """Write data/signs/det/ (YOLO format) and its data.yaml."""
    if DET_OUT.exists():
        shutil.rmtree(DET_OUT)
    for split in ("train", "val"):
        (DET_OUT / "images" / split).mkdir(parents=True)
        (DET_OUT / "labels" / split).mkdir(parents=True)
    pos, neg = bdd_samples("train")
    hard_paths: set[str] = set()
    if HARDNEG_DET.exists():
        hard_paths = {line.split("\t")[0] for line in HARDNEG_DET.read_text(encoding="utf-8").splitlines() if line}
    jobs, synth_jobs = plan_det_jobs(
        pos,
        neg,
        hard_paths,
        ctacke_samples("train"),
        lisa_samples("train"),
        lisa_samples("valid"),
        args.neg_ratio,
        args.synth,
    )
    unboxed = read_unboxed()  # training frames only: evaluation labels never change with a model's detections
    jobs = [(with_unboxed_masks(s, unboxed) if split == "train" else s, split, seed, k) for s, split, seed, k in jobs]
    synth_jobs = [(with_unboxed_masks(s, unboxed), split, seed) for s, split, seed in synth_jobs]
    write_used_for_training("det", [j[0].image for j in jobs if j[1] == "train"] + [j[0].image for j in synth_jobs])
    print(f"BDD positives {len(pos)}, synthetic {len(synth_jobs)}, total jobs {len(jobs)}")
    totals: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for c in pool.map(_det_job, jobs, chunksize=16):
            totals.update(c)
        synth_totals: Counter = Counter()
        for c in pool.map(_synth_job, synth_jobs, chunksize=8):
            synth_totals.update(c)
    data = {
        "path": str(DET_OUT),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(KINDS)),
    }
    (DET_OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    stats = {
        "real_boxes": dict(totals),
        "synthetic_boxes": dict(synth_totals),
        "tiles": {s: len(list((DET_OUT / "images" / s).glob("*.jpg"))) for s in ("train", "val")},
    }
    (DET_OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


# ------------------------------------------------------------------------------------------- classifier

CLS_SIZE = 96
CLS_PAD = 0.15


def value_class(value: int | None) -> str:
    """Classifier folder name for a value (None = the reject class)."""
    return f"v{value}" if value is not None else "none"


def jitter_crop(img: np.ndarray, xyxy: tuple[float, ...], rng: np.random.Generator | None) -> np.ndarray | None:
    """Square-resized crop around a box with CLS_PAD padding; with rng, the box is jittered like a detector's."""
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    if rng is not None:
        cx, cy = (x1 + x2) / 2 + rng.normal(0, 0.05) * bw, (y1 + y2) / 2 + rng.normal(0, 0.05) * bh
        s = float(np.clip(rng.normal(1.0, 0.07), 0.85, 1.2))
        bw, bh = bw * s * float(np.clip(rng.normal(1.0, 0.04), 0.9, 1.1)), bh * s
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    return crop_square(img, (x1, y1, x2, y2))


def crop_square(img: np.ndarray, xyxy: tuple[float, ...]) -> np.ndarray | None:
    """The runtime crop: box + CLS_PAD per side, resized to CLS_SIZE x CLS_SIZE (perception/signs.py mirrors this)."""
    x1, y1, x2, y2 = xyxy
    pw, ph = CLS_PAD * (x2 - x1), CLS_PAD * (y2 - y1)
    h, w = img.shape[:2]
    a, b, c, d = round(x1 - pw), round(y1 - ph), round(x2 + pw), round(y2 + ph)
    a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
    if c - a < 4 or d - b < 4:
        return None
    interp = cv2.INTER_AREA if (c - a) > CLS_SIZE else cv2.INTER_LINEAR
    return cv2.resize(img[b:d, a:c], (CLS_SIZE, CLS_SIZE), interpolation=interp)


def _real_crops(samples: list[Sample], copies: int, seed: int, min_h: float) -> Iterator[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    for s in samples:
        img = cv2.imread(str(s.image))
        if img is None:
            continue
        for b in s.boxes:
            if b.xyxy[3] - b.xyxy[1] < min_h or (b.kind is None and b.cls_label is None):
                continue
            # A speed limit whose number nobody could read is a reject crop: the classifier must learn to say so.
            label = b.cls_label or (value_class(b.value) if b.kind in VALUE_KINDS else "none")
            copies_here = copies if label != "none" else 1
            if label == "none" and b.value is not None:
                label = f"none@{b.value}"  # reject crop of a real limit: its value is kept in the file name
            for _ in range(copies_here):
                crop = jitter_crop(img, b.xyxy, rng)
                if crop is not None:
                    yield label, crop


def bdd_other_records(split: str, limit: int, seed: int, val: bool) -> list[dict]:
    """BDD signs labeled other/not_a_sign, at least 14 px tall, from the val fraction (val=True) or the rest."""
    banned = dev_clip_ids()
    recs = [
        r
        for r in read_labels(split)
        if r["kind"] in ("other", "not_a_sign")
        and r["box"][3] - r["box"][1] >= 14
        and Path(r["image"]).stem not in banned
        and in_val(Path(r["image"])) == val
    ]
    random.Random(seed).shuffle(recs)
    return recs[:limit]


HARDNEG = DATA / "hardneg_cls_curated.jsonl"  # scripts/signs/mine_hard_negatives.py, checked by eye


def hard_negative_records(val: bool) -> list[dict]:
    """Real BDD train signs that read like a speed value but aren't one (checked by eye), by fraction."""
    if not HARDNEG.exists():
        return []
    recs = [json.loads(line) for line in HARDNEG.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in recs if in_val(Path(r["image"])) == val]


def _record_crops(recs: list[dict], copies: int, seed: int) -> Iterator[tuple[str, np.ndarray]]:
    """Reject-class ("none") crops at the boxes of label records, jittered like a detector's boxes."""
    rng = np.random.default_rng(seed)
    for r in recs:
        img = cv2.imread(str(BDD_IMAGES / r["image"]))
        if img is None:
            continue
        for _ in range(copies):
            crop = jitter_crop(img, tuple(r["box"]), rng)
            if crop is not None:
                yield "none", crop


def cls_backgrounds(neg: list[Sample]) -> list[Sample]:
    """Backgrounds for synthetic classifier crops: BDD frames without our signs, training fraction only."""
    return [s for s in neg if not in_val(s.image)][:CLS_BACKGROUNDS]


def _synth_cls_job(args: tuple[int, int, list[str], int]) -> list[tuple[str, np.ndarray]]:
    """Synthetic classifier crops for one value (or the reject class when value == 0)."""
    from scripts.signs.synth import SignRenderer, composite, degrade, warp_sign

    value, count, backgrounds, seed = args
    rng = np.random.default_rng(seed)
    renderer = _renderer_cache.get("r") or _renderer_cache.setdefault("r", SignRenderer())
    out: list[tuple[str, np.ndarray]] = []
    bgs = [np.asarray(b) for b in (cv2.imread(p) for p in backgrounds) if b is not None]
    others = [k for k in KINDS if k not in VALUE_KINDS]
    while len(out) < count:
        bg = bgs[int(rng.integers(len(bgs)))]
        h, w = bg.shape[:2]
        size = float(np.exp(rng.uniform(np.log(16), np.log(130))))  # the app reads no value below 16 px
        if value:
            school = value in (15, 20, 25) and rng.random() < 0.25
            sign = renderer.speed(value, school, rng)
            label = value_class(value)
        else:  # reject class: half other kinds of our signs, half number-bearing look-alikes
            sign = renderer.kind(str(rng.choice(others)), rng) if rng.random() < 0.5 else renderer.distractor(rng)
            label = "none"
        bgr, alpha = warp_sign(sign, size, rng)
        sh, sw = alpha.shape
        if sw + 8 >= w or sh * 3 + 8 >= h:
            continue
        x, y = int(rng.integers(4, w - sw - 4)), int(rng.integers(4, h - 3 * sh - 4))
        patch = bg.copy()
        box = composite(patch, bgr, alpha, x, y, rng, pole=rng.random() < 0.7)
        if box is None:
            continue
        patch = degrade(patch, rng, strength=1.0 if size > 30 else 0.6)
        crop = jitter_crop(patch, box, rng)
        if crop is not None:
            out.append((label, crop))
    return out


def build_cls(args: argparse.Namespace) -> None:
    """Write data/signs/cls/{train,val}/<class>/*.png."""
    if CLS_OUT.exists():
        shutil.rmtree(CLS_OUT)
    classes = [value_class(v) for v in SPEED_VALUES] + ["none"]
    for split in ("train", "val"):
        for c in classes:
            (CLS_OUT / split / c).mkdir(parents=True)
    counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}

    def save(split: str, items: Iterator[tuple[str, np.ndarray]] | list[tuple[str, np.ndarray]], tag: str) -> None:
        for full_label, crop in items:
            # "none@30": a reject crop of a real 30 mph sign nobody could read; evaluation knows its true value.
            label, _, true_value = full_label.partition("@")
            n = counts[split][label]
            suffix = f"_true{true_value}" if true_value else ""
            cv2.imwrite(str(CLS_OUT / split / label / f"{tag}_{n:06d}{suffix}.png"), crop)
            counts[split][label] += 1

    pos, neg = bdd_samples("train")
    train_pos = [s for s in pos if not in_val(s.image)]
    val_pos = [s for s in pos if in_val(s.image)]
    other_train = bdd_other_records("train", args.none_real, 6, val=False)
    hard_train = hard_negative_records(val=False)
    backgrounds = cls_backgrounds(neg)
    lisa_train = lisa_samples("train")
    save("train", _real_crops(train_pos, 4, 1, 14), "bdd")
    save("val", _real_crops(val_pos, 1, 2, 14), "bdd")
    save("train", _real_crops(ctacke_samples("train"), 2, 3, 12), "ctacke")
    save("val", _real_crops(ctacke_samples("val"), 1, 4, 12), "ctacke")
    save("train", _real_crops(lisa_train, 3, 5, 12), "lisa")
    save("train", _record_crops(other_train, 1, 6), "bddother")
    save("train", _record_crops(hard_train, 4, 8), "hardneg")
    # Real look-alikes for validation (threshold choice): every val-fraction "other" sign and hard negative.
    save("val", _record_crops(bdd_other_records("train", 10**6, 9, val=True), 1, 9), "bddother")
    save("val", _record_crops(hard_negative_records(val=True), 1, 10), "hardneg")
    write_used_for_training(
        "cls",
        [s.image for s in train_pos + lisa_train + backgrounds]
        + [BDD_IMAGES / r["image"] for r in other_train + hard_train],
    )
    # Synthetic: every value, plus other kinds for the reject class, on BDD backgrounds without our signs.
    bgs = [str(s.image) for s in backgrounds]
    rng = random.Random(7)
    jobs = []
    for v in (*SPEED_VALUES, 0):
        per = args.synth_per_value if v else args.synth_per_value // 2
        for k in range(0, per, 200):
            jobs.append((v, min(200, per - k), rng.sample(bgs, min(30, len(bgs))), rng.randrange(2**31)))
    val_jobs = [(v, 30, rng.sample(bgs, min(30, len(bgs))), rng.randrange(2**31)) for v in (*SPEED_VALUES, 0)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for items in pool.map(_synth_cls_job, jobs):
            save("train", items, "synth")
        for items in pool.map(_synth_cls_job, val_jobs):
            save("val", items, "synth")
    stats = {s: dict(sorted(c.items())) for s, c in counts.items()}
    (CLS_OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> int:
    """Build one dataset."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["det", "cls"])
    p.add_argument("--neg-ratio", type=float, default=0.3, help="BDD images without our signs, per positive image")
    p.add_argument("--synth", type=int, default=3500, help="synthetic composite frames")
    p.add_argument("--synth-per-value", type=int, default=1200)
    p.add_argument("--none-real", type=int, default=6000, help="real reject-class crops")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    build_det(args) if args.what == "det" else build_cls(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
