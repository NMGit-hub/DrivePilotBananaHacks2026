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
WIDTH = 1280  # frames are resized to this width (the app's detector input), never upscaled
MASK_KINDS = frozenset({"unclear"})
SIGN_CLASS = 9


@dataclass
class Box:
    """One labeled box in source-image pixels. kind None = mask it (unknown sign)."""

    xyxy: tuple[float, float, float, float]
    kind: str | None
    value: int | None = None


@dataclass
class Sample:
    """One source image with its boxes."""

    image: Path
    boxes: list[Box] = field(default_factory=list)
    source: str = ""


def stable_frac(text: str) -> float:
    """A deterministic number in [0, 1) from a string (for train/val splits that never change)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 2**32


def dev_clip_ids() -> set[str]:
    """BDD video ids of the dev clips: never trained on."""
    data = yaml.safe_load((REPO_ROOT / "eval" / "dev_clips.yaml").read_text(encoding="utf-8"))
    return {c["bdd_id"] for c in data["clips"]}


def bdd_samples(split: str) -> tuple[list[Sample], list[Sample]]:
    """(pseudo-labeled images with at least one labeled box, labeled images with none of our signs)."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for rec in read_labels(split):
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


def read_labels(split: str) -> list[dict]:
    """Every pseudo-label of a split from data/signs/bdd_labels/<split>*.jsonl, one per crop key (first wins)."""
    seen: set[str] = set()
    out = []
    for path in sorted(LABELS.glob(f"{split}*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            if rec["key"] not in seen:
                seen.add(rec["key"])
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


def ctacke_samples(split: str) -> list[Sample]:
    """ctacke speed-limit frames (value from the folder name)."""
    out = []
    for lab in sorted((CTACKE / "labels" / split).glob("*/*.txt")):
        value = kind_from_label(lab.parent.name)
        img = CTACKE / "images" / split / lab.parent.name / (lab.stem + ".png")
        if not img.exists() or value is None:
            continue
        h, w = 1080, 1920  # every ctacke frame is 1920x1080 (checked in prepare); boxes are normalized
        boxes = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            boxes.append(Box(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), *value))
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
        th, tw = tile.shape[:2]
        lines = []
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
            counts[b.kind] += 1
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
    boxes = [Box((b.xyxy[0] * s, b.xyxy[1] * s, b.xyxy[2] * s, b.xyxy[3] * s), b.kind, b.value) for b in sample.boxes]
    for b in boxes:
        if b.kind is None:
            mask_box(img, b.xyxy)
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


def _synth_job(args: tuple[Sample, str, int]) -> Counter:
    from scripts.signs.synth import SignRenderer, composite, degrade, warp_sign

    sample, split, seed = args
    rng = np.random.default_rng(seed)
    loaded = load_scaled(sample)
    if loaded is None:
        return Counter()
    img, scaled = loaded
    renderer = _renderer_cache.get("r") or _renderer_cache.setdefault("r", SignRenderer())
    h, w = img.shape[:2]
    kinds, weights = zip(*SYNTH_KIND_WEIGHTS.items(), strict=True)
    p = np.array(weights) / sum(weights)
    boxes: list[Box] = list(scaled.boxes)
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
        box = composite(img, bgr, alpha, x, y, rng, pole=rng.random() < 0.8)
        if box is not None and not any(_iou(box, b.xyxy) > 0.0 for b in boxes):
            boxes.append(Box(box, kind, value))
    img = degrade(img, rng, strength=0.6)
    out = Sample(sample.image, boxes, "synth")
    return write_tiles(out, img, DET_OUT, split, random.Random(seed), keep_empty=0.0)


_renderer_cache: dict = {}


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_det(args: argparse.Namespace) -> None:
    """Write data/signs/det/ (YOLO format) and its data.yaml."""
    if DET_OUT.exists():
        shutil.rmtree(DET_OUT)
    for split in ("train", "val"):
        (DET_OUT / "images" / split).mkdir(parents=True)
        (DET_OUT / "labels" / split).mkdir(parents=True)
    pos, neg = bdd_samples("train")
    rng = random.Random(0)
    jobs: list[tuple[Sample, str, int, float]] = []
    for s in pos:
        split = "val" if stable_frac(s.image.stem) < args.val_frac else "train"
        jobs.append((s, split, rng.randrange(2**31), 0.15))
    rng.shuffle(neg)
    n_neg = min(len(neg), int(args.neg_ratio * len(pos)))
    synth_bg, neg = neg[n_neg : n_neg + args.synth], neg[:n_neg]
    for s in neg:
        split = "val" if stable_frac(s.image.stem) < args.val_frac else "train"
        jobs.append((s, split, rng.randrange(2**31), 0.5))
    for s in ctacke_samples("train"):
        jobs.append((s, "train", rng.randrange(2**31), 0.0))
    for s in lisa_samples("train"):
        jobs.append((s, "train", rng.randrange(2**31), 0.1))
    for s in lisa_samples("valid"):
        jobs.append((s, "val", rng.randrange(2**31), 0.1))
    synth_jobs = [(s, "train", rng.randrange(2**31)) for s in synth_bg]
    print(f"BDD positives {len(pos)}, negatives {len(neg)}, synthetic {len(synth_jobs)}, total jobs {len(jobs)}")
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
            if b.kind is None or b.xyxy[3] - b.xyxy[1] < min_h:
                continue
            label = value_class(b.value) if b.kind in VALUE_KINDS else "none"
            if b.kind in VALUE_KINDS and b.value is None:
                continue  # a speed limit whose value nobody could read: no value label
            for _ in range(copies if label != "none" else 1):
                crop = jitter_crop(img, b.xyxy, rng)
                if crop is not None:
                    yield label, crop


def _bdd_other_crops(split: str, limit: int, seed: int) -> Iterator[tuple[str, np.ndarray]]:
    """Reject-class crops: BDD signs labeled other/not_a_sign (train split only)."""
    path = LABELS / f"{split}.jsonl"
    if not path.exists():
        return
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    recs = [r for r in recs if r["kind"] in ("other", "not_a_sign") and r["box"][3] - r["box"][1] >= 14]
    random.Random(seed).shuffle(recs)
    rng = np.random.default_rng(seed)
    banned = dev_clip_ids()
    for r in recs[:limit]:
        image = BDD_IMAGES / r["image"]
        if image.stem in banned:
            continue
        img = cv2.imread(str(image))
        crop = None if img is None else jitter_crop(img, tuple(r["box"]), rng)
        if crop is not None:
            yield "none", crop


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
        size = float(np.exp(rng.uniform(np.log(14), np.log(130))))
        if value:
            school = value in (15, 20, 25) and rng.random() < 0.25
            sign = renderer.speed(value, school, rng)
            label = value_class(value)
        else:
            sign = renderer.kind(str(rng.choice(others)), rng)
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
        for label, crop in items:
            n = counts[split][label]
            cv2.imwrite(str(CLS_OUT / split / label / f"{tag}_{n:06d}.png"), crop)
            counts[split][label] += 1

    pos, _neg = bdd_samples("train")
    train_pos = [s for s in pos if stable_frac(s.image.stem) >= args.val_frac]
    val_pos = [s for s in pos if stable_frac(s.image.stem) < args.val_frac]
    save("train", _real_crops(train_pos, 4, 1, 14), "bdd")
    save("val", _real_crops(val_pos, 1, 2, 14), "bdd")
    save("train", _real_crops(ctacke_samples("train"), 2, 3, 12), "ctacke")
    save("val", _real_crops(ctacke_samples("val"), 1, 4, 12), "ctacke")
    save("train", _real_crops(lisa_samples("train"), 3, 5, 12), "lisa")
    save("train", _bdd_other_crops("train", args.none_real, 6), "bddother")
    # Synthetic: every value, plus other kinds for the reject class, on BDD backgrounds without our signs.
    _pos, neg = bdd_samples("train")
    bgs = [str(s.image) for s in neg[:400]]
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
    p.add_argument("--val-frac", type=float, default=0.06, help="fraction of BDD train images held out for val")
    p.add_argument("--neg-ratio", type=float, default=0.3, help="BDD images without our signs, per positive image")
    p.add_argument("--synth", type=int, default=3000, help="synthetic composite frames")
    p.add_argument("--synth-per-value", type=int, default=1200)
    p.add_argument("--none-real", type=int, default=6000, help="real reject-class crops")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    build_det(args) if args.what == "det" else build_cls(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
