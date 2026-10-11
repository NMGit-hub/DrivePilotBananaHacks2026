"""Speed-value classifier on its own: readings at the labeled boxes, so value reading is measured without the detector.

    .venv/Scripts/python scripts/signs/eval_values.py                     # threshold from validation only
    .venv/Scripts/python scripts/signs/eval_values.py --views 1 3 5 7     # compare the number of agreeing views
    .venv/Scripts/python scripts/signs/eval_values.py --heldout both --write-metrics   # + held-out scores

Every box is read as perception/signs.py reads a detection: crops of the box from the full frame
(`signs.value_views` shifted/scaled views that must agree, perception.signs.read_value), by the classifier as the
app runs it (its FP16 TensorRT engine when one was built from these weights, else the weights in FP16 on the GPU).

1. Threshold: on real BDD validation boxes (the BDD train split's held-out fraction, never trained on: its speed
   limits with a known value, its other kinds of our signs, every "other"/not-a-sign box at least 14 px tall and the
   curated look-alikes), the lowest probability threshold with no wrong value at it and above. Validation keeps
   boxes the app would not read (14-16 px tall, at the frame border): stricter than the app. ctacke val is
   reported separately and not used: ctacke's val split shares physical signs with its train split. The app's
   value_conf is the larger of this and the lowest threshold with no wrong (or unverifiable) value end to end on
   the tune set (evaluate.py tune), one step higher if a genuine misread sits within 0.02 below it.
2. Held-out (only with --heldout, never while choosing): the hand-checked BDD100K speed limits
   (eval/sign_labels_bdd*.json) the app would read (value_min_px, value_edge_px); right / wrong / withheld.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import SignsConfig, load_config  # noqa: E402
from perception.sign_classes import VALUE_KINDS  # noqa: E402
from perception.signs import (  # noqa: E402
    VALUE_VIEWS,
    check_value_classes,
    engine_path,
    engine_staleness,
    read_value,
    sidecar_path,
    top_reading,
    touches_border,
    value_attempted,
)

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
HELDOUT = {
    "original": REPO_ROOT / "eval" / "sign_labels_bdd.json",
    "fresh": REPO_ROOT / "eval" / "sign_labels_bdd_fresh.json",
}
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)
MIN_H = 14.0  # validation boxes shorter than this are skipped (the app reads >= 16 px: validation is stricter)
BDD_W, BDD_H = 1280, 720

Box = tuple[float, float, float, float]
Item = tuple[Path, Box, int | None]  # (frame, box, true value: None = not a speed sign, any value is wrong)


def sweep(preds: Sequence[tuple[int | None, float] | None], truth: Sequence[int | None]) -> list[dict[str, Any]]:
    """Right / wrong / withheld counts per threshold (truth None = not a speed sign: any value is wrong)."""
    rows = []
    for t in THRESHOLDS:
        right = wrong = withheld = 0
        for pred, g in zip(preds, truth, strict=True):
            v, p = pred if pred is not None else (None, 0.0)
            said = v if (v is not None and p >= t) else None
            if said is None:
                withheld += 1
            elif said == g:
                right += 1
            else:
                wrong += 1
        rows.append({"threshold": t, "right": right, "wrong": wrong, "withheld": withheld})
    return rows


def lowest_safe(rows: list[dict[str, Any]]) -> float | None:
    """The lowest threshold with no wrong value at it and at every higher one; None if even the highest is unsafe."""
    chosen: float | None = None
    for r in sorted(rows, key=lambda r: -r["threshold"]):
        if r["wrong"]:
            break
        chosen = r["threshold"]
    return chosen


def resolve_classifier(weights: Path, imgsz: int, half: bool) -> Path:
    """The classifier the app would run: its TensorRT engine when built from these weights, else the weights."""
    engine = engine_path(weights, (imgsz, imgsz), half)
    if engine.exists() and sidecar_path(engine).exists() and engine_staleness(engine, weights) is None:
        return engine
    return weights


def validation_items() -> dict[str, list[Item]]:
    """Real BDD validation boxes by source: "bdd" (labeled boxes of our kinds), "bddother", "hardneg"."""
    from scripts.signs.prepare_data import bdd_other_records, bdd_samples, hard_negative_records, in_val

    out: dict[str, list[Item]] = {"bdd": [], "bddother": [], "hardneg": []}
    pos, _neg = bdd_samples("train")
    for s in pos:
        if not in_val(s.image):
            continue
        for b in s.boxes:
            if b.kind is None or b.xyxy[3] - b.xyxy[1] < MIN_H:
                continue
            if b.kind in VALUE_KINDS and b.value is None:
                continue  # a speed limit nobody could read: no truth to judge a reading against
            out["bdd"].append((s.image, b.xyxy, b.value if b.kind in VALUE_KINDS else None))
    for r in bdd_other_records("train", 10**6, 9, val=True):
        out["bddother"].append((BDD_IMAGES / r["image"], tuple(r["box"]), None))
    for r in hard_negative_records(val=True):
        out["hardneg"].append((BDD_IMAGES / r["image"], tuple(r["box"]), None))
    return out


def ctacke_items() -> list[Item]:
    """ctacke val speed limits whose number was re-read as readable (reported only)."""
    from scripts.signs.prepare_data import ctacke_samples

    return [(s.image, b.xyxy, b.value) for s in ctacke_samples("val") for b in s.boxes if b.kind and not b.cls_label]


def heldout_items(path: Path, cfg: SignsConfig) -> tuple[list[Item], list[bool]]:
    """Held-out speed limits with a known value that the app would read, and whether each is clearly readable.

    The app's own rules decide which boxes are read: at least value_min_px tall and not within value_edge_px of the
    frame border (BDD frames are 1280x720).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[Item] = []
    readable: list[bool] = []
    for im in data["images"]:
        for x in im["signs"]:
            box = tuple(x["box"])
            if (
                x["kind"] in VALUE_KINDS
                and x["value"] is not None
                and value_attempted(x["kind"], box, BDD_W, cfg.value_min_px)
                and not touches_border(box, BDD_W, BDD_H, cfg.value_edge_px)
            ):
                items.append((BDD_IMAGES / im["image"], box, x["value"]))
                readable.append(bool(x["readable"]))
    return items, readable


def read_items(classify: Any, items: list[Item], imgsz: int, views: int) -> list[tuple[int | None, float] | None]:
    """The app's reading (read_value) of every item, loading each frame once."""
    by_image: dict[Path, list[int]] = defaultdict(list)
    for i, (path, _box, _truth) in enumerate(items):
        by_image[path].append(i)
    out: list[tuple[int | None, float] | None] = [None] * len(items)
    for path, idx in by_image.items():
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"missing frame {path} (downloads: scripts/signs/README.md)")
        for i in idx:
            out[i] = read_value(classify, img, items[i][1], imgsz, views)
    return out


def main(argv: list[str] | None = None) -> int:
    """Pick the value threshold on validation boxes, then score the held-out speed limits."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=None, help="classifier .pt (default: config); its engine if built")
    p.add_argument(
        "--views",
        type=int,
        nargs="+",
        default=None,
        choices=range(1, len(VALUE_VIEWS) + 1),
        help="value_views to compare",
    )
    p.add_argument("--heldout", choices=["original", "fresh", "both", "none"], default="none", help="also score these")
    p.add_argument("--write-metrics", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config()
    s = cfg.signs
    weights = args.weights or cfg.resolve_path(s.classifier_weights)
    model_path = resolve_classifier(weights, s.classifier_imgsz, s.half)
    model = YOLO(str(model_path), task="classify")
    names = dict(model.names)
    check_value_classes(names)
    quant: dict[str, Any] = {"quantize": 16} if s.half else {}

    def classify(crops: list[np.ndarray]) -> list[tuple[int | None, float]]:
        res: Any = model.predict(crops, imgsz=s.classifier_imgsz, device=0, verbose=False, **quant)
        return [top_reading(r.probs.data.float().cpu().numpy(), names) for r in res]

    views_list = args.views or [s.value_views]
    out: dict[str, Any] = {"weights": weights.name, "model": model_path.name, "configured_views": s.value_views}
    print(f"classifier: {model_path.name} ({'FP16' if s.half else 'FP32'})")
    val = validation_items()
    items = [it for src in ("bdd", "bddother", "hardneg") for it in val[src]]
    truth = [it[2] for it in items]
    n_speed = sum(t is not None for t in truth)
    print(f"validation boxes: { {k: len(v) for k, v in val.items()} }, with a speed value {n_speed}")
    if not val["bddother"] or not n_speed:
        raise SystemExit("no real look-alikes or no speed limits in the validation fraction")
    chosen: float | None = None
    for views in views_list:
        rows = sweep(read_items(classify, items, s.classifier_imgsz, views), truth)
        safe = lowest_safe(rows)
        print(f"views {views}: lowest threshold with 0 wrong on BDD validation: {safe}")
        for r in rows:
            print(f"  {r}")
        out[f"validation_views{views}"] = {"sweep": rows, "lowest_safe": safe}
        if views == s.value_views:
            chosen = safe
    out |= {"threshold_from_validation": chosen, "validation_boxes": {k: len(v) for k, v in val.items()}}
    c_items = ctacke_items()
    if c_items:
        c_rows = sweep(read_items(classify, c_items, s.classifier_imgsz, s.value_views), [it[2] for it in c_items])
        print(f"ctacke val (reported only, leaks with ctacke train; {len(c_items)} boxes):")
        for r in c_rows:
            print(f"  {r}")
        out["ctacke_val_sweep"] = c_rows

    sets = ["original", "fresh"] if args.heldout == "both" else ([] if args.heldout == "none" else [args.heldout])
    for name in sets:
        h_items, h_readable = heldout_items(HELDOUT[name], s)
        preds = read_items(classify, h_items, s.classifier_imgsz, s.value_views)
        h_truth = [it[2] for it in h_items]
        r_idx = [i for i, ok in enumerate(h_readable) if ok]
        readable_rows = sweep([preds[i] for i in r_idx], [h_truth[i] for i in r_idx])
        all_rows = sweep(preds, h_truth)
        mistakes = [
            {"truth": h_truth[i], "said": pred[0], "p": round(pred[1], 3)}
            for i, pred in enumerate(preds)
            if pred is not None and pred[0] is not None and pred[0] != h_truth[i]
        ]
        print(f"held-out {name}: speed limits with a value {len(h_items)} (clearly readable {len(r_idx)})")
        for r in readable_rows:
            print(f"  readable {r}")
        for r in all_rows:
            print(f"  all      {r}")
        print(f"  consensus readings that differ from the label (any probability): {mistakes}")
        out[f"heldout_{name}"] = {"readable_sweep": readable_rows, "all_sweep": all_rows, "top1_mistakes": mistakes}
    if args.write_metrics:
        prev = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
        prev["value_classifier"] = out
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    if chosen is None:
        print(f"NO safe threshold: wrong values even at {max(THRESHOLDS)} on BDD validation")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
