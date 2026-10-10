"""Evaluate the sign detector + value classifier exactly as the app runs them (perception/signs.py).

    .venv/Scripts/python scripts/signs/evaluate.py bdd          # hand-checked BDD100K held-out set
    .venv/Scripts/python scripts/signs/evaluate.py public       # ctacke "val" split (public, US dashcam)
    .venv/Scripts/python scripts/signs/evaluate.py all --write-metrics

Every frame goes through SignDetector.detect() at a low detection threshold (0.05) with the value classifier
always run, then the app's thresholds (config.yaml `signs:`) are applied offline, so one pass gives both the
operating point and the precision/recall curves.

Matching: a detection hits a labeled sign of the same kind at IoU >= 0.5 (greedy, most confident first).
Detections on ignore regions (signs of unknown kind, too small to label) count neither way. Speed values: every
reported value is checked; a value on a labeled speed limit with a different number, or on any labeled non-speed
sign, is WRONG. A value on an ignore region can't be checked and is listed separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.sign_classes import KINDS, VALUE_KINDS  # noqa: E402

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
HELDOUT = REPO_ROOT / "eval" / "sign_labels_bdd.json"
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"
IOU_HIT = 0.5


@dataclass
class Gt:
    """A labeled sign (readable: speed limits a person reads with certainty; None for other kinds)."""

    box: tuple[float, float, float, float]
    kind: str
    value: int | None
    readable: bool | None


@dataclass
class Det:
    """One raw detection with the classifier's top value (if any) and its probability."""

    box: tuple[float, float, float, float]
    kind: str
    conf: float
    value: int | None
    value_conf: float | None


@dataclass
class ImageResult:
    """Labels and raw detections of one frame."""

    name: str
    gts: list[Gt]
    ignore: list[tuple[float, float, float, float]]
    dets: list[Det] = field(default_factory=list)
    scale: float = 1.0  # frame width / 1280 (value_min_px is measured at 1280)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two xyxy boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def on_ignore(box: tuple[float, ...], ignore: list[tuple[float, float, float, float]]) -> bool:
    """A detection over an ignore region: its center inside the region, or IoU >= 0.3."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any((g[0] <= cx <= g[2] and g[1] <= cy <= g[3]) or iou(box, g) >= 0.3 for g in ignore)


def match(res: ImageResult, conf_for: dict[str, float]) -> tuple[list[tuple[Det, Gt | None]], list[Gt]]:
    """Greedy same-kind matching at IoU >= IOU_HIT for detections above their kind's threshold.

    Returns ([(detection, matched gt or None)] excluding detections on ignore regions, unmatched gts).
    """
    dets = sorted((d for d in res.dets if d.conf >= conf_for[d.kind]), key=lambda d: -d.conf)
    free = list(range(len(res.gts)))
    pairs: list[tuple[Det, Gt | None]] = []
    for d in dets:
        best, best_iou = None, IOU_HIT
        for i in free:
            g = res.gts[i]
            if g.kind == d.kind and iou(d.box, g.box) >= best_iou:
                best, best_iou = i, iou(d.box, g.box)
        if best is not None:
            free.remove(best)
            pairs.append((d, res.gts[best]))
        elif not on_ignore(d.box, res.ignore):
            pairs.append((d, None))
    return pairs, [res.gts[i] for i in free]


def average_precision(results: list[ImageResult], kind: str) -> tuple[float, int]:
    """VOC all-point AP50 for one kind over all detections (>= 0.05). Returns (AP, number of labeled signs)."""
    n_gt = sum(g.kind == kind for r in results for g in r.gts)
    if n_gt == 0:
        return float("nan"), 0
    scored: list[tuple[float, bool]] = []
    for r in results:
        pairs, _ = match(r, defaultdict(lambda: 0.0))
        scored += [(d.conf, g is not None) for d, g in pairs if d.kind == kind]
    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([s[1] for s in scored]) if scored else np.zeros(0)
    fp = np.cumsum([not s[1] for s in scored]) if scored else np.zeros(0)
    recall = tp / n_gt if len(tp) else np.zeros(0)
    precision = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.zeros(0)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), n_gt


def reported_value(d: Det, scale: float, value_conf: float, min_px: float) -> int | None:
    """The value the app would report for this detection (perception/signs.py rules)."""
    if d.kind not in VALUE_KINDS or d.value is None or d.value_conf is None:
        return None
    if (d.box[3] - d.box[1]) / scale < min_px or d.value_conf < value_conf:
        return None
    return d.value


def summarize(results: list[ImageResult], cfg: Any) -> dict[str, Any]:
    """Per-kind precision/recall at the app's thresholds, AP50, and the speed-value outcome."""
    conf_for = {k: cfg.class_conf.get(k, cfg.conf) for k in KINDS}
    per_kind: dict[str, Counter] = {k: Counter() for k in KINDS}
    value_outcomes: Counter = Counter()
    wrong: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for r in results:
        pairs, missed = match(r, conf_for)
        for d, g in pairs:
            per_kind[d.kind]["tp" if g is not None else "fp"] += 1
            v = reported_value(d, r.scale, cfg.value_conf, cfg.value_min_px)
            if v is not None and (g is None or g.value != v):
                wrong.append(
                    {
                        "image": r.name,
                        "box": [round(x) for x in d.box],
                        "said": v,
                        "truth": None if g is None else g.value,
                        "p": round(d.value_conf or 0, 3),
                    }
                )
        for g in missed:
            per_kind[g.kind]["fn"] += 1
        # Values on ignore regions (unknown signs) can't be checked: listed for a look.
        for d in r.dets:
            if d.conf >= conf_for[d.kind] and on_ignore(d.box, r.ignore):
                v = reported_value(d, r.scale, cfg.value_conf, cfg.value_min_px)
                if v is not None and not any(iou(d.box, g.box) >= IOU_HIT for g in r.gts):
                    unverified.append({"image": r.name, "box": [round(x) for x in d.box], "said": v})
        # Readable speed limits: right / wrong / no value / missed.
        hit = {id(g): d for d, g in pairs if g is not None}
        for g in r.gts:
            if g.kind not in VALUE_KINDS or not g.readable:
                continue
            hd = hit.get(id(g))
            if hd is None:
                value_outcomes[f"{g.kind}:missed"] += 1
                continue
            v = reported_value(hd, r.scale, cfg.value_conf, cfg.value_min_px)
            outcome = "no_value" if v is None else ("right" if v == g.value else "wrong")
            value_outcomes[f"{g.kind}:{outcome}"] += 1
    kinds_out = {}
    for k in KINDS:
        c = per_kind[k]
        ap, n = average_precision(results, k)
        if n == 0 and not c["fp"]:
            continue
        kinds_out[k] = {
            "labeled": n,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "precision": round(c["tp"] / max(1, c["tp"] + c["fp"]), 3),
            "recall": round(c["tp"] / max(1, c["tp"] + c["fn"]), 3),
            "ap50": None if np.isnan(ap) else round(ap, 3),
        }
    readable = sum(v for k, v in value_outcomes.items() if k.startswith("speed_limit:"))
    right = value_outcomes["speed_limit:right"]
    return {
        "per_kind": kinds_out,
        "speed_values": dict(value_outcomes),
        "readable_speed_limits": readable,
        "readable_right_value_rate": round(right / max(1, readable), 3),
        "wrong_values": wrong,
        "unverifiable_values": unverified,
    }


def value_sweep(results: list[ImageResult], cfg: Any) -> list[dict[str, Any]]:
    """Right/wrong value counts on readable speed limits for a range of classifier thresholds."""
    out = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99):
        s = summarize(results, cfg.model_copy(update={"value_conf": t}))
        out.append(
            {
                "value_conf": t,
                "right": s["speed_values"].get("speed_limit:right", 0),
                "wrong_on_readable": s["speed_values"].get("speed_limit:wrong", 0),
                "wrong_total": len(s["wrong_values"]),
                "readable": s["readable_speed_limits"],
            }
        )
    return out


def run_detector(items: list[tuple[str, Path, list[Gt], list[Any]]], cfg: Any) -> tuple[list[ImageResult], dict]:
    """Run SignDetector on every frame (low thresholds so the curves can be computed offline)."""
    from perception.signs import SignDetector

    raw_cfg = cfg.model_copy(update={"conf": 0.05, "class_conf": {}, "value_conf": 0.0, "value_min_px": 1.0})
    det = SignDetector(raw_cfg, REPO_ROOT)
    results: list[ImageResult] = []
    times: list[float] = []
    for name, path, gts, ignore in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        frame = Frame(img, t=0.0, index=len(results), captured_at=time.perf_counter())
        dets = det.detect(frame)
        times.append(det.last_timings_ms["total"])
        r = ImageResult(name, gts, [tuple(b) for b in ignore], scale=frame.width / 1280)
        r.dets = [Det(d.box, d.kind, d.conf, d.value, d.value_conf) for d in dets]
        results.append(r)
    timing = {
        "frames": len(times),
        "median_ms": round(float(np.median(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "model": det.det_path.name,
        "classifier": det.cls_path.name,
    }
    return results, timing


def bdd_items() -> list[tuple[str, Path, list[Gt], list[Any]]]:
    """The hand-checked BDD100K held-out frames."""
    data = json.loads(HELDOUT.read_text(encoding="utf-8"))
    items: list[tuple[str, Path, list[Gt], list[Any]]] = []
    for im in data["images"]:
        gts = [Gt(tuple(s["box"]), s["kind"], s["value"], s["readable"]) for s in im["signs"]]
        items.append((im["image"], BDD_IMAGES / im["image"], gts, im["ignore"]))
    return items


def public_items() -> list[tuple[str, Path, list[Gt], list[Any]]]:
    """ctacke "val" (public, 1920x1080): its speed-limit boxes with the re-read checks of verify_ctacke.py.

    Boxes whose kind can't be told (dark/blurred blobs) are ignore regions; a box whose number can't be read is
    a labeled sign but not "readable" (it counts for detection, not for the value rate).
    """
    from scripts.signs.prepare_data import ctacke_samples

    items: list[tuple[str, Path, list[Gt], list[Any]]] = []
    for s in ctacke_samples("val"):
        gts = [Gt(b.xyxy, b.kind, b.value, b.cls_label is None) for b in s.boxes if b.kind is not None]
        ignore = [b.xyxy for b in s.boxes if b.kind is None]
        items.append((f"{s.image.parent.name}/{s.image.stem}", s.image, gts, ignore))
    return items


def main(argv: list[str] | None = None) -> int:
    """Run the chosen evaluations and print (optionally save) the metrics."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["bdd", "public", "all"])
    p.add_argument("--write-metrics", action="store_true", help=f"save to {METRICS.relative_to(REPO_ROOT)}")
    p.add_argument("--set", action="append", default=[], help="config override, e.g. signs.engine=pytorch")
    args = p.parse_args(argv)
    cfg = load_config(overrides=args.set).signs
    out: dict[str, Any] = {"config": cfg.model_dump(mode="json")}
    sets = {"bdd": bdd_items, "public": public_items}
    for name in ["bdd", "public"] if args.which == "all" else [args.which]:
        results, timing = run_detector(sets[name](), cfg)
        s = summarize(results, cfg)
        s["value_threshold_sweep"] = value_sweep(results, cfg)
        s["timing_eval_loop"] = timing
        out[name] = s
        print(f"== {name}: {len(results)} frames, {timing}")
        for k, v in s["per_kind"].items():
            print(f"  {k:20s} {v}")
        print(f"  speed values: {s['speed_values']}  right-value rate on readable: {s['readable_right_value_rate']}")
        print(f"  WRONG values: {len(s['wrong_values'])} {s['wrong_values'][:10]}")
        print(f"  unverifiable values: {s['unverifiable_values'][:10]}")
        for row in s["value_threshold_sweep"]:
            print(f"    {row}")
    if args.write_metrics:
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
        prev.update(out)
        METRICS.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"metrics -> {METRICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
