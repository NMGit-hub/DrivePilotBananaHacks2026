"""Evaluate the sign detector + value classifier exactly as the app runs them (perception/signs.py).

    .venv/Scripts/python scripts/signs/evaluate.py fresh        # fresh BDD100K held-out slice (headline)
    .venv/Scripts/python scripts/signs/evaluate.py bdd          # original held-out set (add --no-audit)
    .venv/Scripts/python scripts/signs/evaluate.py public       # ctacke "val" split (public, US dashcam)
    .venv/Scripts/python scripts/signs/evaluate.py tune         # BDD train hold-out (pseudo-labels): for thresholds
    .venv/Scripts/python scripts/signs/evaluate.py all --write-metrics

Every frame goes once through SignDetector.raw_detections() (detection threshold 0.05, every speed-kind box
classified); the app's rules are then applied offline by the same perception.signs.select() that detect() uses,
so one pass gives the operating point at config.yaml `signs:` and the sweeps. Raw detections are cached in
data/signs/eval_raw/<set>.json (--cached reuses them, e.g. to sweep thresholds without the GPU).

Matching: a detection hits a labeled sign of the same kind at IoU >= 0.5 (greedy, most confident first).
Detections on ignore regions (signs of unknown kind, too small to label) count neither way.

Speed values: every reported value is judged. It is matched to the labeled sign of ANY kind it overlaps most
(IoU >= 0.5): right if that sign is a speed kind with the same value, unverifiable if it is a speed kind whose
value nobody could read, WRONG otherwise (another value, or a sign that carries no speed). A value matching no
labeled sign is unverifiable if it overlaps an ignore region (IoU >= 0.3) and WRONG elsewhere. Unverifiable values
are listed and must be checked by eye; thresholds are chosen with wrong + unverifiable both at 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import SignsConfig, load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.sign_classes import KINDS, VALUE_KINDS  # noqa: E402
from perception.signs import RawSign, SignDetection, select  # noqa: E402

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
HELDOUT = REPO_ROOT / "eval" / "sign_labels_bdd.json"
HELDOUT_FRESH = REPO_ROOT / "eval" / "sign_labels_bdd_fresh.json"
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"
RAW_CACHE = REPO_ROOT / "data" / "signs" / "eval_raw"
IOU_HIT = 0.5
EVAL_CONF = 0.05  # raw detection threshold of the evaluation (curves are computed from here up)
VALUE_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)

BoxT = tuple[float, float, float, float]
Item = tuple[str, Path, list["Gt"], list[BoxT]]


@dataclass
class Gt:
    """A labeled sign (readable: speed limits a person reads with certainty; None for other kinds)."""

    box: BoxT
    kind: str
    value: int | None
    readable: bool | None


@dataclass
class ImageResult:
    """Labels and raw detections of one frame."""

    name: str
    gts: list[Gt]
    ignore: list[BoxT]
    raw: list[RawSign] = field(default_factory=list)
    width: int = 1280
    height: int = 720


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two xyxy boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def on_ignore(box: tuple[float, ...], ignore: list[BoxT]) -> bool:
    """A detection over an ignore region: its center inside the region, or IoU >= 0.3."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any((g[0] <= cx <= g[2] and g[1] <= cy <= g[3]) or iou(box, g) >= 0.3 for g in ignore)


def match(
    gts: list[Gt], ignore: list[BoxT], dets: list[SignDetection]
) -> tuple[list[tuple[SignDetection, Gt | None]], list[Gt]]:
    """Greedy same-kind matching at IoU >= IOU_HIT, most confident detection first.

    Returns ([(detection, matched gt or None)] without unmatched detections on ignore regions, unmatched gts).
    """
    free = list(range(len(gts)))
    pairs: list[tuple[SignDetection, Gt | None]] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        best, best_iou = None, IOU_HIT
        for i in free:
            o = iou(d.box, gts[i].box)
            if gts[i].kind == d.kind and o >= best_iou:
                best, best_iou = i, o
        if best is not None:
            free.remove(best)
            pairs.append((d, gts[best]))
        elif not on_ignore(d.box, ignore):
            pairs.append((d, None))
    return pairs, [gts[i] for i in free]


def judge_value(d: SignDetection, gts: list[Gt], ignore: list[BoxT]) -> tuple[str, int | None]:
    """("right" | "wrong" | "unverifiable", the labeled value) for a detection that reports a value."""
    best: Gt | None = None
    best_iou = IOU_HIT
    for g in gts:
        o = iou(d.box, g.box)
        if o >= best_iou:
            best, best_iou = g, o
    if best is not None:
        if best.kind not in VALUE_KINDS:
            return "wrong", None
        if best.value is None:
            return "unverifiable", None
        return ("right" if best.value == d.value else "wrong"), best.value
    # Only a real overlap with a region of unknown kind excuses a value (not a centre inside a tiny box).
    return ("unverifiable" if any(iou(d.box, g) >= 0.3 for g in ignore) else "wrong"), None


def average_precision(results: list[ImageResult], kind: str, cfg: SignsConfig) -> tuple[float, int]:
    """VOC all-point AP50 for one kind over all detections >= EVAL_CONF (after the app's duplicate rule)."""
    n_gt = sum(g.kind == kind for r in results for g in r.gts)
    if n_gt == 0:
        return float("nan"), 0
    low = cfg.model_copy(update={"conf": EVAL_CONF, "class_conf": {}})
    scored: list[tuple[float, bool]] = []
    for r in results:
        pairs, _ = match(r.gts, r.ignore, select(r.raw, low, r.width, r.height))
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


def summarize(results: list[ImageResult], cfg: SignsConfig, with_ap: bool = True) -> dict[str, Any]:
    """Per-kind precision/recall at the app's thresholds, AP50, and the judged speed values."""
    per_kind: dict[str, Counter] = {k: Counter() for k in KINDS}
    value_outcomes: Counter = Counter()
    judged: Counter = Counter()
    wrong: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for r in results:
        dets = select(r.raw, cfg, r.width, r.height)
        pairs, missed = match(r.gts, r.ignore, dets)
        for d, g in pairs:
            per_kind[d.kind]["tp" if g is not None else "fp"] += 1
        for g in missed:
            per_kind[g.kind]["fn"] += 1
        for d in dets:
            if d.value is None:
                continue
            outcome, truth = judge_value(d, r.gts, r.ignore)
            judged[outcome] += 1
            rec = {"image": r.name, "box": [round(x) for x in d.box], "kind": d.kind, "said": d.value}
            if outcome == "wrong":
                wrong.append({**rec, "truth": truth, "p": round(d.value_conf or 0.0, 3)})
            elif outcome == "unverifiable":
                unverified.append(rec)
        # Readable speed limits: right / wrong / no value / missed (by the same-kind match).
        hit = {id(g): d for d, g in pairs if g is not None}
        for g in r.gts:
            if g.kind not in VALUE_KINDS or not g.readable:
                continue
            hd = hit.get(id(g))
            if hd is None:
                value_outcomes[f"{g.kind}:missed"] += 1
                continue
            outcome = "no_value" if hd.value is None else ("right" if hd.value == g.value else "wrong")
            value_outcomes[f"{g.kind}:{outcome}"] += 1
    kinds_out: dict[str, dict[str, Any]] = {}
    for k in KINDS:
        c = per_kind[k]
        ap, n = (
            average_precision(results, k, cfg)
            if with_ap
            else (float("nan"), sum(g.kind == k for r in results for g in r.gts))
        )
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
        "frames": len(results),
        "per_kind": kinds_out,
        "speed_values": dict(value_outcomes),
        "readable_speed_limits": readable,
        "readable_right_value_rate": round(right / max(1, readable), 3),
        "reported_values": dict(judged),
        "wrong_values": wrong,
        "unverifiable_values": unverified,
    }


def value_sweep(results: list[ImageResult], cfg: SignsConfig) -> list[dict[str, Any]]:
    """Judged values and readable-limit outcomes for a range of classifier thresholds (value_conf)."""
    out = []
    for t in VALUE_SWEEP:
        s = summarize(results, cfg.model_copy(update={"value_conf": t}), with_ap=False)
        sv = s["speed_values"]
        out.append(
            {
                "value_conf": t,
                "readable": sum(v for k, v in sv.items() if k.split(":")[0] in VALUE_KINDS),  # both speed kinds
                "right_on_readable": sv.get("speed_limit:right", 0) + sv.get("school_speed_limit:right", 0),
                "wrong_total": len(s["wrong_values"]),
                "unverifiable": len(s["unverifiable_values"]),
                # a value that can't be checked may be wrong: thresholds are chosen on both together
                "wrong_or_unverifiable": len(s["wrong_values"]) + len(s["unverifiable_values"]),
            }
        )
    return out


def model_fingerprint(cfg: SignsConfig) -> dict[str, Any]:
    """What raw detections depend on: the weights (sha256; an engine's sidecar ties it to them) and the input."""
    from perception.signs import file_sha256

    det_w, cls_w = (p if p.is_absolute() else REPO_ROOT / p for p in (cfg.weights, cfg.classifier_weights))
    return {
        "detector_sha256": file_sha256(det_w) if det_w.exists() else None,
        "classifier_sha256": file_sha256(cls_w) if cls_w.exists() else None,
        "imgsz": list(cfg.imgsz),
        "classifier_imgsz": cfg.classifier_imgsz,
        "half": cfg.half,
        "engine": cfg.engine,
    }


def _raw_to_json(r: RawSign) -> dict[str, Any]:
    return asdict(r)


def _raw_from_json(d: dict[str, Any]) -> RawSign:
    return RawSign(d["kind"], d["conf"], tuple(d["box"]), d["top_value"], d["top_prob"])


def run_detector(items: list[Item], cfg: SignsConfig) -> tuple[list[ImageResult], dict[str, Any]]:
    """Raw detections of every frame (detection threshold EVAL_CONF, every speed-kind box classified)."""
    from perception.signs import SignDetector

    det = SignDetector(cfg.model_copy(update={"conf": EVAL_CONF, "class_conf": {}}), REPO_ROOT)
    results: list[ImageResult] = []
    times: list[float] = []
    for name, path, gts, ignore in items:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"evaluation frame missing or unreadable: {path} (see scripts/signs/README.md)")
        frame = Frame(img, t=0.0, index=len(results), captured_at=time.perf_counter())
        raw = det.raw_detections(frame, classify_min_px=1.0)
        times.append(det.last_timings_ms["total"])
        results.append(ImageResult(name, gts, ignore, raw, frame.width, frame.height))
    timing: dict[str, Any] = {"frames": len(times), "model": det.det_path.name, "classifier": det.cls_path.name}
    if times:
        timing |= {"median_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}
    return results, timing


def save_raw(name: str, results: list[ImageResult], timing: dict[str, Any], fingerprint: dict[str, Any]) -> None:
    """Cache raw detections of a set (so thresholds can be swept without the GPU), tagged with the models."""
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    data = {
        "fingerprint": fingerprint,
        "timing": timing,
        "frames": {r.name: {"w": r.width, "h": r.height, "raw": [_raw_to_json(x) for x in r.raw]} for r in results},
    }
    (RAW_CACHE / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def load_raw(name: str, items: list[Item], fingerprint: dict[str, Any]) -> tuple[list[ImageResult], dict[str, Any]]:
    """Raw detections from the cache, joined with the current labels.

    Raises if the cache was made with other models or input settings, or lacks a frame.
    """
    data = json.loads((RAW_CACHE / f"{name}.json").read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint:
        raise ValueError(f"cached raw detections of {name} come from other models/settings: re-run without --cached")
    frames = data["frames"]
    results = []
    for n, _path, gts, ignore in items:
        if n not in frames:
            raise KeyError(f"{n} not in the cached raw detections of {name}: re-run without --cached")
        f = frames[n]
        results.append(ImageResult(n, gts, ignore, [_raw_from_json(x) for x in f["raw"]], f["w"], f["h"]))
    return results, data["timing"]


def heldout_items(path: Path, audit: bool = True) -> list[Item]:
    """Frames of a hand-checked BDD100K held-out file (with its detection-triggered audit unless audit=False)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[Item] = []
    for im in data["images"]:
        signs, ignore = list(im["signs"]), list(im["ignore"])
        if audit and im.get("audit"):
            signs += im["audit"]["signs"]
            ignore += im["audit"]["ignore"]
        gts = [Gt(tuple(s["box"]), s["kind"], s["value"], s["readable"]) for s in signs]
        items.append((im["image"], BDD_IMAGES / im["image"], gts, [tuple(b) for b in ignore]))
    return items


def tune_items() -> list[Item]:
    """BDD100K *train*-split frames of the held-out fraction that no build ever trained on, with pseudo-labels.

    For choosing thresholds end to end (detector false alarms reach the classifier too) without touching the
    hand-checked held-out sets. Pseudo-labels: speed values are Claude's readings, so "readable" = it read one.
    """
    from scripts.signs.prepare_data import bdd_samples, in_val, rel_path, training_images

    trained = training_images()
    pos, neg = bdd_samples("train")
    items: list[Item] = []
    for s in pos + neg:
        if not in_val(s.image) or rel_path(s.image) in trained:
            continue
        gts = [Gt(b.xyxy, b.kind, b.value, b.value is not None) for b in s.boxes if b.kind is not None]
        ignore = [b.xyxy for b in s.boxes if b.kind is None]
        items.append((s.image.name, s.image, gts, ignore))
    return items


def public_items() -> list[Item]:
    """ctacke "val" (public, 1920x1080): its speed-limit boxes with the re-read checks of verify_ctacke.py.

    Boxes whose kind can't be told (dark/blurred blobs) are ignore regions; a box whose number can't be read is
    a labeled sign but not "readable" (it counts for detection, not for the value rate).
    """
    from scripts.signs.prepare_data import ctacke_samples

    items: list[Item] = []
    for s in ctacke_samples("val"):
        gts = [Gt(b.xyxy, b.kind, b.value, b.cls_label is None) for b in s.boxes if b.kind is not None]
        ignore = [b.xyxy for b in s.boxes if b.kind is None]
        items.append((f"{s.image.parent.name}/{s.image.stem}", s.image, gts, ignore))
    return items


SETS = {
    "fresh": ("fresh", lambda: heldout_items(HELDOUT_FRESH)),
    "bdd": ("bdd", lambda: heldout_items(HELDOUT, audit=True)),
    "bdd_noaudit": ("bdd", lambda: heldout_items(HELDOUT, audit=False)),
    "public": ("public", public_items),
    "tune": ("tune", tune_items),
}


def main(argv: list[str] | None = None) -> int:
    """Run the chosen evaluations and print (optionally save) the metrics."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=[*SETS, "all"])
    p.add_argument("--write-metrics", action="store_true", help=f"save to {METRICS.relative_to(REPO_ROOT)}")
    p.add_argument("--set", action="append", default=[], help="config override, e.g. signs.value_conf=0.95")
    p.add_argument("--no-audit", action="store_true", help="'bdd' without the detection-triggered audit")
    p.add_argument("--cached", action="store_true", help="reuse data/signs/eval_raw/<set>.json (no GPU)")
    p.add_argument("--show", type=int, default=20, help="wrong/unverifiable values to print")
    p.add_argument("--tag", default="", help="suffix of the raw-detection cache file (to compare models)")
    args = p.parse_args(argv)
    cfg = load_config(overrides=args.set).signs
    names = (
        list(SETS) if args.which == "all" else ["bdd_noaudit" if args.which == "bdd" and args.no_audit else args.which]
    )
    fingerprint = model_fingerprint(cfg)
    out: dict[str, Any] = {"config": cfg.model_dump(mode="json"), "models": fingerprint}
    done: set[str] = set()  # sets whose frames already ran in this invocation (bdd and bdd_noaudit share frames)
    for name in names:
        cache_name, make = SETS[name]
        cache_name += args.tag
        items = make()
        if args.cached or cache_name in done:
            results, timing = load_raw(cache_name, items, fingerprint)
        else:
            results, timing = run_detector(items, cfg)
            save_raw(cache_name, results, timing, fingerprint)
            done.add(cache_name)
        s = summarize(results, cfg)
        s["value_threshold_sweep"] = value_sweep(results, cfg)
        s["timing_eval_loop"] = timing  # raw_detections with every speed box classified: not the app's timing
        out[name] = s
        print(f"== {name}: {len(results)} frames, {timing}")
        for k, v in s["per_kind"].items():
            print(f"  {k:20s} {v}")
        print(f"  readable limits: {s['speed_values']}  right-value rate: {s['readable_right_value_rate']}")
        print(f"  reported values: {s['reported_values']}")
        print(f"  WRONG values: {len(s['wrong_values'])} {s['wrong_values'][: args.show]}")
        print(f"  unverifiable values: {len(s['unverifiable_values'])} {s['unverifiable_values'][: args.show]}")
        for row in s["value_threshold_sweep"]:
            print(f"    {row}")
        for key in ("wrong_total", "wrong_or_unverifiable"):
            safe = [r["value_conf"] for r in s["value_threshold_sweep"] if r[key] == 0]
            s[f"lowest_value_conf_without_{key}"] = min(safe) if safe else None
            print(f"  lowest value_conf with 0 {key}: {s[f'lowest_value_conf_without_{key}']}")
    if args.write_metrics:
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
        prev.update(out)
        METRICS.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"metrics -> {METRICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
