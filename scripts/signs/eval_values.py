"""Speed-value classifier on its own: crops at the labeled boxes, so value reading is measured without the detector.

    .venv/Scripts/python scripts/signs/eval_values.py                 # threshold from validation, then held-out
    .venv/Scripts/python scripts/signs/eval_values.py --weights runs/signs/cls5/weights/best.pt
    .venv/Scripts/python scripts/signs/eval_values.py --write-metrics

1. Threshold: on the classifier's real BDD validation crops (data/signs/cls/val, files bdd_*, bddother_*, hardneg_*:
   speed limits, every "other"/not-a-sign box and every curated look-alike of the BDD train split's held-out
   fraction, never trained on), the lowest probability threshold with no wrong value. ctacke val crops are reported
   separately and not used: ctacke's val split shares physical signs with its train split. The app's value_conf is
   the larger of this and the lowest threshold with no wrong value end to end on the tune set (evaluate.py tune).
2. Held-out: the hand-checked BDD100K speed limits (eval/sign_labels_bdd*.json) cropped from the full frame
   exactly as perception/signs.py crops a detection; right / wrong / withheld per threshold.
The classifier runs as the app runs it: its FP16 TensorRT engine when one was built from these weights (else the
weights in FP16 on the GPU), so the probabilities near the threshold are the app's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from perception.signs import (  # noqa: E402
    check_value_classes,
    crop_square,
    engine_path,
    engine_staleness,
    sidecar_path,
    top_reading,
)

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
CLS_VAL = REPO_ROOT / "data" / "signs" / "cls" / "val"
HELDOUT = {
    "original": REPO_ROOT / "eval" / "sign_labels_bdd.json",
    "fresh": REPO_ROOT / "eval" / "sign_labels_bdd_fresh.json",
}
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
SELECTION_PREFIXES = ("bdd_", "bddother_", "hardneg_")  # real BDD crops used to choose the threshold
REPORT_PREFIXES = ("ctacke_",)  # real but leaky: reported only


def classify(model: Any, crops: list[np.ndarray], imgsz: int, half: bool) -> list[tuple[int | None, float]]:
    """Top-1 (value or None for the reject class, probability) per crop, in the app's precision."""
    names = dict(model.names)
    quant: dict[str, Any] = {"quantize": 16} if half else {}
    out = []
    for c in crops:
        r = model.predict(c, imgsz=imgsz, device=0, verbose=False, **quant)[0]
        out.append(top_reading(r.probs.data.float().cpu().numpy(), names))
    return out


def sweep(preds: Sequence[tuple[int | None, float]], truth: Sequence[int | None]) -> list[dict[str, Any]]:
    """Right / wrong / withheld counts per threshold (truth None = not a speed sign: any value is wrong)."""
    rows = []
    for t in THRESHOLDS:
        right = wrong = withheld = 0
        for (v, p), g in zip(preds, truth, strict=True):
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


def crop_truth(path: Path) -> int | None:
    """True value of a validation crop: its folder's value, or "..._true30.png" (a real 30 trained as "none")."""
    if "_true" in path.stem:
        return int(path.stem.rpartition("_true")[2])
    name = path.parent.name
    return int(name[1:]) if name.startswith("v") and name[1:].isdigit() else None


def val_crops(prefixes: tuple[str, ...]) -> tuple[list[np.ndarray], list[int | None]]:
    """Real validation crops whose file name starts with one of prefixes, with their true values."""
    crops, truth = [], []
    for d in sorted(CLS_VAL.iterdir()):
        for f in sorted(d.glob("*.png")):
            if f.name.startswith(prefixes):
                img = cv2.imread(str(f))
                if img is not None:
                    crops.append(img)
                    truth.append(crop_truth(f))
    return crops, truth


def heldout_crops(path: Path, imgsz: int) -> tuple[list[np.ndarray], list[int], list[bool]]:
    """Held-out speed-limit crops with a known value: (crops, values, clearly readable)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    crops, truth, readable = [], [], []
    for im in data["images"]:
        sl = [x for x in im["signs"] if x["kind"] in ("speed_limit", "school_speed_limit") and x["value"] is not None]
        if not sl:
            continue
        frame = cv2.imread(str(BDD_IMAGES / im["image"]))
        if frame is None:
            raise SystemExit(f"missing held-out frame {im['image']} (download BDD valid: scripts/signs/README.md)")
        for x in sl:
            c = crop_square(frame, tuple(x["box"]), imgsz)
            if c is not None:
                crops.append(c)
                truth.append(x["value"])
                readable.append(bool(x["readable"]))
    return crops, truth, readable


def main(argv: list[str] | None = None) -> int:
    """Pick the value threshold on validation crops, then score the held-out speed limits."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=None, help="classifier .pt (default: config); its engine if built")
    p.add_argument("--heldout", choices=["original", "fresh", "both", "none"], default="both")
    p.add_argument("--write-metrics", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config()
    s = cfg.signs
    weights = args.weights or cfg.resolve_path(s.classifier_weights)
    model_path = resolve_classifier(weights, s.classifier_imgsz, s.half)
    model = YOLO(str(model_path), task="classify")
    check_value_classes(dict(model.names))
    out: dict[str, Any] = {"weights": weights.name, "model": model_path.name}
    print(f"classifier: {model_path.name} ({'FP16' if s.half else 'FP32'})")

    def run(crops: list[np.ndarray]) -> list[tuple[int | None, float]]:
        return classify(model, crops, s.classifier_imgsz, s.half)

    per_prefix = {pre: len(val_crops((pre,))[0]) for pre in SELECTION_PREFIXES}
    crops, truth = val_crops(SELECTION_PREFIXES)
    n_speed = sum(t is not None for t in truth)
    print(f"validation for the threshold: real BDD crops per prefix {per_prefix}, speed values {n_speed}")
    if not per_prefix["bddother_"] or not n_speed:
        raise SystemExit("cls/val has no real look-alikes or no speed limits: rebuild it (prepare_data.py cls)")
    rows = sweep(run(crops), truth)
    chosen = lowest_safe(rows)
    for r in rows:
        print(f"  {r}")
    print(f"lowest value threshold with 0 wrong on BDD validation: {chosen}")
    out |= {"threshold_from_validation": chosen, "validation_sweep": rows, "validation_crops": per_prefix}
    c_crops, c_truth = val_crops(REPORT_PREFIXES)
    if c_crops:
        c_rows = sweep(run(c_crops), c_truth)
        print(f"ctacke val (reported only, leaks with ctacke train; {len(c_crops)} crops):")
        for r in c_rows:
            print(f"  {r}")
        out["ctacke_val_sweep"] = c_rows

    sets = ["original", "fresh"] if args.heldout == "both" else ([] if args.heldout == "none" else [args.heldout])
    for name in sets:
        h_crops, h_truth, h_readable = heldout_crops(HELDOUT[name], s.classifier_imgsz)
        preds = run(h_crops)
        r_idx = [i for i, ok in enumerate(h_readable) if ok]
        readable_rows = sweep([preds[i] for i in r_idx], [h_truth[i] for i in r_idx])
        all_rows = sweep(preds, h_truth)
        mistakes = [
            {"truth": h_truth[i], "said": preds[i][0], "p": round(preds[i][1], 3)}
            for i in range(len(h_crops))
            if preds[i][0] is not None and preds[i][0] != h_truth[i]
        ]
        print(f"held-out {name}: speed limits with a value {len(h_crops)} (clearly readable {len(r_idx)})")
        for r in readable_rows:
            print(f"  readable {r}")
        for r in all_rows:
            print(f"  all      {r}")
        print(f"  top-1 mistakes (any probability): {mistakes}")
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
