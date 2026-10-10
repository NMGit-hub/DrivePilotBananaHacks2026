"""Speed-value classifier on its own: crops at the labeled boxes, so value reading is measured without the detector.

    .venv/Scripts/python scripts/signs/eval_values.py                 # threshold from validation, then held-out
    .venv/Scripts/python scripts/signs/eval_values.py --write-metrics

1. Threshold: on the classifier's validation crops from real signs (BDD100K train-split hold-out and ctacke val,
   data/signs/cls/val, never the BDD held-out), the lowest probability threshold with no wrong value.
2. Held-out: the hand-checked BDD100K speed limits (eval/sign_labels_bdd.json) cropped from the full frame exactly
   as perception/signs.py crops a detection; right / wrong / withheld at that threshold, and a sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from perception.signs import crop_square, parse_value_class  # noqa: E402

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
CLS_VAL = REPO_ROOT / "data" / "signs" / "cls" / "val"
HELDOUT = REPO_ROOT / "eval" / "sign_labels_bdd.json"
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)


def classify(model: Any, crops: list[np.ndarray], imgsz: int) -> list[tuple[int | None, float]]:
    """Top-1 (value or None for the reject class, probability) per crop."""
    out = []
    for c in crops:
        r = model.predict(c, imgsz=imgsz, device=0, verbose=False)[0]
        probs = r.probs.data.float().cpu().numpy()
        top = int(np.argmax(probs))
        out.append((parse_value_class(model.names[top]), float(probs[top])))
    return out


def sweep(preds: list[tuple[int | None, float]], truth: list[int | None]) -> list[dict[str, Any]]:
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


def main(argv: list[str] | None = None) -> int:
    """Pick the value threshold on validation crops, then score the held-out speed limits."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=None, help="classifier .pt (default: config signs.classifier)")
    p.add_argument("--write-metrics", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config()
    s = cfg.signs
    model = YOLO(str(args.weights or cfg.resolve_path(s.classifier_weights)), task="classify")

    # 1. validation crops from real signs (already square, CLS_SIZE): file prefix bdd_/ctacke_ = real
    val_crops, val_truth = [], []
    for d in sorted(CLS_VAL.iterdir()):
        for f in sorted(d.glob("*.png")):
            if f.name.startswith(("bdd_", "ctacke_")):
                img = cv2.imread(str(f))
                if img is not None:
                    val_crops.append(img)
                    # "..._true30.png": a real 30 mph sign trained as "none" (unreadable): reading 30 is right
                    true = f.stem.rpartition("_true")[2] if "_true" in f.stem else None
                    val_truth.append(int(true) if true else parse_value_class(d.name))
    val_rows = sweep(classify(model, val_crops, s.classifier_imgsz), val_truth)
    safe = [r["threshold"] for r in val_rows if r["wrong"] == 0]
    chosen = min(safe) if safe else max(THRESHOLDS)
    print(f"validation (real crops: {len(val_crops)}, speed values {sum(t is not None for t in val_truth)}):")
    for r in val_rows:
        print(f"  {r}")
    print(f"chosen value threshold (lowest with 0 wrong on validation): {chosen}")

    # 2. held-out BDD speed limits
    data = json.loads(HELDOUT.read_text(encoding="utf-8"))
    crops, truth, readable = [], [], []
    for im in data["images"]:
        sl = [x for x in im["signs"] if x["kind"] in ("speed_limit", "school_speed_limit")]
        if not sl:
            continue
        frame = cv2.imread(str(BDD_IMAGES / im["image"]))
        if frame is None:
            raise SystemExit(f"missing held-out frame {im['image']} (download BDD valid: scripts/signs/README.md)")
        for x in sl:
            if x["value"] is None:
                continue
            c = crop_square(frame, tuple(x["box"]), s.classifier_imgsz)
            if c is not None:
                crops.append(c)
                truth.append(x["value"])
                readable.append(bool(x["readable"]))
    preds = classify(model, crops, s.classifier_imgsz)
    all_rows = sweep(preds, truth)
    r_idx = [i for i, ok in enumerate(readable) if ok]
    readable_rows = sweep([preds[i] for i in r_idx], [truth[i] for i in r_idx])
    at = next(r for r in readable_rows if r["threshold"] == chosen)
    print(f"held-out BDD speed limits with a value: {len(crops)} (clearly readable {len(r_idx)})")
    for r in readable_rows:
        print(f"  readable {r}")
    for r in all_rows:
        print(f"  all      {r}")
    mistakes = [
        {"truth": truth[i], "said": preds[i][0], "p": round(preds[i][1], 3)}
        for i in range(len(crops))
        if preds[i][0] is not None and preds[i][0] != truth[i]
    ]
    print(f"top-1 mistakes (any probability): {mistakes}")
    print(f"at {chosen}: {at}")
    if args.write_metrics:
        prev = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
        prev["value_classifier"] = {
            "threshold_from_validation": chosen,
            "validation_sweep": val_rows,
            "heldout_readable_sweep": readable_rows,
            "heldout_all_sweep": all_rows,
            "heldout_top1_mistakes": mistakes,
        }
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
