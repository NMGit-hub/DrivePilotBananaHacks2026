"""Per-frame time of SignDetector.detect() on held-out frames that hold speed limits (the classifier's worst case).

    .venv/Scripts/python scripts/signs/bench_frames.py --write-metrics

bench_signs.py times a whole clip, where most frames have no speed sign, so the value classifier (5 views per speed
sign) rarely runs. This times detect() on every fresh held-out frame with a labeled speed-limit box, after a warmup,
and reports total / classify times over all of them and over the frames where a value was actually read. With
--write-metrics the result and the clip benchmarks found in runs/signs/bench_*.json go to metrics.json "timing".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.sign_classes import VALUE_KINDS  # noqa: E402

BDD_IMAGES = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
FRESH = REPO_ROOT / "eval" / "sign_labels_bdd_fresh.json"
METRICS = REPO_ROOT / "models" / "signs" / "metrics.json"


def stats(values: list[float]) -> dict[str, float]:
    """Median / p95 / max in ms (empty -> {})."""
    if not values:
        return {}
    return {
        "median": round(float(np.median(values)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "max": round(float(np.max(values)), 2),
    }


def main(argv: list[str] | None = None) -> int:
    """Time detect() on the fresh held-out frames with speed limits."""
    from perception.signs import SignDetector

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-metrics", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config()
    det = SignDetector(cfg.signs, cfg.root)
    data = json.loads(FRESH.read_text(encoding="utf-8"))
    frames = [im["image"] for im in data["images"] if any(s["kind"] in VALUE_KINDS for s in im["signs"])]
    images = []
    for rel in frames:
        img = cv2.imread(str(BDD_IMAGES / rel))
        if img is None:
            raise SystemExit(f"missing frame {rel}")
        images.append(img)
    for img in images[:10]:  # warm the GPU clocks and caches
        det.detect(Frame(img, 0.0, 0, time.perf_counter()))
    totals, classify, totals_read, classify_read = [], [], [], []
    for n, img in enumerate(images):
        det.detect(Frame(img, 0.0, n, time.perf_counter()))
        t = det.last_timings_ms
        totals.append(t["total"])
        classify.append(t["classify_total"])
        if t["classified"] > 0:
            totals_read.append(t["total"])
            classify_read.append(t["classify_total"])
    out: dict[str, Any] = {
        "frames": len(images),
        "frames_with_a_speed_sign_classified": len(totals_read),
        "value_views": cfg.signs.value_views,
        "total_ms": stats(totals),
        "classify_ms": stats(classify),
        "total_ms_frames_with_classification": stats(totals_read),
        "classify_ms_frames_with_classification": stats(classify_read),
    }
    print(json.dumps(out, indent=2))
    if args.write_metrics:
        timing: dict[str, Any] = {"engines": [det.det_path.name, det.cls_path.name], "heldout_speed_frames": out}
        built = det.det_path.stat().st_mtime  # clip benchmarks of older engines don't describe these models
        for bench in sorted((REPO_ROOT / "runs" / "signs").glob("bench_*.json")):
            if bench.stat().st_mtime > built:
                timing[bench.stem] = json.loads(bench.read_text(encoding="utf-8"))
        prev = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
        prev["timing"] = timing
        METRICS.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"timing -> {METRICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
