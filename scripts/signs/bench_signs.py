"""Per-frame time of the sign detector on this GPU (TensorRT FP16), alone or while the app's perception runs.

    .venv/Scripts/python scripts/signs/bench_signs.py                          # 30 s of a dev clip, every frame
    .venv/Scripts/python scripts/signs/bench_signs.py --fps 3 --duration 60    # the app's rate (a few fps)
    .venv/Scripts/python scripts/signs/bench_signs.py --with-perception        # + YOLO26s/ByteTrack at 30 fps

--with-perception runs the app's own perception pipeline (scripts/bench_pipeline.py in real-time mode, a separate
process) on the same clip at the same time, so the GPU is shared the way it will be in the app; that process's
fps report shows whether the sign detector slows perception. Results: runs/signs/bench_<stamp>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.signs import SignDetector  # noqa: E402

DEFAULT_CLIP = "data/videos/dev/red_light_rain_school_zone__b61456bb-5bfb50ee.mov"


def main(argv: list[str] | None = None) -> int:
    """Time SignDetector.detect() on a clip and report median / p95 per stage."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path, default=REPO_ROOT / DEFAULT_CLIP)
    p.add_argument("--fps", type=float, default=0.0, help="frames per second to process (0 = every frame, flat out)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds of video")
    p.add_argument("--with-perception", action="store_true", help="run bench_pipeline.py concurrently")
    p.add_argument("--set", action="append", default=[], help="config override (also passed to bench_pipeline)")
    args = p.parse_args(argv)
    cfg = load_config(overrides=args.set)
    if not args.clip.exists():
        raise SystemExit(f"clip not found: {args.clip} (scripts/fetch_dev_clips.py)")
    perception = None
    if args.with_perception:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "bench_pipeline.py"),
            "--clip",
            str(args.clip),
            "--mode",
            "realtime",
            "--duration",
            str(args.duration + 15),
        ]
        for o in args.set:
            cmd += ["--set", o]
        perception = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(12)  # let the perception pipeline warm up and reach steady state first
    det = SignDetector(cfg.signs, cfg.root)
    devices = {w: det._verify_gpu(w, m) for w, m in (("detector", det.det), ("classifier", det.cls))}
    cap = cv2.VideoCapture(str(args.clip))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if args.fps <= 0 else max(1, round(src_fps / args.fps))
    timings: list[dict[str, float]] = []
    n_signs = 0
    i = 0
    t_start = time.perf_counter()
    while i < args.duration * src_fps:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            frame = Frame(img, t=i / src_fps, index=i, captured_at=time.perf_counter())
            n_signs += len(det.detect(frame))
            timings.append(dict(det.last_timings_ms))
            if args.fps > 0:  # pace like the app: one frame every 1/fps seconds of wall time
                wait = t_start + len(timings) / args.fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
        i += 1
    cap.release()

    def stat(key: str) -> dict[str, float]:
        v = np.array([t[key] for t in timings])
        return {
            "median": round(float(np.median(v)), 2),
            "p95": round(float(np.percentile(v, 95)), 2),
            "max": round(float(v.max()), 2),
        }

    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    report: dict[str, Any] = {
        "clip": args.clip.name,
        "frames": len(timings),
        "rate_fps": args.fps or "flat out",
        "signs_detected": n_signs,
        "devices": devices,
        "gpu": torch.cuda.get_device_name(0),
        "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1e6),  # misses TensorRT's own memory
        "gpu_used_mb_whole_device": round((gpu_total - gpu_free) / 1e6),  # every process, TensorRT included
        "ms": {
            k: stat(k) for k in ("total", "detect_total", "inference", "preprocess", "postprocess", "classify_total")
        },
        "with_perception": args.with_perception,
    }
    if perception is not None:
        out, _ = perception.communicate(timeout=300)
        report["perception_tail"] = out.strip().splitlines()[-25:]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPO_ROOT / "runs" / "signs" / f"bench_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "perception_tail"}, indent=2))
    if perception is not None:
        print("\n".join(report["perception_tail"]))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
