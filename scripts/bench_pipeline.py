"""Headless pipeline benchmark: FPS, per-stage latency and GPU memory on a video file, checked against targets.

    .venv/Scripts/python scripts/bench_pipeline.py                       # default dev clip, real-time + lock-step
    .venv/Scripts/python scripts/bench_pipeline.py --clip data/videos/x.mp4 --mode realtime --duration 20
    .venv/Scripts/python scripts/bench_pipeline.py --set perception.imgsz=960

Real-time mode plays the clip at its own speed (frames are skipped if perception falls behind), which is what
the live camera does; lock-step processes every frame as fast as possible (throughput). The report is written
to runs/<stamp>_bench/bench.json, and the exit status is 1 if a target is missed.
Targets (BUILD_PLAN.md M1): real-time processing keeps ≥ 25 of 30 fps; p95 perception time (detect + track +
ego motion) < 40 ms; model weights on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from core.runlog import RunRecorder  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.gpu_monitor import GpuMonitor  # noqa: E402
from perception.pipeline import PerceptionWorker  # noqa: E402

log = get_logger("bench")

DEFAULT_CLIP = "data/videos/dev/stop_sign_turn_lanes_red_light__c4130097-887da34f.mov"
TARGET_REALTIME_FPS_FRACTION = 25 / 30
TARGET_P95_PERCEPTION_MS = 40.0


def run_mode(
    cfg: AppConfig,
    clip: Path,
    realtime: bool,
    duration: float | None,
    detector: Detector,
    gpu: GpuMonitor,
    recorder: RunRecorder,
) -> dict[str, Any]:
    """Run perception over the clip in one mode and return its measurements."""
    source = FileSource(clip, realtime=realtime)
    worker = PerceptionWorker(
        source, detector, EgoMotionEstimator(cfg.ego_motion), recorder=recorder, perf_interval_s=2.0, gpu=gpu
    )
    detector.reset()
    perception_ms: list[float] = []
    worker.add_listener(lambda r: perception_ms.append(r.timings_ms["detect_track_total"] + r.timings_ms["ego_motion"]))
    source.start()
    worker.start()
    deadline = time.monotonic() + duration if duration else None
    while worker.running and (deadline is None or time.monotonic() < deadline):
        time.sleep(0.1)
    worker.stop()
    source.stop()
    if worker.error is not None:
        raise RuntimeError("perception failed") from worker.error

    assert worker.started_at is not None and worker.finished_at is not None
    wall = worker.finished_at - worker.started_at
    stages = worker.run_stats.summary()
    import numpy as np

    arr = np.asarray(perception_ms)
    media_span = source.clock.now() - (cfg.video.file.start_s or 0.0)
    return {
        "mode": "realtime" if realtime else "lockstep",
        "frames_processed": worker.processed,
        "source_frames_decoded": source.stats.decoded,
        "source_frames_dropped": source.stats.dropped,
        "wall_s": round(wall, 2),
        "media_s": round(media_span, 2),
        "processed_fps": round(worker.processed / wall, 2) if wall > 0 else 0.0,
        "source_fps": round(source.fps, 2),
        "perception_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }
        if arr.size
        else {},
        "stages_ms": stages,
        "gpu_samples": worker.gpu_samples,
    }


def print_report(report: dict[str, Any]) -> None:
    """Human-readable summary."""
    print(f"\nclip: {report['clip']}  imgsz={report['imgsz']}  half={report['half']}")
    if report["other_gpu_processes"]:
        print(f"WARNING other GPU users during the run: {', '.join(report['other_gpu_processes'])}")
    for mode in report["modes"]:
        print(
            f"\n[{mode['mode']}] processed {mode['frames_processed']} frames in {mode['wall_s']} s "
            f"= {mode['processed_fps']} fps (source {mode['source_fps']} fps, dropped {mode['source_frames_dropped']})"
        )
        print(f"  {'stage':22s} {'p50':>8s} {'p95':>8s} {'max':>8s}  (ms)")
        for key, s in mode["stages_ms"].items():
            print(f"  {key:22s} {s['p50']:8.2f} {s['p95']:8.2f} {s['max']:8.2f}")
        if mode["gpu_samples"]:
            peak = max(s.get("torch_peak_mb", 0) for s in mode["gpu_samples"])
            dev = max(s.get("device_used_mb", 0) for s in mode["gpu_samples"])
            util = [s.get("gpu_util_pct", 0) for s in mode["gpu_samples"]]
            print(
                f"  GPU: torch peak {peak:.0f} MB, device used max {dev:.0f} MB, util avg {sum(util) / len(util):.0f}%"
            )
    print("\ntargets:")
    for check in report["checks"]:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}: {check['value']} (target {check['target']})")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, default=Path(DEFAULT_CLIP))
    parser.add_argument("--mode", choices=["realtime", "lockstep", "both"], default="both")
    parser.add_argument("--duration", type=float, default=None, help="stop each mode after this many seconds")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(overrides=args.overrides)
    clip = cfg.resolve_path(args.clip)
    with RunRecorder(
        cfg.resolve_path(cfg.logging.runs_dir),
        "bench",
        cfg.redacted_dump(),
        frame_interval_s=cfg.logging.frame_log_interval_s,
    ) as recorder:
        setup_logging(cfg.logging.level, log_file=recorder.log_file, secrets=cfg.secrets.values())
        gpu = GpuMonitor(cfg.perception.device)
        others = gpu.other_gpu_processes()
        detector = Detector(cfg.perception, cfg.root)
        modes = ["realtime", "lockstep"] if args.mode == "both" else [args.mode]
        results = [run_mode(cfg, clip, m == "realtime", args.duration, detector, gpu, recorder) for m in modes]
        others = sorted(set(others) | set(gpu.other_gpu_processes()))

        checks = []
        for r in results:
            if r["mode"] == "realtime":
                fps_target = round(TARGET_REALTIME_FPS_FRACTION * r["source_fps"], 1)
                checks.append(
                    {
                        "name": "real-time processed fps",
                        "value": r["processed_fps"],
                        "target": f">= {fps_target}",
                        "ok": r["processed_fps"] >= fps_target,
                    }
                )
            p95 = r["perception_ms"].get("p95", float("inf"))
            checks.append(
                {
                    "name": f"p95 perception ms ({r['mode']})",
                    "value": round(p95, 1),
                    "target": f"< {TARGET_P95_PERCEPTION_MS}",
                    "ok": p95 < TARGET_P95_PERCEPTION_MS,
                }
            )
        report = {
            "clip": str(clip),
            "imgsz": cfg.perception.imgsz,
            "half": cfg.perception.half,
            "other_gpu_processes": others,
            "modes": results,
            "checks": checks,
        }
        (recorder.dir / "bench.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_report(report)
        print(f"\nreport: {recorder.dir / 'bench.json'}")
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
