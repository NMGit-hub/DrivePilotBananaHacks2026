"""Headless pipeline benchmark: FPS, per-stage latency and GPU memory on a video file, checked against targets.

    .venv/Scripts/python scripts/bench_pipeline.py                       # default dev clip, real-time + lock-step
    .venv/Scripts/python scripts/bench_pipeline.py --clip data/videos/x.mp4 --mode realtime --duration 20
    .venv/Scripts/python scripts/bench_pipeline.py --set perception.imgsz=960

Real-time mode plays the clip at its own speed (frames are skipped if perception falls behind), which is what
the live camera does; lock-step processes every frame as fast as possible (throughput). The report is written
to runs/<stamp>_bench/bench.json, and the exit status is 1 if a target is missed.

Targets (BUILD_PLAN.md M1), checked in real-time mode:
- processed fps >= 25/30 of the source frame rate;
- p95 processing time (detect + track, with ego motion in parallel) < 40 ms;
- p95 latency from frame decode to result (processing + waiting for the worker to be free) < 70 ms.
The detector refuses to start unless its weights are on CUDA in the configured precision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from core.runlog import RunRecorder  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.gpu_monitor import GpuMonitor  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402

log = get_logger("bench")

DEFAULT_CLIP = "data/videos/dev/stop_sign_turn_lanes_red_light__c4130097-887da34f.mov"
TARGET_REALTIME_FPS_FRACTION = 25 / 30
TARGET_P95_PROCESSING_MS = 40.0
TARGET_P95_LATENCY_MS = 70.0


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if not arr.size:
        return {}
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


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
    torch.cuda.reset_peak_memory_stats(cfg.perception.device)
    source = FileSource(clip, realtime=realtime)
    worker = PerceptionWorker(
        source, detector, EgoMotionEstimator(cfg.ego_motion), recorder=recorder, perf_interval_s=2.0, gpu=gpu
    )
    processing_ms: list[float] = []
    media_times: list[float] = []

    def collect(result: PerceptionResult) -> None:
        processing_ms.append(result.timings_ms["processing"])
        media_times.append(result.frame.t)

    worker.add_listener(collect)
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
    return {
        "mode": "realtime" if realtime else "lockstep",
        "frames_processed": worker.processed,
        "frame_errors": worker.frame_errors,
        "source_frames_decoded": source.stats.decoded,
        "source_frames_dropped": source.stats.dropped,
        "wall_s": round(wall, 2),
        "media_s": round(media_times[-1] - media_times[0], 2) if len(media_times) > 1 else 0.0,
        "processed_fps": round(worker.processed / wall, 2) if wall > 0 else 0.0,
        "source_fps": round(source.fps, 2),
        "processing_ms": _percentiles(processing_ms),
        "stages_ms": worker.run_stats.summary(),
        "gpu_samples": worker.gpu_samples,
        "torch_peak_mb": round(torch.cuda.max_memory_allocated(cfg.perception.device) / 1e6, 1),
    }


def build_checks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare the real-time run with the M1 targets."""
    checks: list[dict[str, Any]] = []
    for r in results:
        if r["mode"] != "realtime":
            continue
        fps_target = round(TARGET_REALTIME_FPS_FRACTION * r["source_fps"], 1)
        processing = r["processing_ms"].get("p95", float("inf"))
        latency = r["stages_ms"].get("latency", {}).get("p95", float("inf"))
        checks += [
            {
                "name": "real-time processed fps",
                "value": r["processed_fps"],
                "target": f">= {fps_target}",
                "ok": r["processed_fps"] >= fps_target,
            },
            {
                "name": "p95 processing ms (detect + track || ego)",
                "value": round(processing, 1),
                "target": f"< {TARGET_P95_PROCESSING_MS}",
                "ok": processing < TARGET_P95_PROCESSING_MS,
            },
            {
                "name": "p95 decode-to-result latency ms",
                "value": round(latency, 1),
                "target": f"< {TARGET_P95_LATENCY_MS}",
                "ok": latency < TARGET_P95_LATENCY_MS,
            },
        ]
    return checks


def print_report(report: dict[str, Any]) -> None:
    """Human-readable summary."""
    print(f"\nclip: {report['clip']}  imgsz={report['imgsz']}  half={report['half']}")
    if report["other_gpu_processes"]:
        print(f"WARNING other GPU users during the run: {', '.join(report['other_gpu_processes'])}")
    for mode in report["modes"]:
        print(
            f"\n[{mode['mode']}] processed {mode['frames_processed']} frames in {mode['wall_s']} s "
            f"= {mode['processed_fps']} fps (source {mode['source_fps']} fps, dropped {mode['source_frames_dropped']},"
            f" frame errors {mode['frame_errors']})"
        )
        print(f"  {'stage':22s} {'p50':>8s} {'p95':>8s} {'max':>8s}  (ms)")
        for key, s in mode["stages_ms"].items():
            print(f"  {key:22s} {s['p50']:8.2f} {s['p95']:8.2f} {s['max']:8.2f}")
        util = [s.get("gpu_util_pct", 0.0) for s in mode["gpu_samples"]]
        dev = max((s.get("device_used_mb", 0.0) for s in mode["gpu_samples"]), default=0.0)
        print(
            f"  GPU: torch peak {mode['torch_peak_mb']:.0f} MB, device used max {dev:.0f} MB"
            + (f", util avg {sum(util) / len(util):.0f}%" if util else "")
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
        others = set(gpu.other_gpu_processes())
        detector = Detector(cfg.perception, cfg.root)
        modes = ["realtime", "lockstep"] if args.mode == "both" else [args.mode]
        results = [run_mode(cfg, clip, m == "realtime", args.duration, detector, gpu, recorder) for m in modes]
        others |= set(gpu.other_gpu_processes())
        checks = build_checks(results)
        report = {
            "clip": str(clip),
            "imgsz": cfg.perception.imgsz,
            "half": cfg.perception.half,
            "other_gpu_processes": sorted(others),
            "modes": results,
            "checks": checks,
        }
        (recorder.dir / "bench.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_report(report)
        print(f"\nreport: {recorder.dir / 'bench.json'}")
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
