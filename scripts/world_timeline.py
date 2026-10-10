"""Run the real pipeline on a clip and print the world-state timeline: events plus a snapshot every second.

    .venv/Scripts/python scripts/world_timeline.py --clip data/videos/dev/<name>.mov
    .venv/Scripts/python scripts/world_timeline.py --clip ... --start 15 --every 0.5

File → TensorRT YOLO26s + ByteTrack → ego motion → WorldState, processing every frame (lock-step), so the
timeline is the one the app would build at full frame rate. Used to compare the queue / stop / light logic
with hand-labeled timelines on recorded drives.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capture.file_source import FileSource  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402
from state.world import WorldEvent, WorldSnapshot, WorldState  # noqa: E402


def describe(s: WorldSnapshot) -> str:
    """One line per snapshot."""
    ego = f"stopped since {s.ego_stopped_since:.1f}" if s.ego_stopped and s.ego_stopped_since is not None else "moving"
    parts = [f"{s.t:6.2f}  ego {ego:<18}"]
    sign = "sign" if s.stop_sign_visible else (f"sign@{s.stop_sign_last_seen:.1f}" if s.stop_sign_last_seen else "-")
    parts.append(f"{sign:<10} at_stop={'Y' if s.at_stop_sign else 'n'}")
    if s.lead_vehicle is not None:
        parts.append(f"lead={s.lead_vehicle.label}#{s.lead_vehicle.track_id}")
    if s.queue.active:
        order = ", ".join(
            "US"
            if e.is_ego
            else f"{e.approach.value}:{e.label}#{e.track_id}@{e.arrived:.1f}{'' if e.arrival_known else '?'}"
            for e in s.queue.order
        )
        parts.append(f"queue[{order}]" + (" YOUR TURN" if s.queue.your_turn else ""))
    waiting = [v for v in s.vehicles if v.approach is not None]
    if waiting:
        parts.append(
            "sides: "
            + " ".join(f"{v.approach.value}:{v.label}#{v.track_id}{'(stopped)' if v.stopped else ''}" for v in waiting)  # type: ignore[union-attr]
        )
    if s.light.color.value != "unknown":
        parts.append(f"light={s.light.color.value}{'' if s.light.confident else '?'}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, help="Start time in the clip (s).")
    parser.add_argument("--every", type=float, default=1.0, help="Snapshot interval (s of video).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    setup_logging("WARNING")
    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(args.clip, realtime=False, start_s=args.start)
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(cfg.ego_motion))
    lines: list[str] = []

    def on_event(event: WorldEvent) -> None:
        detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
        lines.append(f"{event.t:6.2f}  ** {event.kind} {detail}")

    world = WorldState(cfg, on_event=on_event)
    next_print = [args.start]

    def listener(result: PerceptionResult) -> None:
        world.on_perception(result)
        if result.frame.t >= next_print[0]:
            snap = world.snapshot()
            if snap is not None:
                lines.append(describe(snap))
            next_print[0] = result.frame.t + args.every

    worker.add_listener(listener)
    source.start()
    worker.start()
    while worker.running:
        time.sleep(0.1)
    worker.stop()
    source.stop()
    if worker.error is not None:
        raise RuntimeError("perception failed") from worker.error
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
