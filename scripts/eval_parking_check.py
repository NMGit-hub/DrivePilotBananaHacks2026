"""What the vision model's spot check (answers/parking.py) does with the parking tracker's spot events (V2-M1).

    .venv/Scripts/python scripts/eval_parking_check.py                                    # the 40 labeled lot clips
    .venv/Scripts/python scripts/eval_parking_check.py --labels eval/parking_labels_crawl_a.yaml \\
        --labels eval/parking_labels_crawl_b.yaml

Every "spot" the geometry finds is checked by the vision model before it is spoken (open? a real stall?
restricted?). This replays the tracker on the cached perception like scripts/eval_parking.py, decodes the frame of
each spot event, asks the model exactly as the watch does (`ParkingWatch._ask_spot`: the space outlined, a
close-up), and reports which events would be spoken: hits kept (good), false spots kept (what the driver would
hear), and false spots the model stopped. Live API calls: one per spot event (a few dozen at most).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
from eval_parking import CACHE_DIR, GPS_DIR, LABELS  # noqa: E402

from answers.parking import ParkingWatch, outlined  # noqa: E402
from answers.vision import VisionError, create_vision_model  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from core.types import Frame  # noqa: E402
from eval.parking_eval import load_labels, replay, score_clip  # noqa: E402
from eval.perception_cache import load_cache, read_frame  # noqa: E402
from nav.gps import GpxReplay  # noqa: E402

VIDEO_DIRS = [REPO_ROOT / "data" / "videos" / d for d in ("parking_lot", "parking_crawl", "dev")]


def find_video(clip_id: str) -> Path | None:
    """The clip's video file."""
    for folder in VIDEO_DIRS:
        path = folder / f"{clip_id}.mov"
        if path.is_file():
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, action="append", default=[])
    parser.add_argument("--hfov", type=float, default=60.0)
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    try:
        model = create_vision_model(cfg)
    except VisionError as exc:
        print(f"no vision model: {exc}")
        return 1
    watch = ParkingWatch(cfg.parking, cfg.vision, model, lambda u: None, lambda: None, lambda m: None, time.monotonic)
    out = args.out or REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_parking_check"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for labels_path in args.labels or [LABELS]:
        for clip in [c for c in load_labels(labels_path) if c.usable]:
            frames = load_cache(CACHE_DIR / f"{clip.id}.pkl", cfg)
            gpx = GPS_DIR / f"{clip.id}.gpx"
            gps = GpxReplay(gpx) if gpx.exists() else None

            def speed(t: float, gps: GpxReplay | None = gps) -> float | None:
                fix = gps.fix_at(t) if gps is not None else None
                return None if fix is None else fix.speed_mps

            events = replay(frames, cfg.parking, args.hfov, speed)
            score = score_clip(clip, events, frames[-1].t, parked_delay_s=cfg.parking.parked_after_s)
            video = find_video(clip.id)
            candidates = {e.space.key: e for e in events if e.kind == "candidate" and e.space is not None}
            for match in score.spots:
                event, space = match.event, match.event.space
                assert space is not None
                # The watch asks at the space's "candidate" event (earlier, the stall better in view), else here.
                asked = candidates.get(space.key, event)
                space = asked.space or space
                cached = min(frames, key=lambda f: abs(f.t - asked.t))
                image = read_frame(video, cached.index) if video is not None else None
                frame = None if image is None else Frame(image=image, t=cached.t, index=cached.index, captured_at=0.0)
                check = watch._ask_spot(space, frame, "lot")  # (exactly what the watch asks)
                spoken = check is None or not check.rejected  # (a failed check is still spoken, hedged)
                kind = "hit" if match.spot is not None and match.spot.target else "labeled" if match.spot else "false"
                row = {
                    "clip": clip.id,
                    "t": round(event.t, 2),
                    "side": space.side,
                    "match": kind,
                    "check": None,
                    "spoken": spoken,
                }
                if check is not None:
                    row["check"] = {
                        "open": check.open,
                        "restriction": check.restriction,
                        "before": check.before,
                        "confidence": check.confidence,
                    }
                rows.append(row)
                print(
                    f"{clip.id} {event.t:6.2f} {space.side:<5} {kind:<7} -> {row['check']} spoken={spoken}", flush=True
                )
                if frame is not None:
                    cv2.imwrite(
                        str(out / "frames" / f"{clip.id}_{event.t:06.2f}_{kind}.jpg"),
                        outlined(frame, space.region, space.footprint),
                    )
    kept = {k: sum(1 for r in rows if r["match"] == k and r["spoken"]) for k in ("hit", "labeled", "false")}
    total = {k: sum(1 for r in rows if r["match"] == k) for k in ("hit", "labeled", "false")}
    summary = {"events": len(rows), "kept": kept, "total": total}
    print(json.dumps(summary))
    (out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
