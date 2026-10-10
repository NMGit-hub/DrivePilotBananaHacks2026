"""Evaluate YOLOPv2 lanes and the drift warning (V2-M5).

    .venv/Scripts/python scripts/eval_lanes.py cache --clip data/videos/highway/<id>.mov ...   # masks at lanes.fps
    .venv/Scripts/python scripts/eval_lanes.py frames --labels eval/lane_labels.yaml            # JPEGs to label
    .venv/Scripts/python scripts/eval_lanes.py index --labels eval/lane_labels.yaml             # lane index score
    .venv/Scripts/python scripts/eval_lanes.py drift --clip ... (--gpx from data/gps/<id>.gpx)  # drift alerts
    .venv/Scripts/python scripts/eval_lanes.py virtual --clip <id>.mov --start 5 --rate 0.25    # virtual drift
    .venv/Scripts/python scripts/eval_lanes.py speed --clip <id>.mov                            # fps with YOLO26s

Masks are cached in runs/lanes_cache/<clip stem>.npz (gitignored) so the tracker can be replayed in seconds.
`--data-root` points at the folder holding data/ and models/ (default: this checkout).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402

log = get_logger("eval_lanes")
CACHE_DIR = REPO_ROOT / "runs" / "lanes_cache"


def cache_path(clip: Path, tag: str = "") -> Path:
    """Where a clip's mask cache lives."""
    return CACHE_DIR / f"{clip.stem}{tag}.npz"


def _config(args: argparse.Namespace) -> AppConfig:
    overrides = list(args.set)
    if args.data_root is not None:
        weights = Path(args.data_root) / "models" / "lanes" / "yolopv2.pt"
        overrides.append(f"lanes.weights={weights.as_posix()}")
    return load_config(REPO_ROOT / "config.yaml", overrides=overrides)


def cmd_cache(args: argparse.Namespace) -> int:
    """Run the lane model over clips and cache the masks."""
    from eval.lanes_eval import build_mask_cache
    from perception.lanes import LaneModel

    cfg = _config(args)
    model = LaneModel(cfg.lanes, cfg.root)
    for clip in args.clip:
        out = cache_path(clip)
        if out.exists() and not args.force:
            print(f"cached   {clip.name}")
            continue
        started = time.perf_counter()
        n = build_mask_cache(model.infer, clip, out, args.fps or cfg.lanes.fps)
        elapsed = time.perf_counter() - started
        print(f"built    {clip.name}: {n} frames in {elapsed:.0f} s", flush=True)
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    """Save frames of clips at the given times (or every --every s) as JPEGs for labeling (two per sheet)."""
    import cv2
    import numpy as np

    out_dir = REPO_ROOT / "runs" / "lane_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[np.ndarray] = []
    names: list[str] = []
    sheet = 0

    def flush() -> None:
        nonlocal sheet, pending, names
        if not pending:
            return
        cv2.imwrite(str(out_dir / f"sheet_{sheet:03d}.jpg"), np.vstack(pending), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"sheet_{sheet:03d}.jpg: {' | '.join(names)}")
        sheet += 1
        pending, names = [], []

    for clip in args.clip:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        t = args.start
        while t < duration - 0.5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, image = cap.read()
            if ok:
                small = cv2.resize(image, (960, round(960 * image.shape[0] / image.shape[1])))
                label = f"{clip.stem[-17:]} t={t:.1f}"
                cv2.putText(small, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(small, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                pending.append(small)
                names.append(label)
                if len(pending) == 2:
                    flush()
            t += args.every
        cap.release()
    flush()
    return 0


def _data_root(args: argparse.Namespace) -> Path:
    return Path(args.data_root) if args.data_root is not None else REPO_ROOT


def cmd_index(args: argparse.Namespace) -> int:
    """Score lane index / count against the hand labels (replaying cached masks)."""
    import yaml

    from eval.lanes_eval import IndexScore, at_time, load_mask_cache, replay, speed_from_gpx

    cfg = _config(args)
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    score = IndexScore()
    root = _data_root(args)
    for clip in labels["clips"]:
        video = root / "data" / "videos" / clip["file"]
        cache = cache_path(video)
        if not cache.exists():
            print(f"no cache for {video.name}: run `cache` first")
            return 1
        steps = replay(
            cfg.lanes, load_mask_cache(cache), args.hfov, speed_from_gpx(root / "data" / "gps" / f"{clip['id']}.gpx")
        )
        for frame in clip["frames"]:
            score.add(clip["id"], frame["t"], frame, at_time(steps, frame["t"]).snapshot)
    if args.verbose:
        for row in score.rows:
            mark = "" if row["got_index"] is None else ("ok " if row["got_index"] == row["index"] else "BAD")
            print(
                f"  {row['clip']} t={row['t']:4.1f}  index {row['index']} got {row['got_index']}  "
                f"count {row['count']} got {row['got_count']}  conf {row['confidence']:.2f} {mark}"
            )
    pct = 100.0 * score.index_right / max(1, score.index_answered)
    print(
        f"lane index: right on {score.index_right}/{score.index_answered} answered frames ({pct:.0f} %), "
        f"answered {score.index_answered}/{score.frames} ({100.0 * score.index_answered / max(1, score.frames):.0f} %, "
        f"abstained on {score.frames - score.index_answered})"
    )
    print(
        f"lane count: right on {score.count_right}/{score.count_answered} answered "
        f"(of {score.count_labeled} frames with a labeled count)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    parser.add_argument("--data-root", type=Path, help="Folder with data/ and models/ (default: this checkout).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache", help="Cache lane masks for clips.")
    p.add_argument("--clip", type=Path, action="append", required=True)
    p.add_argument("--fps", type=float, help="Sampling rate (default lanes.fps).")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_cache)
    p = sub.add_parser("frames", help="Save frames as JPEGs (runs/lane_frames) for hand-labeling.")
    p.add_argument("--clip", type=Path, action="append", required=True)
    p.add_argument("--start", type=float, default=3.0)
    p.add_argument("--every", type=float, default=6.0)
    p.set_defaults(func=cmd_frames)
    p = sub.add_parser("index", help="Score lane index and count against hand labels.")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / "eval" / "lane_labels.yaml")
    p.add_argument("--hfov", type=float, default=60.0, help="Field of view of the clips (BDD100K: ~60°).")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_index)
    args = parser.parse_args(argv)
    setup_logging("WARNING")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
