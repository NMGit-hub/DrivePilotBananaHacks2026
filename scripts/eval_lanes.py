"""Evaluate YOLOPv2 lanes and the drift warning (V2-M5).

    .venv/Scripts/python scripts/eval_lanes.py cache --clip data/videos/highway/<id>.mov ...   # masks at lanes.fps
    .venv/Scripts/python scripts/eval_lanes.py frames --labels eval/lane_labels.yaml            # JPEGs to label
    .venv/Scripts/python scripts/eval_lanes.py index --labels eval/lane_labels.yaml             # lane index score
    .venv/Scripts/python scripts/eval_lanes.py drift [--frames]       # drift alerts + lane changes (GPS from data/gps)
    .venv/Scripts/python scripts/eval_lanes.py virtual --clip <id>.mov --start 15 --rate 0.2 --rate 0.8  # virtual

Masks are cached in runs/lanes_cache/<clip stem>.npz (gitignored) so the tracker can be replayed in seconds.
`--data-root` points at the folder holding data/ and models/ (default: this checkout).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import AppConfig, LanesConfig, load_config  # noqa: E402
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
    from perception.lanes import LaneModel, crop_view, lane_view

    cfg = _config(args)
    model = LaneModel(cfg.lanes, cfg.root)
    view = (0.0, 0.0, 1.0, 1.0)
    if args.hfov is not None:  # a wide camera: the model sees the app's crop (lanes.max_view_deg, view_center_y)
        view = lane_view(args.hfov, cfg.lanes.max_view_deg, cfg.lanes.view_center_y)[0]
        print(f"lane model crop {tuple(round(v, 3) for v in view)} of each frame")
    transform = None if view == (0.0, 0.0, 1.0, 1.0) else (lambda _t, image: crop_view(image, view))
    for clip in args.clip:
        out = cache_path(clip)
        if out.exists() and not args.force:
            print(f"cached   {clip.name}")
            continue
        started = time.perf_counter()
        n = build_mask_cache(model.infer, clip, out, args.fps or cfg.lanes.fps, transform, cfg.lanes, view)
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


def _tracker_hfov(args: argparse.Namespace, lc: LanesConfig) -> float:
    """The lane tracker's field of view: the clips' own, or the crop's for a camera wider than lanes.max_view_deg
    (as the app's: masks cached with `cache --hfov` are of that crop)."""
    from perception.lanes import lane_view

    return lane_view(args.hfov, lc.max_view_deg)[1]


def _load_checked(path: Path, args: argparse.Namespace, lc: LanesConfig) -> list[Any]:
    """A clip's cached masks, refused when they don't match `--hfov`: a wide camera replayed at the crop's field of
    view needs masks of that crop (and the other way round), else the geometry is silently wrong."""
    from eval.lanes_eval import load_mask_cache
    from perception.lanes import lane_view

    masks = load_mask_cache(path)
    full = (0.0, 0.0, 1.0, 1.0)
    cropped = lane_view(args.hfov, lc.max_view_deg)[0] != full
    if masks and (masks[0].view != full) != cropped:
        made = "of a crop" if masks[0].view != full else "of the whole frame"
        raise SystemExit(
            f"{path.name}: masks {made}, but --hfov {args.hfov:g} replays "
            f"{'a crop' if cropped else 'the whole frame'}: rebuild it with `cache --force"
            f"{' --hfov ' + format(args.hfov, 'g') if cropped else ''}`"
        )
    return masks


def _data_root(args: argparse.Namespace) -> Path:
    return Path(args.data_root) if args.data_root is not None else REPO_ROOT


def cmd_index(args: argparse.Namespace) -> int:
    """Score lane index / count against the hand labels (replaying cached masks)."""
    import yaml

    from eval.lanes_eval import IndexScore, at_time, replay, speed_from_gpx

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
            cfg.lanes,
            _load_checked(cache, args, cfg.lanes),
            _tracker_hfov(args, cfg.lanes),
            speed_from_gpx(root / "data" / "gps" / f"{clip['id']}.gpx"),
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


def _clip_id(video: Path) -> str:
    """BDD100K clip id from a video name (`<id>.mov` or `<scene>__<id>.mov`)."""
    return video.stem.split("__")[-1]


def _clips(args: argparse.Namespace) -> list[Path]:
    """Videos to replay: --clip, else every clip of the labels file."""
    if args.clip:
        return [Path(c) for c in args.clip]
    import yaml

    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    return [_data_root(args) / "data" / "videos" / clip["file"] for clip in labels["clips"]]


def cmd_drift(args: argparse.Namespace) -> int:
    """Replay clips with their GPS speed: every drift alert and lane change (and frames of each, with --frames)."""
    import cv2

    from eval.lanes_eval import at_time, draw_lanes, read_frame, replay, speed_from_gpx
    from state.lanes import LaneTracker

    cfg = _config(args)
    root = _data_root(args)
    out_dir = REPO_ROOT / "runs" / "lane_drift"
    total_alerts = 0
    fast_s = 0.0
    for video in _clips(args):
        cache = cache_path(video, args.tag)
        if not cache.exists():
            print(f"no cache for {video.name}{args.tag}: run `cache` first")
            return 1
        masks = _load_checked(cache, args, cfg.lanes)
        speed_at = speed_from_gpx(root / "data" / "gps" / f"{_clip_id(video)}.gpx")
        hfov = _tracker_hfov(args, cfg.lanes)
        tracker = LaneTracker(cfg.lanes, hfov)
        steps = replay(cfg.lanes, masks, hfov, speed_at, tracker)
        valid = sum(1 for s in steps if s.snapshot.valid)
        dt = [b.t - a.t for a, b in itertools.pairwise(steps)]
        fast = sum(
            d
            for d, s in zip(dt, steps, strict=False)
            if s.speed_mps is not None and s.speed_mps >= cfg.lanes.min_speed_mps
        )
        fast_s += fast
        alerts = [(s.t, e) for s in steps for e in s.events]
        changes = list(tracker.lane_changes)
        total_alerts += len(alerts)
        print(
            f"{_clip_id(video)}{args.tag}: {len(alerts)} drift alert(s), {len(changes)} lane change(s); "
            f"lane seen in {100.0 * valid / max(1, len(steps)):.0f} % of {len(steps)} frames; "
            f"{fast:.0f} s at >= {cfg.lanes.min_speed_mps * 2.237:.0f} mph"
        )
        for t, e in alerts:
            print(
                f"    ALERT t={t:5.2f} {e.side:5s} gap {e.gap_m:.2f} m, toward {e.lateral_speed_mps:.2f} m/s, "
                f"offset {e.offset_m:+.2f} m (usual {e.habit_m:+.2f}), {e.speed_mps * 2.237:.0f} mph"
            )
        for t, k in changes:
            print(f"    lane change t={t:5.2f} {'right' if k > 0 else 'left'}")
        if args.frames:
            out_dir.mkdir(parents=True, exist_ok=True)
            moments = [(t, "alert") for t, _ in alerts] + [(t, "change") for t, _ in changes]
            for t, what in moments:
                for dt_s in (-2.0, 0.0):
                    image = read_frame(video, max(0.0, t + dt_s))
                    if image is None:
                        continue
                    snap = at_time(steps, t + dt_s).snapshot
                    name = f"{_clip_id(video)}{args.tag}_{what}_{t:05.1f}_{dt_s:+.0f}.jpg"
                    cv2.imwrite(str(out_dir / name), draw_lanes(image, snap), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"total: {total_alerts} drift alert(s) in {fast_s / 60.0:.1f} min above the speed threshold")
    return 0


def _expected_alert(steps: list[Any], start: float, sign: float, rate: float, cfg: LanesConfig) -> float | None:
    """When the car's side reaches `drift_gap_m` of the line in a virtual slide: the unwarped replay's own position
    (from our usual place, median over ±0.5 s) plus `rate` m/s toward the line from `start`."""
    import numpy as np

    rows = [
        (s.t, s.snapshot.offset_m - s.snapshot.habit_m, s.snapshot.lane_width_m)
        for s in steps
        if s.snapshot.valid and s.snapshot.habit_m is not None and s.snapshot.lane_width_m is not None
    ]
    if not rows:
        return None
    ts = np.array([r[0] for r in rows])
    rel = np.array([r[1] for r in rows])
    width = float(np.median([r[2] for r in rows]))
    threshold = width / 2.0 - cfg.car_width_m / 2.0 - cfg.drift_gap_m
    for t in ts[ts >= start]:
        near = np.abs(ts - t) <= 0.5
        if sign * float(np.median(rel[near])) + rate * (t - start) >= threshold:
            return float(t)
    return None


def cmd_virtual(args: argparse.Namespace) -> int:
    """Virtual drift: a clip warped as if the car slid sideways at --rate m/s from --start (the lane model runs on
    the warped frames). The alert should come within 1 s of when the car's side reaches drift_gap_m of the line;
    a fast slide (a lane change) should stay quiet."""
    from eval.lanes_eval import build_mask_cache, load_mask_cache, replay, speed_from_gpx, warp_lateral
    from perception.lanes import LaneModel
    from state.lanes import LaneTracker

    cfg = _config(args)
    lc = cfg.lanes
    root = _data_root(args)
    model: Any = None
    if _tracker_hfov(args, lc) != args.hfov:
        raise SystemExit("virtual: a camera wider than lanes.max_view_deg isn't supported (its warp needs the crop)")
    for video in args.clip:
        speed_at = speed_from_gpx(root / "data" / "gps" / f"{_clip_id(video)}.gpx")
        base_tracker = LaneTracker(lc, args.hfov)
        base = replay(lc, load_mask_cache(cache_path(video)), args.hfov, speed_at, base_tracker)
        horizon, scale = base_tracker.horizon, base_tracker.scale  # the warp uses the tracker's own geometry
        for side in args.side or ["left", "right"]:
            sign = 1.0 if side == "right" else -1.0
            for rate in args.rate or [0.2]:
                tag = f"_virtual_{side}_{rate:g}_{args.start:g}_{args.max_shift:g}"
                cache = cache_path(video, tag)
                if args.force or not cache.exists():
                    if model is None:
                        model = LaneModel(lc, cfg.root)

                    def transform(
                        t: float,
                        image: Any,
                        sign: float = sign,
                        rate: float = rate,
                        scale: float = scale,
                        horizon: float = horizon,
                    ) -> Any:
                        slid = min(rate * max(0.0, t - args.start), args.max_shift)  # (then the driver holds it)
                        shift = sign * slid / scale  # unscaled metres, like the tracker's
                        return warp_lateral(image, shift, horizon, lc.camera_height_m)

                    build_mask_cache(model.infer, video, cache, args.fps or lc.fps, transform)
                steps = replay(lc, load_mask_cache(cache), args.hfov, speed_at)
                alerts = [(s.t, e) for s in steps for e in s.events]
                expected = _expected_alert(base, args.start, sign, rate, lc)
                fast = rate >= lc.lane_change_mps
                verdict = ""
                hits = [t for t, e in alerts if t >= args.start and e.side == side]
                others = [(t, e.side) for t, e in alerts if not (t >= args.start and e.side == side)]
                if fast:
                    verdict = "ok (quiet)" if not alerts else "FALSE ALERT"
                elif expected is None:
                    verdict = "no expected time (the slide never reaches the line)"
                elif hits:
                    verdict = "ok" if abs(hits[0] - expected) <= 1.0 else f"LATE/EARLY by {hits[0] - expected:+.2f} s"
                else:
                    verdict = "MISSED"
                print(
                    f"{_clip_id(video)} {side} {rate:g} m/s from {args.start:g} s: "
                    f"expected {'-' if expected is None else f'{expected:.2f}'}, "
                    f"alert {'-' if not hits else f'{hits[0]:.2f}'}, other alerts {others}: {verdict}"
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
    p.add_argument(
        "--hfov", type=float, help="The clips' field of view: wider than lanes.max_view_deg = the app's crop."
    )
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
    p = sub.add_parser("drift", help="Drift alerts and lane changes on clips (with their GPS speed).")
    p.add_argument("--clip", type=Path, action="append", help="Videos (default: every clip of --labels).")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / "eval" / "lane_labels.yaml")
    p.add_argument("--tag", default="", help="Cache name suffix (a virtual-drift cache).")
    p.add_argument("--hfov", type=float, default=60.0, help="Field of view of the clips (BDD100K: ~60°).")
    p.add_argument("--frames", action="store_true", help="Save frames at each alert / lane change (runs/lane_drift).")
    p.set_defaults(func=cmd_drift)
    p = sub.add_parser("virtual", help="Virtual drift: a clip warped into a slow sideways slide.")
    p.add_argument("--clip", type=Path, action="append", required=True)
    p.add_argument("--start", type=float, default=15.0, help="The slide starts here (s).")
    p.add_argument("--rate", type=float, action="append", help="Slide speed, m/s (repeatable; default 0.2).")
    p.add_argument("--side", choices=("left", "right"), action="append", help="Default: both.")
    p.add_argument("--fps", type=float, help="Sampling rate (default lanes.fps).")
    p.add_argument("--hfov", type=float, default=60.0)
    p.add_argument("--max-shift", type=float, default=1.2, help="The slide stops after this many metres.")
    p.add_argument("--force", action="store_true", help="Rebuild the warped mask caches.")
    p.set_defaults(func=cmd_virtual)
    args = parser.parse_args(argv)
    setup_logging("WARNING")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
