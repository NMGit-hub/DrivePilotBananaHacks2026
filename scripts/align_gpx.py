"""Align a GPX track (GPSLogger) with a phone video: writes a copy of the GPX with the `video_start` waypoint.

    .venv/Scripts/python scripts/align_gpx.py data/videos/own/VID_20261011_143022.mp4 data/gps/20261011.gpx
    .venv/Scripts/python scripts/align_gpx.py VIDEO GPX --no-refine            # timestamps only (no GPU)
    .venv/Scripts/python scripts/align_gpx.py VIDEO GPX --start 2026-10-11T21:30:22.4Z   # name has no time

The start comes from the file name `VID_YYYYMMDD_HHMMSS` (local time in --tz) or the MP4's creation_time, then
is refined by matching the first clean pull-away from a stop in the video (ego motion, from the perception cache;
built on the GPU for the first --search-s seconds if missing) and in the GPS speed. See nav/align.py for how
and why. Writes `<gpx stem>.aligned.gpx` next to the GPX (or --out; the original is never modified) and prints
a summary with the command that replays the drive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from nav.align import (  # noqa: E402
    DEFAULT_TZ,
    AlignError,
    RefineSettings,
    align,
    format_summary,
    parse_manual_start,
    perception_frames,
    time_zone,
)

if TYPE_CHECKING:
    from eval.perception_cache import CachedFrame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="The phone video (VID_YYYYMMDD_HHMMSS.mp4).")
    parser.add_argument("gpx", type=Path, help="The GPX track recorded during the drive (GPSLogger).")
    parser.add_argument("--out", type=Path, help="Aligned GPX to write (default: <gpx stem>.aligned.gpx).")
    parser.add_argument("--tz", default=DEFAULT_TZ, help=f"Time zone of the file name's time (default {DEFAULT_TZ}).")
    parser.add_argument(
        "--start-from",
        choices=("auto", "name", "creation"),
        default="auto",
        help="Timestamp source: the file name, the MP4 creation_time, or auto (name if present).",
    )
    parser.add_argument(
        "--creation-is",
        choices=("auto", "start", "end"),
        default="auto",
        help="Whether creation_time marks the start or the end of the recording (auto: decided from the name).",
    )
    parser.add_argument("--start", help="The video's first frame as an ISO time (no offset = --tz local time).")
    parser.add_argument("--no-refine", action="store_true", help="Timestamps only: no motion matching (no GPU).")
    parser.add_argument("--search-s", type=float, default=180.0, help="Look for a pull-away in this much video.")
    parser.add_argument("--max-shift-s", type=float, default=10.0, help="Search the GPS within ± this of the start.")
    parser.add_argument(
        "--min-confidence",
        choices=("high", "medium"),
        default="high",
        help="medium: also apply a match only the pull-away supports (e.g. parked when recording began).",
    )
    parser.add_argument("--cache", type=Path, help="Perception cache file to read or build (default: data/cache/).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    for path in (args.video, args.gpx):
        if not path.is_file():
            print(f"not found: {path}", file=sys.stderr)
            return 1
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    settings = RefineSettings(
        max_shift_s=args.max_shift_s, min_confidence=cast(Literal["high", "medium"], args.min_confidence)
    )

    def progress(t: float) -> None:
        print(f"perception: {t:.0f} / {args.search_s:.0f} s of video", flush=True)

    def frames(duration_s: float | None) -> tuple[list[CachedFrame], Path | None]:
        print(f"reading ego motion for the first {args.search_s:.0f} s of the video...", flush=True)
        cached, path, built = perception_frames(cfg, args.video, args.search_s, duration_s, args.cache, progress)
        print(f"perception cache: {path} ({'built' if built else 'reused'}, {len(cached)} frames)", flush=True)
        return cached, path

    try:
        time_zone(args.tz)
        result = align(
            args.video,
            args.gpx,
            args.out,
            tz=args.tz,
            prefer=args.start_from,
            creation_is=args.creation_is,
            manual_start=parse_manual_start(args.start, args.tz) if args.start else None,
            frames=None if args.no_refine else frames,
            stopped_after_s=cfg.ego_motion.stopped_after_s,
            search_s=args.search_s,
            settings=settings,
        )
    except AlignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(result, args.tz, REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
