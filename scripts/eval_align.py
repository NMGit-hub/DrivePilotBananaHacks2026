"""Validate `scripts/align_gpx.py`'s refinement on BDD100K clips whose `video_start` is known (CPU: cached perception).

    .venv/Scripts/python scripts/eval_align.py                       # shifts -9..+8 s, recorded GPS speed
    .venv/Scripts/python scripts/eval_align.py --positions           # speed from positions (GPX without speed)
    .venv/Scripts/python scripts/eval_align.py --min-confidence medium --shifts -6,-3,2,5

Every clip with a perception cache (`data/cache/perception/<name>.pkl`, `scripts/cache_perception.py`) and a GPX
with a `video_start` waypoint (`data/gps/<id>.gpx`, `scripts/bdd_info_to_gpx.py`) is aligned again from a start
moved off the truth by each shift; the error is the refined start minus the true one. Clips where the refinement
finds no usable stop and pull-away keep the shifted start: that's reported as "none" (correct behavior).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from eval.perception_cache import load_cache  # noqa: E402
from nav.align import GpxTrack, RefineSettings, read_gpx, refine_from_frames  # noqa: E402
from nav.gps import VIDEO_START  # noqa: E402


def clips(root: Path) -> list[tuple[str, Path, Path]]:
    """(name, perception cache, GPX) for every cached clip with a GPX."""
    out = []
    for cache in sorted((root / "data" / "cache" / "perception").glob("*.pkl")):
        gpx = root / "data" / "gps" / f"{cache.stem.split('__')[-1]}.gpx"
        if gpx.is_file():
            out.append((cache.stem, cache, gpx))
    return out


def true_start(gpx: Path) -> datetime | None:
    """The GPX's `video_start` waypoint time (None if it has none)."""
    import gpxpy

    with gpx.open(encoding="utf-8") as fh:
        parsed = gpxpy.parse(fh)
    times = [w.time for w in parsed.waypoints if w.name == VIDEO_START and w.time is not None]
    return times[0] if times else None


def without_speed(track: GpxTrack) -> GpxTrack:
    """The track as if the GPX had no speed field (speed from positions)."""
    segments = tuple(tuple(replace(p, speed_mps=None) for p in segment) for segment in track.segments)
    return replace(track, segments=segments, speed_origin="none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shifts", default="-9,-7,-5,-3,-1,2,4,6,8", help="Start errors to recover (s).")
    parser.add_argument("--positions", action="store_true", help="Ignore the GPX speed (speed from positions).")
    parser.add_argument("--min-confidence", choices=("high", "medium"), default="high")
    parser.add_argument("--only", default="", help="Only clips whose name contains this.")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml")
    settings = RefineSettings(min_confidence=cast(Literal["high", "medium"], args.min_confidence))
    shifts = [float(x) for x in args.shifts.split(",")]
    errors: list[float] = []
    tries = 0
    for name, cache, gpx in clips(REPO_ROOT):
        if args.only and args.only not in name:
            continue
        truth = true_start(gpx)
        if truth is None:
            continue
        frames = load_cache(cache)
        t = np.array([f.t for f in frames])
        stopped = np.array([f.ego_stopped for f in frames])
        track = read_gpx(gpx)
        if args.positions:
            track = without_speed(track)
        cells = []
        note = ""
        for shift in shifts:
            tries += 1
            result = refine_from_frames(
                t, stopped, cfg.ego_motion.stopped_after_s, track, truth + timedelta(seconds=shift), settings
            )
            if result.shift_s is None:
                cells.append(f"{shift:+.0f}: none ")
                note = note or " / ".join(result.notes)[:110]
            else:
                error = shift + result.shift_s
                errors.append(error)
                cells.append(f"{shift:+.0f}:{error:+.2f}{result.confidence[0]}")
        print(f"{name[-34:]:34s} " + " ".join(cells) + (f"  | {note}" if note and not errors[-1:] else ""))
    if errors:
        abs_err = np.abs(np.array(errors))
        print(
            f"\napplied {len(abs_err)} of {tries} tries: within 0.5 s {int((abs_err <= 0.5).sum())}, "
            f"median |error| {np.median(abs_err):.2f} s, max {abs_err.max():.2f} s"
        )
    else:
        print(f"\nno refinement applied in {tries} tries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
