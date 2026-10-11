"""The car dashcam's horizontal field of view, from turns: how far the image swings vs how far the GPS heading turns.

    .venv/Scripts/python scripts/dashcam_calibrate.py D:/Dashcam --turns 12

Turning by an angle dtheta (about the vertical axis) moves the image's centre point sideways by f * tan(dtheta); summed
over the frames of a turn, the centre's sideways travel is ~f * (total heading change). The per-frame homographies
come from the app's own ego-motion estimator (perception/ego_motion.py); the heading change from the clip's overlay
GPS (scripts/dashcam_gpx.py scan), measured over the straight stretches before and after the turn (the positions
are quantized to an arc-second, so long baselines). f = summed travel / heading change; hfov = 2 atan(W / 2f). The
median over several turns is reported (forward motion adds parallax that averages out over left and right turns).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from nav.dashcam_overlay import FRAME_W, TrackSample, smooth_track  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache" / "dashcam"
MPH = 0.44704


@dataclass(frozen=True, slots=True)
class Turn:
    """A turn between two straight stretches of one clip."""

    clip: str
    t0: float  # end of the straight stretch before
    t1: float  # start of the straight stretch after
    heading_change_deg: float  # signed (+ = right / clockwise)


def headings(samples: list[TrackSample]) -> tuple[np.ndarray, np.ndarray]:
    """(t, heading in degrees clockwise from north) between consecutive smoothed positions."""
    t = np.array([s.t for s in samples])
    lat = np.array([s.lat for s in samples])
    lon = np.array([s.lon for s in samples])
    north = np.diff(lat) * 111_320.0
    east = np.diff(lon) * 111_320.0 * math.cos(math.radians(float(lat.mean())))
    return (t[:-1] + t[1:]) / 2, np.degrees(np.arctan2(east, north))


def _bearing(samples: list[TrackSample], a: float, b: float) -> float | None:
    """Heading from the position at time a to the one at time b (None if too close: under 40 m)."""
    pa = min(samples, key=lambda s: abs(s.t - a))
    pb = min(samples, key=lambda s: abs(s.t - b))
    north = (pb.lat - pa.lat) * 111_320.0
    east = (pb.lon - pa.lon) * 111_320.0 * math.cos(math.radians(pa.lat))
    if math.hypot(north, east) < 40.0:
        return None
    return math.degrees(math.atan2(east, north))


def find_turns(clip: str, min_turn_deg: float = 60.0) -> list[Turn]:
    """Turns of at least `min_turn_deg` with 8 s of straight, moving road before and after."""
    readings = json.loads((CACHE / f"{clip}.json").read_text(encoding="utf-8"))["readings"]
    if len(readings) < 40:
        return []
    samples = smooth_track([TrackSample(r["t"], r["mph"] * MPH, r["lat"], r["lon"]) for r in readings])
    speed = {round(s.t, 1): s.speed_mps for s in samples}
    turns: list[Turn] = []
    t_end = samples[-1].t
    t = samples[0].t + 8.0
    while t < t_end - 20.0:
        before = _bearing(samples, t - 8.0, t)
        before_check = _bearing(samples, t - 8.0, t - 4.0)
        found = None
        for span in (6.0, 8.0, 10.0, 12.0):
            t1 = t + span
            after = _bearing(samples, t1, t1 + 8.0)
            after_check = _bearing(samples, t1 + 4.0, t1 + 8.0)
            if None in (before, after, before_check, after_check):
                continue
            assert before is not None and after is not None and before_check is not None and after_check is not None
            straight = abs(_wrap(before - before_check)) < 8.0 and abs(_wrap(after - after_check)) < 8.0
            change = _wrap(after - before)
            moving = min(speed.get(round(x, 1), 0.0) for x in np.arange(t - 8.0, t1 + 8.0, 1.0)) > 2.5
            if straight and moving and abs(change) >= min_turn_deg:
                found = Turn(clip, t, t1, change)
                break
        if found is not None:
            turns.append(found)
            t = found.t1 + 8.0
        else:
            t += 1.0
    return turns


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def image_swing_px(path: Path, t0: float, t1: float) -> float | None:
    """Summed sideways travel (px, + = scene moves left = we turn right) of the image centre between t0 and t1."""
    import av

    from perception.ego_motion import EgoMotionEstimator

    cfg = load_config()
    ego = EgoMotionEstimator(cfg.ego_motion)
    total, used = 0.0, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        assert tb is not None
        container.seek(int(max(0.0, t0 - 1.0) / tb), stream=stream, backward=True)
        index = 0
        for frame in container.decode(stream):
            t = float(frame.pts * tb) if frame.pts is not None else 0.0
            if t < t0 - 0.5:
                continue
            if t > t1:
                break
            image = frame.to_ndarray(format="bgr24")
            result = ego.update(Frame(image, t, index, time.perf_counter()))
            index += 1
            if t >= t0 and result.homography is not None:
                h = result.homography
                cx, cy = image.shape[1] / 2.0, image.shape[0] / 2.0
                p = h @ np.array([cx, cy, 1.0])
                total += cx - p[0] / p[2]
                used += 1
    return total if used > 10 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--turns", type=int, default=12, help="Measure this many turns (from different clips).")
    args = parser.parse_args(argv)
    clips = sorted(p.stem for p in CACHE.glob("dm_*.json"))
    turns: list[Turn] = []
    for clip in clips:
        found = find_turns(clip)
        if found:
            turns.append(max(found, key=lambda x: abs(x.heading_change_deg)))
    print(f"{len(turns)} clips with a clean turn; measuring {min(args.turns, len(turns))}")
    step = max(1, len(turns) // max(1, args.turns))
    results = []
    for turn in turns[::step][: args.turns]:
        swing = image_swing_px(args.folder / f"{turn.clip}.mp4", turn.t0, turn.t1)
        if swing is None:
            continue
        f = swing / math.radians(turn.heading_change_deg)
        if f <= 0:
            print(
                f"  {turn.clip} {turn.t0:.0f}-{turn.t1:.0f} s: turn {turn.heading_change_deg:+.0f} deg, swing "
                f"{swing:+.0f} px: opposite signs, skipped"
            )
            continue
        hfov = math.degrees(2 * math.atan(FRAME_W / (2 * f)))
        results.append((f, hfov))
        print(
            f"  {turn.clip} {turn.t0:.0f}-{turn.t1:.0f} s: turn {turn.heading_change_deg:+.0f} deg, image swing "
            f"{swing:+.0f} px -> f {f:.0f} px, hfov {hfov:.1f} deg"
        )
    if results:
        fs = np.array([r[0] for r in results])
        f_med = float(np.median(fs))
        print(
            f"median f {f_med:.0f} px (IQR {np.percentile(fs, 25):.0f}-{np.percentile(fs, 75):.0f}): "
            f"hfov {math.degrees(2 * math.atan(FRAME_W / (2 * f_med))):.1f} deg over {len(results)} turns"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
