"""The car dashcam's horizontal field of view, from turns: how far the image swings vs how far the GPS heading turns.

    .venv/Scripts/python scripts/dashcam_calibrate.py D:/Dashcam --turns 12

Turning by an angle dtheta (about the vertical axis) moves the image's centre point sideways by f * tan(dtheta); summed
over the frames of a turn, the centre's sideways travel is ~f * (total heading change). The per-frame homographies
come from the app's own ego-motion estimator (perception/ego_motion.py); the heading change from the clip's overlay
GPS (scripts/dashcam_gpx.py scan), measured over the straight stretches before and after the turn (the positions
are quantized to an arc-second, so long baselines). f = summed travel / heading change; hfov = 2 atan(W / 2f). The
median over several turns is reported (forward motion adds parallax that averages out over left and right turns).

A frame whose homography is missing (too few tracked points, RANSAC failed) or implausible (non-finite, or the centre
jumping more than MAX_FRAME_SWING_PX in one frame) adds nothing to the sum, so a turn with many such frames
under-reads the swing (f too small, hfov too wide). Turns whose failed fraction exceeds --max-missing (default
MAX_MISSING_FRACTION) are reported but left out of the median.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from core.types import Frame  # noqa: E402
from nav.dashcam_overlay import FRAME_W, TrackSample, smooth_track  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache" / "dashcam"
MPH = 0.44704
MAX_MISSING_FRACTION = 0.10  # skip a turn when more than this fraction of its frames has no usable homography
MAX_FRAME_SWING_PX = 150.0  # a centre jump beyond this in one frame (25 fps) is a bad homography, not a turn
MIN_USED_FRAMES = 10  # fewer usable frames than this: no measurement at all

log = get_logger(__name__)


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


@dataclass(frozen=True, slots=True)
class Swing:
    """The image centre's summed sideways travel over a turn, and how many frames contributed to it."""

    total_px: float  # + = scene moves left = we turn right
    frames: int  # frames inside the turn window
    used: int  # frames with a usable homography

    @property
    def missing(self) -> int:
        """Frames whose homography was missing or rejected."""
        return self.frames - self.used

    @property
    def coverage(self) -> float:
        """Fraction of the turn's frames that contributed (0 for an empty window)."""
        return self.used / self.frames if self.frames else 0.0

    def rejection(self, max_missing: float = MAX_MISSING_FRACTION) -> str | None:
        """Why this swing can't be trusted (None if it can)."""
        if self.used < MIN_USED_FRAMES:
            return f"only {self.used} usable frames"
        if self.missing > max_missing * self.frames:
            return f"homography missing on {self.missing}/{self.frames} frames (> {max_missing:.0%})"
        return None


def frame_swing_px(homography: np.ndarray | None, cx: float, cy: float) -> float | None:
    """Sideways travel of the point (cx, cy) under one frame-to-frame homography (None if missing or implausible)."""
    if homography is None or not np.all(np.isfinite(homography)):
        return None
    p = homography @ np.array([cx, cy, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    dx = float(cx - p[0] / p[2])
    if not math.isfinite(dx) or abs(dx) > MAX_FRAME_SWING_PX:
        return None
    return dx


def sum_swing(homographies: Iterable[np.ndarray | None], cx: float, cy: float) -> Swing:
    """Sum the centre's sideways travel over a turn's frame-to-frame homographies, counting the failed ones."""
    total, frames, used = 0.0, 0, 0
    for h in homographies:
        frames += 1
        dx = frame_swing_px(h, cx, cy)
        if dx is not None:
            total += dx
            used += 1
    return Swing(total, frames, used)


def focal_px(swing_px: float, heading_change_deg: float) -> float:
    """Focal length from the summed swing and the GPS heading change (<= 0 when their signs disagree)."""
    return swing_px / math.radians(heading_change_deg)


def hfov_deg(f_px: float, width: int = FRAME_W) -> float:
    """Horizontal field of view (degrees) of a pinhole camera with focal length f_px and image width `width`."""
    return math.degrees(2 * math.atan(width / (2 * f_px)))


def image_swing_px(path: Path, t0: float, t1: float) -> Swing:
    """The image centre's sideways travel between t0 and t1, from the app's ego-motion homographies."""
    import av

    from perception.ego_motion import EgoMotionEstimator

    cfg = load_config()
    ego = EgoMotionEstimator(cfg.ego_motion)
    homographies: list[np.ndarray | None] = []
    cx, cy = FRAME_W / 2.0, 0.0
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
            if t >= t0:
                cx, cy = image.shape[1] / 2.0, image.shape[0] / 2.0
                homographies.append(result.homography)
    swing = sum_swing(homographies, cx, cy)
    log.info(
        "turn swing",
        clip=path.stem,
        t0=round(t0, 1),
        t1=round(t1, 1),
        swing_px=round(swing.total_px, 1),
        frames=swing.frames,
        used=swing.used,
        coverage=round(swing.coverage, 3),
    )
    return swing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--turns", type=int, default=12, help="Measure this many turns (from different clips).")
    parser.add_argument(
        "--max-missing",
        type=float,
        default=MAX_MISSING_FRACTION,
        help="Skip a turn when more than this fraction of its frames has no usable homography.",
    )
    args = parser.parse_args(argv)
    setup_logging("WARNING")
    clips = sorted(p.stem for p in CACHE.glob("dm_*.json"))
    turns: list[Turn] = []
    for clip in clips:
        found = find_turns(clip)
        if found:
            turns.append(max(found, key=lambda x: abs(x.heading_change_deg)))
    print(f"{len(turns)} clips with a clean turn; measuring {min(args.turns, len(turns))}", flush=True)
    step = max(1, len(turns) // max(1, args.turns))
    print(f"{'clip':<22} {'window s':>11} {'turn deg':>9} {'swing px':>9} {'f px':>6} {'hfov':>6} {'cover':>6}  status")
    accepted: list[float] = []
    rejected = 0
    for turn in turns[::step][: args.turns]:
        swing = image_swing_px(args.folder / f"{turn.clip}.mp4", turn.t0, turn.t1)
        f = focal_px(swing.total_px, turn.heading_change_deg)
        reason = swing.rejection(args.max_missing)
        if reason is None and f <= 0:
            reason = "swing and turn have opposite signs"
        f_txt, hfov_txt = (f"{f:6.0f}", f"{hfov_deg(f):6.1f}") if f > 0 else (f"{'-':>6}", f"{'-':>6}")
        print(
            f"{turn.clip:<22} {turn.t0:5.0f}-{turn.t1:<5.0f} {turn.heading_change_deg:+9.0f} {swing.total_px:+9.0f} "
            f"{f_txt} {hfov_txt} {swing.coverage:6.1%}  {'ok' if reason is None else 'SKIPPED: ' + reason}",
            flush=True,
        )
        if reason is None:
            accepted.append(f)
        else:
            rejected += 1
    print(f"{len(accepted)} turns accepted, {rejected} rejected (max missing {args.max_missing:.0%})")
    if accepted:
        fs = np.array(accepted)
        f_med = float(np.median(fs))
        q25, q75 = float(np.percentile(fs, 25)), float(np.percentile(fs, 75))
        print(
            f"median f {f_med:.0f} px (IQR {q25:.0f}-{q75:.0f}): hfov {hfov_deg(f_med):.1f} deg "
            f"(IQR {hfov_deg(q75):.1f}-{hfov_deg(q25):.1f}) over {len(accepted)} turns"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
