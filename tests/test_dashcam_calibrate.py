"""The dashcam FOV calibration's pure parts (scripts/dashcam_calibrate.py): per-frame swing, coverage, f and hfov."""

from __future__ import annotations

import math

import numpy as np

from scripts.dashcam_calibrate import (
    MAX_FRAME_SWING_PX,
    MAX_MISSING_FRACTION,
    MIN_USED_FRAMES,
    Swing,
    focal_px,
    frame_swing_px,
    hfov_deg,
    sum_swing,
)

CX, CY = 960.0, 640.0


def _shift(dx: float) -> np.ndarray:
    """A homography that moves every point by dx to the right."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _yaw(f: float, deg: float) -> np.ndarray:
    """The image homography of a pure camera rotation about the vertical axis (K R K^-1)."""
    a = math.radians(deg)
    k = np.array([[f, 0.0, CX], [0.0, f, CY], [0.0, 0.0, 1.0]])
    r = np.array([[math.cos(a), 0.0, -math.sin(a)], [0.0, 1.0, 0.0], [math.sin(a), 0.0, math.cos(a)]])
    return k @ r @ np.linalg.inv(k)


def test_frame_swing_sign_and_size() -> None:
    assert frame_swing_px(_shift(-5.0), CX, CY) == 5.0  # scene moves left: + (turning right)
    assert frame_swing_px(_shift(3.0), CX, CY) == -3.0
    assert frame_swing_px(np.eye(3), CX, CY) == 0.0


def test_frame_swing_rejects_missing_and_implausible() -> None:
    assert frame_swing_px(None, CX, CY) is None
    assert frame_swing_px(_shift(-(MAX_FRAME_SWING_PX + 1.0)), CX, CY) is None
    bad = np.eye(3)
    bad[0, 2] = np.nan
    assert frame_swing_px(bad, CX, CY) is None
    degenerate = np.eye(3)
    degenerate[2] = [0.0, 0.0, 0.0]  # maps the centre to infinity
    assert frame_swing_px(degenerate, CX, CY) is None


def test_sum_swing_counts_failed_frames() -> None:
    swing = sum_swing([_shift(-2.0), None, _shift(-3.0), _shift(-1000.0)], CX, CY)
    assert swing == Swing(total_px=5.0, frames=4, used=2)
    assert swing.missing == 2 and swing.coverage == 0.5
    assert sum_swing([], CX, CY).coverage == 0.0


def test_rejection_threshold() -> None:
    frames = 100
    allowed = int(MAX_MISSING_FRACTION * frames)
    assert Swing(100.0, frames, frames - allowed).rejection() is None  # exactly at the limit: kept
    reason = Swing(100.0, frames, frames - allowed - 1).rejection()
    assert reason is not None and "missing" in reason
    assert Swing(100.0, frames, 80).rejection(max_missing=0.25) is None  # the CLI option loosens it
    few = Swing(10.0, MIN_USED_FRAMES - 1, MIN_USED_FRAMES - 1).rejection()
    assert few is not None and "usable" in few


def test_rotation_recovers_focal_length_and_fov() -> None:
    f_true, per_frame_deg, n = 850.0, 0.6, 150  # a 90 degree right turn over 6 s at 25 fps
    swing = sum_swing([_yaw(f_true, per_frame_deg)] * n, CX, CY)
    assert swing.coverage == 1.0
    f = focal_px(swing.total_px, per_frame_deg * n)
    assert math.isclose(f, f_true, rel_tol=0.01)  # small per-frame angles: tan(a) ~ a
    assert math.isclose(hfov_deg(f), hfov_deg(f_true), abs_tol=0.5)
    assert focal_px(-swing.total_px, per_frame_deg * n) < 0  # signs disagree: caller skips it


def test_hfov_formula() -> None:
    assert math.isclose(hfov_deg(960.0, 1920), 90.0)
    assert math.isclose(hfov_deg(849.0), 97.0, abs_tol=0.5)  # the first 6-turn estimate
