"""ui/motion.py: easing, frame sizes, the stinger's reveal."""

from __future__ import annotations

import numpy as np
import pytest

from ui.motion import (
    Shot,
    ease_in_out_cubic,
    ease_out_back,
    ease_out_cubic,
    end_part,
    how_part,
    progress,
    shots_part,
    stinger,
    title_part,
)

W, H = 320, 206


def test_easing_starts_at_0_and_ends_at_1() -> None:
    for ease in (ease_out_cubic, ease_in_out_cubic, ease_out_back):
        assert ease(0.0) == pytest.approx(0.0) and ease(1.0) == pytest.approx(1.0)
        assert ease(-1.0) == pytest.approx(0.0) and ease(2.0) == pytest.approx(1.0)  # clamped
    assert max(ease_out_back(x / 100) for x in range(101)) > 1.0  # the pop overshoots
    assert progress(1.5, 1.0, 1.0) == 0.5 and progress(0.0, 1.0, 1.0) == 0.0 and progress(5.0, 1.0, 1.0) == 1.0


def test_every_part_draws_a_bgr_frame_of_the_reel_size() -> None:
    shot = Shot(np.full((90, 160, 3), 120, np.uint8), "Tells you when it's green", "reads lights")
    for frame in (
        title_part(W, H, 1.0, 3.6),
        how_part(W, H, 3.0, 7.5),
        shots_part(W, H, 1.0, 3.2, [shot]),
        end_part(W, H, 0.5, 1.8, "Phone + laptop"),
    ):
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    assert title_part(W, H, 0.0, 3.6).std() < title_part(W, H, 2.0, 3.6).std()  # the title builds up


def test_the_stinger_reveals_the_next_scene_behind_its_band() -> None:
    prev, nxt = np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 200, np.uint8)
    start, end = stinger(prev, nxt, 0.0, 0.7, "02", "Lane"), stinger(prev, nxt, 0.7, 0.7, "02", "Lane")
    assert (start == prev).mean() > 0.95  # before: the previous scene
    assert (end == nxt).mean() > 0.95  # after: the next one
