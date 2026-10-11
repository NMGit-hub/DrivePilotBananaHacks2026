"""scripts/showcase_reel.py: item parsing, title cards, letterboxing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.showcase_reel import Item, fit, parse_item, title_card


def test_parse_item() -> None:
    assert parse_item("a/b.mp4=Rain|What did that sign say?") == Item(
        Path("a/b.mp4"), "Rain", "What did that sign say?"
    )
    assert parse_item("b.mp4=Night") == Item(Path("b.mp4"), "Night", "")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_item("b.mp4")


def test_title_card_is_a_dark_bgr_frame_with_text() -> None:
    card = title_card(640, 410, "Rain", "What did that sign say?")
    assert card.shape == (410, 640, 3) and card.dtype == np.uint8
    assert card[5, 5].tolist() == [0x10, 0x0D, 0x0B]  # the page's background (#0b0d10), as BGR
    assert card.max() > 200  # the title is drawn


def test_fit_letterboxes_to_the_reel_size() -> None:
    same = np.full((410, 640, 3), 50, np.uint8)
    assert fit(same, 640, 410) is same
    wide = fit(np.full((360, 640, 3), 50, np.uint8), 640, 410)
    assert wide.shape == (410, 640, 3) and wide[0, 320].tolist() == [0, 0, 0] and wide[205, 320].tolist() == [50] * 3
