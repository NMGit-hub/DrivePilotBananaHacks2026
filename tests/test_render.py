"""The showcase renderer's drawing (ui/render.py): geometry, the page's lane easing and highlight, text wrapping,
sprite pasting, and a whole frame drawn without errors."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from ui.render import (
    Renderer,
    apply_highlight,
    card_sprite,
    dash_segments,
    ease_lanes,
    gamma_lut,
    layout_captions,
    make_geometry,
    output_rate,
    paste,
    pill_sprite,
    rgba,
    rounded_rect_points,
    stack_from_bottom,
    wrap_text,
)
from ui.timeline import ALERT, ANSWER, QUESTION, Caption, Shown


def lane(x0: float, x1: float) -> list[list[float]]:
    return [[x0 + (x1 - x0) * i / 7, 1.0 - 0.4 * i / 7] for i in range(8)]


def lanes_packet(shift: float = 0.0, drift: str | None = None) -> dict:
    return {
        "left": lane(0.3 + shift, 0.48 + shift),
        "right": lane(0.8 + shift, 0.52 + shift),
        "left_next": lane(0.0, 0.45),
        "right_next": None,
        "others": [lane(0.95, 0.55)],
        "drift": drift,
        "target": {"lane": "left", "moves": 1},
    }


def test_geometry_dashcam_crop_and_even_sizes() -> None:
    g = make_geometry(1920, 1280, 1280, crop_bottom=45)
    assert (g.keep_h, g.out_w, g.out_h) == (1235, 1280, 822)
    x, y = g.px(0.5, 1235 / 1280)  # the top of the text bar lands on the output's bottom edge
    assert x == pytest.approx(640.0)
    assert y == pytest.approx(822.0)
    full = make_geometry(1280, 720)
    assert (full.out_w, full.out_h, full.fs) == (1280, 720, 16)
    odd = make_geometry(1280, 720, 1001)
    assert odd.out_w % 2 == 0 and odd.out_h % 2 == 0
    with pytest.raises(ValueError):
        make_geometry(100, 40, crop_bottom=40)


def test_ease_lanes_moves_part_way_and_restarts() -> None:
    assert ease_lanes(None, None) is None
    first = ease_lanes(None, lanes_packet())
    assert first == lanes_packet()
    moved = ease_lanes(first, lanes_packet(shift=0.1), k=0.5)
    assert moved is not None
    assert moved["left"][0][0] == pytest.approx(0.35)  # half way from 0.3 to 0.4
    assert moved["target"] == {"lane": "left", "moves": 1}
    changed = lanes_packet()
    changed["others"] = []
    assert ease_lanes(moved, changed) == changed  # a different set of lines: start over
    assert ease_lanes(moved, None) is None


def test_apply_highlight_marks_only_that_object() -> None:
    overlay = {
        "items": [
            {"kind": "vehicle", "id": 3, "box": [0, 0, 1, 1]},
            {"kind": "vehicle", "id": 4, "box": [0, 0, 1, 1]},
            {"kind": "light", "box": [0, 0, 1, 1]},
        ]
    }
    marked = apply_highlight(overlay, 4)
    assert [item.get("pulse", False) for item in marked["items"]] == [False, True, False]
    assert "pulse" not in overlay["items"][1]  # the recorded packet is left alone
    assert apply_highlight(overlay, None) is overlay


def test_wrap_text() -> None:
    def measure(text: str) -> float:
        return float(len(text))

    assert wrap_text("one two three four", measure, 9) == ["one two", "three", "four"]
    assert wrap_text("", measure, 9) == []
    assert wrap_text("abcdefghijkl x", measure, 5) == ["abcde", "fghij", "kl x"]


def test_stack_from_bottom() -> None:
    assert stack_from_bottom([10, 20], bottom=100, gap=5) == [65, 80]
    assert stack_from_bottom([], 100, 5) == []


def test_rounded_rect_points_and_dashes() -> None:
    pts = rounded_rect_points(0, 0, 100, 50, 10)
    assert pts[:, 0].min() == pytest.approx(0) and pts[:, 0].max() == pytest.approx(100)
    assert pts[:, 1].min() == pytest.approx(0) and pts[:, 1].max() == pytest.approx(50)
    assert len(rounded_rect_points(0, 0, 10, 10, 0)) == 4
    pieces = dash_segments(np.array([[0, 0], [100, 0]], np.float32), 10, 10)
    assert len(pieces) == 5
    assert pieces[1][0][0] == pytest.approx(20) and pieces[1][-1][0] == pytest.approx(30)


def test_paste_blends_and_clips() -> None:
    dst = np.zeros((10, 10, 3), np.uint8)
    sprite = np.zeros((4, 4, 4), np.uint8)
    sprite[..., 2] = 200  # red (BGRA)
    sprite[..., 3] = 255
    paste(dst, sprite, 8, 8)  # partly off the image
    assert dst[9, 9].tolist() == [0, 0, 200]
    assert dst[7, 7].tolist() == [0, 0, 0]
    paste(dst, sprite, 0, 0, opacity=0.5)
    assert dst[0, 0, 2] == 100
    paste(dst, sprite, -20, -20)  # entirely off: nothing happens


def test_sprites_have_alpha_and_sizes() -> None:
    pill = pill_sprite("1st", rgba("#0b0d10"), rgba("#fbbf24"), 16, 700)
    assert pill.shape[2] == 4 and pill.shape[0] == 24
    assert pill[pill.shape[0] // 2, 2, 3] > 0 and pill[0, 0, 3] < 255  # rounded corner
    card = card_sprite(ALERT, "Light's green. Check before you go.", 300, 20)
    assert card.shape[1] <= 300 and card.shape[2] == 4


def test_layout_captions_newest_lowest() -> None:
    g = make_geometry(1280, 720)
    q = Caption(QUESTION, [(0.0, "What's that?")], 0.0, 1.0, 1)
    a = Caption(ANSWER, [(1.0, "A stop sign. " * 8)], 1.0, 2.0, 1)
    layout = layout_captions(g, [Shown(q, "What's that?", 1.0), Shown(a, a.parts[0][1], 0.5)])
    (q_sprite, _, qy, _), (a_sprite, _, ay, a_alpha) = layout.cards
    assert qy + q_sprite.shape[0] < ay  # the question above its answer
    assert ay + a_sprite.shape[0] <= g.out_h
    assert a_alpha == 0.5
    assert a_sprite.shape[1] <= g.out_w * 0.5 + 1
    assert layout.top == qy


def test_renderer_draws_a_frame() -> None:
    g = make_geometry(1920, 1280, 960, crop_bottom=45)
    renderer = Renderer(g)
    image = np.full((1280, 1920, 3), 90, np.uint8)
    overlay = {
        "items": [
            {
                "kind": "vehicle",
                "id": 1,
                "box": [0.4, 0.5, 0.5, 0.6],
                "color": "amber",
                "label": "car",
                "badge": "1st",
                "timer": 2.0,
            },
            {"kind": "person", "id": 2, "box": [0.1, 0.5, 0.15, 0.7], "color": "red", "label": "pedestrian"},
            {"kind": "light", "box": [0.5, 0.1, 0.52, 0.15], "color": "green", "label": "GREEN"},
            {"kind": "spot", "box": [0.7, 0.6, 0.9, 0.7], "color": "green", "label": "Open spot"},
            {"kind": "vehicle", "id": 5, "box": [0.0, 0.0, 0.0, 0.0], "color": "white"},  # empty: skipped
        ],
        "you": {"text": "YOU: 2nd", "color": "white"},
        "lanes": lanes_packet(drift="right"),
    }
    hud = {
        "speed_mph": 31,
        "speed_limit": 25,
        "school_limit": 20,
        "banner": {"text": "Turn right", "distance": "in 500 ft", "lanes": [{"valid": True, "ways": ["right"]}]},
        "roundabout": {"exit": 2, "passed": 1},
    }
    card = Caption(ALERT, [(0.0, "Light's green.")], 0.0, 1.0)
    out = renderer.draw(image, 1.0, overlay, hud, [Shown(card, "Light's green.", 1.0)], listening=True, highlight=1)
    assert out.shape == (g.out_h, g.out_w, 3)
    assert not np.array_equal(out, renderer.base(image))  # something was drawn
    blank = renderer.draw(image, 1.1, None, None)
    assert blank.shape == out.shape


def test_gamma_and_title() -> None:
    lut = gamma_lut(2.0)
    assert lut[0] == 0 and lut[255] == 255 and lut[64] == 128  # sqrt: brighter shadows
    assert np.array_equal(gamma_lut(1.0), np.arange(256, dtype=np.uint8))
    g = make_geometry(640, 360)
    bright = Renderer(g, gamma=2.0).base(np.full((360, 640, 3), 64, np.uint8))
    assert int(bright[0, 0, 0]) == 128
    titled = Renderer(g, title="Night", title_from=10.0, title_s=5.0)
    assert [titled.title_alpha(t) for t in (9.0, 10.0, 15.0, 15.25, 16.0)] == [0.0, 1.0, 1.0, 0.5, 0.0]
    assert Renderer(g).title_alpha(10.0) == 0.0


def test_output_rate_snaps() -> None:
    assert output_rate(25.0) == Fraction(25)
    assert output_rate(30.0274) == Fraction(30)
    assert output_rate(29.97) == Fraction(30000, 1001)
    assert output_rate(12.5) == Fraction(25, 2)
