"""Draw the V2 page's overlays into video frames, for the showcase videos (scripts/render_showcase.py).

The page (ui/static/v2.html) draws `ui.overlay.frame_packet` on a canvas over the video and shows the HUD
(`hud_packet`) and the conversation as HTML. Here the same things are drawn into the frame itself, with the page's
colors and proportions: shapes with OpenCV (anti-aliased, with the page's opacities), text and badges with Pillow
(Segoe UI, the page's system-ui font on Windows), rendered at 2x and shrunk so their edges are smooth, and cached
(the same labels repeat frame after frame).

- our lane filled (green when it is the one to be in), its two lines white with a dark halo, the line we drift
  toward red and that side's screen edge pulsing; the other lanes' lines dashed gray; the target lane in green
  with "move one lane left";
- tracked objects: boxes in their color (green = go / your turn, red = danger, amber = watch this, white = tracked,
  gray = parked), 4-way order badges with their wait timers, labels for people, signs and lights, open parking
  spots filled green, the object an answer is about pulsing, "YOU: 2nd" at the bottom;
- the HUD: speed-limit sign and speed badge top right, the navigation banner top center;
- the conversation as caption cards at the bottom left (the page's feed): "You" in blue, "DrivePilot" in white,
  alerts amber on brown with an amber bar; and "DrivePilot" with a green "Listening…" while the driver speaks.

`Renderer.draw` does one frame. Pure helpers (geometry, lane easing, highlight, text wrapping, sprite pasting) are
tested in tests/test_render.py.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui.timeline import ALERT, ANSWER, QUESTION, Shown

# The page's palette (v2.html :root)
BG = "#0b0d10"
PANEL = "#14171c"
TEXT = "#e8ebef"
MUTED = "#8b94a3"
YOU = "#7cc4ff"
ALERT_BG = "#3a2a0a"
ALERT_FG = "#ffc24d"
LISTEN = "#4ade80"
GREEN = "#34d399"
RED = "#f87171"
AMBER = "#fbbf24"
WHITE = "#f1f5f9"
GRAY = "#94a3b8"
COLORS = {"green": GREEN, "red": RED, "amber": AMBER, "white": WHITE, "gray": GRAY}

RGBA = tuple[int, int, int, int]
SS = 2  # sprites are drawn at this scale and shrunk (smooth edges)
LANE_EASE = 0.45  # the page's easing toward each new lane reading, per drawn frame
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
FONT_FILES = ((850, "seguibl.ttf"), (650, "segoeuib.ttf"), (550, "seguisb.ttf"), (0, "segoeui.ttf"))
ARROWS = {
    "left": "←",
    "slight left": "↖",
    "sharp left": "↰",
    "straight": "↑",
    "slight right": "↗",
    "right": "→",
    "sharp right": "↱",
    "uturn": "↶",
}
CARD_LABELS = {QUESTION: "YOU", ANSWER: "DRIVEPILOT", ALERT: "ALERT"}


def rgba(color: str, alpha: float = 1.0) -> RGBA:
    """ "#rrggbb" (+ opacity 0..1) -> an (r, g, b, a) tuple for Pillow."""
    n = int(color.lstrip("#"), 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255, round(255 * alpha)


def bgr(color: str) -> tuple[int, int, int]:
    """ "#rrggbb" -> an OpenCV (b, g, r) color."""
    r, g, b, _ = rgba(color)
    return b, g, r


# ------------------------------------------------------------------------------------------------ geometry


@dataclass(frozen=True, slots=True)
class Geometry:
    """Where the source frame goes in the output: the top `keep_h` rows of a `src_w` x `src_h` frame (a dashcam's
    burned-in text bar below them is cropped), scaled to `out_w` x `out_h`. Overlay coordinates are fractions of the
    whole source frame."""

    src_w: int
    src_h: int
    keep_h: int
    out_w: int
    out_h: int

    @property
    def sx(self) -> float:
        """Output pixels per source pixel, across."""
        return self.out_w / self.src_w

    @property
    def sy(self) -> float:
        """Output pixels per source pixel, down."""
        return self.out_h / self.keep_h

    def px(self, nx: float, ny: float) -> tuple[float, float]:
        """A point in normalized source-frame coordinates -> output pixels."""
        return nx * self.src_w * self.sx, ny * self.src_h * self.sy

    @property
    def fs(self) -> int:
        """The page's label size for this video height (v2.html: max(11, vh / 46))."""
        return max(11, round(self.out_h / 46))

    @property
    def lw(self) -> float:
        """One page pixel for line widths (the page multiplies by the screen's pixel density)."""
        return max(1.0, self.out_h / 600)

    @property
    def u(self) -> float:
        """One CSS pixel of the page's HUD and feed at this video size (the page at a ~720 px tall video)."""
        return self.out_h / 720


def output_rate(fps: float) -> Fraction:
    """The output frame rate: the source's, snapped to 24/25/30/50/60 or their NTSC (x 1000/1001) rates."""
    for base in (24, 25, 30, 50, 60):
        if abs(fps - base * 1000 / 1001) < 0.005:
            return Fraction(base * 1000, 1001)
        if abs(fps - base) < 0.05:
            return Fraction(base)
    return Fraction(fps).limit_denominator(1001)


def make_geometry(src_w: int, src_h: int, out_w: int | None = None, crop_bottom: int = 0) -> Geometry:
    """The output geometry: `out_w` wide (default: the source's), height to keep the aspect ratio of the kept rows,
    both even (H.264 with 4:2:0 chroma needs even sizes)."""
    keep_h = src_h - max(0, crop_bottom)
    if keep_h <= 0:
        raise ValueError(f"cropping {crop_bottom} rows leaves nothing of a {src_h}-row frame")
    width = out_w if out_w is not None else src_w
    width -= width % 2
    height = round(keep_h * width / src_w)
    height -= height % 2
    return Geometry(src_w, src_h, keep_h, width, height)


# --------------------------------------------------------------------------------------------- page logic


def ease_lanes(shown: dict[str, Any] | None, target: dict[str, Any] | None, k: float = LANE_EASE) -> dict | None:
    """The page's easeLanes: move the drawn lane lines a share `k` toward the latest reading (so they glide between
    the lane model's ~12 updates a second); start over when the set of lines changes."""
    if not target:
        return None
    if not shown or len(shown.get("others", [])) != len(target.get("others", [])):
        return copy.deepcopy(target)

    def mix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        if len(a) != len(b):
            return copy.deepcopy(b)
        return [[p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1])] for p, q in zip(a, b, strict=True)]

    out = dict(shown)
    out["left"] = mix(shown["left"], target["left"])
    out["right"] = mix(shown["right"], target["right"])
    for key in ("left_next", "right_next"):
        out[key] = (
            mix(shown[key], target[key]) if target.get(key) and shown.get(key) else copy.deepcopy(target.get(key))
        )
    out["others"] = [mix(a, b) for a, b in zip(shown["others"], target["others"], strict=True)]
    out["drift"] = target.get("drift")
    out["target"] = target.get("target")
    return out


def apply_highlight(overlay: dict[str, Any], track_id: int | None) -> dict[str, Any]:
    """The overlay with the object an answer is about marked to pulse (as `frame_packet(highlight=...)`)."""
    if track_id is None:
        return overlay
    items = [
        {**item, "pulse": True}
        if item.get("id") == track_id and item.get("kind") in ("vehicle", "person", "sign")
        else item
        for item in overlay.get("items", [])
    ]
    return {**overlay, "items": items}


def wrap_text(text: str, measure: Callable[[str], float], max_width: float) -> list[str]:
    """Greedy word wrap to `max_width` (as `measure` measures); a word longer than a line is split."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}" if current else word
        if measure(trial) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
            current = ""
        while measure(word) > max_width and len(word) > 1:  # split a word too long for a line
            cut = len(word) - 1
            while cut > 1 and measure(word[:cut]) > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


def stack_from_bottom(heights: Sequence[int], bottom: float, gap: float) -> list[float]:
    """Top y of cards stacked upward from `bottom` (the last card lowest), `gap` apart."""
    tops: list[float] = []
    y = bottom
    for h in reversed(heights):
        y -= h
        tops.append(y)
        y -= gap
    return tops[::-1]


def rounded_rect_points(x1: float, y1: float, x2: float, y2: float, r: float, n: int = 6) -> np.ndarray:
    """A rounded rectangle as a closed polygon (float points, clockwise from the top-left corner's arc)."""
    r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    if r < 0.5:
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    pts: list[tuple[float, float]] = []
    corners = (
        (x1 + r, y1 + r, math.pi),
        (x2 - r, y1 + r, 1.5 * math.pi),
        (x2 - r, y2 - r, 0.0),
        (x1 + r, y2 - r, 0.5 * math.pi),
    )
    for cx, cy, a0 in corners:
        for i in range(n + 1):
            a = a0 + (math.pi / 2) * i / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.float32)


def paste(dst: np.ndarray, sprite: np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite onto a BGR image at (x, y) (clipped to the image), in place."""
    if opacity <= 0.0:
        return
    h, w = sprite.shape[:2]
    xi, yi = round(x), round(y)
    x1, y1 = max(xi, 0), max(yi, 0)
    x2, y2 = min(xi + w, dst.shape[1]), min(yi + h, dst.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    part = sprite[y1 - yi : y2 - yi, x1 - xi : x2 - xi]
    a = part[..., 3:4].astype(np.float32) * (min(opacity, 1.0) / 255.0)
    roi = dst[y1:y2, x1:x2]
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + part[..., :3].astype(np.float32) * a + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ sprites


@lru_cache(maxsize=64)
def font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Segoe UI (the page's system-ui on Windows) at a pixel size and CSS weight; Pillow's own font elsewhere."""
    name = next(file for floor, file in FONT_FILES if weight >= floor)
    for candidate in (FONT_DIR / name, FONT_DIR / "segoeui.ttf"):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size)


@lru_cache(maxsize=16)
def symbol_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Segoe UI Symbol (the lane arrows: Segoe UI itself has only some of them, e.g. no diagonal ones)."""
    path = FONT_DIR / "seguisym.ttf"
    return ImageFont.truetype(str(path), size) if path.is_file() else font(size, 700)


def _length(f: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> float:
    return float(f.getlength(text))


def _canvas(w: float, h: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (max(SS, math.ceil(w / SS) * SS), max(SS, math.ceil(h / SS) * SS)), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _shrink(img: Image.Image) -> np.ndarray:
    """2x canvas -> BGRA array at the output size."""
    small = img.resize((img.width // SS, img.height // SS), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.uint8)
    return np.ascontiguousarray(arr[..., [2, 1, 0, 3]])


def _text_layer(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer)


@lru_cache(maxsize=4096)
def pill_sprite(text: str, fg: RGBA, bg: RGBA, size: int, weight: int = 700) -> np.ndarray:
    """The page's pill(): `text` in a rounded label of height 1.5 x `size`."""
    f = font(size * SS, weight)
    h = size * 1.5 * SS
    w = _length(f, text) + size * 0.9 * SS
    img, d = _canvas(w, h)
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=img.height / 2, fill=bg)
    layer, t = _text_layer(img.size)
    t.text((size * 0.45 * SS, img.height / 2 + 0.5 * SS), text, font=f, fill=fg, anchor="lm")
    return _shrink(Image.alpha_composite(img, layer))


@lru_cache(maxsize=256)
def speed_sprite(mph: int, u: float) -> np.ndarray:
    """The speed badge (number over "MPH")."""
    fv, fu = font(round(26 * u * SS), 700), font(round(11 * u * SS), 600)
    value, unit = str(mph), "MPH"
    pad_x, pad_y = 12 * u * SS, 6 * u * SS
    vh, uh = 26 * 1.05 * u * SS, 11 * 1.45 * u * SS
    w = max(74 * u * SS, max(_length(fv, value), _length(fu, unit)) + 2 * pad_x)
    h = pad_y * 2 + vh + uh
    img, d = _canvas(w, h)
    d.rounded_rectangle(
        (0, 0, img.width - 1, img.height - 1),
        radius=12 * u * SS,
        fill=rgba("#0c1016", 0.82),
        outline=(255, 255, 255, 20),
        width=max(1, round(u * SS)),
    )
    layer, t = _text_layer(img.size)
    cx = img.width / 2
    t.text((cx, pad_y + vh / 2), value, font=fv, fill=rgba(TEXT), anchor="mm")
    t.text((cx, pad_y + vh + uh / 2), unit, font=fu, fill=rgba(MUTED), anchor="mm")
    return _shrink(Image.alpha_composite(img, layer))


@lru_cache(maxsize=256)
def limit_sprite(value: int, school: bool, u: float) -> np.ndarray:
    """The US speed-limit sign badge: black on white (yellow-green for a school limit), a black border inside a
    white edge."""
    s = u * SS
    edge, border = 2 * s, 2 * s
    inner_w = 64 * s
    fl, fv = font(round(10 * s), 900), font(round(30 * s), 900)
    label_h = 10 * 1.05 * s
    inner_h = border * 2 + 5 * s + label_h * 2 + 2 * s + 30 * s + 6 * s
    img, d = _canvas(inner_w + 2 * edge, inner_h + 2 * edge)
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=9 * s, fill=(255, 255, 255, 255))
    d.rounded_rectangle((edge, edge, edge + inner_w, edge + inner_h), radius=7 * s, fill=(0, 0, 0, 255))
    face = (0xE6, 0xF2, 0x4A, 255) if school else (255, 255, 255, 255)
    d.rounded_rectangle(
        (edge + border, edge + border, edge + inner_w - border, edge + inner_h - border), radius=5 * s, fill=face
    )
    layer, t = _text_layer(img.size)
    cx = img.width / 2
    y = edge + border + 5 * s
    t.text((cx, y + label_h / 2), "SPEED", font=fl, fill=(0, 0, 0, 255), anchor="mm")
    t.text((cx, y + label_h * 1.5), "LIMIT", font=fl, fill=(0, 0, 0, 255), anchor="mm")
    t.text((cx, y + label_h * 2 + 2 * s + 15 * s), str(value), font=fv, fill=(0, 0, 0, 255), anchor="mm")
    return _shrink(Image.alpha_composite(img, layer))


@lru_cache(maxsize=256)
def banner_sprite(text: str, distance: str, lanes: tuple[tuple[bool, str], ...], u: float) -> np.ndarray:
    """The navigation banner: lane arrows (the ones that work for the maneuver in green), the step, the distance."""
    s = u * SS
    fstep, fdist, farrow = font(round(20 * s), 650), font(round(18 * s), 700), symbol_font(round(20 * s))
    pad_x, pad_y, gap = 18 * s, 10 * s, 14 * s
    lane_w, lane_h, lane_gap = 30 * s, 34 * s, 4 * s
    lanes_w = len(lanes) * lane_w + max(0, len(lanes) - 1) * lane_gap
    parts = [w for w in (lanes_w, _length(fstep, text), _length(fdist, distance)) if w > 0]
    w = pad_x * 2 + sum(parts) + gap * (len(parts) - 1)
    h = pad_y * 2 + max(lane_h if lanes else 0, 20 * 1.45 * s)
    img, d = _canvas(w, h)
    d.rounded_rectangle(
        (0, 0, img.width - 1, img.height - 1), radius=12 * s, fill=rgba("#0c1016", 0.82), outline=(255, 255, 255, 20)
    )
    layer, t = _text_layer(img.size)
    x, cy = pad_x, img.height / 2
    for valid, way in lanes:
        box = (x, cy - lane_h / 2, x + lane_w, cy + lane_h / 2)
        d.rounded_rectangle(box, radius=5 * s, fill=rgba(GREEN, 0.9) if valid else rgba(GRAY, 0.18))
        fg = rgba("#062418") if valid else rgba(WHITE, 0.45)
        t.text((x + lane_w / 2, cy), ARROWS.get(way, "↑"), font=farrow, fill=fg, anchor="mm")
        x += lane_w + lane_gap
    if lanes:
        x += gap - lane_gap
    t.text((x, cy), text, font=fstep, fill=rgba(TEXT), anchor="lm")
    x += _length(fstep, text) + gap
    t.text((x, cy), distance, font=fdist, fill=rgba(GREEN), anchor="lm")
    return _shrink(Image.alpha_composite(img, layer))


@lru_cache(maxsize=64)
def ring_sprite(exit_no: int, passed: int, u: float) -> np.ndarray:
    """The roundabout diagram: we enter at the bottom, our exit in green, the exits passed dimmed."""
    s = u * SS
    pad = 6 * s
    size = 72 * s
    fu = font(round(11 * s), 600)
    w, h = size + 2 * pad, size * 76 / 72 + 2 * pad + 16 * s
    img, d = _canvas(w, h)
    d.rounded_rectangle(
        (0, 0, img.width - 1, img.height - 1), radius=12 * s, fill=rgba("#0c1016", 0.82), outline=(255, 255, 255, 20)
    )
    n, r, c = max(exit_no + 1, 4), 26 * s, 36 * s
    ox, oy = pad, pad
    d.ellipse((ox + c - r, oy + c - r, ox + c + r, oy + c + r), outline=rgba(GRAY, 0.55), width=round(7 * s))
    d.line((ox + c, oy + c + r, ox + c, oy + c + r + 10 * s), fill=rgba(WHITE), width=round(5 * s))
    for k in range(1, exit_no + 1):
        a = math.pi / 2 - 2 * math.pi * k / n
        x1, y1 = ox + c + r * math.cos(a), oy + c + r * math.sin(a)
        x2, y2 = ox + c + (r + 10 * s) * math.cos(a), oy + c + (r + 10 * s) * math.sin(a)
        color = rgba(GREEN) if k == exit_no else rgba(GRAY, 0.45) if k <= passed else rgba(WHITE)
        d.line((x1, y1, x2, y2), fill=color, width=round((7 if k == exit_no else 4) * s))
    layer, t = _text_layer(img.size)
    t.text((img.width / 2, oy + size * 76 / 72 + 8 * s), f"exit {exit_no}", font=fu, fill=rgba(MUTED), anchor="mm")
    return _shrink(Image.alpha_composite(img, layer))


def card_sprite(
    kind: str, text: str, max_w: int, size: int, image: np.ndarray | None = None, image_w: int = 0
) -> np.ndarray:
    """A caption card (the page's feed entry): a muted "YOU" / "DRIVEPILOT" / "ALERT" label over the text, which
    wraps to `max_w` pixels; a question in blue, an answer in white, an alert amber on brown with an amber bar.
    `image`: a picture under an answer (the remembered frame it came from), `image_w` wide."""
    return _card_sprite(kind, text, max_w, size, image_w if image is not None else 0, _ImageKey(image))


class _ImageKey:
    """Hashes an optional image by identity (the card cache keeps the same thumbnail object)."""

    __slots__ = ("image",)

    def __init__(self, image: np.ndarray | None) -> None:
        self.image = image

    def __hash__(self) -> int:
        return id(self.image)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ImageKey) and other.image is self.image


@lru_cache(maxsize=512)
def _card_sprite(kind: str, text: str, max_w: int, size: int, image_w: int, image_key: _ImageKey) -> np.ndarray:
    s = SS
    alert = kind == ALERT
    weight = 600 if kind in (QUESTION, ALERT) else 400
    color = rgba(YOU) if kind == QUESTION else rgba(ALERT_FG) if alert else rgba(TEXT)
    f = font(round(size * s), weight)
    fl = font(round(size * 0.56 * s), 600)
    pad_x, pad_y = size * 0.62 * s, size * 0.48 * s
    bar = size * 0.17 * s if alert else 0.0
    label = CARD_LABELS.get(kind, "DRIVEPILOT")
    label_h = size * 0.56 * 1.5 * s
    line_h = size * 1.3 * s
    lines = wrap_text(text, lambda x: _length(f, x), max_w * s - 2 * pad_x - bar)
    text_w = max([_length(f, line) for line in lines] + [_length(fl, label)])
    image = image_key.image
    img_h = 0.0
    if image is not None and image_w > 0:
        img_h = image_w * s * image.shape[0] / image.shape[1]
        text_w = max(text_w, image_w * s)
    w = text_w + 2 * pad_x + bar
    h = pad_y * 2 + label_h + line_h * len(lines) + (img_h + pad_y * 0.8 if img_h else 0.0)
    canvas, d = _canvas(w, h)
    radius = size * (0.3 if alert else 0.5) * s
    if alert:
        d.rounded_rectangle((0, 0, canvas.width - 1, canvas.height - 1), radius=radius, fill=rgba(ALERT_BG, 0.93))
        d.rounded_rectangle((0, 0, bar * 1.6, canvas.height - 1), radius=radius, fill=rgba(ALERT_FG))
        d.rectangle((bar, 0, bar * 1.6, canvas.height - 1), fill=rgba(ALERT_BG, 0.93))
    else:
        d.rounded_rectangle(
            (0, 0, canvas.width - 1, canvas.height - 1),
            radius=radius,
            fill=rgba(PANEL, 0.88),
            outline=(255, 255, 255, 24),
            width=max(1, round(s * size / 22)),
        )
    layer, t = _text_layer(canvas.size)
    x = bar + pad_x
    y = pad_y
    t.text((x, y + label_h / 2), label, font=fl, fill=rgba(MUTED), anchor="lm")
    y += label_h
    for line in lines:
        t.text((x, y + line_h / 2), line, font=f, fill=color, anchor="lm")
        y += line_h
    out = Image.alpha_composite(canvas, layer)
    if image is not None and img_h:
        thumb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pic = Image.fromarray(thumb).resize((round(image_w * s), round(img_h)), Image.Resampling.LANCZOS)
        mask = Image.new("L", pic.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, pic.width - 1, pic.height - 1), radius=6 * s, fill=255)
        out.paste(pic, (round(x), round(y + pad_y * 0.8)), mask)
    return _shrink(out)


# ------------------------------------------------------------------------------------------------ shapes


def _fixed(points: np.ndarray, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Float points -> OpenCV's fixed-point (4 fractional bits) int32, shifted by (dx, dy)."""
    return np.round((points + np.array([dx, dy], np.float32)) * 16.0).astype(np.int32)


def _bounds(points: Sequence[np.ndarray], margin: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    allpts = np.vstack(points)
    x1 = max(0, math.floor(float(allpts[:, 0].min()) - margin))
    y1 = max(0, math.floor(float(allpts[:, 1].min()) - margin))
    x2 = min(w, math.ceil(float(allpts[:, 0].max()) + margin))
    y2 = min(h, math.ceil(float(allpts[:, 1].max()) + margin))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def blend_shapes(
    img: np.ndarray,
    alpha: float,
    points: Sequence[np.ndarray],
    draw: Callable[[np.ndarray, float, float], None],
    margin: float = 8.0,
) -> None:
    """Draw shapes at an opacity: `draw(canvas, dx, dy)` draws onto a copy of the region around `points` (shifted
    by (dx, dy)), which is then blended back at `alpha` (overlapping shapes of one call don't add up)."""
    if not points or alpha <= 0.0:
        return
    box = _bounds(points, margin, img.shape[1], img.shape[0])
    if box is None:
        return
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    canvas = roi.copy()
    draw(canvas, -x1, -y1)
    cv2.addWeighted(canvas, min(alpha, 1.0), roi, 1.0 - min(alpha, 1.0), 0.0, dst=roi)


def _polyline(
    canvas: np.ndarray, pts: np.ndarray, dx: float, dy: float, color: str, width: float, closed: bool = False
) -> None:
    cv2.polylines(canvas, [_fixed(pts, dx, dy)], closed, bgr(color), max(1, round(width)), cv2.LINE_AA, shift=4)


def _fill(canvas: np.ndarray, pts: np.ndarray, dx: float, dy: float, color: str) -> None:
    cv2.fillPoly(canvas, [_fixed(pts, dx, dy)], bgr(color), cv2.LINE_AA, shift=4)


def _fill_shape(pts: np.ndarray, color: str, canvas: np.ndarray, dx: float, dy: float) -> None:
    _fill(canvas, pts, dx, dy, color)


def _outline_shape(pts: np.ndarray, color: str, width: float, canvas: np.ndarray, dx: float, dy: float) -> None:
    _polyline(canvas, pts, dx, dy, color, width, closed=True)


def dash_segments(pts: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    """The visible pieces of a dashed polyline (`dash` on, `gap` off along its length)."""
    pieces: list[np.ndarray] = []
    if len(pts) < 2 or dash <= 0:
        return pieces
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(along[-1])
    start = 0.0
    while start < total:
        end = min(start + dash, total)
        at = np.concatenate(([start], along[(along > start) & (along < end)], [end]))
        xs = np.interp(at, along, pts[:, 0])
        ys = np.interp(at, along, pts[:, 1])
        pieces.append(np.stack([xs, ys], axis=1).astype(np.float32))
        start = end + gap
    return pieces


# ------------------------------------------------------------------------------------------------ layers


def draw_lanes(img: np.ndarray, g: Geometry, lanes: dict[str, Any], t: float) -> None:
    """Our lane (filled; green when it's the one to be in), the target lane, the other lines dashed, our two lines
    with a dark halo (the one we drift toward red) and the drift side's screen edge pulsing."""

    def at(line: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array([g.px(p[0], p[1]) for p in line], np.float32)

    lw = g.lw
    left, right = at(lanes["left"]), at(lanes["right"])
    target = lanes.get("target") or {}
    goal = target.get("lane")
    lane = np.vstack([left, right[::-1]])
    fill = GREEN if goal == "ours" else WHITE
    blend_shapes(img, 0.3 if goal == "ours" else 0.18, [lane], lambda c, dx, dy: _fill(c, lane, dx, dy, fill), 2)
    if goal in ("left", "right"):
        inner = left if goal == "left" else right
        outer_raw = lanes.get("left_next" if goal == "left" else "right_next")
        if outer_raw:
            beside = np.vstack([at(outer_raw), inner[::-1]])
            blend_shapes(img, 0.32, [beside], lambda c, dx, dy: _fill(c, beside, dx, dy, GREEN), 2)
        fs = max(11, round(g.out_h / 40))
        n = int(target.get("moves") or 1)
        plural = "s" if n > 1 else ""
        text = f"← move {n} lane{plural} left" if goal == "left" else f"move {n} lane{plural} right →"
        sprite = pill_sprite(text, rgba("#062418"), rgba(GREEN), fs, 700)
        bx, by = inner[min(2, len(inner) - 1)]
        sw = sprite.shape[1]
        x = max(8, bx - sw - fs) if goal == "left" else min(g.out_w - sw - 8, bx + fs)
        paste(img, sprite, x, by - fs * 2.2)
    others = [at(line) for line in lanes.get("others", []) if line]
    if others:
        dashes = [piece for line in others for piece in dash_segments(line, 8 * lw, 8 * lw)]

        def draw_others(c: np.ndarray, dx: float, dy: float) -> None:
            for piece in dashes:
                _polyline(c, piece, dx, dy, GRAY, 2 * lw)

        if dashes:
            blend_shapes(img, 0.6, dashes, draw_others)
    drift = lanes.get("drift")
    ours = {"left": left, "right": right}

    def halos(c: np.ndarray, dx: float, dy: float) -> None:
        for side, pts in ours.items():
            _polyline(c, pts, dx, dy, BG, (9 if drift == side else 7) * lw)

    def lines(c: np.ndarray, dx: float, dy: float) -> None:
        for side, pts in ours.items():
            red = drift == side
            _polyline(c, pts, dx, dy, RED if red else WHITE, (5 if red else 3) * lw)

    blend_shapes(img, 0.55, [left, right], halos, 10 * lw)
    blend_shapes(img, 0.95, [left, right], lines, 10 * lw)
    if drift in ("left", "right"):
        p = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 2)
        width = max(1, round(g.out_w * 0.07))
        ramp = np.linspace(0.25 + 0.45 * p, 0.0, width, dtype=np.float32)[None, :, None]
        region = img[:, :width] if drift == "left" else img[:, g.out_w - width :]
        if drift == "right":
            ramp = ramp[:, ::-1]
        red = np.array(bgr(RED), np.float32)
        region[:] = (region.astype(np.float32) * (1.0 - ramp) + red * ramp + 0.5).astype(np.uint8)


def draw_items(img: np.ndarray, g: Geometry, items: Sequence[dict[str, Any]], t: float) -> None:
    """Every tracked object: its box (spots filled), its order badge and wait timer, its label."""
    fs = g.fs
    lw = g.lw
    for item in items:
        bx1, by1, bx2, by2 = item["box"]
        x1, y1 = g.px(bx1, by1)
        x2, y2 = g.px(bx2, by2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        color_name = item.get("color", "white")
        color = COLORS.get(color_name, WHITE)
        kind = item.get("kind")
        width = 1.2 if kind == "vehicle" and color_name == "gray" else 3.0 if kind == "spot" else 2.0
        alpha = 0.55 if color_name == "gray" else 0.95
        if item.get("pulse"):
            p = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.5)
            width, alpha = 3 + 4 * p, 0.6 + 0.4 * p
        if kind == "spot":
            area = rounded_rect_points(x1, y1, x2, y2, 6 * lw)
            blend_shapes(img, 0.22, [area], partial(_fill_shape, area, color), 2)
        outline = rounded_rect_points(x1, y1, x2, y2, min(6 * lw, w / 4, h / 4))
        blend_shapes(img, alpha, [outline], partial(_outline_shape, outline, color, width * lw), width * lw + 2)
        tx, ty = x1, y1 - fs * 1.6
        if ty < 2:
            ty = y1 + 2
        if item.get("badge"):
            fg = rgba("#231703") if color_name == "amber" else rgba(BG)
            sprite = pill_sprite(str(item["badge"]), fg, rgba(color), fs, 700)
            paste(img, sprite, tx, ty)
            tx += sprite.shape[1] + 4
            timer = item.get("timer")
            if isinstance(timer, int | float):
                sprite = pill_sprite(f"{timer:.1f} s", rgba(WHITE), rgba("#0c1016", 0.78), fs, 700)
                paste(img, sprite, tx, ty)
                tx += sprite.shape[1] + 4
        label = item.get("label")
        if label and (kind != "vehicle" or item.get("badge") or item.get("pulse")):
            solid = kind in ("light", "spot")
            sprite = pill_sprite(
                str(label),
                rgba(BG) if solid else rgba(WHITE),
                rgba(color) if solid else rgba("#0c1016", 0.78),
                round(fs * (0.85 if item.get("badge") else 0.9)),
                700,
            )
            paste(img, sprite, tx, ty)


def draw_you(img: np.ndarray, g: Geometry, you: dict[str, Any], bottom: float) -> None:
    """Our own 4-way badge, bottom center, above the caption cards (`bottom`: where they start)."""
    size = round(g.fs * 1.4)
    green = you.get("color") == "green"
    sprite = pill_sprite(
        str(you.get("text", "")),
        rgba("#062418") if green else rgba(WHITE),
        rgba(GREEN) if green else rgba("#0c1016", 0.85),
        size,
        700,
    )
    y = min(g.out_h - size * 2.4, bottom - sprite.shape[0] - 8)
    paste(img, sprite, g.out_w / 2 - sprite.shape[1] / 2, y)


def draw_hud(img: np.ndarray, g: Geometry, hud: dict[str, Any] | None) -> None:
    """The corner (speed-limit sign, speed, roundabout) and the navigation banner."""
    if not hud:
        return
    u = g.u
    x_right, y = g.out_w - 12 * u, 12 * u
    school = hud.get("school_limit")
    shown = school if isinstance(school, int) else hud.get("speed_limit")
    sprites: list[np.ndarray] = []
    if isinstance(shown, int):
        sprites.append(limit_sprite(shown, isinstance(school, int), round(u, 3)))
    if isinstance(hud.get("speed_mph"), int):
        sprites.append(speed_sprite(hud["speed_mph"], round(u, 3)))
    ring = hud.get("roundabout")
    if ring:
        sprites.append(ring_sprite(int(ring["exit"]), int(ring.get("passed", 0)), round(u, 3)))
    for sprite in sprites:
        paste(img, sprite, x_right - sprite.shape[1], y)
        y += sprite.shape[0] + 10 * u
    banner = hud.get("banner")
    if banner:
        lanes = tuple(
            (bool(lane.get("valid")), (lane.get("ways") or ["straight"])[0]) for lane in banner.get("lanes") or []
        )
        sprite = banner_sprite(str(banner.get("text", "")), str(banner.get("distance", "")), lanes, round(u, 3))
        paste(img, sprite, g.out_w / 2 - sprite.shape[1] / 2, 12 * u)


def draw_brand(
    img: np.ndarray, g: Geometry, listening: bool, title: str | None = None, title_alpha: float = 1.0
) -> None:
    """ "DrivePilot" top left, with the page's green "Listening…" while the driver is asking, and an optional title
    under it (what the clip shows, e.g. "Night · highway")."""
    u = g.u
    size = max(11, round(14 * u))
    sprite = pill_sprite("DrivePilot", rgba(TEXT), rgba("#0c1016", 0.72), size, 700)
    x, y = 12 * u, 12 * u
    paste(img, sprite, x, y)
    if listening:
        tag = pill_sprite("Listening…", rgba(BG), rgba(LISTEN), size, 600)
        paste(img, tag, x + sprite.shape[1] + 6 * u, y)
    if title and title_alpha > 0:
        label = pill_sprite(title, rgba(WHITE), rgba("#0c1016", 0.72), max(12, round(18 * u)), 600)
        paste(img, label, x, y + sprite.shape[0] + 8 * u, title_alpha)


def gamma_lut(gamma: float) -> np.ndarray:
    """A lookup table that brightens (gamma > 1) or darkens (< 1) an 8-bit image: 255 * (x / 255) ** (1 / gamma)."""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.round(255.0 * np.power(x, 1.0 / max(gamma, 1e-3))), 0, 255).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class CardLayout:
    """Caption cards placed for one frame: (sprite, x, y, opacity) each, and the top of the stack."""

    cards: list[tuple[np.ndarray, float, float, float]]
    top: float


def layout_captions(g: Geometry, shown: Sequence[Shown], images: dict[str, np.ndarray] | None = None) -> CardLayout:
    """The caption cards stacked from the bottom-left corner up, newest lowest."""
    u = g.u
    size = max(12, round(21 * u))
    max_w = round(min(g.out_w * 0.5, 640 * u))
    image_w = round(min(240 * u, max_w * 0.6))
    sprites = []
    for item in shown:
        key = item.caption.image_key
        image = images.get(key) if images is not None and key is not None else None
        sprites.append(card_sprite(item.caption.kind, item.text, max_w, size, image, image_w))
    margin = 16 * u
    tops = stack_from_bottom([s.shape[0] for s in sprites], g.out_h - margin, 8 * u)
    cards = [(sprite, margin, top, item.alpha) for sprite, top, item in zip(sprites, tops, shown, strict=True)]
    return CardLayout(cards, tops[0] if tops else float(g.out_h))


class Renderer:
    """Draws frames of one video: keeps the eased lane lines from frame to frame, as the page does. `gamma`: brighten
    the video picture for viewing (night footage; display only, perception saw the original); `title`: a label under
    the brand for the first `title_s` seconds after `title_from`."""

    def __init__(
        self,
        geometry: Geometry,
        brand: bool = True,
        gamma: float = 1.0,
        title: str | None = None,
        title_from: float = 0.0,
        title_s: float = 5.0,
    ) -> None:
        self.g = geometry
        self.brand = brand
        self.title, self.title_from, self.title_s = title, title_from, title_s
        self._lut = gamma_lut(gamma) if abs(gamma - 1.0) > 1e-3 else None
        self._lanes: dict[str, Any] | None = None

    def base(self, image: np.ndarray) -> np.ndarray:
        """The source frame cropped to the kept rows and scaled to the output size (brightened with `gamma`)."""
        g = self.g
        kept = image[: g.keep_h]
        if kept.shape[1] == g.out_w and kept.shape[0] == g.out_h:
            out = kept.copy()
        else:
            interpolation = cv2.INTER_AREA if g.out_w < g.src_w else cv2.INTER_CUBIC
            out = cv2.resize(kept, (g.out_w, g.out_h), interpolation=interpolation)
        if self._lut is not None:
            cv2.LUT(out, self._lut, dst=out)
        return out

    def title_alpha(self, t: float) -> float:
        """The title's opacity at video time t: shown for `title_s`, then fading out over half a second."""
        age = t - self.title_from
        if not self.title or age < 0:
            return 0.0
        return max(0.0, min(1.0, (self.title_s + 0.5 - age) / 0.5))

    def draw(
        self,
        image: np.ndarray,
        t: float,
        overlay: dict[str, Any] | None,
        hud: dict[str, Any] | None,
        shown: Sequence[Shown] = (),
        listening: bool = False,
        highlight: int | None = None,
        images: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """One output frame (BGR): the source frame with the page's overlay, HUD and caption cards."""
        g = self.g
        out = self.base(image)
        packet = (
            apply_highlight(overlay, highlight) if overlay is not None else {"items": [], "you": None, "lanes": None}
        )
        self._lanes = ease_lanes(self._lanes, packet.get("lanes"))
        if self._lanes is not None:
            draw_lanes(out, g, self._lanes, t)
        draw_items(out, g, packet.get("items", []), t)
        cards = layout_captions(g, shown, images)
        if packet.get("you"):
            draw_you(out, g, packet["you"], cards.top)
        draw_hud(out, g, hud)
        if self.brand:
            draw_brand(out, g, listening, self.title, self.title_alpha(t))
        for sprite, x, y, alpha in cards.cards:
            paste(out, sprite, x, y, alpha)
        return out
