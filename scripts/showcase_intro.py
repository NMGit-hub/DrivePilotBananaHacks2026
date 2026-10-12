"""Intro slides for the showcase reel (scripts/showcase_reel.py --slide): what DrivePilot is, how the phone (here the
car's dashcam) and the laptop work together, and what it does, drawn in the V2 page's colors.

    .venv/Scripts/python scripts/showcase_intro.py runs/showcase/v2/intro --size 1280x822 ^
        --shots runs/showcase/final/1_traffic_light_b_t034.5.png runs/showcase/v2/A_lane_guidance_t1_t092.0.png ...

Writes intro_1.png (title), intro_2.png (how it works: a diagram), intro_3.png (what it does: up to 3 frames of the
showcase, each with a caption).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ui.render import ALERT_FG, BG, GREEN, MUTED, PANEL, TEXT, YOU, font, rgba  # noqa: E402

RGB = tuple[int, int, int]


def color(hex_color: str) -> RGB:
    """ "#rrggbb" -> (r, g, b)."""
    r, g, b, _ = rgba(hex_color)
    return r, g, b


def canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """A slide on the page's background."""
    img = Image.new("RGB", (w, h), color(BG))
    return img, ImageDraw.Draw(img)


def title_slide(w: int, h: int) -> Image.Image:
    """DrivePilot, what it is, in two lines."""
    img, d = canvas(w, h)
    u = h / 720
    cx, cy = w / 2, h / 2
    d.text((cx, cy - 95 * u), "DrivePilot", font=font(round(76 * u), 850), fill=color(TEXT), anchor="mm")
    d.text(
        (cx, cy - 15 * u),
        "A voice co-pilot that watches the road",
        font=font(round(34 * u), 600),
        fill=color(YOU),
        anchor="mm",
    )
    lines = (
        "Ask out loud. It looks at the road and answers in a second or two,",
        "and speaks up on its own when something matters.",
        "All it needs: a phone on the dash and a laptop.",
    )
    for i, line in enumerate(lines):
        d.text((cx, cy + (55 + 36 * i) * u), line, font=font(round(24 * u), 400), fill=color(MUTED), anchor="mm")
    d.text((cx, h - 40 * u), "Banana Hacks 2026", font=font(round(18 * u), 600), fill=color(MUTED), anchor="mm")
    return img


def _box(d: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], u: float, edge: str = "#2a3240") -> None:
    d.rounded_rectangle(xy, radius=16 * u, fill=color(PANEL), outline=color(edge), width=max(2, round(2 * u)))


def _arrow(d: ImageDraw.ImageDraw, x1: float, x2: float, y: float, u: float, label: str) -> None:
    d.line((x1, y, x2 - 14 * u, y), fill=color(MUTED), width=max(3, round(4 * u)))
    d.polygon([(x2, y), (x2 - 18 * u, y - 11 * u), (x2 - 18 * u, y + 11 * u)], fill=color(MUTED))
    d.text(((x1 + x2) / 2, y - 22 * u), label, font=font(round(16 * u), 600), fill=color(MUTED), anchor="mm")


def how_slide(w: int, h: int) -> Image.Image:
    """Phone on the dash -> laptop (the models) -> the driver's ears, with what each part does."""
    img, d = canvas(w, h)
    u = h / 720
    d.text((w / 2, 70 * u), "How it works", font=font(round(44 * u), 800), fill=color(TEXT), anchor="mm")
    top, bottom = 150 * u, 560 * u
    col_w = (w - 2 * 60 * u - 2 * 90 * u) / 3
    xs = [60 * u + i * (col_w + 90 * u) for i in range(3)]
    titles = ("Phone on the dash", "Laptop (GPU)", "You")
    subtitles = ("in these clips: the car's dashcam", "everything runs here, live", "")
    bullets = (
        ("Camera video", "GPS (speed, route)", "Microphone:", '"Hey DrivePilot, ..."'),
        (
            "Cars, lights, people:",
            "YOLO26 + ByteTrack",
            "Lanes: YOLOPv2",
            "US signs: own detector",
            "Signs, parking: Claude",
            "Routes: Mapbox",
        ),
        ("Spoken answers", "Warnings it says on", "its own", "The screen: overlays"),
    )
    edges = (YOU, GREEN, ALERT_FG)
    for x, t, s, items, edge in zip(xs, titles, subtitles, bullets, edges, strict=True):
        _box(d, (x, top, x + col_w, bottom), u, edge)
        d.text((x + col_w / 2, top + 40 * u), t, font=font(round(26 * u), 700), fill=color(edge), anchor="mm")
        if s:
            d.text((x + col_w / 2, top + 74 * u), s, font=font(round(16 * u), 400), fill=color(MUTED), anchor="mm")
        for i, item in enumerate(items):
            d.text(
                (x + 26 * u, top + (120 + 40 * i) * u),
                item,
                font=font(round(21 * u), 400),
                fill=color(TEXT),
                anchor="lm",
            )
    mid = (top + bottom) / 2
    _arrow(d, xs[0] + col_w + 8 * u, xs[1] - 8 * u, mid, u, "USB")
    _arrow(d, xs[1] + col_w + 8 * u, xs[2] - 8 * u, mid, u, "voice")
    note = "Every clip that follows is our own dashcam footage, played through the real app: nothing is edited in."
    d.text((w / 2, h - 70 * u), note, font=font(round(19 * u), 400), fill=color(MUTED), anchor="mm")
    return img


def shots_slide(w: int, h: int, shots: list[tuple[Path, str]]) -> Image.Image:
    """Up to 3 frames of the showcase, each with a caption: what it does."""
    img, d = canvas(w, h)
    u = h / 720
    d.text((w / 2, 70 * u), "What it does", font=font(round(44 * u), 800), fill=color(TEXT), anchor="mm")
    n = max(1, len(shots))
    gap = 30 * u
    tile_w = (w - 2 * 50 * u - (n - 1) * gap) / n
    for i, (path, caption) in enumerate(shots[:3]):
        shot = Image.open(path).convert("RGB")
        tile_h = tile_w * shot.height / shot.width
        x, y = 50 * u + i * (tile_w + gap), max(150 * u, (h - tile_h - 90 * u) / 2 + 30 * u)  # (centered)
        img.paste(shot.resize((round(tile_w), round(tile_h)), Image.Resampling.LANCZOS), (round(x), round(y)))
        d.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=10 * u, outline=color("#2a3240"), width=2)
        for k, line in enumerate(caption.split("|")):
            d.text(
                (x + tile_w / 2, y + tile_h + (34 + 30 * k) * u),
                line,
                font=font(round(21 * u), 600 if k == 0 else 400),
                fill=color(TEXT if k == 0 else MUTED),
                anchor="mm",
            )
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", default="1280x822", help="WxH (the showcase videos' size).")
    parser.add_argument("--shots", nargs="*", default=[], metavar="PNG=CAPTION", help="Frames for slide 3.")
    args = parser.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for item in args.shots:
        path, _, caption = item.partition("=")
        shots.append((Path(path), caption))
    slides = [title_slide(w, h), how_slide(w, h)] + ([shots_slide(w, h, shots)] if shots else [])
    for i, slide in enumerate(slides, 1):
        out = args.out_dir / f"intro_{i}.png"
        slide.save(out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
