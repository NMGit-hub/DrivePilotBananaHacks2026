"""Motion graphics for the showcase reel (scripts/showcase_reel.py --motion): an animated intro (title over a moving
road, "how it works" with the phone -> laptop -> you diagram building itself, "what it does" with frames of the
scenes), stinger transitions between scenes and an end card. Every function draws one frame for a time `t` (seconds
into its part), so the reel can render any length; colors and fonts are the V2 page's (ui/render.py).

Drawing is done with Pillow at 2x and shrunk (smooth edges), as ui/render.py does for its sprites.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from ui.render import ALERT_FG, BG, GREEN, MUTED, PANEL, TEXT, YOU, font, rgba

SS = 2  # supersampling
RGBA = tuple[int, int, int, int]


# ------------------------------------------------------------------------------------------------ easing


def clamp01(x: float) -> float:
    """x limited to 0..1."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def ease_out_cubic(x: float) -> float:
    """Fast start, soft landing."""
    x = clamp01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    """Soft start and landing."""
    x = clamp01(x)
    return 4 * x**3 if x < 0.5 else 1.0 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x: float, overshoot: float = 1.6) -> float:
    """Lands a little past the target and settles (a pop)."""
    x = clamp01(x)
    c3 = overshoot + 1.0
    return 1.0 + c3 * (x - 1.0) ** 3 + overshoot * (x - 1.0) ** 2


def progress(t: float, start: float, duration: float) -> float:
    """How far (0..1) an animation starting at `start` and lasting `duration` is at time t."""
    return clamp01((t - start) / duration) if duration > 0 else float(t >= start)


def alpha(color: str, a: float) -> RGBA:
    """A page color at opacity a (0..1)."""
    return rgba(color, clamp01(a))


# ------------------------------------------------------------------------------------------------ canvas


class Canvas:
    """A 2x RGBA drawing surface for one frame; `frame()` gives the BGR image at the output size."""

    def __init__(self, w: int, h: int, background: str = BG) -> None:
        self.w, self.h = w, h
        self.img = Image.new("RGBA", (w * SS, h * SS), rgba(background))
        self.u = h / 720 * SS  # one "page pixel" at this size, supersampled

    def layer(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        """A transparent layer to draw on (composited with `put`)."""
        lay = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        return lay, ImageDraw.Draw(lay)

    def put(self, lay: Image.Image, opacity: float = 1.0) -> None:
        """Composite a layer at an opacity."""
        if opacity <= 0.0:
            return
        if opacity < 1.0:
            a = lay.getchannel("A").point(lambda v: round(v * opacity))
            lay.putalpha(a)
        self.img.alpha_composite(lay)

    def paste_image(
        self, bgr: np.ndarray, x: float, y: float, w: float, opacity: float = 1.0, radius: float = 0
    ) -> None:
        """A BGR picture `w` wide (2x units) at (x, y), rounded corners, at an opacity."""
        pic = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1])).convert("RGBA")
        h = w * pic.height / pic.width
        pic = pic.resize((max(1, round(w)), max(1, round(h))), Image.Resampling.BILINEAR)
        mask = Image.new("L", pic.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, pic.width - 1, pic.height - 1), radius=radius, fill=255)
        if opacity < 1.0:
            mask = mask.point(lambda v: round(v * clamp01(opacity)))
        pic.putalpha(mask)
        self.img.alpha_composite(pic, (round(x), round(y)))

    def frame(self) -> np.ndarray:
        """The frame at the output size (BGR uint8)."""
        small = self.img.resize((self.w, self.h), Image.Resampling.LANCZOS).convert("RGB")
        return np.ascontiguousarray(np.asarray(small)[..., ::-1])


# ------------------------------------------------------------------------------------------------ pieces


def draw_road(c: Canvas, t: float, opacity: float = 1.0, horizon: float = 0.62) -> None:
    """A dark road in perspective: edge lines to a vanishing point, lane dashes streaming toward us, a soft glow."""
    lay, d = c.layer()
    W, H = c.img.size
    vx, vy = W / 2, H * horizon
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for k in range(10, 0, -1):  # radial glow at the vanishing point
        r = W * 0.05 * k
        gd.ellipse((vx - r, vy - r * 0.55, vx + r, vy + r * 0.55), fill=alpha(GREEN, 0.012))
    c.put(glow, opacity)
    for side in (-1, 1):
        for lane in (0.55, 1.6):
            d.line((vx, vy, vx + side * W * lane, H), fill=alpha(MUTED, 0.35), width=round(2 * c.u))
    # Dashes: positions along the road at depth z (1 = near), moving toward us.
    for lane_x in (-1.08, 1.08):
        for k in range(14):
            z0 = ((k / 14.0) + t * 0.35) % 1.0
            z1 = min(1.0, z0 + 0.025)
            p0, p1 = z0**2.2, z1**2.2  # perspective: near dashes spread out
            x0, y0 = vx + lane_x * W * 0.5 * p0, vy + (H - vy) * p0
            x1, y1 = vx + lane_x * W * 0.5 * p1, vy + (H - vy) * p1
            d.line((x0, y0, x1, y1), fill=alpha(TEXT, 0.25 + 0.5 * p0), width=max(1, round(c.u * (1 + 5 * p0))))
    c.put(lay, opacity)


def draw_text_reveal(
    c: Canvas,
    text: str,
    cx: float,
    cy: float,
    size: float,
    weight: int,
    color: str,
    t: float,
    start: float,
    stagger: float = 0.045,
    rise: float = 26.0,
) -> None:
    """Text centered at (cx, cy) whose letters rise and fade in one after another (from `start`)."""
    f = font(round(size), weight)
    total = f.getlength(text)
    x = cx - total / 2
    lay, d = c.layer()
    for i, ch in enumerate(text):
        p = progress(t, start + i * stagger, 0.55)
        if p > 0:
            dy = (1.0 - ease_out_back(p)) * rise * c.u
            d.text((x, cy + dy), ch, font=f, fill=alpha(color, ease_out_cubic(p)), anchor="lm")
        x += f.getlength(ch)
    c.put(lay)


def draw_text(
    c: Canvas, text: str, x: float, y: float, size: float, weight: int, color: str, a: float = 1.0, anchor: str = "mm"
) -> None:
    """Plain text at an opacity."""
    if a <= 0:
        return
    lay, d = c.layer()
    d.text((x, y), text, font=font(round(size), weight), fill=alpha(color, a), anchor=anchor)
    c.put(lay)


def draw_waveform(c: Canvas, cx: float, cy: float, t: float, a: float, bars: int = 13, color: str = GREEN) -> None:
    """A voice waveform: bars bouncing like speech."""
    if a <= 0:
        return
    lay, d = c.layer()
    bw, gap = 5 * c.u, 4 * c.u
    x0 = cx - (bars * bw + (bars - 1) * gap) / 2
    for i in range(bars):
        env = math.sin(math.pi * (i + 0.5) / bars)
        h = (6 + 26 * env * (0.45 + 0.55 * abs(math.sin(t * 7.0 + i * 0.9) * math.sin(t * 3.1 + i * 0.4)))) * c.u
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((x, cy - h / 2, x + bw, cy + h / 2), radius=bw / 2, fill=alpha(color, a))
    c.put(lay)


def draw_pill(
    c: Canvas, text: str, cx: float, cy: float, size: float, fg: str, bg: str, a: float, scale: float = 1.0
) -> None:
    """A rounded label, centered, scaled (pop-in) and faded."""
    if a <= 0 or scale <= 0:
        return
    f = font(round(size * scale), 650)
    w, h = f.getlength(text) + size * 1.1 * scale, size * 1.7 * scale
    lay, d = c.layer()
    d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=h / 2, fill=alpha(bg, a))
    d.text((cx, cy), text, font=f, fill=alpha(fg, a), anchor="mm")
    c.put(lay)


def draw_arrow(c: Canvas, x1: float, x2: float, y: float, p: float, t: float, label: str, a: float = 1.0) -> None:
    """An arrow that draws itself left to right (`p` 0..1), with data dots flowing along it once drawn."""
    if p <= 0:
        return
    lay, d = c.layer()
    u = c.u
    xe = x1 + (x2 - x1) * ease_in_out_cubic(p)
    d.line((x1, y, xe - 10 * u, y), fill=alpha(MUTED, 0.9 * a), width=round(4 * u))
    if p > 0.85:
        d.polygon([(xe + 4 * u, y), (xe - 16 * u, y - 11 * u), (xe - 16 * u, y + 11 * u)], fill=alpha(MUTED, a))
        for k in range(3):
            q = (t * 0.9 + k / 3.0) % 1.0
            x = x1 + (x2 - x1 - 24 * u) * q
            r = 5 * u
            d.ellipse((x - r, y - r, x + r, y + r), fill=alpha(GREEN, a * math.sin(math.pi * q)))
    d.text(((x1 + x2) / 2, y - 24 * u), label, font=font(round(16 * u), 650), fill=alpha(MUTED, a * p), anchor="mm")
    c.put(lay)


def draw_card(
    c: Canvas, box: tuple[float, float, float, float], edge: str, title: str, subtitle: str, a: float, dy: float = 0.0
) -> None:
    """A rounded card with a colored edge and a title, offset by dy (slide-in)."""
    if a <= 0:
        return
    x1, y1, x2, y2 = box
    y1, y2 = y1 + dy, y2 + dy
    u = c.u
    lay, d = c.layer()
    d.rounded_rectangle(
        (x1, y1, x2, y2), radius=18 * u, fill=alpha(PANEL, 0.96 * a), outline=alpha(edge, a), width=round(2.5 * u)
    )
    d.text(((x1 + x2) / 2, y1 + 42 * u), title, font=font(round(27 * u), 750), fill=alpha(edge, a), anchor="mm")
    if subtitle:
        d.text(((x1 + x2) / 2, y1 + 76 * u), subtitle, font=font(round(16 * u), 400), fill=alpha(MUTED, a), anchor="mm")
    c.put(lay)


def draw_icon(c: Canvas, kind: str, cx: float, cy: float, a: float, color: str = TEXT) -> None:
    """Small line icons: camera, pin (GPS), mic, chip, speaker, screen."""
    if a <= 0:
        return
    u = c.u
    lay, d = c.layer()
    col, w = alpha(color, a), round(2.5 * u)
    s = 13 * u
    if kind == "camera":
        d.rounded_rectangle((cx - s, cy - s * 0.7, cx + s, cy + s * 0.75), radius=4 * u, outline=col, width=w)
        d.ellipse((cx - s * 0.42, cy - s * 0.35, cx + s * 0.42, cy + s * 0.45), outline=col, width=w)
        d.rectangle((cx - s * 0.45, cy - s * 0.95, cx + s * 0.1, cy - s * 0.7), fill=col)
    elif kind == "pin":
        d.ellipse((cx - s * 0.6, cy - s, cx + s * 0.6, cy + s * 0.2), outline=col, width=w)
        d.polygon([(cx - s * 0.45, cy - s * 0.05), (cx + s * 0.45, cy - s * 0.05), (cx, cy + s)], fill=col)
        d.ellipse((cx - s * 0.2, cy - s * 0.6, cx + s * 0.2, cy - s * 0.2), fill=col)
    elif kind == "mic":
        d.rounded_rectangle(
            (cx - s * 0.35, cy - s, cx + s * 0.35, cy + s * 0.25), radius=s * 0.35, outline=col, width=w
        )
        d.arc((cx - s * 0.7, cy - s * 0.6, cx + s * 0.7, cy + s * 0.55), 0, 180, fill=col, width=w)
        d.line((cx, cy + s * 0.55, cx, cy + s), fill=col, width=w)
    elif kind == "speaker":
        d.polygon(
            [
                (cx - s, cy - s * 0.35),
                (cx - s * 0.45, cy - s * 0.35),
                (cx + s * 0.1, cy - s),
                (cx + s * 0.1, cy + s),
                (cx - s * 0.45, cy + s * 0.35),
                (cx - s, cy + s * 0.35),
            ],
            outline=col,
            width=w,
        )
        for r in (0.55, 0.9):
            d.arc((cx - s * r + s * 0.3, cy - s * r, cx + s * r + s * 0.3, cy + s * r), -50, 50, fill=col, width=w)
    elif kind == "screen":
        d.rounded_rectangle((cx - s, cy - s * 0.7, cx + s, cy + s * 0.55), radius=3 * u, outline=col, width=w)
        d.line((cx - s * 0.5, cy + s, cx + s * 0.5, cy + s), fill=col, width=w)
    c.put(lay)


# ------------------------------------------------------------------------------------------------ parts


@dataclass(frozen=True, slots=True)
class Shot:
    """A frame of a scene for the "what it does" part, with a two-line caption."""

    image: np.ndarray
    title: str
    subtitle: str = ""


def title_part(w: int, h: int, t: float, duration: float) -> np.ndarray:
    """DrivePilot over a moving road: letters rising in, the tagline, a voice waveform; fades out at the end."""
    c = Canvas(w, h)
    W, H = c.img.size
    u = c.u
    out = 1.0 - progress(t, duration - 0.35, 0.35)
    draw_road(c, t, opacity=0.9 * out)
    lay = Image.new("RGBA", c.img.size, (0, 0, 0, 0))  # (the text fades out with everything)
    sub = Canvas(w, h, background=BG)
    sub.img = lay
    draw_text_reveal(sub, "DrivePilot", W / 2, H * 0.36, 84 * u, 850, TEXT, t, 0.15)
    tag = ease_out_cubic(progress(t, 0.9, 0.6))
    draw_text(
        sub,
        "A voice co-pilot that watches the road",
        W / 2,
        H * 0.36 + (66 + 18 * (1 - tag)) * u,
        32 * u,
        600,
        YOU,
        tag,
    )
    wave = ease_out_cubic(progress(t, 1.5, 0.5))
    draw_waveform(sub, W / 2, H * 0.36 + 140 * u, t, 0.9 * wave)
    draw_text(sub, '"Hey DrivePilot, ..."', W / 2, H * 0.36 + 190 * u, 20 * u, 400, MUTED, wave)
    c.put(lay, out)
    return c.frame()


def how_part(w: int, h: int, t: float, duration: float) -> np.ndarray:
    """Phone -> laptop -> you: cards sliding up one after another, arrows drawing themselves with data flowing, the
    models popping into the laptop card, a spoken answer appearing in the last card."""
    c = Canvas(w, h)
    W, H = c.img.size
    u = c.u
    out = 1.0 - progress(t, duration - 0.35, 0.35)
    lay = Image.new("RGBA", c.img.size, (0, 0, 0, 0))
    s = Canvas(w, h)
    s.img = lay
    head = ease_out_cubic(progress(t, 0.0, 0.5))
    draw_text(s, "How it works", W / 2, (66 - 20 * (1 - head)) * u, 44 * u, 800, TEXT, head)
    top, bottom = 140 * u, 600 * u
    col_w = (W - 2 * 56 * u - 2 * 92 * u) / 3
    xs = [56 * u + i * (col_w + 92 * u) for i in range(3)]
    cards = (
        (YOU, "Phone on the dash", "here: the car's dashcam", 0.25),
        (GREEN, "Laptop (GPU)", "everything runs here, live", 1.1),
        (ALERT_FG, "You", "hear it, see it", 2.6),
    )
    for x, (edge, title, subtitle, start) in zip(xs, cards, strict=True):
        p = ease_out_cubic(progress(t, start, 0.6))
        draw_card(s, (x, top, x + col_w, bottom), edge, title, subtitle, p, dy=(1 - p) * 60 * u)
    # Phone: what it sends.
    for i, (icon, label) in enumerate(
        (("camera", "Camera video"), ("pin", "GPS: speed, route"), ("mic", "Your voice"))
    ):
        p = ease_out_cubic(progress(t, 0.6 + 0.18 * i, 0.4))
        y = top + (140 + 62 * i) * u
        draw_icon(s, icon, xs[0] + 44 * u, y, p)
        draw_text(s, label, xs[0] + 76 * u, y, 21 * u, 400, TEXT, p, anchor="lm")
    mid = (top + bottom) / 2
    draw_arrow(s, xs[0] + col_w + 10 * u, xs[1] - 10 * u, mid, progress(t, 0.85, 0.5), t, "USB")
    # Laptop: the models, popping in.
    chips = ("YOLO26 + ByteTrack", "YOLOPv2 lanes", "US sign detector", "Claude vision", "Mapbox routes")
    for i, chip in enumerate(chips):
        p = progress(t, 1.5 + 0.22 * i, 0.4)
        draw_pill(
            s,
            chip,
            xs[1] + col_w / 2,
            top + (130 + 58 * i) * u,
            19 * u,
            TEXT,
            "#1f2a24",
            ease_out_cubic(p),
            scale=0.6 + 0.4 * ease_out_back(p),
        )
    draw_arrow(s, xs[1] + col_w + 10 * u, xs[2] - 10 * u, mid, progress(t, 2.5, 0.5), t, "voice")
    # You: a spoken alert and the screen.
    p = ease_out_cubic(progress(t, 3.0, 0.5))
    draw_icon(s, "speaker", xs[2] + 44 * u, top + 140 * u, p)
    draw_text(s, "Spoken answers", xs[2] + 76 * u, top + 140 * u, 21 * u, 400, TEXT, p, anchor="lm")
    q = progress(t, 3.4, 0.45)
    draw_pill(
        s,
        "Light's green.",
        xs[2] + col_w / 2,
        top + 212 * u,
        19 * u,
        "#231703",
        ALERT_FG,
        ease_out_cubic(q),
        scale=0.6 + 0.4 * ease_out_back(q),
    )
    draw_waveform(
        s, xs[2] + col_w / 2, top + 268 * u, t, 0.85 * ease_out_cubic(progress(t, 3.6, 0.4)), bars=9, color=ALERT_FG
    )
    r = ease_out_cubic(progress(t, 3.9, 0.5))
    draw_icon(s, "screen", xs[2] + 44 * u, top + 336 * u, r)
    draw_text(s, "Overlays on screen", xs[2] + 76 * u, top + 336 * u, 21 * u, 400, TEXT, r, anchor="lm")
    note = ease_out_cubic(progress(t, 4.4, 0.6))
    draw_text(
        s,
        "Everything that follows is our own dashcam footage, played through the real app.",
        W / 2,
        H - 58 * u,
        19 * u,
        400,
        MUTED,
        note,
    )
    c.put(lay, out)
    return c.frame()


def shots_part(w: int, h: int, t: float, duration: float, shots: Sequence[Shot]) -> np.ndarray:
    """What it does: up to 3 frames flying in from below one after another, slowly zooming, with captions."""
    c = Canvas(w, h)
    W, H = c.img.size
    u = c.u
    out = 1.0 - progress(t, duration - 0.3, 0.3)
    lay = Image.new("RGBA", c.img.size, (0, 0, 0, 0))
    s = Canvas(w, h)
    s.img = lay
    head = ease_out_cubic(progress(t, 0.0, 0.45))
    draw_text(s, "What it does", W / 2, (66 - 20 * (1 - head)) * u, 44 * u, 800, TEXT, head)
    n = max(1, min(3, len(shots)))
    gap = 30 * u
    tile_w = (W - 2 * 48 * u - (n - 1) * gap) / n
    for i, shot in enumerate(shots[:3]):
        p = ease_out_cubic(progress(t, 0.25 + 0.25 * i, 0.6))
        if p <= 0:
            continue
        zoom = 1.0 + 0.05 * progress(t, 0.25, duration)
        tw = tile_w * zoom
        th = tw * shot.image.shape[0] / shot.image.shape[1]
        x = 48 * u + i * (tile_w + gap) - (tw - tile_w) / 2
        y = (H - th) / 2 - 10 * u + (1 - p) * 80 * u
        s.paste_image(shot.image, x, y, tw, opacity=p, radius=12 * u)
        draw_text(s, shot.title, x + tw / 2, y + th + 34 * u, 22 * u, 700, TEXT, p)
        draw_text(s, shot.subtitle, x + tw / 2, y + th + 64 * u, 19 * u, 400, MUTED, p)
    c.put(lay, out)
    return c.frame()


def stinger(prev: np.ndarray, nxt: np.ndarray, t: float, duration: float, number: str, label: str) -> np.ndarray:
    """A transition: a diagonal accent band sweeps across, revealing the next scene behind it, carrying its number and
    name; `prev` and `nxt` are BGR frames of the output size."""
    h, w = prev.shape[:2]
    p = ease_in_out_cubic(progress(t, 0.0, duration))
    slant = 0.35 * h
    band = 0.26 * w
    edge = -slant + (w + 2 * slant + band) * p  # the band's leading edge (x at the top): off screen at both ends
    xs = np.arange(w)[None, :]
    ys = np.arange(h)[:, None]
    lead = edge - slant * (ys / h)  # x of the leading edge on each row (leaning right)
    out = np.where((xs < lead - band)[..., None], nxt, prev).astype(np.uint8)  # behind the band: the next scene
    band_mask = (xs >= lead - band) & (xs < lead)
    g = np.clip((lead - xs) / band, 0, 1)  # 0 at the leading edge, 1 at the trailing edge
    c0, c1 = np.array(rgba(GREEN)[:3][::-1], np.float32), np.array(rgba(YOU)[:3][::-1], np.float32)
    colors = (c0[None, None, :] * (1 - g[..., None]) + c1[None, None, :] * g[..., None]).astype(np.uint8)
    out = np.where(band_mask[..., None], colors, out)
    # The label rides in the band's middle.
    canvas = Image.fromarray(np.ascontiguousarray(out[..., ::-1])).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    u = h / 720
    cx = float(edge - band / 2 - slant * 0.5)
    if -band < cx < w + band:
        a = math.sin(math.pi * p)
        d.text((cx, h / 2 - 24 * u), number, font=font(round(54 * u), 850), fill=alpha("#062418", a), anchor="mm")
        d.text((cx, h / 2 + 30 * u), label, font=font(round(22 * u), 700), fill=alpha("#062418", a), anchor="mm")
    return np.ascontiguousarray(np.asarray(canvas.convert("RGB"))[..., ::-1])


def end_part(w: int, h: int, t: float, duration: float, line: str) -> np.ndarray:
    """The end card: the road, DrivePilot scaling in, one line under it."""
    c = Canvas(w, h)
    W, H = c.img.size
    u = c.u
    draw_road(c, t + 3.0, opacity=0.8)
    p = ease_out_back(progress(t, 0.0, 0.6))
    draw_text(
        c, "DrivePilot", W / 2, H * 0.42, 70 * u * (0.7 + 0.3 * p), 850, TEXT, ease_out_cubic(progress(t, 0, 0.4))
    )
    q = ease_out_cubic(progress(t, 0.35, 0.5))
    draw_text(c, line, W / 2, H * 0.42 + 70 * u, 24 * u, 600, YOU, q)
    return c.frame()
