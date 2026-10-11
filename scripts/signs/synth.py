"""Synthetic US signs: MUTCD renders with any speed value, warped and degraded like a dashcam sees them.

Two uses (scripts/signs/prepare_data.py):
- value-classifier crops for every posted value 5-85 mph (real data covers only a few values well);
- detector composites: signs pasted into real BDD100K frames, for kinds the real data has few of.

Templates are public-domain MUTCD renders from Wikimedia Commons (data/signs/templates/, fetched by
scripts/signs/fetch_sources.py); numbers are drawn with Overpass (SIL OFL, a Highway Gothic look-alike) and a few
Windows fonts for variety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = REPO_ROOT / "data" / "signs" / "templates"
FONTS = REPO_ROOT / "data" / "signs" / "fonts"
WIN_FONTS = Path("C:/Windows/Fonts")

# kind -> template files (MUTCD codes). Speed limits are rendered from R2-1 / S5-1 with the number replaced.
KIND_TEMPLATES: dict[str, list[str]] = {
    "stop": ["MUTCD_R1-1.png"],
    "yield": ["MUTCD_R1-2.png"],
    "do_not_enter": ["MUTCD_R5-1.png"],
    "wrong_way": ["MUTCD_R5-1a.png"],
    "no_turn_on_red": ["MUTCD_R10-11.png", "MUTCD_R10-11a.png", "MUTCD_R10-11b.png"],
    "one_way": ["MUTCD_R6-1R.png", "MUTCD_R6-1L.png", "MUTCD_R6-2R.png"],
    "no_left_turn": ["MUTCD_R3-2.png"],
    "no_right_turn": ["MUTCD_R3-1.png"],
    "no_u_turn": ["MUTCD_R3-4.png"],
    "pedestrian_crossing": ["MUTCD_W11-2.png"],
    "school_zone": ["MUTCD_S1-1.png"],
}
YELLOW_GREEN = (198, 255, 0)  # fluorescent yellow-green used on school and pedestrian signs


def _fonts(size: int) -> list[ImageFont.FreeTypeFont]:
    """Number fonts: Overpass Bold/SemiBold (Highway Gothic-like) weighted most, plus condensed Windows fonts."""
    out: list[ImageFont.FreeTypeFont] = []
    overpass = FONTS / "Overpass.ttf"
    if overpass.exists():
        for weight in (b"Bold", b"SemiBold", b"Bold", b"ExtraBold"):
            f = ImageFont.truetype(str(overpass), size)
            f.set_variation_by_name(weight)
            out.append(f)
    for name in ("ARIALNB.TTF", "bahnschrift.ttf", "GOTHICB.TTF", "arialbd.ttf"):
        if (WIN_FONTS / name).exists():
            out.append(ImageFont.truetype(str(WIN_FONTS / name), size))
    if not out:
        raise FileNotFoundError("no number fonts found (run scripts/signs/fetch_sources.py)")
    return out


def _number_rows(gray: np.ndarray, top_frac: float) -> tuple[int, int]:
    """Rows (y0, y1) of the tallest run of dark text rows below top_frac of the height: the big number."""
    h, w = gray.shape
    inner = gray[:, int(0.12 * w) : int(0.88 * w)]
    dark = (inner < 100).sum(axis=1) > 2
    runs: list[tuple[int, int]] = []
    start = None
    for y in range(int(top_frac * h), int(0.97 * h)):
        if dark[y] and start is None:
            start = y
        elif not dark[y] and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, int(0.97 * h)))
    if not runs:
        raise ValueError("no text rows found in template")
    return max(runs, key=lambda r: r[1] - r[0])


@dataclass
class SpeedTemplate:
    """A speed-limit template with its number erased, and where the number goes."""

    image: Image.Image  # RGBA, number area painted white
    rows: tuple[int, int]
    cols: tuple[int, int]


def load_speed_template(name: str, top_frac: float) -> SpeedTemplate:
    """Load R2-1 / S5-1 and erase its printed number."""
    img = Image.open(TEMPLATES / name).convert("RGBA")
    gray = np.asarray(img.convert("L"))
    alpha = np.asarray(img)[:, :, 3]
    gray = np.where(alpha > 0, gray, 255)
    y0, y1 = _number_rows(gray, top_frac)
    band = gray[y0:y1, int(0.08 * img.width) : int(0.92 * img.width)]
    xs = np.where((band < 100).any(axis=0))[0] + int(0.08 * img.width)
    x0, x1 = int(xs.min()), int(xs.max())
    draw = ImageDraw.Draw(img)
    draw.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), fill=(255, 255, 255, 255))
    return SpeedTemplate(img, (y0, y1), (x0, x1))


class SignRenderer:
    """Renders clean (unwarped) RGBA signs of every kind; speed limits with any value."""

    def __init__(self) -> None:
        self.r2 = load_speed_template("MUTCD_R2-1.png", 0.45)
        self.s5 = load_speed_template("MUTCD_S5-1.png", 0.45)
        self.templates = {k: [Image.open(TEMPLATES / f).convert("RGBA") for f in v] for k, v in KIND_TEMPLATES.items()}
        # Look-alikes that carry numbers but are not speed limits (route shields, advisory/exit plates, NIGHT and
        # MINIMUM speeds, other text signs): never labeled, so both networks learn to leave them alone.
        self.distractors = [Image.open(f).convert("RGBA") for f in sorted((TEMPLATES / "distractors").glob("*.png"))]

    def speed(self, value: int, school: bool, rng: np.random.Generator) -> Image.Image:
        """A SPEED LIMIT (or SCHOOL SPEED LIMIT) sign showing `value`."""
        t = self.s5 if school else self.r2
        img = t.image.copy()
        y0, y1 = t.rows
        x0, x1 = t.cols
        height = int((y1 - y0) * rng.uniform(0.92, 1.08))
        fonts = _fonts(int(height * 1.38))
        font = fonts[int(rng.integers(len(fonts)))]
        text = str(value)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_w = (x1 - x0) * (1.25 if len(text) == 2 else 0.8)
        scale = min(1.0, max_w / max(1, tw))
        if scale < 1.0 or abs(th - height) > 0.1 * height:
            font = font.font_variant(size=max(8, int(font.size * min(scale, height / max(1, th)))))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), text, fill=(0, 0, 0, 255), font=font)
        return img

    def kind(self, kind: str, rng: np.random.Generator) -> Image.Image:
        """A sign of a non-speed kind (variants picked at random)."""
        choices = self.templates[kind]
        img = choices[int(rng.integers(len(choices)))].copy()
        if kind in ("pedestrian_crossing",) and rng.random() < 0.5:
            img = recolor_yellow(img, YELLOW_GREEN)
        return img

    def distractor(self, rng: np.random.Generator) -> Image.Image:
        """A random number-bearing look-alike (route shield, advisory plate, ...), sometimes under a direction plate."""
        if not self.distractors:
            raise FileNotFoundError(f"no distractor templates in {TEMPLATES / 'distractors'}")
        img = self.distractors[int(rng.integers(len(self.distractors)))]
        east = next((d for d in self.distractors if d.width > 1.8 * d.height and d.height < 0.5 * img.height), None)
        if east is not None and img.height >= img.width * 0.8 and rng.random() < 0.25:
            plate = east.resize((img.width, max(1, round(east.height * img.width / east.width))))
            stacked = Image.new("RGBA", (img.width, img.height + plate.height), (0, 0, 0, 0))
            stacked.paste(plate, (0, 0), plate)
            stacked.paste(img, (0, plate.height), img)
            return stacked
        return img.copy()

    def any(self, kind: str, value: int | None, rng: np.random.Generator) -> Image.Image:
        """Render any taxonomy kind."""
        if kind == "speed_limit":
            return self.speed(value or 35, False, rng)
        if kind == "school_speed_limit":
            return self.speed(value or 20, True, rng)
        return self.kind(kind, rng)


def recolor_yellow(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Swap the yellow field of a warning sign for another color (yellow-green school/pedestrian signs)."""
    a = np.asarray(img).copy()
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    yellow = (r > 180) & (g > 150) & (b < 90)
    a[yellow, :3] = color
    return Image.fromarray(a)


def warp_sign(img: Image.Image, height_px: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Weather, perspective and scale a clean sign. Returns (BGR image, alpha 0-1 float) at about height_px tall."""
    a = np.asarray(img).astype(np.float32)
    rgb, alpha = a[:, :, :3], a[:, :, 3] / 255.0
    # Aging / dirt / retroreflective sheen: tint, contrast, a soft brightness gradient.
    tint = rng.uniform(0.85, 1.05, size=3) * rng.uniform(0.8, 1.05)
    rgb = (rgb * tint).astype(np.float32)
    gy = np.linspace(rng.uniform(0.8, 1.15), rng.uniform(0.8, 1.15), rgb.shape[0])[:, None, None]
    rgb = np.clip(rgb * gy, 0, 255)
    h, w = alpha.shape
    # Perspective: yaw (sign turned away), small pitch and roll.
    yaw = math.radians(rng.uniform(-50, 50))
    roll = math.radians(rng.normal(0, 4))
    near = 1.0 + 0.18 * abs(math.sin(yaw))
    far = 1.0 - 0.18 * abs(math.sin(yaw))
    wscale = max(0.35, math.cos(yaw))
    left_h, right_h = (near, far) if yaw > 0 else (far, near)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    cx, cy = w / 2, h / 2
    pts = np.array(
        [
            [cx - wscale * w / 2, cy - left_h * h / 2],
            [cx + wscale * w / 2, cy - right_h * h / 2],
            [cx + wscale * w / 2, cy + right_h * h / 2],
            [cx - wscale * w / 2, cy + left_h * h / 2],
        ],
        dtype=np.float32,
    )
    rot = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    pts = ((pts - center) @ rot.T + center).astype(np.float32)
    pts += rng.normal(0, 0.01 * h, pts.shape).astype(np.float32)
    # Scale to the target height in one warp (area interpolation happens via a pre-shrink below).
    s = height_px / (h * max(left_h, right_h))
    pts = pts * s
    pts -= pts.min(axis=0)
    out_w, out_h = math.ceil(pts[:, 0].max()) + 2, math.ceil(pts[:, 1].max()) + 2
    m = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    # Shrinking by > 2x in one perspective warp aliases; pre-shrink with INTER_AREA first.
    pre = max(1.0, 0.5 / s)
    if pre > 1.0:
        size = (max(4, int(w / pre)), max(4, int(h / pre)))
        rgb = np.asarray(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), dtype=np.float32)
        alpha = np.asarray(cv2.resize(alpha, size, interpolation=cv2.INTER_AREA), dtype=np.float32)
        m = m @ np.diag([pre, pre, 1.0]).astype(np.float64)
    bgr = cv2.warpPerspective(rgb[:, :, ::-1], m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
    alp = cv2.warpPerspective(alpha, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return bgr, np.clip(alp, 0, 1)


def composite(
    bg: np.ndarray, sign: np.ndarray, alpha: np.ndarray, x: int, y: int, rng: np.random.Generator, pole: bool
) -> tuple[float, float, float, float] | None:
    """Paste a warped sign into a BGR frame at top-left (x, y), lit to match; returns its box or None if clipped."""
    h, w = alpha.shape
    H, W = bg.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H or w < 4 or h < 4:
        return None
    region = bg[y : y + h, x : x + w].astype(np.float32)
    # Match the scene's brightness: dark frames get darker signs (unless headlight-lit), bright ones stay bright.
    scene = float(bg.mean()) / 128.0
    gain = np.clip(scene * rng.uniform(0.7, 1.25), 0.25, 1.3)
    if scene < 0.5 and rng.random() < 0.4:
        gain = rng.uniform(0.9, 1.4)  # retroreflective sign in headlights
    s = np.clip(sign.astype(np.float32) * gain, 0, 255)
    if pole:
        pw = max(1, round(w * rng.uniform(0.05, 0.09)))
        px = x + w // 2 - pw // 2
        py1 = min(H, y + h + int(h * rng.uniform(1.0, 3.0)))
        shade = np.float32(rng.uniform(60, 150) * min(1.0, scene + 0.2))
        bg[y + h - 1 : py1, px : px + pw] = shade
    a = alpha[:, :, None]
    bg[y : y + h, x : x + w] = np.clip(region * (1 - a) + s * a, 0, 255).astype(np.uint8)
    ys, xs = np.where(alpha > 0.5)
    if len(xs) == 0:
        return None
    return float(x + xs.min()), float(y + ys.min()), float(x + xs.max() + 1), float(y + ys.max() + 1)


def degrade(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Camera and codec effects on a BGR image: blur, motion blur, noise, JPEG."""
    out = img
    if rng.random() < 0.7:
        sigma = rng.uniform(0.2, 1.1) * strength
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    if rng.random() < 0.2 * strength:
        k = int(rng.integers(3, 7))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(1, 6) * strength, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(rng.uniform(25, 90))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else None
    return np.asarray(decoded) if decoded is not None else out
