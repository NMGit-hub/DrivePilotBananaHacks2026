"""GPS and speed from the Mercedes-Benz dashcam's burned-in overlay, so its clips replay with GPS like a GPX.

The car's dashcam (MBUX, files `dm_YYYYMMDD_HHMMSS.mp4`: 2-minute clips, 1920x1280, 25 fps, every frame a keyframe,
no audio and no data track) writes one line of text into the bottom 45 px of every frame:

    2026-09-14   07:54:20   27.4 mph                 ...                 47°36'35"N -122°-7'-16"W

in a fixed monospaced font on a 15-px grid: the left block starts at x = 29, the right one at x = 1548. The time is
local (Pacific) and updates once a second, the speed (0.1 mph, from the car) about twice a second, the position
once a second but only in whole arc-seconds (about 31 m north-south and 21 m east-west here).

Reading it (`OverlayReader`): each 15 x 22 px cell is compared with glyph templates (normalized correlation); a cell
with almost no ink is a space. The templates (`GLYPHS_PATH`, built by `scripts/dashcam_gpx.py --build-glyphs`) are
learned from the footage itself: every clip's date is known from its file name, which labels the digits, and the
fixed punctuation labels the rest. `parse_left` / `parse_right` turn the strings into values, refusing anything
malformed (a frame is skipped, never guessed).

Turning the readings into a track (`smooth_track`): the speed is used as read; the position is quantized, so each
axis is smoothed with a constant-velocity Kalman filter and smoother (RTS) whose measurement noise is the
quantization (1/sqrt(12) of an arc-second), which recovers it to a few metres along a straight road.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from core.logging_setup import get_logger

log = get_logger(__name__)

FRAME_W, FRAME_H = 1920, 1280
ROW0, ROW1 = 1250, 1272  # the text rows (the bar is 1235-1279)
PITCH = 15
LEFT_X0, LEFT_CELLS = 29, 33  # date, time, speed + "mph"
RIGHT_X0, RIGHT_CELLS = 1548, 25  # latitude, longitude
OVERLAY_TOP = 1235  # rows from here down are the text bar (mask or crop it for perception)
INK = 45  # gray level counted as ink (the bar is ~25; dim glyphs such as "-" peak at ~60)
MIN_INK = 3  # cells with fewer ink pixels are spaces
MIN_SCORE = 0.6  # a cell whose best template correlates less is unreadable
FAINT = 65  # ...unless it is this faint: then it is noise in an empty cell
GLYPHS_PATH = Path(__file__).with_name("dashcam_glyphs.npz")
TZ = ZoneInfo("America/Los_Angeles")
NAME_RE = re.compile(r"dm_(\d{8})_(\d{6})", re.IGNORECASE)

# (the date's dim dashes may read as "?": the date is known from the file name anyway)
_LEFT_RE = re.compile(r"^(\d{4})[-? ](\d{2})[-? ](\d{2}) +(\d{2}):(\d{2}):(\d{2}) +(\d{1,3}\.\d) mph$")
# The minus signs are dim (sometimes read as blanks); the hemisphere letter gives the sign anyway.
_RIGHT_RE = re.compile(
    r"^-?(\d{1,2})°[- ]?(\d{1,2})'[- ]?(\d{1,2})\"([NS]) +-? ?(\d{1,3})°[- ]?(\d{1,2})'[- ]?(\d{1,2})\"([EW])$"
)


MX, MY = 2, 1  # cells come with this margin: glyphs sit a pixel or two differently in the two blocks
CELL_H, CELL_W = ROW1 - ROW0, PITCH


def cells(gray: np.ndarray, x0: int, count: int) -> list[np.ndarray]:
    """The text cells of one block: `count` patches of (22 + 2*MY) x (15 + 2*MX) px (the cell and a margin) from a
    full-height grayscale frame."""
    band = gray[ROW0 - MY : ROW1 + MY]
    left, need = x0 - MX, x0 + count * PITCH + MX
    if need > band.shape[1]:  # a long longitude runs off the frame's right edge: pad with the bar's black
        band = np.pad(band, ((0, 0), (0, need - band.shape[1])), constant_values=int(np.median(band)))
    return [band[:, left + k * PITCH : left + (k + 1) * PITCH + 2 * MX] for k in range(count)]


def _core(patch: np.ndarray) -> np.ndarray:
    """The cell itself, without its margin."""
    return patch[MY : MY + CELL_H, MX : MX + CELL_W]


def _norm(patch: np.ndarray) -> np.ndarray | None:
    p = patch.astype(np.float32)
    p = p - p.mean()
    n = float(np.linalg.norm(p))
    return p / n if n > 1e-6 else None


@dataclass(frozen=True, slots=True)
class Glyphs:
    """Templates: one normalized patch per character."""

    chars: str
    patches: np.ndarray  # (n, 22, 15) float32, each zero-mean, unit-norm

    @classmethod
    def load(cls, path: Path = GLYPHS_PATH) -> Glyphs:
        """The saved templates."""
        data = np.load(path)
        return cls(str(data["chars"]), data["patches"].astype(np.float32))

    def save(self, path: Path = GLYPHS_PATH) -> None:
        """Save the templates (small: a few kB)."""
        np.savez_compressed(path, chars=np.array(self.chars), patches=self.patches)

    @classmethod
    def learn(cls, labeled: list[tuple[str, np.ndarray]]) -> Glyphs:
        """Templates from labeled cells: the mean normalized patch per character (spaces ignored)."""
        groups: dict[str, list[np.ndarray]] = {}
        for char, patch in labeled:
            if char == " ":
                continue
            p = _norm(_core(patch))
            if p is not None:
                groups.setdefault(char, []).append(p)
        chars = "".join(sorted(groups))
        patches = []
        for char in chars:
            mean = np.mean(groups[char], axis=0)
            patches.append(mean / max(float(np.linalg.norm(mean)), 1e-6))
        return cls(chars, np.stack(patches).astype(np.float32))

    def read(self, patch: np.ndarray) -> tuple[str, float]:
        """(character, correlation) for one cell; (" ", 1.0) for a blank one, ("?", score) if unsure."""
        if int((_core(patch) > INK).sum()) < MIN_INK:
            return " ", 1.0
        windows = [
            w
            for dy in range(2 * MY + 1)
            for dx in range(2 * MX + 1)
            if (w := _norm(patch[dy : dy + CELL_H, dx : dx + CELL_W])) is not None
        ]
        if not windows:
            return " ", 1.0
        scores = np.tensordot(self.patches, np.stack(windows), axes=([1, 2], [1, 2])).max(axis=1)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < MIN_SCORE and int(_core(patch).max()) < FAINT:
            return " ", 1.0  # compression noise in an empty cell (a dim "-" still matches its template at ~0.9)
        return (self.chars[best] if score >= MIN_SCORE else "?"), score


@dataclass(frozen=True, slots=True)
class Reading:
    """What one frame's overlay says."""

    local_time: datetime  # naive local (Pacific) wall time, whole seconds
    speed_mph: float
    lat: float  # degrees (whole arc-seconds)
    lon: float


def _dms(deg: str, minutes: str, seconds: str) -> float:
    """Degrees from the overlay's signed fields ("-122", "-7", "-16" -> -122.1211)."""
    d, m, s = int(deg), int(minutes), int(seconds)
    sign = -1.0 if d < 0 or m < 0 or s < 0 else 1.0
    return sign * (abs(d) + abs(m) / 60.0 + abs(s) / 3600.0)


def parse_left(text: str) -> tuple[datetime, float] | None:
    """(local time, speed in mph) from the left block, None if malformed."""
    m = _LEFT_RE.match(text.strip())
    if m is None:
        return None
    y, mo, d, hh, mm, ss, speed = m.groups()
    try:
        when = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
    except ValueError:
        return None
    value = float(speed)
    return (when, value) if 0.0 <= value <= 160.0 else None


def parse_right(text: str) -> tuple[float, float] | None:
    """(lat, lon) in degrees from the right block, None if malformed or impossible."""
    m = _RIGHT_RE.match(re.sub(r" +", " ", text.strip()))
    if m is None:
        return None
    lat, lon = _dms(*m.group(1, 2, 3)), _dms(*m.group(5, 6, 7))
    if int(m.group(2)) >= 60 or int(m.group(3)) >= 60 or int(m.group(6)) >= 60 or int(m.group(7)) >= 60:
        return None
    lat = -lat if m.group(4) == "S" else lat
    lon = -lon if m.group(8) == "W" else lon
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


class OverlayReader:
    """Reads the overlay of full frames (BGR or grayscale, 1920 x 1280)."""

    def __init__(self, glyphs: Glyphs | None = None) -> None:
        self.glyphs = glyphs or Glyphs.load()

    def text(self, gray: np.ndarray, x0: int, count: int) -> tuple[str, float]:
        """The block's characters and its worst cell score."""
        chars, worst = [], 1.0
        for patch in cells(gray, x0, count):
            char, score = self.glyphs.read(patch)
            chars.append(char)
            worst = min(worst, score)
        return "".join(chars), worst

    def read(self, image: np.ndarray) -> Reading | None:
        """The frame's reading, None when any field is unreadable or malformed."""
        if image.shape[0] != FRAME_H or image.shape[1] != FRAME_W:
            return None
        gray = image if image.ndim == 2 else _gray(image)
        left, _ = self.text(gray, LEFT_X0, LEFT_CELLS)
        right, _ = self.text(gray, RIGHT_X0, RIGHT_CELLS)
        if "?" in left[10:] or "?" in right:
            return None
        parsed_left = parse_left(left)
        parsed_right = parse_right(right)
        if parsed_left is None or parsed_right is None:
            return None
        (when, speed), (lat, lon) = parsed_left, parsed_right
        return Reading(when, speed, lat, lon)


def _gray(image: np.ndarray) -> np.ndarray:
    import cv2

    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def clip_start(name: str) -> datetime | None:
    """The clip's first frame as an aware datetime, from `dm_YYYYMMDD_HHMMSS` (local Pacific time)."""
    m = NAME_RE.search(name)
    if m is None:
        return None
    try:
        local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return local.replace(tzinfo=TZ)


# ------------------------------------------------------------------------------------------------ track


@dataclass(frozen=True, slots=True)
class TrackSample:
    """One reading on the clip's clock."""

    t: float  # seconds of video
    speed_mps: float
    lat: float
    lon: float


def _rts_1d(t: np.ndarray, z: np.ndarray, meas_sd: float, accel_sd: float) -> np.ndarray:
    """Positions smoothed by a constant-velocity Kalman filter + Rauch-Tung-Striebel smoother (one axis)."""
    n = len(t)
    xs = np.zeros((n, 2))
    ps = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0])
    p = np.diag([meas_sd**2, 25.0])
    r = meas_sd**2
    preds_x, preds_p, fs = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))
    for i in range(n):
        dt = t[i] - t[i - 1] if i else 0.0
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = accel_sd**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        x = f @ x
        p = f @ p @ f.T + q
        preds_x[i], preds_p[i], fs[i] = x, p, f
        k = p[:, 0] / (p[0, 0] + r)
        x = x + k * (z[i] - x[0])
        p = p - np.outer(k, p[0, :])
        xs[i], ps[i] = x, p
    for i in range(n - 2, -1, -1):
        c = ps[i] @ fs[i + 1].T @ np.linalg.inv(preds_p[i + 1])
        xs[i] = xs[i] + c @ (xs[i + 1] - preds_x[i + 1])
        ps[i] = ps[i] + c @ (ps[i + 1] - preds_p[i + 1]) @ c.T
    return xs[:, 0]


def smooth_track(samples: list[TrackSample], accel_sd: float = 1.5) -> list[TrackSample]:
    """The samples with their quantized positions smoothed (metres-scale; speed and times unchanged)."""
    if len(samples) < 3:
        return samples
    t = np.array([s.t for s in samples])
    lat = np.array([s.lat for s in samples])
    lon = np.array([s.lon for s in samples])
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    north = (lat - lat0) * m_per_deg_lat
    east = (lon - lon0) * m_per_deg_lon
    q_north = m_per_deg_lat / 3600.0 / math.sqrt(12.0)  # quantization noise of a whole arc-second
    q_east = m_per_deg_lon / 3600.0 / math.sqrt(12.0)
    north_s = _rts_1d(t, north, q_north, accel_sd)
    east_s = _rts_1d(t, east, q_east, accel_sd)
    return [
        TrackSample(s.t, s.speed_mps, lat0 + n / m_per_deg_lat, lon0 + e / m_per_deg_lon)
        for s, n, e in zip(samples, north_s, east_s, strict=True)
    ]


def video_start_utc(name: str) -> datetime | None:
    """The clip's first frame in UTC (for the GPX `video_start` waypoint)."""
    start = clip_start(name)
    return None if start is None else start.astimezone(ZoneInfo("UTC"))


def sample_time(start: datetime, t: float) -> datetime:
    """The UTC time of video time `t` in a clip that started at `start`."""
    return (start + timedelta(seconds=t)).astimezone(ZoneInfo("UTC"))
