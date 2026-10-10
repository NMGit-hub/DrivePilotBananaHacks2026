"""Name a vehicle's color from its image crop ("the white car on the left").

Dashcam footage has strong color casts (blue at dawn and in shade, orange under street lights), and dark paint
has unstable hue, so:
1. the frame is normalized first: white balance (gray world: scale the channels so the frame's average is
   neutral; clamped so a frame full of trees or brick can't flip colors) and exposure (dashcams under-expose to
   keep the sky, so a white car at dawn is as dark as a silver one at noon: scale so the frame's median
   brightness is typical; clamped);
2. only the body panel is sampled: below the windows (glass reflects the sky) and above the wheels;
3. each pixel is judged by its chroma (max - min channel), absolute and relative to its brightness, not HSV
   saturation, which explodes for dark pixels; if enough pixels are clearly colored, their hues vote for a
   name, otherwise the car is white / silver / gray / black by brightness.
Checked by eye on 41 vehicle crops from 5 dev clips (dawn, overcast, sun).
Deliberately coarse: lighting and reflections make fine distinctions unreliable. Crops too small to judge
return "" and answers fall back to "the car on the left".
"""

from __future__ import annotations

import cv2
import numpy as np

HUE_NAMES = (  # OpenCV hue is 0-179: (upper limit, name)
    (8, "red"),
    (22, "orange"),
    (34, "yellow"),
    (85, "green"),
    (130, "blue"),
    (160, "purple"),
    (180, "red"),
)
CHROMA_MIN = 40.0  # max - min channel (0-255, after normalization) for a pixel to count as colored...
REL_CHROMA_MIN = 0.3  # ...and as a fraction of its brightness (pale champagne/silver paint is not "orange")
COLORED_SHARE = 0.4  # this share of the panel must be colored to name a hue
MAX_WB_CHANGE = 0.35  # white-balance gains stay within 1 +- this
TARGET_MEDIAN = 100.0  # exposure: the frame's median brightness is scaled to this...
EXPOSURE_RANGE = (0.8, 2.0)  # ...by a factor in this range
BRIGHTNESS_NAMES = ((65.0, "black"), (115.0, "gray"), (185.0, "silver"))  # below the limit; above: white


def frame_gains(image: np.ndarray) -> np.ndarray:
    """Per-channel (B, G, R) gains for a frame: gray-world white balance times exposure, both clamped."""
    small = image[::8, ::8].reshape(-1, 3).astype(np.float32)
    means = small.mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1.0), 1.0 - MAX_WB_CHANGE, 1.0 + MAX_WB_CHANGE)
    median = float(np.median((small * gains).max(axis=1)))
    exposure = float(np.clip(TARGET_MEDIAN / max(median, 1.0), *EXPOSURE_RANGE))
    return gains * exposure


def color_name(image: np.ndarray, box: tuple[float, float, float, float], gains: np.ndarray | None = None) -> str:
    """Coarse color name of the vehicle in `box` (BGR image), or "" if the crop is too small.

    `gains`: `frame_gains(image)` (computed if not given; pass them when naming several vehicles in one frame).
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    # Body panel: 20-80 % of the width, 40-75 % of the height (below the windows, above the wheels).
    cx1, cx2 = int(x1 + 0.2 * (x2 - x1)), int(x2 - 0.2 * (x2 - x1))
    cy1, cy2 = int(y1 + 0.40 * (y2 - y1)), int(y1 + 0.75 * (y2 - y1))
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w_img, cx2), min(h_img, cy2)
    if cx2 - cx1 < 4 or cy2 - cy1 < 3:
        return ""
    if gains is None:
        gains = frame_gains(image)
    pixels = np.clip(image[cy1:cy2, cx1:cx2].reshape(-1, 3).astype(np.float32) * gains, 0, 255)
    brightest, darkest = pixels.max(axis=1), pixels.min(axis=1)
    chroma = brightest - darkest
    colored = (chroma >= CHROMA_MIN) & (chroma >= REL_CHROMA_MIN * brightest)
    if colored.mean() >= COLORED_SHARE:
        hsv = cv2.cvtColor(pixels[colored].astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        votes: dict[str, int] = {}
        limits = np.array([limit for limit, _ in HUE_NAMES])
        for index, count in zip(
            *np.unique(np.searchsorted(limits, hsv[:, 0], side="right"), return_counts=True), strict=True
        ):
            name = HUE_NAMES[min(int(index), len(HUE_NAMES) - 1)][1]
            votes[name] = votes.get(name, 0) + int(count)
        return max(votes, key=lambda k: votes[k])
    value = float(np.median(brightest))
    return next((name for limit, name in BRIGHTNESS_NAMES if value < limit), "white")
