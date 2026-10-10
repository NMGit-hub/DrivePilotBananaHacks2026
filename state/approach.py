"""Which intersection approach is a vehicle on (left / ahead / right of us), or is it the car right in front of us?

Pure geometry on the forward camera image, with zones in config (fractions of the frame, so they work at any
resolution; calibrate them once for the real mount):
- lead vehicle: the car directly in front of us in our lane: centered and large (close). While one is there we
  are not at the stop line ourselves.
- left / right: bottom-center x beyond `side_x`; must be seen side-on (box wider than tall by `min_side_aspect`)
  because cars waiting on the cross street face across our view. This keeps cars parked along our own street
  (seen from behind) out of the queue.
- ahead: the middle band, seen roughly head-on (aspect below `max_ahead_aspect`), not too small (far down the
  road) and not the lead vehicle.
Vehicles whose box bottom is outside the road band (`min_bottom_y`..`max_bottom_y`) are ignored.
"""

from __future__ import annotations

from core.config import IntersectionConfig
from perception.detector import VEHICLE_CLASSES, Track
from state.stop_queue import Approach


def classify_approach(track: Track, frame_w: int, frame_h: int, zones: IntersectionConfig) -> Approach | str | None:
    """Return Approach.LEFT/AHEAD/RIGHT, "lead" for the car directly in front of us, or None (not relevant)."""
    if track.cls not in VEHICLE_CLASSES:
        return None
    x_bc, y_bc = track.bottom_center
    xf, yf = x_bc / frame_w, y_bc / frame_h
    h_frac = track.height / frame_h
    aspect = track.width / max(track.height, 1.0)
    if not (zones.min_bottom_y <= yf <= zones.max_bottom_y) or h_frac < zones.min_height:
        return None
    if abs(xf - 0.5) <= zones.lead_center_band and h_frac >= zones.lead_min_height:
        return "lead"
    if xf < zones.side_x:
        return Approach.LEFT if aspect >= zones.min_side_aspect else None
    if xf > 1.0 - zones.side_x:
        return Approach.RIGHT if aspect >= zones.min_side_aspect else None
    return Approach.AHEAD if aspect <= zones.max_ahead_aspect else None


def is_clipped(track: Track, frame_w: int, frame_h: int, margin_px: float = 3.0) -> bool:
    """True if the box touches the left/right/bottom frame edge (the vehicle is partly out of view)."""
    x1, _, x2, y2 = track.box
    return x1 <= margin_px or x2 >= frame_w - margin_px or y2 >= frame_h - margin_px
