"""Which lane are we in? The vision model reads it off the latest frame, for navigation's lane advice.

PLAN.md "Combining route and camera": the route says which lanes are valid, the camera says which lane we're in,
and the difference is the instruction ("move one lane left"). Lane counting by the model is not reliable
(PROGRESS.md: it said 2 lanes instead of 3 on one of two runs), so a reading is only used when the model says
it is sure, and the navigation handler only uses it when its lane count matches the route's. Anything else
(a timeout, low confidence, nonsense numbers) means "I can't tell exactly which lane you're in".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from answers.scene import encode_jpeg
from answers.vision import ImagePart, VisionError, VisionModel, VisionRequest
from core.config import VisionConfig
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)

MAX_LANES = 8
SYSTEM_PROMPT = (
    "You look at one dashcam image from a car (the camera is behind the windshield, looking forward) and count "
    "lanes. Be conservative: when lane lines are hidden by traffic, worn, wet, or the view is unclear, say so "
    'with confidence "low" instead of guessing.'
)
QUESTION = (
    "Count the lanes for traffic going in our direction: the lanes between the left edge (solid line, median or "
    "barrier) and the right edge line, separated by dashed lane lines. Do not count shoulders, bike lanes, "
    "parking lanes or lanes for oncoming traffic. Then say which of those lanes our car is in, counted from the "
    'left (1 = leftmost). Reply as {"lanes": <int>, "our_lane": <int>, "confidence": "high" | "medium" | "low"}.'
)


def parse_lane_reading(data: dict[str, Any]) -> tuple[int, int] | None:
    """(lanes, our lane from the left) from the model's JSON, or None unless it is confident and consistent."""
    lanes, ours, confidence = data.get("lanes"), data.get("our_lane"), data.get("confidence")
    if confidence != "high":
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (lanes, ours)):
        return None
    assert isinstance(lanes, int) and isinstance(ours, int)
    if not 1 <= lanes <= MAX_LANES or not 1 <= ours <= lanes:
        return None
    return lanes, ours


class VisionLaneReader:
    """`nav.handler.LaneReader`: (lanes in our direction, our lane from the left) or None."""

    def __init__(
        self,
        model: VisionModel,
        frame: Callable[[], Frame | None],
        cfg: VisionConfig,
        on_frame: Callable[[Frame], None] | None = None,
    ) -> None:
        self.model = model
        self._frame = frame
        self.cfg = cfg
        self._on_frame = on_frame  # e.g. save the frame sent to the model in the run folder

    def __call__(self) -> tuple[int, int] | None:
        """Ask the model about the latest frame (blocks for up to the vision deadlines)."""
        frame = self._frame()
        if frame is None:
            return None
        if self._on_frame is not None:
            self._on_frame(frame)
        image = ImagePart(encode_jpeg(frame.image, self.cfg.latest_width, self.cfg.jpeg_quality), "Now")
        request = VisionRequest(
            system=SYSTEM_PROMPT, text=QUESTION, images=(image,), max_tokens=60, temperature=self.cfg.temperature
        )
        try:
            data = self.model.complete_json(request, self.cfg.first_token_timeout_s, self.cfg.total_timeout_s)
        except VisionError as exc:
            log.warning("lane reading failed", error=str(exc))
            return None
        reading = parse_lane_reading(data)
        log.info("lane reading", raw=data, used=reading is not None)
        return reading
