"""Vision-model answers about the scene (PLAN.md: signs, parking, lights, road awareness, lane counts).

`SceneHandler.answer` builds one request: the latest frame at full resolution plus 1-2 earlier frames at lower
resolution (motion and things just passed), a close-up of the traffic light when the question is about lights,
and a short text context: local day and time (time-limited signs), whether we are stopped, what the tracker
sees (cars by side, the 4-way-stop order, the light state, pedestrians) and, from M6, speed and the next
navigation step. The model is told to answer in 1-2 spoken sentences, left/right from the driver's seat, to
say what it can't see, and never to give a bare command.

The reply streams back sentence by sentence (speech starts on the first). A first-token deadline and a total
deadline turn a slow or failed call into a short spoken fallback; a last safety net rewrites a bare "go".
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime

import cv2
import numpy as np

from answers.sentences import sentences
from answers.vision import ImagePart, VisionError, VisionModel, VisionRequest, VisionTimeout
from core.config import AppConfig
from core.logging_setup import get_logger
from core.types import Frame
from state.stop_queue import Approach
from state.traffic_light import LightColor
from state.world import WorldSnapshot
from voice.router import Route

log = get_logger(__name__)

SYSTEM_PROMPT = """You are DrivePilot, a voice co-pilot riding in a car. You see the road through a dashcam on the \
windshield, looking forward. The driver asks you questions out loud while driving, and your answer is read aloud.

Rules:
- Answer in one or two short sentences of plain spoken English. No lists, markdown, or emojis.
- Left and right are from the driver's seat (image left is the driver's left).
- Never give a bare command such as "go", "merge now" or "you're clear". The driver decides: say what you see, \
and when it matters add "check before you go".
- If you can't see something clearly (too far, glare, blur, out of view), say so instead of guessing.
- Read signs exactly. Quote only words you can actually read; never fill in text you can't make out. For \
time-limited rules (parking, turn restrictions, school zones) use the day and time given.
- To count lanes, count the lanes in your direction between the solid edge lines, using the dashed lane lines \
and the cars in them; say which one you are in, counted from the left. If the edges aren't visible, say so.
- The context comes from an automatic tracker and can be wrong: if it disagrees with what you see, trust what \
you see. Don't guess why we are stopped (traffic, a light, a sign) unless you can see it.
- Add "check before you go" only when the question is about going, turning or crossing.
- Don't mention images, frames, cameras or the tracker; speak as if you are looking at the road.
- Use US units (mph, feet, miles)."""

FALLBACK_TIMEOUT = "Sorry, I couldn't get a clear answer in time."
FALLBACK_ERROR = "Sorry, I can't reach the vision model right now."
FALLBACK_EMPTY = "Sorry, I didn't get an answer to that."
SAFE_GO = "It looks clear from here, but check before you go."
LIGHT_WORDS = re.compile(r"\b(?:light|lights|signal|arrow|green|red|yellow|amber)\b", re.IGNORECASE)
BARE_GO = re.compile(
    r"^\s*(?:yes[,.!]?\s*)?(?:go|go now|go ahead|you can go|you may go|it'?s safe to go|merge now|you'?re clear|"
    r"you are clear|all clear|clear to go)[.!]*\s*$",
    re.IGNORECASE,
)
WHERE = {Approach.LEFT: "on the left", Approach.AHEAD: "across the intersection", Approach.RIGHT: "on the right"}


def spoken_time(now: datetime) -> str:
    """ "Saturday 5:40 PM"."""
    hour = now.hour % 12 or 12
    return f"{now:%A} {hour}:{now:%M} {'AM' if now.hour < 12 else 'PM'}"


def encode_jpeg(image: np.ndarray, max_width: int, quality: int) -> bytes:
    """Downscale to `max_width` (never up) and JPEG-encode."""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def light_crop(image: np.ndarray, box: tuple[float, float, float, float], height: int = 256) -> np.ndarray | None:
    """A close-up of a traffic light: the box padded by half its size, upscaled to `height` pixels."""
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.5, (y2 - y1) * 0.5
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(w_img, int(x2 + pad_x)), min(h_img, int(y2 + pad_y))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    scale = height / crop.shape[0]
    return cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)


def describe_world(snapshot: WorldSnapshot) -> list[str]:
    """What the tracker knows, as short context lines for the prompt."""
    lines = []
    if not snapshot.ego_motion_confident:
        lines.append("Unsure whether we are moving.")
    elif snapshot.ego_stopped:
        since = snapshot.ego_stopped_since
        lines.append(f"We are stopped (for {snapshot.t - since:.0f} s)." if since is not None else "We are stopped.")
    else:
        lines.append("We are moving.")
    if snapshot.stop_sign_visible:
        lines.append("A stop sign is in view.")
    elif snapshot.stop_sign_last_seen is not None and snapshot.t - snapshot.stop_sign_last_seen < 15:
        lines.append(f"A stop sign was seen {snapshot.t - snapshot.stop_sign_last_seen:.0f} s ago.")

    # Whether other cars are moving is only reliable while we stand still (our own motion is cancelled only
    # approximately); otherwise just say where they are and let the model judge from the frames.
    def motion(stopped: bool) -> str:
        if not (snapshot.ego_stopped and snapshot.ego_motion_confident):
            return ""
        return " (stopped)" if stopped else " (moving)"

    cars = []
    if snapshot.lead_vehicle is not None:
        lead = snapshot.lead_vehicle
        cars.append(f"{lead.label} directly ahead of us{motion(lead.stopped)}")
    for v in snapshot.vehicles:
        if v.approach is not None and len(cars) < 6:
            cars.append(f"{v.label} {WHERE[v.approach]}{motion(v.stopped)}")
    if cars:
        lines.append("Vehicles near the intersection: " + "; ".join(cars) + ".")
    queue_ = snapshot.queue
    if queue_.active and len(queue_.order) > 1:
        names = ["us" if e.is_ego else f"{e.label} {WHERE.get(e.approach, '')}".strip() for e in queue_.order]
        lines.append("4-way stop order (who goes first): " + ", then ".join(names) + ".")
    light = snapshot.light
    if light.color is not LightColor.UNKNOWN:
        lines.append(f"Traffic light facing us looks {light.color.value}{'' if light.confident else ' (unsure)'}.")
    if snapshot.people_ahead:
        lines.append(f"{snapshot.people_ahead} pedestrian{'s' if snapshot.people_ahead > 1 else ''} ahead.")
    return lines


class SceneHandler:
    """Answers scene questions with the vision model (the assistant's `SceneAnswerer`)."""

    def __init__(
        self,
        cfg: AppConfig,
        model: VisionModel,
        frames: Callable[[int, float], list[Frame]],
        extra_context: Callable[[], list[str]] | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self._frames = frames
        self._extra = extra_context
        self._now = now

    # ---------------------------------------------------------------------------------- prompt

    def build_request(self, question: str, snapshot: WorldSnapshot) -> VisionRequest:
        """The images and text for one question."""
        vcfg = self.cfg.vision
        frames = self._frames(vcfg.frames, vcfg.frame_spacing_s)
        if not frames and snapshot.frame is not None:
            frames = [snapshot.frame]
        images: list[ImagePart] = []
        latest = frames[-1] if frames else None
        for frame in frames[:-1]:
            age = (latest.t - frame.t) if latest is not None else 0.0
            images.append(ImagePart(encode_jpeg(frame.image, vcfg.older_width, vcfg.jpeg_quality), f"{age:.1f} s ago"))
        if latest is not None:
            images.append(ImagePart(encode_jpeg(latest.image, vcfg.latest_width, vcfg.jpeg_quality), "Now"))
            if LIGHT_WORDS.search(question) and snapshot.light.boxes:
                crop = light_crop(latest.image, snapshot.light.boxes[0])
                if crop is not None:
                    images.append(ImagePart(encode_jpeg(crop, 512, 92), "Close-up of the traffic light ahead"))
        context = [f"Local time: {spoken_time(self._now())}.", *describe_world(snapshot)]
        if self._extra is not None:
            context += self._extra()
        if latest is None:
            context.append("No camera image is available right now.")
        text = "Context:\n" + "\n".join(f"- {line}" for line in context) + f"\n\nDriver's question: {question}"
        return VisionRequest(system=SYSTEM_PROMPT, text=text, images=tuple(images), max_tokens=self._max_tokens())

    def _max_tokens(self) -> int:
        provider = self.cfg.vision_model
        if provider == "claude":
            return self.cfg.vision.claude.max_tokens
        if provider == "gemini":
            return self.cfg.vision.gemini.max_tokens
        return 300

    # ---------------------------------------------------------------------------------- answer

    def answer(self, route: Route, snapshot: WorldSnapshot) -> Iterator[str]:
        """Spoken sentences, as they arrive."""
        request = self.build_request(route.question or route.text, snapshot)
        return self._safe(sentences(self._pieces(request)))

    def _pieces(self, request: VisionRequest) -> Iterator[str]:
        """The model's stream, with a first-token and a total deadline (the provider runs on its own thread)."""
        vcfg = self.cfg.vision
        items: queue.Queue[str | BaseException | None] = queue.Queue()  # text piece, error, or None = done

        def run() -> None:
            try:
                for piece in self.model.stream(request):
                    items.put(piece)
                items.put(None)
            except Exception as exc:  # noqa: BLE001 - any provider failure is handed to the consumer below
                items.put(exc)

        threading.Thread(target=run, name=f"vision-{self.model.name}", daemon=True).start()
        start = time.perf_counter()
        first = True
        while True:
            elapsed = time.perf_counter() - start
            limit = (vcfg.first_token_timeout_s if first else vcfg.total_timeout_s) - elapsed
            try:
                item = items.get(timeout=max(0.0, limit))
            except queue.Empty:
                what = "no reply" if first else "reply too slow"
                raise VisionTimeout(f"{self.model.name}: {what} after {elapsed:.1f} s") from None
            if item is None:
                return
            if isinstance(item, BaseException):
                if isinstance(item, VisionError):
                    raise item
                raise VisionError(f"{self.model.name}: {type(item).__name__}: {item}") from item
            if first:
                log.info("vision first token", provider=self.model.name, ms=round(elapsed * 1000))
                first = False
            yield item

    def _safe(self, stream: Iterator[str]) -> Iterator[str]:
        """Fallbacks for failures, and the bare-"go" safety net."""
        said = False
        try:
            for sentence in stream:
                if BARE_GO.match(sentence):
                    log.warning("rewrote a bare go from the model", sentence=sentence)
                    sentence = SAFE_GO
                said = True
                yield sentence
        except VisionError as exc:
            log.warning("vision answer failed", error=str(exc), after_sentences=said)
            if not said:
                yield FALLBACK_TIMEOUT if isinstance(exc, VisionTimeout) else FALLBACK_ERROR
        else:
            if not said:
                log.warning("vision answer was empty")
                yield FALLBACK_EMPTY
