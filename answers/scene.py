"""Vision-model answers about the scene (PLAN.md: signs, parking, lights, road awareness, lane counts).

`SceneHandler.answer` builds one request: the latest frame at full resolution plus 1-2 earlier frames (smaller,
or full-size and further back for "what did that sign say?"), close-ups of up to three traffic lights for light
questions, and a short text context: local day and time (time-limited signs), whether we are stopped, what the
tracker sees (cars by side, the right-of-way verdict, the light, pedestrians) and, from M6, speed and the next
navigation step. The model is told to answer in 1-2 spoken sentences, left/right from the driver's seat, to say
what it can't see, and never to vouch for traffic.

The reply streams back sentence by sentence (speech starts on the first). Safety nets on every sentence:
- anything that gives permission ("go ahead", "you're good to go", "you can merge now", a bare "Go.") is
  replaced by "I can't judge moving traffic, so check both ways yourself before you go.";
- for questions about going, turning, merging or waiting, any vouching for traffic ("it's clear", "nothing's
  coming") is replaced the same way, and the answer always ends with that sentence.
Deadlines and "stop" cancel the provider request itself (`guarded_stream`); spoken fallbacks cover failures.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from datetime import datetime

import cv2
import numpy as np

from answers.local import SCHOOL_BUS_RULE, ahead_of_us, describe
from answers.sentences import sentences
from answers.vision import (
    ImagePart,
    VisionCancelled,
    VisionError,
    VisionModel,
    VisionRequest,
    VisionTimeout,
    guarded_stream,
)
from core.config import AppConfig
from core.logging_setup import get_logger
from core.types import Frame
from state.stop_queue import Approach
from state.traffic_light import LightColor
from state.world import WorldSnapshot
from voice.router import Route, RouteKind

log = get_logger(__name__)

SYSTEM_PROMPT = """You are DrivePilot, a voice co-pilot riding in a car. You see the road through a dashcam on the \
windshield, looking forward. The driver asks you questions out loud while driving, and your answer is read aloud.

Rules:
- Answer in one or two short sentences of plain spoken English. No lists, markdown, or emojis.
- Left and right are from the driver's seat (image left is the driver's left).
- You cannot judge moving traffic or gaps. Never say or imply that it is safe or clear to go, turn, merge, pull \
out, cross or change lanes, and never say "go", "go ahead", "you're good to go", "you can go" or "you're clear". \
Describe what you see, and when the question is about going, turning or merging, end with "check both ways \
yourself".
- If you can't see something clearly (too far, glare, blur, out of view), say so instead of guessing.
- Read signs exactly. Quote only words you can actually read; never fill in text you can't make out, and never \
name a sign (like "no turn on red") unless you can read those words on it. Electronic message boards are usually \
too small to read: say so. For time-limited rules (parking, turn restrictions, school zones) use the day and \
time given.
- A school bus stopped with its red lights flashing or its STOP arm out is loading or unloading children: \
traffic must stop and wait until it moves on. Never suggest passing or going around it.
- To count lanes, count the lanes in your direction between the solid edge lines, using the dashed lane lines \
and the cars in them; say which one you are in, counted from the left. If the edges aren't visible, say so.
- The context comes from an automatic tracker and can be wrong: if it disagrees with what you see, trust what \
you see. It reads only the main color of the light facing us; it cannot read arrows. Don't guess why we are \
stopped (traffic, a light, a sign) unless you can see it.
- Don't mention images, frames, cameras or the tracker; speak as if you are looking at the road.
- Use US units (mph, feet, miles)."""

NO_VOUCH = "I can't judge moving traffic, so check both ways yourself before you go."
FALLBACK_TIMEOUT = "Sorry, I couldn't get a clear answer in time."
FALLBACK_ERROR = "Sorry, I can't reach the vision model right now."
FALLBACK_EMPTY = "Sorry, I didn't get an answer to that."
LOST_REST = "Sorry, I lost the rest of that."
LIGHT_WORDS = re.compile(r"\b(?:light|lights|signal|signals|arrow|arrows|green|red|yellow|amber)\b", re.IGNORECASE)
SIGN_WORDS = re.compile(r"\b(?:sign|signs|said|say|says|read|speed limit|limit|plate)\b", re.IGNORECASE)
# Questions about *our* moving off (the router's gap checks are covered by the route kind already).
GO_QUESTION = re.compile(
    r"\b(?:(?:can|could|should|may|do|shall) (?:i|we) (?:still )?(?:go|turn|merge|pull out|pull in|cross|proceed|pass|"
    r"change lanes?|get over|move over|make it)|(?:is it|it is|am i|are we|is that) (?:safe|clear|ok|okay|good)|"
    r"(?:safe|clear|ok|okay) to|(?:anyone|anybody|anything|someone|somebody|cars?|traffic) coming|room to|gap)\b",
    re.IGNORECASE,
)
# Questions about passing / going around something; a school bus in the answer then gets the fixed rule.
PASS_QUESTION = re.compile(
    r"\b(?:pass|passing|overtake|overtaking|go(?:ing)? around|get around|drive around|go past|get past|get by)\b",
    re.IGNORECASE,
)
SCHOOL_BUS = re.compile(r"\bschool ?bus(?:es)?\b", re.IGNORECASE)
# Permission in any sentence, whatever the question.
PERMISSION = re.compile(
    r"^\W*(?:yes\W+|ok(?:ay)?\W+|sure\W+)?go\b(?! to\b)"  # "Go.", "Yes, go." (not "Go to the next exit")
    r"|^\W*(?:yes\W+|ok(?:ay)?\W+)?(?:merge|turn|pull out|cross|pass|change lanes)(?: \w+)? now\b"  # "Merge now."
    r"|\bgo ahead\b|\bgo for it\b|\bgood to go\b|\bgo now\b|\bgo(?:,| -|—)? it'?s clear\b"
    r"|\byou(?:'re| are) (?:good|clear|all clear|safe|fine|ok|okay)\b(?! to (?:park|stop))"
    r"|\ball clear\b"
    r"|\byou (?:can|may|could|should) (?:go|merge|turn|pull out|proceed|cross|pass|change lanes|make it|get over)\b"
    r"|\b(?:safe|clear|ok|okay|fine) (?:for you )?to "
    r"(?:go|merge|turn|pull out|proceed|cross|pass|change lanes|get over)\b",
    re.IGNORECASE,
)
# Vouching for traffic: not allowed in answers to go/turn/merge/wait questions ("isn't clear" is fine).
VOUCHING = re.compile(
    r"(?<!not )(?<!n't )\b(?:clear|safe|empty)\b"
    r"|\b(?:nothing(?:'s| is) coming|no (?:one|cars?|traffic|vehicles?)(?:'s| is)? coming|nobody(?:'s| is) coming|"
    r"no traffic)\b",
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


def _side(box: tuple[float, float, float, float], width: int) -> str:
    center = (box[0] + box[2]) / 2 / max(width, 1)
    return "on the left" if center < 0.4 else "on the right" if center > 0.6 else "in the middle"


def describe_world(snapshot: WorldSnapshot) -> list[str]:
    """What the tracker knows, as short context lines for the prompt (verdicts, not raw data)."""
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
    queue = snapshot.queue
    if queue.active:
        lines.append("We are at a stop sign (the tracker can't tell whether it is an all-way stop).")
        if queue.deadlock:
            lines.append("Right of way: several cars stopped together; nobody has a clear right of way.")
        elif queue.your_turn:
            lines.append("Right of way: by arrival order it is our turn; moving traffic is not judged.")
        else:
            ahead = ahead_of_us(queue)
            names = ", then ".join(describe(e) for e in ahead)
            lines.append(f"Right of way: not our turn yet; {len(ahead)} car(s) go before us: {names}.")
        for entry in queue.dropped:
            lines.append(f"Lost track of {describe(entry)}; it may still be there.")
    light = snapshot.light
    if light.color is not LightColor.UNKNOWN:
        lines.append(f"Traffic light facing us looks {light.color.value}{'' if light.confident else ' (unsure)'}.")
    if snapshot.people_ahead:
        lines.append(f"{snapshot.people_ahead} pedestrian{'s' if snapshot.people_ahead > 1 else ''} ahead.")
    return lines


def is_go_question(route: Route) -> bool:
    """Questions where vouching for traffic is dangerous: gap checks, "should I wait", go/turn/merge wording."""
    return route.kind is RouteKind.GAP_CHECK or route.intent == "should_wait" or bool(GO_QUESTION.search(route.text))


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
        about_sign = bool(SIGN_WORDS.search(question))
        spacing = vcfg.sign_frame_spacing_s if about_sign else vcfg.frame_spacing_s
        older_width = vcfg.latest_width if about_sign else vcfg.older_width  # signs just passed: keep them legible
        frames = self._frames(vcfg.frames, spacing)
        if not frames and snapshot.frame is not None:
            frames = [snapshot.frame]
        images: list[ImagePart] = []
        latest = frames[-1] if frames else None
        for frame in frames[:-1]:
            age = (latest.t - frame.t) if latest is not None else 0.0
            images.append(ImagePart(encode_jpeg(frame.image, older_width, vcfg.jpeg_quality), f"{age:.1f} s ago"))
        if latest is not None:
            images.append(ImagePart(encode_jpeg(latest.image, vcfg.latest_width, vcfg.jpeg_quality), "Now"))
            if LIGHT_WORDS.search(question):
                for box in snapshot.light.boxes[:3]:
                    crop = light_crop(latest.image, box)
                    if crop is not None:
                        label = f"Close-up of the traffic light {_side(box, latest.image.shape[1])}"
                        images.append(ImagePart(encode_jpeg(crop, 512, 92), label))
        when = vcfg.time_override or spoken_time(self._now())
        context = [f"Local time: {when}.", *describe_world(snapshot)]
        if self._extra is not None:
            context += self._extra()
        if latest is None:
            context.append("No camera image is available right now.")
        text = "Context:\n" + "\n".join(f"- {line}" for line in context) + f"\n\nDriver's question: {question}"
        return VisionRequest(
            system=SYSTEM_PROMPT,
            text=text,
            images=tuple(images),
            max_tokens=self._max_tokens(),
            temperature=vcfg.temperature,
        )

    def _max_tokens(self) -> int:
        by_provider = {"claude": self.cfg.vision.claude.max_tokens, "gemini": self.cfg.vision.gemini.max_tokens}
        return by_provider.get(self.model.name, 300)

    # ---------------------------------------------------------------------------------- answer

    def answer(self, route: Route, snapshot: WorldSnapshot, cancel: threading.Event | None = None) -> Iterator[str]:
        """Spoken sentences, as they arrive (stops early if `cancel` is set)."""
        request = self.build_request(route.question or route.text, snapshot)
        vcfg = self.cfg.vision
        pieces = guarded_stream(self.model, request, vcfg.first_token_timeout_s, vcfg.total_timeout_s, cancel)
        passing = bool(PASS_QUESTION.search(route.text))
        return self._safe(sentences(pieces), strict=is_go_question(route), passing=passing)

    def _safe(self, stream: Iterator[str], strict: bool, passing: bool = False) -> Iterator[str]:
        """Safety nets on every sentence, and spoken fallbacks for failures."""
        said = False
        no_vouch_said = False
        try:
            for sentence in stream:
                if passing and SCHOOL_BUS.search(sentence):
                    # "Should I go around them?" with a school bus in view: the rule, not the model's judgment.
                    log.warning("school bus in a passing answer: replaced by the rule", sentence=sentence)
                    yield SCHOOL_BUS_RULE
                    return
                if PERMISSION.search(sentence) or (strict and VOUCHING.search(sentence)):
                    log.warning("replaced a sentence that vouched for traffic", sentence=sentence, strict=strict)
                    if no_vouch_said:
                        continue
                    sentence, no_vouch_said = NO_VOUCH, True
                said = True
                yield sentence
        except VisionCancelled:
            return
        except VisionError as exc:
            log.warning("vision answer failed", error=str(exc), after_sentences=said)
            if said:
                yield LOST_REST
            else:
                yield FALLBACK_TIMEOUT if isinstance(exc, VisionTimeout) else FALLBACK_ERROR
            return
        if not said:
            log.warning("vision answer was empty")
            yield FALLBACK_EMPTY
            return
        if strict and not no_vouch_said:
            yield NO_VOUCH
