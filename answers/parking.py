"""Find me a parking spot (V2-M1): the watch that turns parking events (state/parking.py) into speech.

The user's rules (2026-10-10, V2_BUILD_PLAN.md "Decisions"):
- Lots and street parking; street parking only when asked. When we crawl into what looks like a lot (and the
  vision model agrees it is one), DrivePilot offers help ("Looks like a parking lot. Want help finding a spot?")
  and listens a few seconds for a plain "yes" / "no" (no wake word needed for that reply).
- Skip handicap, reserved, fire-lane and no-parking spots; announce EV-only spots as such.
- Either side, as far ahead as it can see reliably (state/parking.py only reports a space once it can see into
  it, so a small car hidden behind a bigger one isn't an "open spot").
- After a spot: announce the nearest, keep watching, speak again only if we pass it, it's taken, or a closer one
  opens; also "A car is pulling out on your right."
- The watch ends when we've parked, on "stop looking", or after 5 minutes.
- Phrasing: "Open spot on your left, two cars ahead, just after the white SUV."

Every candidate space is double-checked by the vision model on the frame where it was found (structured JSON, like
answers/lanes.py): is it really empty (a tiny car, a motorcycle, a cart), a real stall (not a walkway or driving
lane), and restricted? The check starts at the "candidate" event, so its answer is usually back when the space is
confirmed ("spot"); if the model fails, the spot is still announced, hedged ("Looks like an open spot ...").

Which spot is spoken: the current one (the last we told the driver about) stays current until we pass it or it's
taken; a confirmed spot nearer than the current one is spoken at once ("a closer one opens"); others are kept and
the nearest is spoken when the current one is passed or taken. "Cars ahead" is counted when the spot is spoken.
Thread-safe: world events arrive on the perception thread, questions on the voice thread, checks on a worker, and
a timer thread (while watching) handles passing a spot and the 5-minute timeout.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from answers.scene import encode_jpeg
from answers.speech import Priority, Utterance
from answers.vision import ImagePart, VisionError, VisionModel, VisionRequest
from core.config import ParkingConfig, VisionConfig
from core.logging_setup import get_logger
from core.types import Frame
from state.parking import ParkedCar, ParkingEvent, Space
from state.world import WorldEvent, WorldSnapshot

log = get_logger(__name__)

OFFER = "Looks like a parking lot. Want help finding a spot?"
ARMED = "Okay, I'll look for an open spot."
ARMED_STREET = "Okay, I'll look for a spot along the street."
ALREADY = "I'm already looking for a spot."
STOPPED = "Okay, I'll stop looking."
NOT_LOOKING = "I wasn't looking for a spot."
TIMED_OUT = "I've stopped looking for a parking spot."
TAKEN = "That spot's been taken."
DECLINED = "Okay."
NO_CAMERA = "I can't look for a spot right now: I don't have the camera."
RESTRICTIONS = ("none", "handicap", "ev", "reserved", "compact", "fire_lane", "no_parking", "loading", "unknown")
SKIP = {"handicap", "reserved", "fire_lane", "no_parking", "loading"}
_NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_MISSING, _RUNNING = object(), object()  # states of a space's model check (else a SpotCheck, or None: failed)
TICK_S = 0.5
NEVER = float("-inf")  # no offer waiting for its reply

CHECK_SYSTEM = (
    "You check one dashcam image from a car that is looking for a parking space. A green outline marks one parking "
    "space on the ground between two parked vehicles. Be strict: a space with anything in it, or that you can't "
    'see well enough, is not open; say "low" confidence when unsure.'
)
CHECK_LOT = (
    "Look at the space marked by the green outline on the ground, on the {side} side. Is it a real parking stall "
    "we could pull into (not a driving lane, walkway, crosswalk, island or driveway), and is it completely empty: "
    "no car of any size, no motorcycle, scooter, cart, cone, barrier or person in it? Is it restricted by paint or "
    "a sign: handicap "
    "(blue paint, wheelchair symbol), EV charging, reserved, compact, fire lane, no parking or loading? Also name "
    'the parked vehicle right before the space (nearer to us), e.g. "white SUV" or "gray sedan". Reply as '
    '{{"open": true|false, "restriction": "none"|"handicap"|"ev"|"reserved"|"compact"|"fire_lane"|"no_parking"|'
    '"loading"|"unknown", "before": "<color and type>" or null, "confidence": "high"|"medium"|"low"}}.'
)
CHECK_STREET = (
    "Look at the curb space marked by the green outline on the ground, on the {side} side. Is it a legal, empty "
    "curb space long enough for a car: no vehicle, motorcycle or person in it, not a driveway, crosswalk, fire "
    "hydrant zone, bus stop or painted (red/yellow/blue) curb, and no sign forbidding parking now? Name the parked "
    "vehicle right "
    'before the space (nearer to us). Reply as {{"open": true|false, "restriction": "none"|"handicap"|"ev"|'
    '"reserved"|"compact"|"fire_lane"|"no_parking"|"loading"|"unknown", "before": "<color and type>" or null, '
    '"confidence": "high"|"medium"|"low"}}.'
)
LOT_SYSTEM = "You look at one dashcam image and say where the car is. Be conservative."
LOT_QUESTION = (
    "Is the car driving inside a parking lot or parking garage (rows of parked cars in marked stalls), rather than "
    'on a street with cars parked along the curb? Reply as {"place": "lot"|"street"|"other", "confidence": '
    '"high"|"medium"|"low"}.'
)


@dataclass(frozen=True, slots=True)
class SpotCheck:
    """The vision model's verdict on one candidate space."""

    open: bool
    restriction: str
    before: str | None
    confidence: str

    @property
    def rejected(self) -> bool:
        """Not a spot to announce: taken, unsure, or restricted (EV-only is announced as such)."""
        return not self.open or self.confidence == "low" or self.restriction in SKIP


def parse_spot_check(data: dict[str, Any]) -> SpotCheck | None:
    """A SpotCheck from the model's JSON, or None if it's malformed."""
    is_open, restriction = data.get("open"), data.get("restriction", "unknown")
    before, confidence = data.get("before"), data.get("confidence")
    if not isinstance(is_open, bool) or confidence not in ("high", "medium", "low"):
        return None
    if restriction not in RESTRICTIONS:
        restriction = "unknown"
    if not isinstance(before, str) or not before.strip() or len(before) > 40:
        before = None
    return SpotCheck(is_open, str(restriction), before.strip() if before else None, str(confidence))


def _count(n: int) -> str:
    return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else str(n)


def describe(space: Space, cars_before: int, before: str | None, check: SpotCheck | None) -> str:
    """ "Open spot on your left, two cars ahead, just after the white SUV." (hedged without a model check)"""
    kind = "spot"
    if check is not None and check.restriction == "ev":
        kind = "EV charging spot"
    elif check is not None and check.restriction == "compact":
        kind = "compact spot"
    what = f"{_count(space.stalls).capitalize()} open {kind}s" if space.stalls > 1 else f"Open {kind}"
    text = f"{what} on your {space.side}"
    if cars_before > 1:
        text += f", {_count(cars_before)} cars ahead"
    if before:
        text += f", just after the {before}"
    text += "."
    if check is None:  # the model couldn't double-check it
        text = ("Looks like an " if text.startswith("Open") else "Looks like ") + text[0].lower() + text[1:]
    return text


def outlined(
    frame: Frame, region: tuple[float, float, float, float], polygon: tuple[tuple[float, float], ...] = ()
) -> np.ndarray:
    """The frame with the space outlined in green (what the model is asked about): its ground footprint if known,
    else its box."""
    image = frame.image.copy()
    thick = max(2, image.shape[1] // 400)
    if polygon:
        points = np.array([[round(x), round(y)] for x, y in polygon], np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=thick)
        return image
    x1, y1, x2, y2 = (round(v) for v in region)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thick)
    return image


def crop(image: np.ndarray, region: tuple[float, float, float, float], context: float = 1.5) -> np.ndarray:
    """A close-up around the region (with context), at least 384 px wide."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = region
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) * (1 + context), 160) / 2
    half_h = max((y2 - y1) * (1 + context), 120) / 2
    a, b = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    c, d = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    out = image[b:d, a:c]
    if out.shape[1] < 384:
        out = cv2.resize(out, None, fx=384 / out.shape[1], fy=384 / out.shape[1], interpolation=cv2.INTER_CUBIC)
    return out


@dataclass(slots=True)
class _Spot:
    """A confirmed, approved space."""

    space: Space
    check: SpotCheck | None
    spoken: bool = False

    def same_place(self, space: Space) -> bool:
        """The same place along our path (re-detected under other neighbour ids)."""
        return self.space.side == space.side and abs(self.space.along - space.along) < 0.6 * space.pitch


class ParkingWatch:
    """Looks for a parking spot when asked (or after a yes to the lot offer) and says what it finds."""

    def __init__(
        self,
        cfg: ParkingConfig,
        vision_cfg: VisionConfig,
        model: VisionModel | None,
        say: Callable[[Utterance], None],
        snapshot: Callable[[], WorldSnapshot | None],
        set_mode: Callable[[str], None],
        clock: Callable[[], float],
        expect_reply: Callable[[float], None] | None = None,
        quiet: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cfg = cfg
        self.vision_cfg = vision_cfg
        self.model = model
        self._say = say
        self._snapshot = snapshot
        self._set_mode = set_mode
        self._clock = clock
        self.expect_reply = expect_reply  # the voice pipeline: listen without the wake word for a few seconds
        self._quiet = quiet
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parking-check")
        self._generation = 0
        self.active = False
        self.mode = "lot"
        self._offered_at: float | None = None
        self._awaiting_reply_until = NEVER
        self._check_times: deque[float] = deque()
        self._clear()

    def _clear(self) -> None:
        """Forget the spots of the current search (caller holds the lock, or construction)."""
        self._armed_at = 0.0
        self._checks: dict[tuple[str, int, int], object] = {}  # _RUNNING, a SpotCheck, or None (failed)
        self._pending: dict[tuple[str, int, int], Space] = {}  # confirmed spaces waiting for their check
        self._spots: list[_Spot] = []
        self._rejected: list[Space] = []
        self._current: _Spot | None = None

    def reset(self) -> None:
        """Forget everything (new video segment): the watch ends."""
        with self._lock:
            self.active = False
            self._generation += 1
            self._offered_at = None
            self._awaiting_reply_until = NEVER
            self._clear()

    # ------------------------------------------------------------------------------------- voice

    def arm(self, street: bool = False) -> str:
        """ "Find me a parking spot": start looking. Returns what to say."""
        snap = self._snapshot()
        if snap is None:
            return NO_CAMERA
        with self._lock:
            if self.active:
                return ALREADY
            self.active = True
            self._generation += 1
            self._clear()
            self._armed_at = self._clock()
            self._awaiting_reply_until = NEVER
            generation = self._generation
        if street:
            self._use_mode("street")
        elif snap.parking.lot_like:
            self._use_mode("lot")
        else:
            self._pool.submit(self._choose_mode, snap.frame, generation)
        threading.Thread(target=self._run_timer, args=(generation,), name="parking-timer", daemon=True).start()
        log.info("parking watch armed", street=street)
        return ARMED_STREET if street else ARMED

    def cancel(self) -> str:
        """ "Stop looking" (or "I found a spot"). Returns what to say."""
        was = self.stop_silently()
        log.info("parking watch cancelled", was_active=was)
        return STOPPED if was else NOT_LOOKING

    def stop_silently(self) -> bool:
        """End the watch without a word ("never mind", we parked). True if it was on."""
        with self._lock:
            was, self.active = self.active, False
            self._generation += 1
        return was

    def awaiting_reply(self) -> bool:
        """An offer was just made and its plain yes/no is still welcome."""
        with self._lock:
            return self._clock() <= self._awaiting_reply_until

    def on_reply(self, yes: bool) -> str:
        """The plain answer to the lot offer."""
        with self._lock:
            self._awaiting_reply_until = NEVER
        return self.arm() if yes else DECLINED

    # ------------------------------------------------------------------------------------ events

    def on_world_event(self, event: WorldEvent) -> None:
        """WorldState callback (perception thread)."""
        if event.kind == "segment_reset":
            self.reset()
            return
        if not event.kind.startswith("parking_") or not isinstance(event.payload, tuple):
            return
        parking_event, frame = event.payload
        if not isinstance(parking_event, ParkingEvent):
            return
        kind = parking_event.kind
        if kind == "lot":
            self._maybe_offer(frame)
            return
        with self._lock:
            active, generation = self.active, self._generation
        if not active:
            return
        space = parking_event.space
        if kind == "parked":
            if self.stop_silently():
                log.info("parking watch ended: parked")
        elif kind == "candidate" and space is not None:
            self._start_check(space, frame, generation)
        elif kind == "spot" and space is not None:
            self._on_spot(space, frame, generation)
        elif kind == "taken" and space is not None:
            self._on_taken(space, generation)
        elif kind == "pulling_out" and parking_event.car is not None:
            self._on_pull_out(parking_event.car)

    # ----------------------------------------------------------------------------- mode, offer

    def _use_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode
        self._set_mode(mode)

    def _choose_mode(self, frame: Frame | None, generation: int) -> None:
        place = self._ask_place(frame)
        with self._lock:
            current = self.active and self._generation == generation
        if current:
            self._use_mode("street" if place == "street" else "lot")

    def _ask_place(self, frame: Frame | None) -> str | None:
        """ "lot" | "street" | "other" from the vision model (None if unsure or unavailable)."""
        if self.model is None or frame is None:
            return None
        cfg = self.vision_cfg
        image = ImagePart(encode_jpeg(frame.image, cfg.latest_width, cfg.jpeg_quality), "Now")
        request = VisionRequest(
            system=LOT_SYSTEM, text=LOT_QUESTION, images=(image,), max_tokens=40, temperature=cfg.temperature
        )
        try:
            data = self.model.complete_json(request, cfg.first_token_timeout_s, cfg.total_timeout_s)
        except VisionError as exc:
            log.warning("lot check failed", error=str(exc))
            return None
        place = data.get("place")
        log.info("lot check", raw=data)
        return str(place) if place in ("lot", "street", "other") and data.get("confidence") != "low" else None

    def _maybe_offer(self, frame: Frame | None) -> None:
        now = self._clock()
        with self._lock:
            if self.active or self._quiet():
                return
            if self._offered_at is not None and now - self._offered_at < self.cfg.offer_cooldown_s:
                return
            self._offered_at = now  # (one offer per lot visit, even while the check runs)
        self._pool.submit(self._offer_if_lot, frame)

    def _offer_if_lot(self, frame: Frame | None) -> None:
        if self._ask_place(frame) != "lot":
            return
        with self._lock:
            if self.active:
                return
            self._awaiting_reply_until = self._clock() + self.cfg.reply_window_s + 4.0  # (+ saying the offer)
        self._say(Utterance(OFFER, "alert", Priority.INFO, t=self._clock()))
        if self.expect_reply is not None:
            self.expect_reply(self.cfg.reply_window_s)

    # ---------------------------------------------------------------------------------- checks

    def _start_check(self, space: Space, frame: Frame | None, generation: int) -> bool:
        """Ask the model about a candidate space (once per space); False if no check was started."""
        now = time.monotonic()
        with self._lock:
            if space.key in self._checks or self._known(space):
                return False
            while self._check_times and now - self._check_times[0] > 60.0:
                self._check_times.popleft()
            if len(self._check_times) >= self.cfg.max_checks_per_min:
                log.info("parking check skipped: rate limit")
                return False
            self._check_times.append(now)
            self._checks[space.key] = _RUNNING
            mode = self.mode
        self._pool.submit(self._check, space, frame, mode, generation)
        return True

    def _check(self, space: Space, frame: Frame | None, mode: str, generation: int) -> None:
        verdict = self._ask_spot(space, frame, mode)
        with self._lock:
            if self._generation != generation:
                return
            self._checks[space.key] = verdict  # a SpotCheck, or None if the model failed
            pending = self._pending.pop(space.key, None)
        log.info("parking check", side=space.side, verdict=verdict)
        if pending is not None:
            self._decide(pending, verdict, generation)

    def _ask_spot(self, space: Space, frame: Frame | None, mode: str) -> SpotCheck | None:
        if self.model is None or frame is None:
            return None
        cfg = self.vision_cfg
        marked = outlined(frame, space.region, space.footprint)
        images = (
            ImagePart(encode_jpeg(marked, cfg.latest_width, cfg.jpeg_quality), "Now, the space outlined"),
            ImagePart(encode_jpeg(crop(marked, space.outline), 512, cfg.jpeg_quality), "Close-up of the space"),
        )
        question = (CHECK_LOT if mode == "lot" else CHECK_STREET).format(side=space.side)
        request = VisionRequest(
            system=CHECK_SYSTEM, text=question, images=images, max_tokens=80, temperature=cfg.temperature
        )
        try:
            data = self.model.complete_json(request, cfg.first_token_timeout_s, cfg.total_timeout_s)
        except VisionError as exc:
            log.warning("parking check failed", error=str(exc))
            return None
        return parse_spot_check(data)

    # ----------------------------------------------------------------------------------- spots

    def _on_spot(self, space: Space, frame: Frame | None, generation: int) -> None:
        """A confirmed space: decide now if its check is done, or when it returns."""
        with self._lock:
            state = self._checks.get(space.key, _MISSING)
            if state is _RUNNING:
                self._pending[space.key] = space
                return
        if state is _MISSING and self._start_check(space, frame, generation):
            with self._lock:
                state = self._checks.get(space.key, _MISSING)
                if state is _RUNNING:
                    self._pending[space.key] = space
                    return
        self._decide(space, state if isinstance(state, SpotCheck) else None, generation)

    def _decide(self, space: Space, check: SpotCheck | None, generation: int) -> None:
        """A confirmed space with its check: keep it (if the model agrees or couldn't say) and speak if it's next."""
        with self._lock:
            if not self.active or self._generation != generation or self._known(space):
                return
            if check is not None and check.rejected:
                self._rejected.append(space)
                log.info("parking spot skipped", side=space.side, check=check)
                return
            self._spots.append(_Spot(space, check))
        self._speak_next(generation)

    def _known(self, space: Space) -> bool:
        """Kept, spoken or rejected already (caller holds the lock)."""
        return any(s.same_place(space) for s in self._spots) or any(
            r.side == space.side and abs(r.along - space.along) < 0.6 * space.pitch for r in self._rejected
        )

    def _speak_next(self, generation: int) -> None:
        """Speak the nearest kept spot still ahead, unless the current one is still ahead and nearer."""
        snap = self._snapshot()
        along = snap.parking.along if snap is not None else 0.0
        with self._lock:
            if not self.active or self._generation != generation:
                return
            current = self._current
            if current is not None and current.space.along <= along:
                current = self._current = None  # passed
            ahead = [s for s in self._spots if not s.spoken and s.space.along > along]
            if not ahead:
                return
            nearest = min(ahead, key=lambda s: s.space.along)
            if current is not None and current.space.along <= nearest.space.along:
                return  # the spot we told about is still ahead and nearer
            nearest.spoken = True
            self._current = nearest
        space = nearest.space
        z_now = space.along - along
        row = snap.parking.rows.get(space.side, ()) if snap is not None else ()
        cars_before = sum(1 for c in row if c.z < z_now) if row else space.cars_before
        check = nearest.check
        before = check.before if check is not None and check.before else self._label(snap, space.near)
        text = describe(space, cars_before, before, check)
        log.info("parking spot", text=text, checked=check is not None)
        self._say(Utterance(text, "alert", Priority.REQUESTED, t=self._clock()))

    def _on_taken(self, space: Space, generation: int) -> None:
        with self._lock:
            spot = next((s for s in self._spots if s.same_place(space)), None)
            if spot is None:
                return
            self._spots.remove(spot)
            self._rejected.append(space)  # (don't announce it again)
            was_current = self._current is spot
            if was_current:
                self._current = None
        if was_current:
            self._say(Utterance(TAKEN, "alert", Priority.REQUESTED, t=self._clock()))
            self._speak_next(generation)

    def _on_pull_out(self, car: ParkedCar) -> None:
        snap = self._snapshot()
        if snap is None or car.z < 2.0:
            return
        before = sum(1 for c in snap.parking.rows.get(car.side, ()) if c.z < car.z)  # parked cars nearer than it
        where = "just ahead" if before == 0 else f"{_count(before + 1)} cars ahead"
        text = f"A car is pulling out on your {car.side}, {where}."
        self._say(Utterance(text, "alert", Priority.REQUESTED, t=self._clock()))

    @staticmethod
    def _label(snap: WorldSnapshot | None, car: ParkedCar) -> str | None:
        if snap is None:
            return None
        for vehicle in snap.vehicles:
            if vehicle.track_id == car.track_id and vehicle.color:
                return vehicle.label
        return None

    # ------------------------------------------------------------------------------------ timer

    def _run_timer(self, generation: int) -> None:
        """While watching: speak the next spot once the current one is passed; end after the timeout."""
        while True:
            time.sleep(TICK_S)
            with self._lock:
                if not self.active or self._generation != generation:
                    return
                expired = self._clock() - self._armed_at >= self.cfg.watch_timeout_s
                if expired:
                    self.active = False
            if expired:
                log.info("parking watch timed out")
                self._say(Utterance(TIMED_OUT, "alert", Priority.REQUESTED, t=self._clock()))
                return
            self._speak_next(generation)

    def close(self) -> None:
        """Stop the worker (app shutdown)."""
        self.stop_silently()
        self._pool.shutdown(wait=False, cancel_futures=True)
