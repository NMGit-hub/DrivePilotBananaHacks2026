"""Showcase video: a clip played through DrivePilot, with the V2 page's overlays drawn into the video, captions for
the questions, answers and alerts, and the spoken audio (Piper), written as an MP4.

    .venv/Scripts/python scripts/render_showcase.py data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov ^
        --start 26 --end 40 --ask "28=tell me when it's green" --set video.file.hfov_deg=60
    .venv/Scripts/python scripts/render_showcase.py D:/Dashcam/dm_20260919_152229.mp4 --start 30 --end 50 ^
        --ask "36=what does that sign say?" --set video.file.hfov_deg=110 --png 40

Pass 1 plays the clip in lock-step (every frame, deterministic, like eval/run_sheet.py) through the app's real
pieces: perception (YOLO26s + ByteTrack, ego motion), the world state, YOLOPv2 lanes (with drift warnings, given our
GPS speed), the sign detector (sign alerts, the speed-limit badge), the speeding watch, the parking watch, scene
memory and the `Assistant` (local answers, the Claude vision model with the location and sign context). It asks
each `--ask` question at its video time and records, for every frame from --start, the page's overlay
(`ui.overlay.frame_packet`) and HUD (`hud_packet`), and everything DrivePilot said.

Then every spoken line is synthesized with Piper (answers and alerts in the app's voice, alerts after the app's
chime; the driver's question in another Piper voice, "Hey Drive Pilot", the app's listening blip, the question) and
placed in time like the real app (ui/timeline.py): the question ends just before its moment, the answer follows
after the measured answer time, alerts at their event, interrupting answers.

Pass 2 decodes the clip again and draws each frame (ui/render.py): overlays, HUD, caption cards, "Listening…",
and encodes H.264 (libx264, CRF 18) + AAC audio with PyAV. Dashcam clips (`dm_*`, 1920x1280) have a burned-in text
bar in their bottom 45 rows (date, time, speed, GPS): it is cropped from the output (--keep-overlay-bar keeps it)
and masked (black) for perception (--no-mask-bar to leave it).

Output: runs/showcase/<clip>_<start>-<end>.mp4 (or --out), with a .json timeline (every line, when it plays) and
.log next to it; --png T saves the rendered frame at video time T as a PNG too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.assistant import Assistant  # noqa: E402
from answers.auto_alerts import AutoSwitches, SpeedingWatch, WrongLaneWatch  # noqa: E402
from answers.lanes import CameraLaneReader, VisionLaneReader  # noqa: E402
from answers.parking import ParkingWatch  # noqa: E402
from answers.scene import SceneHandler  # noqa: E402
from answers.speech import Priority, TextSpeaker, Utterance  # noqa: E402
from answers.vision import VisionError, VisionModel, create_reading_model, create_vision_model  # noqa: E402
from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from core.types import Frame  # noqa: E402
from nav.gps import GpsFix, GpxReplay  # noqa: E402
from nav.handler import NavHandler, speed_lines  # noqa: E402
from nav.location import LocationContext, mapbox_reverser  # noqa: E402
from nav.progress import Progress  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.lanes import lane_view  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402
from state.lanes import LanesSnapshot, LaneTracker  # noqa: E402
from state.memory import SceneMemory  # noqa: E402
from state.pedestrians import DrivableArea  # noqa: E402
from state.signs import SignTracker  # noqa: E402
from state.world import WorldState  # noqa: E402
from ui.overlay import frame_packet, hud_packet, lane_target  # noqa: E402
from ui.render import Renderer, make_geometry, output_rate  # noqa: E402
from ui.timeline import (  # noqa: E402
    ALERT,
    ANSWER,
    QUESTION,
    FramePicker,
    Line,
    Placed,
    Window,
    active,
    after_answers,
    answer_delay,
    build_captions,
    mix,
    question_caption,
    schedule,
    visible_captions,
)
from voice.router import RouteKind, parse_yes_no  # noqa: E402

log = get_logger("scripts.render_showcase")

DASHCAM_SIZE = (1920, 1280)
DASHCAM_BAR_TOP = 1235  # nav/dashcam_overlay.py OVERLAY_TOP: rows from here down are the burned-in text bar
DASHCAM_DIR = Path("D:/Dashcam")
# The dashcam's camera (PROGRESS.md "own dashcam footage"): ~96° across (scripts/dashcam_calibrate.py, 18 turns), and
# pitched up (horizon ~70 % down the frame), so the lane model's crop sits low.
DASHCAM_OVERRIDES = ("video.file.hfov_deg=96", "lanes.view_center_y=0.65", "perception.hood_bottom=0.93")
VIDEOS = REPO_ROOT / "data" / "videos"
GPS_DIR = REPO_ROOT / "data" / "gps"
OUT_DIR = REPO_ROOT / "runs" / "showcase"
DRIVER_VOICES = REPO_ROOT / "data" / "wakeword" / "piper_voices"
LANES_FRESH_S = 0.5  # (app.py: lane detection older than this isn't shown)
STALE_AFTER_NAV = {"nav_rerouted", "nav_not_this_exit", "nav_arrived", "nav_roundabout"}  # (app.py)
GPS_TICK_S = 0.5  # (app.py: the speeding watch runs on this tick)
ENDPOINT_S = 0.3  # the driver stops talking this long before the question is asked (end-of-question detection)
ALERT_LATENCY_S = 0.15  # an alert's first sound after its event (pre-rendered clips: < 0.3 s)
AUDIO_RATE = 48000
THUMB_W = 480
RECORD_EARLY_S = 0.1  # frame states are recorded from this long before --start


# ------------------------------------------------------------------------------------------------ inputs


def resolve_clip(name: str) -> Path:
    """A clip path, or a file name (or part of one) under data/videos/ or D:/Dashcam."""
    path = Path(name)
    if path.is_file():
        return path
    matches = [p for p in VIDEOS.rglob("*") if p.is_file() and name.lower() in p.name.lower()]
    if not matches and DASHCAM_DIR.is_dir():
        matches = [p for p in DASHCAM_DIR.glob("*.mp4") if name.lower() in p.name.lower()]
    if len(matches) != 1:
        raise SystemExit(f"clip {name!r}: {len(matches)} matching files (need exactly one)")
    return matches[0]


def is_dashcam(path: Path, width: int, height: int) -> bool:
    """The car's own dashcam (burned-in text bar at the bottom): `dm_` files or its 1920x1280 frames."""
    return path.name.startswith("dm_") or (width, height) == DASHCAM_SIZE


def default_gpx(clip: Path) -> Path | None:
    """data/gps/<stem>.gpx (dashcam clips), or data/gps/<BDD id>.gpx for dev clips named "<what>__<id>"."""
    for stem in (clip.stem, clip.stem.split("__")[-1]):
        candidate = GPS_DIR / f"{stem}.gpx"
        if candidate.is_file():
            return candidate
    return None


def parse_ask(item: str) -> tuple[float, str]:
    """ "22.5=who got here first?" -> (22.5, "who got here first?") (as app.py's --ask)."""
    t, sep, question = item.partition("=")
    if not sep or not question.strip():
        raise argparse.ArgumentTypeError(f"expected <seconds>=<question>, got {item!r}")
    return float(t), question.strip()


@dataclass(frozen=True, slots=True)
class ClipInfo:
    """What the renderer needs to know about a clip."""

    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.frames / self.fps if self.fps > 0 else 0.0


def probe(path: Path) -> ClipInfo:
    """Frame rate, frame count and size of a clip."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"can't open {path}")
        return ClipInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


class MaskedFileSource(FileSource):
    """A lock-step file source whose frames have the rows from `mask_from` down blacked out (the dashcam's text bar,
    so perception never sees its text)."""

    def __init__(self, path: Path, mask_from: int | None, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self.mask_from = mask_from

    def _publish(self, frame: Frame) -> None:
        if self.mask_from is not None and frame.image.shape[0] > self.mask_from:
            image = frame.image.copy()
            image[self.mask_from :] = 0
            frame = replace(frame, image=image)
        super()._publish(frame)


# ------------------------------------------------------------------------------------------------ pass 1


@dataclass(slots=True)
class FrameState:
    """What the page would show with one frame."""

    t: float
    overlay: dict[str, Any]
    hud: dict[str, Any]


@dataclass(slots=True)
class Asked:
    """One --ask question and its answer."""

    t: float
    question: str
    group: int
    answer: str | None = None
    source: str = ""
    route: str = ""
    first_ms: float | None = None
    sentences: list[str] = field(default_factory=list)
    highlight: int | None = None
    image: np.ndarray | None = None  # the remembered frame the answer came from (a thumbnail)
    image_t: float | None = None


@dataclass(slots=True)
class Recording:
    """Pass 1's result."""

    frames: list[FrameState]
    asked: list[Asked]
    said: list[Utterance]  # everything DrivePilot said on its own (alerts, parking), at its media time
    seconds: float = 0.0  # wall time of the pass
    processed: int = 0


@dataclass(slots=True)
class Models:
    """The models pass 1 uses (built once)."""

    detector: Detector
    vision: VisionModel | None
    reading: VisionModel | None
    lane_model: Any
    sign_model: Any
    reverser: Any
    routing: Any = None  # Mapbox (navigation), None without a token or with --no-nav


def build_models(cfg: AppConfig, lanes: bool, signs: bool, nav: bool = True) -> Models:
    """The detector, lane and sign models, the vision models, the reverse geocoder and the routing client, as
    eval/run_sheet.py and app.py build them."""
    try:
        vision: VisionModel | None = create_vision_model(cfg)
    except VisionError as exc:
        log.warning("vision model not available: model questions will say so", error=str(exc))
        vision = None
    reading = create_reading_model(cfg) if vision is not None else None
    detector = Detector(cfg.perception, cfg.root)
    sign_model = None
    if signs and cfg.sign_alerts.enabled and cfg.resolve_path(cfg.signs.weights).is_file():
        from perception.signs import SignDetector

        sign_model = SignDetector(cfg.signs, cfg.root)
    lane_model = None
    if lanes and cfg.lanes.enabled and cfg.resolve_path(cfg.lanes.weights).is_file():
        from perception.lanes import LaneModel

        lane_model = LaneModel(cfg.lanes, cfg.root, hfov_deg=cfg.effective_hfov_deg)
    token = cfg.secrets.mapbox_access_token
    reverser = (
        mapbox_reverser(token.get_secret_value(), cfg.nav.timeout_s)
        if token is not None and cfg.vision.location
        else None
    )
    routing = None
    if nav and token is not None:
        from nav.mapbox import MapboxClient

        routing = MapboxClient(token.get_secret_value(), cfg.nav.profile, cfg.nav.timeout_s, cfg.nav.country)
    return Models(detector, vision, reading, lane_model, sign_model, reverser, routing)


def run_pass(
    cfg: AppConfig,
    clip: Path,
    models: Models,
    asks: list[tuple[float, str]],
    t_from: float,
    t_render: float,
    t_end: float,
    gps: GpxReplay | None,
    mask_from: int | None,
    destination: str | None = None,
    keep_share: float | None = None,
    destination_at: float = -math.inf,
) -> Recording:
    """Play the clip in lock-step from `t_from` (warm-up: the tracker and the light's state get their history) and
    record the page's overlay and HUD for every frame from `t_render` to `t_end`, the answers and what was said.
    `destination`: navigate there from the first GPS fix on, silently (a route already running when the video
    starts); "take me to ..." as a question works too, with navigation on (a GPX track and a Mapbox token).
    `keep_share`: answer thumbnails keep only this top share of the frame's rows (the dashcam's text bar is cropped;
    a share, not rows: remembered frames are downscaled)."""
    speaker = TextSpeaker()
    world_holder: list[WorldState] = []
    memory = SceneMemory(cfg.memory)
    latest_tracks: list[Any] = [()]
    lanes = LaneTracker(cfg.lanes, lane_view(cfg.effective_hfov_deg, cfg.lanes.max_view_deg)[1])
    signs = SignTracker(cfg.sign_alerts)
    switches = AutoSwitches.from_config(cfg.alerts.auto)
    speeding = SpeedingWatch(cfg.alerts.auto)
    wrong_lane = WrongLaneWatch(cfg.alerts.auto)
    drivable: list[DrivableArea | None] = [None]
    clock = [t_from]
    marks = {"lane": -1e9, "sign": -1e9, "gps": -1e9, "fix": -1e9}
    pending_destination = [destination]
    current: list[Asked | None] = [None]
    asked: list[Asked] = []
    frames: list[FrameState] = []
    in_ask: set[int] = set()  # uids of what was said while answering a question

    def fix_now() -> GpsFix | None:
        return gps.fix_at(clock[0]) if gps is not None else None

    def gps_speed(t: float) -> float | None:
        fix = gps.fix_at(t) if gps is not None else None
        return None if fix is None else fix.speed_mps

    location = (
        LocationContext(models.reverser, fix_now, cfg.vision.location_refresh_m)
        if models.reverser is not None and cfg.vision.location
        else None
    )

    def lanes_view(t: float) -> LanesSnapshot | None:
        snap = lanes.snapshot()
        return snap if snap.valid and 0.0 <= t - snap.t <= LANES_FRESH_S else None

    def latest_frame() -> Frame | None:
        snap = world_holder[0].snapshot() if world_holder else None
        return snap.frame if snap is not None else None

    def on_nav_alert(kind: str, text: str) -> None:  # (app.py _on_nav_alert)
        drops = kind in STALE_AFTER_NAV
        assistant.say(Utterance(text, "alert", Priority.REQUESTED, clip=kind, t=clock[0], drops_answers=drops))

    nav: NavHandler | None = None
    if models.routing is not None and gps is not None:
        vision_reader = VisionLaneReader(models.vision, latest_frame, cfg.vision) if models.vision is not None else None
        nav = NavHandler(
            cfg.nav,
            models.routing,
            fix_now,
            on_nav_alert,
            CameraLaneReader(lambda: lanes_view(clock[0]), vision_reader),
            turn_prompts=lambda: switches.enabled("turns"),
        )

    def progress_now() -> Progress | None:
        return nav.progress if nav is not None and nav.route is not None else None

    def extra_context() -> list[str]:  # as the app's (speed and route, location, signs, lanes)
        lines = nav.extra_context() if nav is not None else speed_lines(fix_now())
        lines = lines + (location.context() if location is not None else [])
        return lines + signs.context(clock[0]) + (lanes.context() if lanes_view(clock[0]) is not None else [])

    def on_highlight(track_id: int) -> None:
        if current[0] is not None:
            current[0].highlight = track_id

    def on_image(frame: Frame) -> None:
        if current[0] is not None:
            image = frame.image if keep_share is None else frame.image[: int(frame.image.shape[0] * keep_share)]
            scale = THUMB_W / image.shape[1]
            current[0].image = cv2.resize(image, (THUMB_W, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            current[0].image_t = frame.t

    scene = (
        SceneHandler(
            cfg,
            models.vision,
            lambda n, s: world_holder[0].recent_frames(n, s),
            extra_context,
            tracks=lambda: latest_tracks[0],
            memory=memory,
            on_highlight=on_highlight,
            on_image=on_image,
            reading_model=models.reading,
        )
        if models.vision is not None
        else None
    )
    assistant = Assistant(cfg, lambda: world_holder[0].snapshot(), speaker, nav=nav, model=scene)
    assistant.on_highlight = on_highlight
    assistant.alerts.switches = switches
    world = WorldState(
        cfg, on_event=assistant.on_world_event, speed_source=gps_speed, drivable_source=lambda: drivable[0]
    )
    world_holder.append(world)
    parking = ParkingWatch(
        cfg.parking,
        cfg.vision,
        models.vision,
        assistant.say,
        world.snapshot,
        world.set_parking_mode,
        lambda: clock[0],
        quiet=lambda: assistant.alerts.quiet,
    )
    assistant.parking = parking
    pending = sorted(asks)
    done = threading.Event()
    source = MaskedFileSource(clip, mask_from, realtime=False, start_s=t_from)
    stopper: list[threading.Thread] = []
    pace: list[tuple[float, float] | None] = [None]  # (video t, wall time) when the parking watch started

    def say_alert(text: str, priority: Priority, t: float, clip_name: str | None = None) -> None:
        if priority < Priority.REQUESTED and assistant.alerts.quiet:
            return
        assistant.say(Utterance(text, "alert", priority, clip=clip_name, t=t))

    def ask(t: float, question: str) -> None:
        if location is not None:
            location.wait_idle()  # (lock-step: a lookup started a few frames ago has had no real time yet)
        record = Asked(t=t, question=question, group=len(asked) + 1)
        asked.append(record)
        current[0] = record
        before = len(speaker.said)
        reply = assistant.ask(question)
        current[0] = None
        record.answer, record.source, record.first_ms = reply.text, reply.source, reply.first_ms
        record.route = f"{reply.route.kind.value}/{reply.route.intent}"
        for u in speaker.said[before:]:
            if u.kind == "answer":  # (an alert said meanwhile, e.g. by the parking watch's thread, stays an alert)
                in_ask.add(u.uid)
                record.sentences.append(u.text)
        log.info("asked", t=t, question=question, answer=reply.text, source=reply.source, first_ms=reply.first_ms)
        print(f"  {t:6.2f} s  you: {question}\n           drivepilot ({reply.source}): {reply.text}", flush=True)

    def on_result(result: PerceptionResult) -> None:
        if done.is_set():
            return
        frame = result.frame
        t = frame.t
        clock[0] = t
        if t >= t_end:
            done.set()
            stopper.append(threading.Thread(target=source.stop, name="stop-source", daemon=True))
            stopper[-1].start()  # (not on this thread: stop() joins the source, which waits for us in lock-step)
            return
        if location is not None:
            location.update()
        if models.lane_model is not None and t - marks["lane"] >= 1.0 / cfg.lanes.fps - 1e-6:
            marks["lane"] = t
            masks = models.lane_model(frame)
            for drift in lanes.update(t, masks, gps_speed(t)):  # drift warnings, as the app's lanes thread
                say_alert(drift.text, Priority.SAFETY, drift.t, "drift")
            drivable[0] = DrivableArea(masks.t, masks.drivable, frame.segment, masks.view)
        if models.sign_model is not None and t - marks["sign"] >= 1.0 / cfg.sign_alerts.fps - 1e-6:
            marks["sign"] = t
            for sign in signs.update(t, models.sign_model.detect(frame)):  # as the app's sign thread
                say_alert(sign.text, Priority.INFO, sign.t)
            for seen in signs.last_frame:
                value = f"{seen.value} mph" if seen.value is not None else None
                memory.add_sign(frame, seen.id, seen.kind, seen.box, text=value)
        world.on_perception(result)
        latest_tracks[0] = tuple(result.tracks)
        memory.add(frame, result.tracks)
        if t - marks["gps"] >= GPS_TICK_S - 1e-6:  # the app's GPS tick: navigation, then the wrong-lane and
            marks["gps"] = t  # speeding watches (app.py _housekeeping, _feed_gps, _auto_alerts)
            fix = fix_now()
            if nav is not None and fix is not None and fix.t != marks["fix"]:
                marks["fix"] = fix.t
                nav.on_fix(fix)
            progress = progress_now() if fix is not None else None
            if nav is not None:
                step = (id(nav.route), progress.next_step.index) if progress and progress.next_step else None
                to_next = progress.to_next_m if progress is not None else None
                text = wrong_lane.update(t, lane_target(lanes_view(t), progress), step, to_next)
                if text is not None and switches.enabled("wrong_lane"):
                    say_alert(text, Priority.REQUESTED, t, "wrong_lane")
            sign_snap = signs.snapshot()
            text = speeding.update(
                t,
                fix.speed_mps if fix is not None else None,
                sign_snap.limit,
                sign_snap.limit_t,
                fix.heading if fix is not None else None,
            )
            if text is not None and switches.enabled("speeding"):
                say_alert(text, Priority.INFO, t, "speeding")
            if (
                pending_destination[0] and nav is not None and fix is not None and t >= destination_at
            ):  # (--destination)
                query, pending_destination[0] = pending_destination[0], None
                said = nav.set_destination(query)
                log.info("destination set", query=query, answer=said)
                print(f"  {t:6.2f} s  destination {query!r}: {said}", flush=True)
        while pending and t >= pending[0][0]:
            ask(*pending.pop(0))
            pace[0] = None  # (video time stood still while answering: the parking watch's pacing starts over)
        if t >= t_render - RECORD_EARLY_S:  # (the first output frame may show a source frame just before --start)
            view = lanes_view(t)
            progress = progress_now()
            overlay = frame_packet(
                world.snapshot(),
                result.tracks,
                frame.width,
                frame.height,
                parking_active=parking.active,
                lanes=view,
                lane_goal=lane_target(view, progress),
                hood_bottom=cfg.perception.hood_bottom,
            )
            fix = fix_now()
            sign_snap = signs.snapshot()
            school_t = sign_snap.school_limit_t
            recent = school_t is not None and t - school_t <= cfg.sign_alerts.repeat_s
            hud = hud_packet(
                progress,
                fix.speed_mps if fix is not None else None,
                sign_snap.limit,
                sign_snap.school_limit if recent else None,
            )
            frames.append(FrameState(t, overlay, hud))
        # The parking watch's model checks run in the background in real time: while it is on, don't let the video
        # run ahead of the wall clock (so its answers land at realistic video times).
        if parking.active:
            anchor = pace[0]
            if anchor is None:
                anchor = pace[0] = (t, time.perf_counter())
            ahead = (t - anchor[0]) - (time.perf_counter() - anchor[1])
            if ahead > 0:
                time.sleep(ahead)
        else:
            pace[0] = None

    start = time.perf_counter()
    worker = PerceptionWorker(source, models.detector, EgoMotionEstimator(cfg.ego_motion))
    worker.add_listener(on_result)
    source.start()
    worker.start()
    worker.join()
    for thread in stopper:  # (FileSource's close isn't safe to run twice at once)
        thread.join()
    source.stop()
    parking.close()
    if worker.error is not None:
        raise RuntimeError(f"perception failed on {clip.name}") from worker.error
    for t, question in pending:
        log.warning("question past the end of the render", t=t, question=question)
    said = [u for u in speaker.said if u.uid not in in_ask]
    return Recording(frames, asked, said, time.perf_counter() - start, worker.processed)


# ------------------------------------------------------------------------------------------------ speech


class Voices:
    """Piper: DrivePilot's voice (config.yaml voice.tts) and the driver's (another Piper voice), at AUDIO_RATE."""

    def __init__(self, cfg: AppConfig, driver_voice: str | None) -> None:
        from voice.tts import Tts, alert_chime, listen_tone

        self.tts = Tts(cfg.voice.tts, cfg.root)
        self.tts.warmup()
        rate = self.tts.rate
        self.chime = alert_chime(rate, cfg.voice.tts.chime_volume)
        self.blip = listen_tone(rate, cfg.voice.tts.chime_volume)
        self.gap = np.zeros(round(0.04 * rate), np.float32)
        self.driver: Any = None
        if driver_voice:
            path = DRIVER_VOICES / f"{driver_voice}.onnx"
            if path.is_file():
                self.driver = Tts(cfg.voice.tts.model_copy(update={"voice": path, "length_scale": 1.0}), cfg.root)
            else:
                log.warning("driver voice not found: questions are silent", path=str(path))

    def _to_out(self, audio: np.ndarray, rate: int) -> np.ndarray:
        from voice.audio import resample

        return resample(audio.astype(np.float32), rate, AUDIO_RATE)

    def answer(self, text: str) -> tuple[np.ndarray, float]:
        """An answer sentence (and how long synthesizing it took)."""
        start = time.perf_counter()
        audio = self.tts.synthesize(text)
        return self._to_out(audio, self.tts.rate), time.perf_counter() - start

    def alert(self, text: str) -> np.ndarray:
        """An alert: the chime, then the words (as voice/speaker.py)."""
        return self._to_out(np.concatenate((self.chime, self.gap, self.tts.synthesize(text))), self.tts.rate)

    def question(self, text: str) -> tuple[np.ndarray, float]:
        """The driver's "Hey Drive Pilot, <question>" with the app's listening blip after the wake word, and when
        (seconds into it) the wake word has been heard; silent without a driver voice."""
        if self.driver is None:
            return np.zeros(0, np.float32), 0.0
        from voice.audio import resample

        rate = self.driver.rate
        if len(text.split()) <= 3 and parse_yes_no(text) is not None:  # a reply to an offer: no wake word (the app
            return self._to_out(self.driver.synthesize(text), rate), 0.0  # listens for it a few seconds)
        wake = self.driver.synthesize("Hey Drive Pilot.")
        pause = np.zeros(round(0.12 * rate), np.float32)
        blip = self.blip if rate == self.tts.rate else resample(self.blip, self.tts.rate, rate)
        after = np.zeros(round(0.22 * rate), np.float32)
        words = self.driver.synthesize(text)
        audio = np.concatenate((wake, pause, blip, after, words))
        return self._to_out(audio, rate), (len(wake) + len(pause)) / rate


def silent_stand_in(text: str) -> np.ndarray:
    """--no-audio: silence as long as the line would take to say (~15 characters a second), so captions keep their
    timing."""
    return np.zeros(round(AUDIO_RATE * max(1.0, len(text) / 15.0)), np.float32)


@dataclass(slots=True)
class Timeline:
    """Every line placed in time, the caption cards, the listening and highlight windows, and the audio."""

    placed: list[Placed]
    captions: list[Any]
    listening: list[Window]
    highlights: list[Window]
    images: dict[str, np.ndarray]
    audio: Any
    lines: list[dict[str, Any]]


def build_timeline(
    cfg: AppConfig, rec: Recording, voices: Voices | None, t0: float, t1: float, answer_delay_s: float | None
) -> Timeline:
    """Synthesize every line, place them (ui/timeline.py) and mix the audio track from t0 to t1."""
    lines: list[Line] = []
    audio: dict[int, np.ndarray] = {}
    uid = iter(range(1, 1_000_000))
    listening: list[Window] = []
    images: dict[str, np.ndarray] = {}
    caption_image: dict[int, str] = {}
    routing: list[tuple[float, float]] = []  # (asked, answer ready) of navigation questions
    for a in rec.asked:
        q_uid = next(uid)
        q_audio, wake_s = voices.question(a.question) if voices is not None else (np.zeros(0, np.float32), 0.0)
        q_dur = len(q_audio) / AUDIO_RATE if len(q_audio) else 1.2
        q_start = a.t - ENDPOINT_S - q_dur
        lines.append(Line(QUESTION, question_caption(a.question), q_start, q_dur, group=a.group, uid=q_uid))
        audio[q_uid] = q_audio
        listening.append(Window(q_start + wake_s, a.t))
        tts_s = 0.0
        sentences: list[Line] = []
        for i, sentence in enumerate(a.sentences):
            s_uid = next(uid)
            if voices is not None:
                s_audio, synth_s = voices.answer(sentence)
                tts_s = synth_s if i == 0 else tts_s
            else:
                s_audio = silent_stand_in(sentence)
            audio[s_uid] = s_audio
            sentences.append(Line(ANSWER, sentence, 0.0, len(s_audio) / AUDIO_RATE, group=a.group, uid=s_uid))
        delay = answer_delay_s if answer_delay_s is not None else answer_delay(a.source, a.first_ms, tts_s)
        lines += [replace(line, want_t=a.t + delay, asked_t=a.t) for line in sentences]
        if a.route.startswith(f"{RouteKind.DESTINATION.value}/"):  # (only a new route sets guidance in motion)
            routing.append((a.t, a.t + delay))
        if a.image is not None:
            key = f"q{a.group}"
            images[key] = a.image
            caption_image[a.group] = key
    for u in rec.said:
        s_uid = next(uid)
        kind = ALERT if u.kind == "alert" else ANSWER
        if voices is not None:
            s_audio = voices.alert(u.text) if kind == ALERT else voices.answer(u.text)[0]
        else:
            s_audio = silent_stand_in(u.text)
        audio[s_uid] = s_audio
        t = u.t
        if (u.clip or "").startswith("nav") or u.clip == "wrong_lane":  # guidance for a route a question just set
            moved = after_answers(t, routing)
            t = moved + GPS_TICK_S / 2 if moved != t else t  # (on the app's next GPS tick)
        lines.append(
            Line(
                kind,
                u.text,
                t + ALERT_LATENCY_S,
                len(s_audio) / AUDIO_RATE,
                priority=int(u.priority),
                drops_answers=u.drops_answers,
                uid=s_uid,
            )
        )
    placed = schedule(lines)
    captions = build_captions(placed)
    for card in captions:
        if card.kind == ANSWER and card.group in caption_image:
            card.image_key = caption_image[card.group]
    highlights: list[Window] = []
    for a in rec.asked:
        if a.highlight is None:
            continue
        starts = [p.start for p in placed if p.line.kind == ANSWER and p.line.group == a.group]
        if starts:
            highlights.append(Window(min(starts), min(starts) + cfg.ui.highlight_s, a.highlight))
    track = mix(placed, lambda line: audio.get(line.uid, np.zeros(0, np.float32)), AUDIO_RATE, t0, t1)
    summary = [
        {
            "kind": p.line.kind,
            "text": p.line.text,
            "want_t": round(p.line.want_t, 2),
            "start": round(p.start, 2),
            "end": round(p.end, 2),
            "cut": p.cut,
            "priority": p.line.priority,
            "group": p.line.group,
        }
        for p in placed
    ]
    return Timeline(placed, captions, listening, highlights, images, track, summary)


# ------------------------------------------------------------------------------------------------ pass 2


class StateLookup:
    """The recorded frame state nearest a decoded frame's time."""

    def __init__(self, frames: list[FrameState], fps: float) -> None:
        self.frames = frames
        self.ts = [f.t for f in frames]
        self.tolerance = 0.6 / fps
        self.misses = 0

    def at(self, t: float) -> FrameState | None:
        """The state recorded for the frame at t (the latest earlier one if none matches)."""
        if not self.frames:
            return None
        i = bisect_left(self.ts, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.ts)), key=lambda j: abs(self.ts[j] - t))
        if abs(self.ts[best] - t) <= self.tolerance:
            return self.frames[best]
        self.misses += 1
        earlier = max(0, i - 1)
        return self.frames[earlier]


def encode(
    clip: Path,
    out: Path,
    renderer: Renderer,
    lookup: StateLookup,
    timeline: Timeline,
    t_start: float,
    t_end: float,
    rate: Fraction,
    audio: bool,
    pngs: list[float],
    crf: int,
) -> int:
    """Decode the clip from t_start to t_end, draw it at a constant frame rate on the video's own clock (each output
    frame shows the latest source frame at its time, as ffmpeg's -vsync cfr: a clip with irregular timestamps keeps
    its pictures in step with the audio, which is placed in video time) and write the MP4 (+ the audio track).
    Returns the number of frames written."""
    import av

    container = av.open(str(out), "w", container_options={"movflags": "+faststart"})
    vstream = container.add_stream("libx264", rate=rate)
    vstream.width, vstream.height, vstream.pix_fmt = renderer.g.out_w, renderer.g.out_h, "yuv420p"
    vstream.options = {"crf": str(crf), "preset": "medium", "tune": "film"}
    astream = container.add_stream("aac", rate=AUDIO_RATE, layout="mono") if audio else None
    if astream is not None:
        astream.bit_rate = 160_000
    cap = cv2.VideoCapture(str(clip))
    seek = max(0.0, t_start - 1.0)
    if seek > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
    step = 1.0 / float(rate)

    def decoded() -> Iterator[tuple[float, np.ndarray]]:
        while cap.grab():
            t = max(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 0.0)  # (as FileSource)
            if t < t_start - step / 2:
                continue
            ok, image = cap.retrieve()
            if not ok:
                return
            yield t, image

    picker = FramePicker(decoded(), step / 2)
    slots = max(1, round((t_end - t_start) / step))
    written = 0
    audio_pts = 0
    wanted = sorted(pngs)
    try:
        for k in range(slots):
            tau = t_start + k * step
            picked = picker.at(tau)
            if picked is None or (picker.exhausted and tau - picked[0] > 2 * step):
                break  # (the clip ended early)
            t_src, image = picked
            state = lookup.at(t_src)
            hl = active(timeline.highlights, tau)
            frame = renderer.draw(
                image,
                tau,
                state.overlay if state is not None else None,
                state.hud if state is not None else None,
                visible_captions(timeline.captions, tau),
                listening=active(timeline.listening, tau) is not None,
                highlight=hl.value if hl is not None else None,
                images=timeline.images,
            )
            while wanted and tau >= wanted[0] - step / 2:
                png = out.with_name(f"{out.stem}_t{wanted.pop(0):05.1f}.png")
                cv2.imwrite(str(png), frame)
            vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = written
            for packet in vstream.encode(vf):
                container.mux(packet)
            written += 1
            if astream is not None:  # interleave: the audio of this frame's time slot
                chunk = timeline.audio.chunk(tau, t_start + (k + 1) * step)
                af = av.AudioFrame.from_ndarray(chunk[None, :], format="fltp", layout="mono")
                af.sample_rate = AUDIO_RATE
                af.pts = audio_pts
                audio_pts += len(chunk)
                for packet in astream.encode(af):
                    container.mux(packet)
            if written % 100 == 0:
                print(f"  rendered {written} frames ({tau:.1f} s)", flush=True)
        for packet in vstream.encode():
            container.mux(packet)
        if astream is not None:
            for packet in astream.encode():
                container.mux(packet)
    finally:
        cap.release()
        container.close()
    return written


# ------------------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="A video file, or a name (part of one) under data/videos/ or D:/Dashcam.")
    parser.add_argument("--gpx", type=Path, default=None, help="GPX track (default: data/gps/<clip>.gpx if present).")
    parser.add_argument("--no-gpx", action="store_true", help="Don't use a GPX track even if there is one.")
    parser.add_argument("--start", type=float, default=0.0, help="Render from this video time (s).")
    parser.add_argument("--end", type=float, default=None, help="Render until this video time (s; default: the end).")
    parser.add_argument(
        "--warmup", type=float, default=8.0, help="Run perception this long before --start (-1: from the clip start)."
    )
    parser.add_argument("--ask", action="append", default=[], type=parse_ask, metavar="T=QUESTION")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output MP4 (default: runs/showcase/<clip>_<s>-<e>.mp4)."
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width (height keeps the aspect ratio).")
    parser.add_argument("--no-audio", action="store_true", help="Video only (no Piper, no audio track).")
    parser.add_argument("--driver-voice", default="en_US-joe-medium", help="Piper voice of the questions ('' = none).")
    parser.add_argument("--answer-delay", type=float, default=None, help="Fixed question -> answer delay (s).")
    parser.add_argument("--keep-overlay-bar", action="store_true", help="Dashcam: keep the burned-in text bar.")
    parser.add_argument("--no-mask-bar", action="store_true", help="Dashcam: let perception see the text bar.")
    parser.add_argument("--no-lanes", action="store_true")
    parser.add_argument("--no-signs", action="store_true")
    parser.add_argument("--no-nav", action="store_true", help="Without navigation (Mapbox).")
    parser.add_argument(
        "--destination", default=None, help="Navigate there from the first GPS fix (silently: the route is on already)."
    )
    parser.add_argument(
        "--destination-at", type=float, default=-math.inf, metavar="T", help="Set --destination at this video time."
    )
    parser.add_argument("--no-brand", action="store_true", help="No 'DrivePilot' tag top left.")
    parser.add_argument("--title", default=None, help="A label under the tag for the first 5 s (e.g. 'Night · I-90').")
    parser.add_argument(
        "--brighten", type=float, default=1.0, metavar="GAMMA", help="Brighten the picture (night: 1.6; display only)."
    )
    parser.add_argument("--png", action="append", type=float, default=[], metavar="T", help="Also save the frame at T.")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, bigger).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    clip = resolve_clip(args.clip)
    info = probe(clip)
    dashcam = is_dashcam(clip, info.width, info.height)
    t_start = max(0.0, args.start)
    t_end = min(args.end if args.end is not None else info.duration, info.duration)
    if t_end <= t_start:
        raise SystemExit(f"nothing to render: --start {t_start} >= --end {t_end} (clip: {info.duration:.1f} s)")
    t_from = 0.0 if args.warmup < 0 else max(0.0, t_start - args.warmup)
    out = args.out or OUT_DIR / f"{clip.stem}_{t_start:g}-{t_end:g}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    overrides = (list(DASHCAM_OVERRIDES) if dashcam else []) + args.overrides  # (--set wins)
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=overrides)
    setup_logging("INFO", log_file=out.with_suffix(".log"), secrets=cfg.secrets.values())
    gpx_path = None if args.no_gpx else (args.gpx or default_gpx(clip))
    gps = GpxReplay(gpx_path) if gpx_path is not None and gpx_path.is_file() else None
    bar_top = round(DASHCAM_BAR_TOP * info.height / DASHCAM_SIZE[1])  # (a fixed share of the frame, at any size)
    crop = info.height - bar_top if dashcam and not args.keep_overlay_bar else 0
    mask_from = bar_top if dashcam and not args.no_mask_bar else None
    print(
        f"{clip.name}: {info.width}x{info.height} at {info.fps:.2f} fps, {info.duration:.1f} s; rendering "
        f"{t_start:g}-{t_end:g} s (perception from {t_from:g} s), hfov {cfg.effective_hfov_deg:g}°, "
        f"GPX: {gpx_path.name if gps is not None and gpx_path is not None else 'none'}"
        + (", dashcam (text bar cropped)" if crop > 0 else ""),
        flush=True,
    )
    for t, _ in args.ask:
        if not t_from <= t < t_end:
            print(f"  note: question at {t:g} s is outside the processed {t_from:g}-{t_end:g} s", flush=True)

    models = build_models(cfg, lanes=not args.no_lanes, signs=not args.no_signs, nav=not args.no_nav)
    navigation = (
        "on"
        if models.routing is not None and gps is not None
        else "off (needs a GPX track)"
        if models.routing
        else "off"
    )
    print(
        f"models: lanes {'on' if models.lane_model else 'off'}, signs {'on' if models.sign_model else 'off'}, "
        f"vision {models.vision.name if models.vision else 'off'}, navigation {navigation}",
        flush=True,
    )
    keep_share = bar_top / info.height if crop > 0 else None
    rec = run_pass(
        cfg,
        clip,
        models,
        sorted(args.ask),
        t_from,
        t_start,
        t_end,
        gps,
        mask_from,
        args.destination,
        keep_share,
        args.destination_at,
    )
    print(
        f"pass 1: {rec.processed} frames in {rec.seconds:.1f} s ({rec.processed / max(rec.seconds, 1e-6):.1f} fps), "
        f"{len(rec.frames)} recorded, {len(rec.said)} line(s) said on its own",
        flush=True,
    )
    for u in rec.said:
        print(f"  {u.t:6.2f} s  {u.kind}: {u.text}", flush=True)

    voices = None if args.no_audio else Voices(cfg, args.driver_voice or None)
    timeline = build_timeline(cfg, rec, voices, t_start, t_end, args.answer_delay)
    for p in timeline.placed:
        if p.line.kind == QUESTION and p.start < t_start - 1e-6:
            better = math.floor(p.start - 0.5)
            print(f"  note: '{p.line.text}' is spoken from {p.start:.1f} s, before --start (try --start {better:g})")
        if p.line.kind == QUESTION:
            talking = [o for o in timeline.placed if o.line.kind != QUESTION and o.start < p.end and p.start < o.end]
            if talking:
                print(f"  note: '{p.line.text}' is spoken over '{talking[0].line.text[:40]}' (move the question?)")
        if p.end > t_end + 1e-6 and p.start < t_end:
            better = math.ceil(p.end + 1)
            print(f"  note: '{p.line.text[:50]}' plays until {p.end:.1f} s, past --end (try --end {better:g})")
    geometry = make_geometry(info.width, info.height, args.width, crop)
    renderer = Renderer(geometry, brand=not args.no_brand, gamma=args.brighten, title=args.title, title_from=t_start)
    lookup = StateLookup(rec.frames, info.fps)
    rate = output_rate(info.fps)
    start = time.perf_counter()
    written = encode(clip, out, renderer, lookup, timeline, t_start, t_end, rate, not args.no_audio, args.png, args.crf)
    seconds = time.perf_counter() - start
    if lookup.misses:
        log.warning("frames without a matching recorded state", misses=lookup.misses)
    meta = {
        "clip": str(clip),
        "start": t_start,
        "end": t_end,
        "perception_from": t_from,
        "size": [geometry.out_w, geometry.out_h],
        "fps": str(rate),
        "frames": written,
        "audio": not args.no_audio,
        "gpx": str(gpx_path) if gps is not None else None,
        "hfov_deg": cfg.effective_hfov_deg,
        "overrides": overrides,
        "asked": [
            {
                "t": a.t,
                "question": a.question,
                "answer": a.answer,
                "source": a.source,
                "route": a.route,
                "first_ms": None if a.first_ms is None else round(a.first_ms, 1),
                "highlight": a.highlight,
                "image_t": a.image_t,
            }
            for a in rec.asked
        ],
        "said": [{"t": round(u.t, 2), "kind": u.kind, "text": u.text, "priority": u.priority.name} for u in rec.said],
        "timeline": timeline.lines,
        "state_misses": lookup.misses,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"pass 2: {written} frames in {seconds:.1f} s -> {out}", flush=True)
    for line in timeline.lines:
        print(f"  {line['start']:6.2f}-{line['end']:6.2f} s  {line['kind']}: {line['text']}", flush=True)
    return 0 if written > 0 and math.isfinite(seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
