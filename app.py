"""DrivePilot: the V1 app (PLAN.md), everything in one process.

    python -m app                                       # as config.yaml says (video.source: file | camera)
    python -m app --video data/videos/dev/clip.mov      # replay a recorded drive (+ --gpx track.gpx)
    python -m app --camera                              # the phone camera over USB (scrcpy)
    python -m app --no-voice --no-audio                 # typed questions only, answers as text only
    python -m app --ask "22.5=who got here first?"      # typed questions at video times (repeatable)
    python -m app --print-config                        # the resolved config (keys masked)
    python -m app --set vision_model=gemini             # dotted-key overrides, repeatable
    python -m app --camera --wake hey                   # "hey" instead of "Hey DrivePilot" (or: none, no wake word)
    python -m app --camera --destination "Costco"       # navigate there once the phone sends a GPS fix

The screen is at http://localhost:8000/ (video plus the question and answer as text; space = pause, ←/→ = skip
5 s, Q = type a question; the destination box, the wake mode and the spoken alerts are in its side panel). The
phone's GPS page is http://localhost:8000/gps after `adb reverse tcp:8000 tcp:8000`.

Threads: capture → perception (YOLO26s + ByteTrack ∥ ego motion on the GPU) → world state (its own thread) →
alert engine; the world thread hands frames to the lanes thread (YOLOPv2 + lane tracker: drift warnings, the
drivable area for pedestrian warnings) and the sign thread; microphone → wake word → end of question → Whisper
→ assistant → speaker; a housekeeping thread feeds GPS fixes to navigation, asks scripted questions and updates
the page's status line; the web server.
Every run writes a review folder, runs/<time>_<label>/: the log, events (questions, answers, alerts, world
events, GPS fixes, latencies), detections, perf, frames (one a second plus every frame sent to the vision
model) and summary.json (latencies against PLAN.md's targets).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from answers.alerts import LIGHT_GREEN, SAFETY_ALERT_TEXTS, YOUR_TURN
from answers.assistant import CAMERA_LOST, Assistant
from answers.auto_alerts import FIXED_TEXTS as AUTO_ALERT_TEXTS
from answers.auto_alerts import AutoSwitches, SpeedingWatch, WrongLaneWatch
from answers.lanes import CameraLaneReader, VisionLaneReader
from answers.parking import ParkingWatch
from answers.scene import SceneHandler, encode_jpeg
from answers.speech import Priority, Speaker, TextSpeaker, Utterance
from answers.vision import VisionModel
from capture.base import VideoSource
from core.config import WAKE_MODES, AppConfig, ConfigError, load_config
from core.latency import LatencyBook
from core.logging_setup import get_logger, redact, setup_logging, to_jsonable
from core.runlog import RunRecorder
from core.types import Frame
from nav.gps import GpsFix, GpsSource, PhoneGps
from nav.handler import FIXED_ALERTS, NavHandler
from nav.location import LocationContext, Reverser, mapbox_reverser
from nav.mapbox import Place, RoutingClient
from nav.phrasing import in_distance, maneuver_phrase
from perception.detector import Detector, Track
from perception.ego_motion import EgoMotionEstimator
from perception.frame_worker import FrameWorker
from perception.gpu_monitor import GpuMonitor
from perception.lanes import LaneMasks, LaneWorker, lane_view
from perception.pipeline import AsyncListener, PerceptionResult, PerceptionWorker
from perception.sign_worker import SignWorker
from state.lanes import DRIFT_TEXTS, LanesSnapshot, LaneTracker
from state.memory import SceneMemory
from state.pedestrians import DrivableArea
from state.signs import SignTracker
from state.signs import alert_texts as sign_alert_texts
from state.world import WorldEvent, WorldSnapshot, WorldState
from ui.hub import UiHub
from ui.overlay import frame_packet, hud_packet, lane_target

if TYPE_CHECKING:
    from voice.pipeline import HeardQuestion

log = get_logger("app")

TICK_S = 0.1
GPS_INTERVAL_S = 0.5
STATUS_INTERVAL_S = 0.5
STALE_CAMERA_S = 1.5  # a live camera with no processed frame for this long is "lost"
CAMERA_RETRY_S = 3.0  # first wait after a failed reconnect (doubles, up to CAMERA_RETRY_MAX_S)
CAMERA_RETRY_MAX_S = 30.0
CAMERA_HEALTHY_S = 30.0  # running this long after a reconnect resets the backoff
MODEL_START_TIMEOUT_S = 30.0  # start() waits at most this long for the lane model, then the sign model, to build
LANES_FRESH_S = 0.5  # lane detection older than this (vs. the frame shown / the question) isn't used
STALE_AFTER_NAV = {"nav_rerouted", "nav_not_this_exit", "nav_arrived", "nav_roundabout"}
LaneModelFactory = Callable[[], Callable[[Frame], LaneMasks]]
FIXED_ALERT_TEXTS = (  # pre-rendered: play at once
    YOUR_TURN,
    LIGHT_GREEN,
    *SAFETY_ALERT_TEXTS,
    *DRIFT_TEXTS,
    *FIXED_ALERTS,
    *sign_alert_texts(list(range(5, 90, 5))),
    *AUTO_ALERT_TEXTS,
)


def parse_ask(item: str) -> tuple[float, str]:
    """ "22.5=who got here first?" -> (22.5, "who got here first?")."""
    t, sep, question = item.partition("=")
    if not sep or not question.strip():
        raise argparse.ArgumentTypeError(f"expected <seconds>=<question>, got {item!r}")
    return float(t), question.strip()


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _p90(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))], 3)


@dataclass(slots=True)
class QuestionRecord:
    """One question as the run log and summary keep it."""

    qid: int
    text: str
    via: str  # "voice" | "typed" | "scripted"
    t: float  # media time when asked
    route: str = ""
    source: str = ""
    answer: str | None = None
    answer_ms: float | None = None  # routing + answering until the whole answer was handed to the speaker
    first_audio_s: float | None = None  # end of the question → first sound of the answer
    ended_at: float | None = None  # perf_counter when the question ended (the driver stopped talking / pressed Enter)
    voice: dict[str, Any] = field(default_factory=dict)  # wake/end/STT timings for spoken questions


class ScreenSpeaker:
    """Speaks through the real speaker and mirrors everything said to the page and the run log."""

    def __init__(self, inner: Speaker, latency: LatencyBook, hub: UiHub | None, recorder: RunRecorder | None) -> None:
        self.inner = inner
        self.latency = latency
        self.hub = hub
        self.recorder = recorder

    def say(self, utterance: Utterance) -> None:
        """Speak, show and record."""
        self.latency.on_say(utterance.uid, utterance.kind, utterance.clip)
        self.inner.say(utterance)
        if self.hub is not None:
            self.hub.publish(utterance.kind, text=utterance.text, t=round(utterance.t, 2))
        if self.recorder is not None:
            self.recorder.event(
                "alert" if utterance.kind == "alert" else "said",  # "answer" is the question's whole answer
                t=utterance.t,
                text=utterance.text,
                priority=utterance.priority.name,
                clip=utterance.clip,
            )

    def stop(self) -> None:
        """Stop speaking."""
        self.inner.stop()
        if self.hub is not None:
            self.hub.publish("stopped")

    def stop_answers(self) -> None:
        """Stop answering; alerts keep playing."""
        self.inner.stop_answers()
        if self.hub is not None:
            self.hub.publish("stopped")


class DrivePilotApp:
    """Wires capture, perception, world state, the assistant, voice, navigation, the page and the run log."""

    def __init__(
        self,
        cfg: AppConfig,
        make_source: Callable[[], VideoSource],
        detector: Detector,
        speaker: Speaker,
        *,
        ego: EgoMotionEstimator | None = None,
        vision: VisionModel | None = None,
        routing: RoutingClient | None = None,
        gps: GpsSource | None = None,
        recorder: RunRecorder | None = None,
        hub: UiHub | None = None,
        gpu: GpuMonitor | None = None,
        status: dict[str, str] | None = None,
        scripted: Sequence[tuple[float, str]] = (),
        exit_at_end: bool = False,
        lane_model: LaneModelFactory | None = None,
        sign_model: Callable[[], Any] | None = None,
        location: Reverser | None = None,
        reading_vision: VisionModel | None = None,
    ) -> None:
        self.cfg = cfg
        self._make_source = make_source
        self.source = make_source()
        self.detector = detector
        self.ego = ego or EgoMotionEstimator(cfg.ego_motion)
        self.speaker = speaker
        self.vision = vision
        self.routing = routing
        self.gps = gps
        self.recorder = recorder
        self.hub = hub
        self.gpu = gpu
        self.status: dict[str, str] = status if status is not None else {}  # shown on the page; shared
        self.exit_at_end = exit_at_end
        self.latency = LatencyBook()
        self.questions: dict[int, QuestionRecord] = {}
        self._qids = itertools.count(1)
        self._scripted = sorted(scripted)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._answering: set[threading.Thread] = set()
        self.worker: PerceptionWorker | None = None
        self.voice: Any = None  # VoicePipeline, attached with attach_voice()
        self._mic: Any = None
        self.camera_restarts = 0
        self._reconnect_failures = 0  # in a row (backoff)
        self._judged_worker: PerceptionWorker | None = None
        self.camera_state = ""
        self._last_fix_t: float | None = None
        self._last_wake_at = 0.0  # perf_counter of the latest wake word
        self._fps_mark: tuple[float, int] = (time.perf_counter(), 0)
        self._started_at: float | None = None
        # Spoken alerts nobody asked for (alerts.auto): switched on the page; two watches run on the GPS tick
        self.switches = AutoSwitches.from_config(cfg.alerts.auto)
        self.wrong_lane = WrongLaneWatch(cfg.alerts.auto)
        self.speeding = SpeedingWatch(cfg.alerts.auto)
        self.pending_destination: str | None = None  # --destination: set once there's a GPS fix

        # V2-M5: fed by the lanes thread; a wide camera's lane model sees only its center (the crop's view)
        self.lanes = LaneTracker(cfg.lanes, lane_view(cfg.effective_hfov_deg, cfg.lanes.max_view_deg)[1])
        self._lane_model = lane_model  # builds YOLOPv2 on the lanes thread (None: lanes off)
        self._sign_model = sign_model  # builds the sign detector on the signs thread (None: sign alerts off)
        self.lane_worker: LaneWorker | None = None
        self._drivable: DrivableArea | None = None  # the lane model's latest drivable area (pedestrian warnings)
        self._segment = 0  # the video segment the world thread is in (lane results from an older one are dropped)
        self.world = WorldState(
            cfg,
            on_event=self._on_world_event,
            speed_source=self._gps_speed_at,
            drivable_source=self._current_drivable,
        )
        self.memory = SceneMemory(cfg.memory)  # V2: "what was that sign?"
        self.signs = SignTracker(cfg.sign_alerts)  # V2-M6: what the sign detector read (limits, school zones)
        self.sign_worker: SignWorker | None = None
        self._latest_tracks: Sequence[Track] = ()
        self._last_result: PerceptionResult | None = None  # the latest frame the world state took in
        self._highlight: tuple[int, float] | None = None  # (track id, until perf_counter): pulsed on the page
        self._image_ids = itertools.count(1)
        self._world_thread = AsyncListener(self._world_update, name="world", max_pending=2)
        self.screen = ScreenSpeaker(speaker, self.latency, hub, recorder)
        # Where we are, for the vision model (looked up in the background from the GPS; see nav/location.py)
        self.location = (
            LocationContext(location, self.gps_fix, cfg.vision.location_refresh_m) if location is not None else None
        )
        self.nav: NavHandler | None = None
        if routing is not None:
            vision_reader = (
                VisionLaneReader(vision, self._latest_frame, cfg.vision, on_frame=self._save_model_frame)
                if vision is not None
                else None
            )
            lane_reader = CameraLaneReader(lambda: self.lanes_view(self.media_time()), vision_reader)
            self.nav = NavHandler(
                cfg.nav,
                routing,
                self.gps_fix,
                self._on_nav_alert,
                lane_reader,
                turn_prompts=lambda: self.switches.enabled("turns"),
            )
        self.scene = (
            SceneHandler(
                cfg,
                vision,
                self._model_frames,
                self._extra_context,
                tracks=lambda: self._latest_tracks,
                memory=self.memory,
                on_highlight=self.highlight,
                on_image=self._show_image,
                reading_model=reading_vision,
            )
            if vision is not None
            else None
        )
        self.assistant = Assistant(
            cfg, self._snapshot, self.screen, nav=self.nav, model=self.scene, unavailable=self._unavailable
        )
        self.assistant.on_highlight = self.highlight
        self.assistant.alerts.switches = self.switches
        self.parking = ParkingWatch(  # V2: "find me a parking spot"
            cfg.parking,
            cfg.vision,
            vision,
            self.assistant.say,
            self._snapshot,
            self.world.set_parking_mode,
            self.media_time,
            expect_reply=self._expect_reply,
            quiet=lambda: self.assistant.alerts.quiet,
        )
        self.assistant.parking = self.parking

    # ------------------------------------------------------------------------------------ wiring

    def speaking(self) -> bool:
        """DrivePilot is talking (or about to): the open-mic wake modes ignore the microphone meanwhile."""
        speaking = getattr(self.speaker, "speaking", None)
        return bool(speaking()) if callable(speaking) else False

    def attach_voice(self, pipeline: Any, mic: Any) -> None:
        """Use a `VoicePipeline` (built with this app's `on_heard` / `on_wake` / `on_nothing`) on `mic`."""
        self.voice, self._mic = pipeline, mic

    def _expect_reply(self, seconds: float) -> None:
        """DrivePilot asked a yes/no question (the parking-lot offer): once it has finished saying it (the
        microphone must not hear our own voice), listen a few seconds for a plain answer, no wake word needed."""

        def run() -> None:
            wait_idle = getattr(self.speaker, "wait_idle", None)
            if callable(wait_idle):
                wait_idle(8.0)
            if self.voice is not None and not self._stop.is_set():
                self.voice.expect_reply(seconds)
                if self.hub is not None:
                    self.hub.publish("listening")

        threading.Thread(target=run, name="expect-reply", daemon=True).start()

    def media_time(self) -> float:
        """The pipeline clock: video time in file mode, wall time in camera mode."""
        return self.source.clock.now()

    def gps_fix(self) -> GpsFix | None:
        """The current GPS fix (GPX replay at video time, or the phone's latest)."""
        return self.gps.fix_at(self.media_time()) if self.gps is not None else None

    def _gps_speed_at(self, t: float) -> float | None:
        """Our GPS speed (m/s) at media time t, None without a fresh fix (the world state's parking tracker)."""
        fix = self.gps.fix_at(t) if self.gps is not None else None
        return None if fix is None else fix.speed_mps

    def _camera_lost(self) -> bool:
        worker = self.worker
        return self.source.is_live and (worker is None or worker.stale(STALE_CAMERA_S)) and self._started_at is not None

    def _snapshot(self) -> WorldSnapshot | None:
        return None if self._camera_lost() else self.world.snapshot()

    def _unavailable(self) -> str | None:
        return CAMERA_LOST if self._camera_lost() else None

    def _latest_frame(self) -> Frame | None:
        if self._camera_lost():
            return None
        snap = self.world.snapshot()
        return snap.frame if snap is not None else None

    def _model_frames(self, count: int, spacing_s: float) -> list[Frame]:
        frames = self.world.recent_frames(count, spacing_s)
        for frame in frames:
            self._save_model_frame(frame)
        return frames

    def _save_model_frame(self, frame: Frame) -> None:
        if self.recorder is not None:
            self.recorder.save_frame(frame.image, f"model_t{frame.t:08.2f}")

    def _make_worker(self) -> PerceptionWorker:
        worker = PerceptionWorker(self.source, self.detector, self.ego, recorder=self.recorder, gpu=self.gpu)
        worker.add_listener(self._world_thread)
        if self.hub is not None and not self.cfg.ui.overlays:  # (V2: the world thread offers frame + overlay)
            worker.add_listener(lambda result: self.hub.offer_frame(result.frame) if self.hub else None)
        return worker

    # ----------------------------------------------------------------------------- world + alerts

    def _world_update(self, result: PerceptionResult) -> None:
        """World-state thread: update, and time any alert it raises from this frame's capture. Then, for the V2
        page, hand the frame to the screen with the overlay built from this very update."""
        if result.new_segment:  # (before the world update: its pedestrian check must not see the old segment's road)
            self._segment = result.frame.segment
            self.lanes.reset()
            self._drivable = None
        with self.latency.detection(result.frame.captured_at):
            self.world.on_perception(result)
        self._latest_tracks = tuple(result.tracks)
        try:
            if result.new_segment:
                self.memory.reset()
            self.memory.add(result.frame, result.tracks)
        except Exception:
            log.exception("scene memory failed")
        self._last_result = result
        if self.sign_worker is not None:
            if result.new_segment:
                self.signs.reset()
            self.sign_worker.offer(result.frame)
        if self.lane_worker is not None:
            self.lane_worker.offer(result.frame)
        if self.hub is not None and self.cfg.ui.overlays:
            self.hub.offer_frame(result.frame, self._overlay(result))

    def _overlay(self, result: PerceptionResult) -> dict[str, Any] | None:
        """The V2 page's overlay for this result (with the highlight, and when its pulse ends on the wall clock)."""
        frame = result.frame
        highlight = self._highlight
        left = highlight[1] - time.perf_counter() if highlight is not None else 0.0
        lanes = self.lanes_view(frame.t)
        try:
            overlay = frame_packet(
                self.world.snapshot(),
                result.tracks,
                frame.width,
                frame.height,
                parking_active=self.parking.active,
                highlight=highlight[0] if highlight is not None and left > 0 else None,
                lanes=lanes,
                lane_goal=lane_target(lanes, self.nav.progress if self.nav is not None and self.nav.route else None),
            )
        except Exception:
            log.exception("overlay failed")
            return None
        if left > 0:
            overlay["pulse_until"] = round(time.time() + left, 3)  # (the page stops pulsing then, even when paused)
        return overlay

    def _on_world_event(self, event: WorldEvent) -> None:
        if self.recorder is not None:
            self.recorder.event("world", t=event.t, event=event.kind, **event.detail)
        self.assistant.on_world_event(event)

    def _on_nav_alert(self, kind: str, text: str) -> None:
        # Guidance the driver asked for by setting a destination: kept in quiet mode. A new route, an exit to
        # skip or arrival make an answer in progress (about the old route) stale.
        self.assistant.say(
            Utterance(
                text=text,
                kind="alert",
                priority=Priority.REQUESTED,
                clip=kind,
                t=self.media_time(),
                drops_answers=kind in STALE_AFTER_NAV,
            )
        )

    # ---------------------------------------------------------------------------------- questions

    def on_heard(self, heard: HeardQuestion) -> None:
        """A transcribed spoken question (`HeardQuestion`, voice answer thread)."""
        if heard.reply and not self.assistant.takes_reply(heard.text):
            self.on_nothing("reply")  # not a yes/no to the offer (a passenger?): ignored
            return
        lag = heard.end_t - heard.speech_end_t if heard.speech_end_t is not None else 0.0
        origin = heard.end_detected_at - max(lag, 0.0)  # when the driver stopped talking
        timings = {
            "wake_t": round(heard.wake_t, 2),
            "endpoint_s": round(lag, 3),
            "stt_ms": round(heard.stt_ms),
            "speculative": heard.speculative,
            "transcript_s": round(heard.transcript_at - origin, 3),
        }
        if self._last_wake_at > origin:
            # The driver said the wake word again after this question ended (it took long to transcribe): they've
            # moved on, and an answer now would talk over the new question.
            self._superseded(heard.text, origin, timings)
            return
        with self._lock:
            self._answering.add(threading.current_thread())  # the end of a replay waits for it
        self._answer(heard.text, "voice", origin, timings)

    def _superseded(self, text: str, origin: float, timings: dict[str, Any]) -> None:
        qid = next(self._qids)
        t = self.media_time()
        record = QuestionRecord(qid, text, "voice", t, source="superseded", voice=timings, ended_at=origin)
        with self._lock:
            self.questions[qid] = record
        log.info("question superseded by a newer one", text=text)
        if self.recorder is not None:
            self.recorder.event("question", t=t, qid=qid, text=text, via="voice", superseded=True, **timings)
        if self.hub is not None:
            # `superseded`: the page keeps its listening light on (the driver is asking the newer question now).
            self.hub.publish("question", text=text, via="voice", t=round(t, 2), superseded=True)
            self.hub.publish("note", text="(skipped: you asked something else)")

    def on_wake(self) -> None:
        """The wake word fired: stop talking (so "Hey DrivePilot, stop" works) and show that we're listening."""
        self._last_wake_at = time.perf_counter()
        self.assistant.interrupt()
        listening = getattr(self.speaker, "listening", None)
        if callable(listening):
            listening()
        if self.hub is not None:
            self.hub.publish("listening")

    def on_nothing(self, reason: str) -> None:
        """The wake word fired but no question followed."""
        if self.hub is not None:
            self.hub.publish("nothing", reason=reason)

    def ask_typed(self, text: str, via: str = "typed") -> None:
        """A typed question (page, scripted): answered on its own thread."""
        thread = threading.Thread(
            target=self._answer, args=(text, via, time.perf_counter(), {}), name="answer-typed", daemon=True
        )
        with self._lock:
            self._answering.add(thread)
        thread.start()

    def _answer(self, text: str, via: str, origin: float, timings: dict[str, Any]) -> None:
        qid = next(self._qids)
        t = self.media_time()
        record = QuestionRecord(qid, text, via, t, voice=timings, ended_at=origin)
        with self._lock:
            self.questions[qid] = record
        if self.hub is not None:
            self.hub.publish("question", text=text, via=via, t=round(t, 2))
        if self.recorder is not None:
            self.recorder.event("question", t=t, qid=qid, text=text, via=via, **timings)
        try:
            with self.latency.question(qid, origin):
                reply = self.assistant.ask(text)
        finally:
            with self._lock:
                self._answering.discard(threading.current_thread())
        record.route = f"{reply.route.kind.value}/{reply.route.intent}"
        record.source, record.answer, record.answer_ms = reply.source, reply.text, round(reply.elapsed_ms, 1)
        if self.recorder is not None:
            self.recorder.event(
                "answer",
                t=t,
                qid=qid,
                text=reply.text,
                route=record.route,
                source=reply.source,
                ms=record.answer_ms,
                first_ms=None if reply.first_ms is None else round(reply.first_ms, 1),
            )
        if reply.text is None and self.hub is not None:
            self.hub.publish("nothing", reason="no answer")

    def on_speech_record(self, record: Any) -> None:
        """The audio speaker played (or dropped) an utterance (`SpeechRecord`): measure and log its latency."""
        measured = self.latency.on_played(record.uid, record.sound_at)
        if measured is None:
            return
        if measured.kind == "answer":
            with self._lock:
                question = self.questions.get(int(measured.key))
            if question is not None:
                question.first_audio_s = round(measured.seconds, 3)
        if self.recorder is not None:
            self.recorder.event("latency", what=measured.kind, key=measured.key, ms=round(measured.seconds * 1000.0, 1))

    # -------------------------------------------------------------------------------- page actions

    def toggle_pause(self) -> bool | None:
        """Space: pause/resume a replay (None for the live camera)."""
        if self.source.is_live:
            return None
        paused = self.source.toggle_pause()
        if self.recorder is not None:
            self.recorder.event("pause" if paused else "resume", t=self.media_time())
        return paused

    def seek(self, delta_s: float) -> bool:
        """←/→: skip a replay (False for the live camera)."""
        if self.source.is_live:
            return False
        self.source.seek_relative(delta_s)
        if self.recorder is not None:
            self.recorder.event("seek", t=self.media_time(), delta_s=delta_s)
        return True

    def gps_fix_from_phone(
        self, lat: float, lon: float, speed: float | None, heading: float | None, accuracy: float | None
    ) -> bool:
        """A fix from the phone page; False unless phone GPS is in use."""
        if not isinstance(self.gps, PhoneGps):
            return False
        self.gps.push(lat, lon, speed, heading, accuracy)
        return True

    def set_wake_mode(self, mode: str) -> bool:
        """The page's wake-mode switch: "hey_drivepilot", "hey" or "none". False without voice input."""
        if mode not in WAKE_MODES:
            raise ValueError(f"unknown wake mode {mode!r}")
        if self.voice is None:
            return False
        self.voice.set_mode(mode)
        if self.recorder is not None:
            self.recorder.event("wake_mode", t=self.media_time(), mode=mode)
        self._publish_status()
        return True

    def set_alert(self, name: str, on: bool) -> None:
        """The page's switch for one unrequested alert (answers/auto_alerts.py: AUTO_NAMES)."""
        self.switches.set(name, on)
        log.info("alert switch", alert=name, on=on)
        if self.recorder is not None:
            self.recorder.event("alert_switch", t=self.media_time(), alert=name, on=on)
        self._publish_status()

    def suggest_places(self, query: str) -> list[dict[str, Any]]:
        """Places for the page's destination box as the driver types (best match first)."""
        if self.nav is None:
            return []
        return [
            {"name": p.name, "address": p.address, "lat": p.location[0], "lon": p.location[1], "kind": p.kind}
            for p in self.nav.suggest(query)
        ]

    def set_destination(self, query: str, place: Place | None = None) -> bool:
        """The page's destination box: navigate to what was typed (or the suggestion picked), answered aloud like
        "I'm going to ..." (on its own thread: geocoding and routing take a moment). False without navigation."""
        if self.nav is None:
            return False
        thread = threading.Thread(target=self._destination, args=(query, place), name="destination", daemon=True)
        with self._lock:
            self._answering.add(thread)
        thread.start()
        return True

    def cancel_route(self) -> bool:
        """The page's "End route" button."""
        if self.nav is None:
            return False
        self._say_answer(self.nav.cancel())
        self._publish_status()
        return True

    def _destination(self, query: str, place: Place | None) -> None:
        text = f"Take me to {place.name if place is not None else query}"
        qid = next(self._qids)
        t = self.media_time()
        record = QuestionRecord(qid, text, "page", t, route="destination/page", source="nav")
        with self._lock:
            self.questions[qid] = record
        if self.hub is not None:
            self.hub.publish("question", text=text, via="typed", t=round(t, 2))
        if self.recorder is not None:
            self.recorder.event("question", t=t, qid=qid, text=text, via="page")
        start = time.perf_counter()
        try:
            nav = self.nav
            answer = nav.set_destination(query, place) if nav is not None else "Navigation isn't available right now."
        except Exception:
            log.exception("destination failed", query=query)
            answer = "Sorry, something went wrong setting the destination."
        finally:
            with self._lock:
                self._answering.discard(threading.current_thread())
        record.answer, record.answer_ms = answer, round((time.perf_counter() - start) * 1000.0, 1)
        if self.recorder is not None:
            self.recorder.event("answer", t=t, qid=qid, text=answer, route=record.route, source="nav")
        self._say_answer(answer)
        self._publish_status()

    def _say_answer(self, text: str) -> None:
        self.assistant.say(Utterance(text, "answer", Priority.ANSWER, t=self.media_time()))

    # ------------------------------------------------------------------------------- housekeeping

    def _housekeeping(self) -> None:
        next_gps = next_status = 0.0
        while not self._stop.wait(TICK_S):
            now = time.perf_counter()
            try:
                self._ask_scripted()
                if now >= next_gps:
                    next_gps = now + GPS_INTERVAL_S
                    self._feed_gps()
                    if self.location is not None:
                        self.location.update()
                    self._auto_alerts()
                    self._start_pending_destination()
                if now >= next_status:
                    next_status = now + STATUS_INTERVAL_S
                    self._publish_status()
            except Exception:
                log.exception("housekeeping failed")

    def _auto_alerts(self) -> None:
        """The wrong-lane and speeding watches (answers/auto_alerts.py), every GPS tick. They always follow the
        state, so switching one on doesn't replay an old condition; only what is switched on is said."""
        t = self.media_time()
        fix = self.gps_fix()
        nav = self.nav
        progress = nav.progress if nav is not None and nav.route is not None and fix is not None else None
        target = lane_target(self.lanes_view(t), progress)
        step = progress.next_step.index if progress is not None and progress.next_step is not None else None
        text = self.wrong_lane.update(t, target, step, progress.to_next_m if progress is not None else None)
        if text is not None and self.switches.enabled("wrong_lane"):
            self._say_auto("wrong_lane", text, Priority.REQUESTED)  # (guidance for the route the driver set)
        signs = self.signs.snapshot()
        text = self.speeding.update(t, fix.speed_mps if fix is not None else None, signs.limit, signs.limit_t)
        if text is not None and self.switches.enabled("speeding"):
            self._say_auto("speeding", text, Priority.INFO)

    def _say_auto(self, kind: str, text: str, priority: Priority) -> None:
        if priority < Priority.REQUESTED and self.assistant.alerts.quiet:
            log.info("alert suppressed: quiet mode", kind=kind)
            return
        log.info("alert", kind=kind, text=text, priority=priority.name)
        self.assistant.say(Utterance(text, "alert", priority, clip=kind, t=self.media_time()))

    def _start_pending_destination(self) -> None:
        """--destination: navigate there as soon as there's a GPS fix."""
        if self.pending_destination is None or self.nav is None or self.gps_fix() is None:
            return
        query, self.pending_destination = self.pending_destination, None
        self.set_destination(query)

    def _ask_scripted(self) -> None:
        if not self._scripted or self.worker is None or self.worker.processed == 0:
            return
        t = self.media_time()
        while self._scripted and t >= self._scripted[0][0]:
            _, question = self._scripted.pop(0)
            self.ask_typed(question, via="scripted")

    def _feed_gps(self) -> None:
        fix = self.gps_fix()
        if fix is None or fix.t == self._last_fix_t:
            return
        self._last_fix_t = fix.t
        if self.recorder is not None:
            self.recorder.event(
                "gps", t=self.media_time(), lat=round(fix.lat, 6), lon=round(fix.lon, 6), speed=fix.speed_mps
            )
        if self.nav is not None:
            self.nav.on_fix(fix)

    def _nav_line(self) -> str | None:
        nav = self.nav
        if nav is None or nav.route is None or nav.progress is None:
            return None
        progress = nav.progress
        if progress.next_step is None or progress.to_next_m is None:
            return None
        return f"{maneuver_phrase(progress.next_step)} {in_distance(progress.to_next_m)}"

    def _gps_state(self) -> str:
        if self.gps is None:
            return "off"
        return "ok" if self.gps_fix() is not None else "no fix"

    def _publish_status(self) -> None:
        if self.hub is None:
            return
        worker = self.worker
        now, processed = time.perf_counter(), worker.processed if worker is not None else 0
        then, before = self._fps_mark
        fps = (processed - before) / (now - then) if now > then and processed >= before else 0.0
        self._fps_mark = (now, processed)
        for name, model in (("lanes", self.lane_worker), ("signs", self.sign_worker)):
            if model is not None and model.error is not None:
                self.status[name] = "failed (see the log)"  # the thread couldn't build its model
        self.hub.set_status(
            t=round(self.media_time(), 1),
            live=self.source.is_live,
            paused=bool(getattr(self.source, "paused", False)),
            fps=round(fps, 1),
            camera=self.camera_state or ("end of the replay" if getattr(self.source, "at_end", False) else None),
            gps=self._gps_state(),
            nav=self._nav_line(),
            quiet=self.assistant.alerts.quiet,
            hud=self._hud(),
            wake_mode=self.voice.mode if self.voice is not None else None,
            auto=self.switches.as_dict(),
            route=self._route_status(),
            **self.status,
        )

    def _route_status(self) -> dict[str, Any] | None:
        """The page's destination box: where we're going and how far; None when not navigating, {"off": True}
        when navigation isn't available (no Mapbox token)."""
        nav = self.nav
        if nav is None:
            return {"off": True}
        place, progress = nav.place, nav.progress
        if nav.route is None or place is None:
            return None
        out: dict[str, Any] = {"name": place.name, "address": place.address}
        if progress is not None:
            out["minutes"] = max(1, round(progress.remaining_s / 60))
            out["miles"] = round(progress.remaining_m / 1609.344, 1)
        return out

    def highlight(self, track_id: int) -> None:
        """Pulse a tracked object on the page for `ui.highlight_s` (an answer is about it)."""
        self._highlight = (track_id, time.perf_counter() + self.cfg.ui.highlight_s)
        if self.recorder is not None:
            self.recorder.event("highlight", t=self.media_time(), track_id=track_id)
        last = self._last_result
        if self.hub is not None and self.cfg.ui.overlays and last is not None:
            self.hub.offer_frame(last.frame, self._overlay(last))  # (a paused replay sends no new frames)

    def _show_image(self, frame: Frame) -> None:
        """A remembered frame an answer came from: shown on the page as a thumbnail."""
        if self.hub is None:
            return
        image_id = next(self._image_ids)
        self.hub.put_image(image_id, encode_jpeg(frame.image, 480, 80))
        self.hub.publish("thumbnail", image=image_id, t=round(frame.t, 1))

    def _extra_context(self) -> list[str]:
        """What the vision model's prompt adds: navigation and speed, what the sign detector read, and the lane
        count and our lane from lane detection (when it is sure)."""
        lines = self.nav.extra_context() if self.nav is not None else []
        if self.location is not None:
            lines += self.location.context()
        t = self.media_time()
        lanes = self.lanes.context() if self.lanes_view(t) is not None else []
        return lines + self.signs.context(t) + lanes

    def _current_drivable(self) -> DrivableArea | None:
        """The lane model's latest drivable area, if it is from the current video segment (world thread)."""
        road = self._drivable
        return road if road is not None and road.segment == self._segment else None

    def lanes_view(self, t: float) -> LanesSnapshot | None:
        """Lane detection's latest, if it saw our lane at most LANES_FRESH_S before media time `t`."""
        snap = self.lanes.snapshot()
        return snap if snap.valid and 0.0 <= t - snap.t <= LANES_FRESH_S else None

    def _hud(self) -> dict[str, Any] | None:
        """The V2 page's banner, speed, speed limit and roundabout diagram."""
        try:
            fix = self.gps_fix()
            progress = self.nav.progress if self.nav is not None and self.nav.route is not None else None
            signs = self.signs.snapshot()
            school_t = signs.school_limit_t
            recent = school_t is not None and self.media_time() - school_t <= self.cfg.sign_alerts.repeat_s
            school = signs.school_limit if recent else None  # (a school limit applies near the school only)
            return hud_packet(progress, fix.speed_mps if fix is not None else None, signs.limit, school)
        except Exception:
            log.exception("hud failed")
            return None

    # ---------------------------------------------------------------------------------- lifecycle

    def _start_signs(self) -> None:
        """The fine-tuned sign detector (V2-M6) on its own thread, if the app was given it (`sign_model`, see
        `sign_model_factory`); otherwise sign alerts are off and everything else runs."""
        if self._sign_model is None:
            self.status.setdefault("signs", "off")
            return
        self.sign_worker = SignWorker(self._sign_model, self._on_signs, self.cfg.sign_alerts.fps).start()
        self.status["signs"] = "on"

    def _start_lanes(self) -> None:
        """YOLOPv2 lanes (V2-M5) on their own thread, if the app was given the model (`lane_model`, see
        `lane_model_factory`): our lane on the page, the lane count for lane guidance, drift warnings, the drivable
        area for pedestrian warnings."""
        if self._lane_model is None:
            self.status.setdefault("lanes", "off")
            return
        self.lane_worker = LaneWorker(self._lane_model, self._on_lanes, self.cfg.lanes.fps).start()
        self.status["lanes"] = "on"

    def _on_lanes(self, frame: Frame, masks: LaneMasks) -> None:
        """Lanes thread: the masks feed the lane tracker (drift warnings: safety, spoken at once) and the
        pedestrian check's drivable area."""
        if frame.segment != self._segment:
            return  # a frame from before a seek / loop restart, finished after it: its lanes and road are stale
        self._drivable = DrivableArea(masks.t, masks.drivable, frame.segment, masks.view)
        for event in self.lanes.update(frame.t, masks, self._gps_speed_at(frame.t)):
            if self.recorder is not None:
                self.recorder.event(
                    "drift",
                    t=event.t,
                    side=event.side,
                    gap_m=round(event.gap_m, 2),
                    lateral_mps=round(event.lateral_speed_mps, 2),
                    offset_m=round(event.offset_m, 2),
                    habit_m=round(event.habit_m, 2),
                    speed_mps=round(event.speed_mps, 1),
                )
            self.assistant.say(Utterance(event.text, "alert", Priority.SAFETY, clip="drift", t=event.t))

    def _on_signs(self, frame: Frame, detections: Sequence[Any]) -> None:
        """Sign thread: what the detector read becomes alerts, memory close-ups and the page's limit badge."""
        events = self.signs.update(frame.t, detections)
        for view in self.signs.last_frame:
            text = f"{view.value} mph" if view.value is not None else None
            self.memory.add_sign(frame, view.id, view.kind, view.box, text=text)
        quiet = self.assistant.alerts.quiet  # (sign alerts are INFO: muted in quiet mode, like "you're up")
        for event in events:
            if self.recorder is not None:
                self.recorder.event("sign", t=event.t, sign=event.kind, value=event.value, text=event.text, quiet=quiet)
            if not quiet:
                self.assistant.say(Utterance(event.text, "alert", Priority.INFO, t=event.t))

    @staticmethod
    def _await_model(worker: FrameWorker[Any] | None) -> bool:
        """Wait until a model thread has built its model (or failed to); False if it is still building."""
        if worker is not None and not worker.ready.wait(MODEL_START_TIMEOUT_S):
            log.warning("model still starting; the app goes on anyway", worker=worker.name)
            return False
        return True

    def start(self) -> None:
        """Start every thread."""
        if self.hub is not None:
            self.hub.start()
        # Build the lane and sign models one at a time, each before the detector starts running: both capture CUDA
        # graphs while they load (the lane model's own; ultralytics' TensorRT loader one per engine load, 17 for
        # the sign model, in global capture mode), and GPU work from another thread during a capture breaks it
        # (seen in the bench: built side by side, both models failed to start). The first seconds of a drive
        # shouldn't lose frames to model warm-ups either. Nothing reloads an engine after this (checked).
        self._start_lanes()
        if self._await_model(self.lane_worker):
            self._start_signs()
            self._await_model(self.sign_worker)
        elif self._sign_model is not None:  # building it now would break the lane model's capture (and its own)
            log.warning("sign detector not started: the lane model is still loading")
            self.status["signs"] = "off (lane model still loading)"
        self._world_thread.start()
        self.source.start()
        self.worker = self._make_worker()
        self.worker.start()
        self._started_at = time.perf_counter()
        housekeeping = threading.Thread(target=self._housekeeping, name="housekeeping", daemon=True)
        housekeeping.start()
        self._threads.append(housekeeping)
        if self.voice is not None and self._mic is not None:
            self._mic.start()
            self.voice.start(self._mic)
        if self.recorder is not None:
            self.recorder.event("start", source=self.source.name, live=self.source.is_live, status=dict(self.status))
        log.info("running", source=self.source.name, **self.status)

    def run(self) -> int:
        """Start, supervise (restart a lost camera; stop at the end of a file if asked) and stop. Exit code."""
        self.start()
        code = 0
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
                worker = self.worker
                if worker is None or worker.running:
                    continue
                if worker.error is not None:
                    if self.source.is_live:  # keep trying: voice, navigation and the page keep working meanwhile
                        self._restart_camera()
                        continue
                    log.error("perception stopped", error=repr(worker.error))
                    code = 1
                    break
                if self.exit_at_end:  # the file ended
                    self._finish_answers()
                    break
                if self.camera_state != "replay finished":
                    self.camera_state = "replay finished"
                    log.info("replay finished; the page stays up until Ctrl+C")
        except KeyboardInterrupt:
            log.info("stopping (Ctrl+C)")
        finally:
            self.stop()
        return code

    def _restart_camera(self) -> None:
        self.camera_restarts += 1
        self.camera_state = "camera lost, reconnecting"
        log.warning("camera lost; restarting", attempt=self.camera_restarts)
        if self.recorder is not None:
            self.recorder.event("camera_restart", attempt=self.camera_restarts)
        worker = self.worker
        if worker is not self._judged_worker:  # how long did the camera last since the previous reconnect?
            self._judged_worker = worker
            ran_s = (
                worker.last_result_at - worker.started_at
                if worker is not None and worker.started_at is not None and worker.last_result_at is not None
                else 0.0
            )
            if ran_s >= CAMERA_HEALTHY_S:
                self._reconnect_failures = 0
            elif self.camera_restarts > 1:
                self._reconnect_failures += 1
        if self._reconnect_failures:  # wait longer after each reconnect that failed or didn't last
            self._stop.wait(min(CAMERA_RETRY_S * 2 ** (self._reconnect_failures - 1), CAMERA_RETRY_MAX_S))
        self.source.stop()
        try:
            source = self._make_source()
            source.start()
        except Exception as exc:  # noqa: BLE001 - any failure to reconnect: try again shortly
            self._reconnect_failures += 1
            log.warning("camera restart failed", error=str(exc), failures=self._reconnect_failures)
            return
        self.source = source
        self.worker = self._make_worker()
        self.worker.start()
        self.camera_state = ""

    def _finish_answers(self, timeout: float = 30.0) -> None:
        """Let answers in progress finish and be spoken (end of a scripted replay)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                busy = [t for t in self._answering if t.is_alive()]
            if not busy:
                break
            time.sleep(0.05)
        wait_idle = getattr(self.speaker, "wait_idle", None)
        if callable(wait_idle):
            wait_idle(max(0.0, deadline - time.perf_counter()))

    def stop(self) -> None:
        """Stop every thread and write the summary."""
        self._stop.set()
        self.parking.close()
        if self.sign_worker is not None:
            self.sign_worker.stop()
        if self.lane_worker is not None:
            self.lane_worker.stop()
        if self.voice is not None:
            self.voice.stop()
        if self._mic is not None:
            self._mic.stop()
        if self.worker is not None:
            self.worker.stop()
        self._world_thread.stop()
        self.source.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        close = getattr(self.speaker, "close", None)
        if callable(close):
            close()
        if self.hub is not None:
            self.hub.stop()
        summary = self.summary()
        if self.recorder is not None:
            text = redact(json.dumps(to_jsonable(summary), indent=2))
            (self.recorder.dir / "summary.json").write_text(text, encoding="utf-8")
        log.info("stopped", questions=len(self.questions), answers_timed=len(self.latency.answers))

    def summary(self) -> dict[str, Any]:
        """Latencies against PLAN.md's targets, plus perception and voice counters."""
        with self._lock:
            questions = list(self.questions.values())
        by_source: dict[str, list[float]] = {}
        for q in questions:
            if q.first_audio_s is not None and q.via == "voice":
                by_source.setdefault(q.source, []).append(q.first_audio_s)
        alerts = [m.seconds * 1000.0 for m in self.latency.alerts if not m.key.startswith("nav_")]
        worker = self.worker
        perception: dict[str, Any] = {}
        if worker is not None and worker.started_at is not None:
            elapsed = (worker.finished_at or time.perf_counter()) - worker.started_at
            stages = worker.run_stats.summary()
            perception = {
                "frames": worker.processed,
                "fps": round(worker.processed / elapsed, 1) if elapsed > 0 else None,
                "processing_p95_ms": round(stages.get("processing", {}).get("p95", float("nan")), 1),
                "latency_p95_ms": round(stages.get("latency", {}).get("p95", float("nan")), 1),
                "source_dropped": self.source.stats.dropped,
                "frame_errors": worker.frame_errors,
            }
        voice: dict[str, Any] = {}
        if self.voice is not None:
            voice = {"wakes": self.voice.wakes, "timeouts": self.voice.timeouts, "empty": self.voice.empty_transcripts}
        return {
            "questions": [
                {
                    "qid": q.qid,
                    "t": round(q.t, 2),
                    "via": q.via,
                    "text": q.text,
                    "route": q.route,
                    "source": q.source,
                    "answer": q.answer,
                    "answer_ms": q.answer_ms,
                    "first_audio_s": q.first_audio_s,
                    **q.voice,
                }
                for q in questions
            ],
            "first_audio_s": {
                source: {"n": len(values), "median": _median(values), "p90": _p90(values)}
                for source, values in sorted(by_source.items())
            },
            "alert_ms": {"n": len(alerts), "median": _median(alerts), "max": round(max(alerts), 1) if alerts else None},
            "perception": perception,
            "voice": voice,
            "camera_restarts": self.camera_restarts,
            "models": {  # the lanes and sign threads: frames a second, last model time
                name: {"fps": model.rate(), "frames": model.processed, "last_ms": model.last_ms}
                for name, model in (("lanes", self.lane_worker), ("signs", self.sign_worker))
                if model is not None
            },
            "status": self.status,
        }


# ------------------------------------------------------------------------------------------- building


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(prog="python -m app", description="DrivePilot dashcam voice co-pilot")
    parser.add_argument("--config", type=Path, default=None, help="config file (default: repo config.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set video.source=camera (repeatable)",
    )
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, default=None, help="replay this video file (video.source=file)")
    source.add_argument("--camera", action="store_true", help="use the phone camera (video.source=camera)")
    parser.add_argument("--gpx", type=Path, default=None, help="GPX track recorded with --video")
    parser.add_argument("--no-voice", action="store_true", help="no microphone: typed questions only")
    parser.add_argument("--no-audio", action="store_true", help="no speech output: answers on the page only")
    parser.add_argument("--no-ui", action="store_true", help="don't start the web page")
    parser.add_argument("--ask", action="append", default=[], type=parse_ask, metavar="T=QUESTION")
    parser.add_argument("--exit-at-end", action="store_true", help="stop when the video file ends")
    parser.add_argument("--label", default=None, help="run folder label (default: the video name or 'camera')")
    parser.add_argument(
        "--wake",
        choices=WAKE_MODES,
        default=None,
        help='wake word: "hey_drivepilot" (default), "hey", or "none" (voice.wake_mode)',
    )
    parser.add_argument("--destination", default=None, help="navigate here once there's a GPS fix")
    return parser


def apply_cli(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """--video / --camera / --gpx on top of the config (paths relative to the current folder)."""
    video, file = cfg.video, cfg.video.file
    if args.video is not None:
        file = file.model_copy(update={"path": args.video.resolve()})
        video = video.model_copy(update={"source": "file", "file": file})
    if args.gpx is not None:
        file = file.model_copy(update={"gpx_path": args.gpx.resolve()})
        video = video.model_copy(update={"file": file})
    if args.camera:
        video = video.model_copy(update={"source": "camera"})
    update: dict[str, Any] = {"video": video}
    if getattr(args, "wake", None) is not None:
        update["voice"] = cfg.voice.model_copy(update={"wake_mode": args.wake})
    return cfg.model_copy(update=update)


def _source_factory(cfg: AppConfig, hold_at_end: bool = False) -> Callable[[], VideoSource]:
    if cfg.video.source == "camera":
        from capture.camera_source import ScrcpyCameraSource
        from core.clock import WallClock

        return lambda: ScrcpyCameraSource(cfg.video.camera, WallClock())
    from capture.file_source import FileSource

    f = cfg.video.file
    return lambda: FileSource(
        cfg.resolve_path(f.path), realtime=f.realtime, start_s=f.start_s, loop=f.loop, hold_at_end=hold_at_end
    )


def _gps_source(cfg: AppConfig) -> GpsSource | None:
    kind = cfg.gps_source
    if kind == "gpx":
        from nav.gps import GpxReplay

        path = cfg.video.file.gpx_path
        assert path is not None
        return GpxReplay(cfg.resolve_path(path), cfg.video.file.gpx_offset_s)
    if kind == "phone":
        return PhoneGps(cfg.gps.stale_s)
    return None


def lane_model_factory(cfg: AppConfig, status: dict[str, str]) -> LaneModelFactory | None:
    """What builds YOLOPv2 on the lanes thread, or None (lanes off, or the weights aren't installed)."""
    if not cfg.lanes.enabled:
        status["lanes"] = "off"
        return None
    if not cfg.resolve_path(cfg.lanes.weights).is_file():
        status["lanes"] = "off (no model)"
        return None

    def factory() -> Callable[[Frame], LaneMasks]:
        from perception.lanes import LaneModel

        return LaneModel(cfg.lanes, cfg.root, hfov_deg=cfg.effective_hfov_deg)

    return factory


def sign_model_factory(cfg: AppConfig, status: dict[str, str]) -> Callable[[], Any] | None:
    """What builds the fine-tuned sign detector on the signs thread, or None (sign alerts off, or no model)."""
    if not cfg.sign_alerts.enabled:
        status["signs"] = "off"
        return None
    if not cfg.resolve_path(cfg.signs.weights).is_file():
        status["signs"] = "off (no model)"
        return None

    def factory() -> Any:
        from perception.signs import SignDetector

        return SignDetector(cfg.signs, cfg.root)

    return factory


def _vision(cfg: AppConfig, status: dict[str, str]) -> VisionModel | None:
    from answers.vision import VisionError, create_vision_model

    try:
        model = create_vision_model(cfg)
    except VisionError as exc:
        log.warning("vision model not available: scene questions will say so", error=str(exc))
        status["vision"] = "off"
        return None
    _check_model(model, "vision", cfg.vision.temperature, status)
    return model


def _reading_vision(cfg: AppConfig, status: dict[str, str]) -> VisionModel | None:
    """The second model for sign, parking and scene-description questions (`vision.reading`), if configured."""
    from answers.vision import VisionError, create_reading_model

    try:
        model = create_reading_model(cfg)
    except VisionError as exc:
        log.warning("reading model not available: the main vision model answers everything", error=str(exc))
        return None
    if model is not None:
        _check_model(model, "reading", cfg.vision.reading.temperature, status)
    return model


def _check_model(model: VisionModel, key: str, temperature: float | None, status: dict[str, str]) -> None:
    """A tiny request in the background: verifies the key and the model, and opens the connection before the first
    question (same settings as real questions). `status[key]` says how it went."""
    from answers.vision import VisionError, VisionRequest

    label = f"{model.name} {getattr(model, 'model', '')}".strip()
    status[key] = f"{label} (checking)"

    def check() -> None:
        start = time.perf_counter()
        try:
            request = VisionRequest(
                system="Reply with one word.", text="Say OK.", max_tokens=5, temperature=temperature
            )
            model.complete(request, 5.0, 10.0)
        except VisionError as exc:
            log.warning("vision model check failed", which=key, error=str(exc))
            status[key] = f"{label} (failed: {exc})"
            return
        status[key] = label
        log.info("vision model ready", which=key, model=label, ms=round((time.perf_counter() - start) * 1000))

    threading.Thread(target=check, name=f"{key}-check", daemon=True).start()


def _routing(cfg: AppConfig, status: dict[str, str]) -> RoutingClient | None:
    token = cfg.secrets.mapbox_access_token
    if token is None:
        log.warning("MAPBOX_ACCESS_TOKEN is not set: navigation is off")
        status["nav"] = "off"
        return None
    from nav.mapbox import MapboxClient

    return MapboxClient(token.get_secret_value(), cfg.nav.profile, cfg.nav.timeout_s, cfg.nav.country)


def _reverser(cfg: AppConfig) -> Reverser | None:
    """Reverse geocoding for the vision model's location line (needs the Mapbox token; off in config: None)."""
    token = cfg.secrets.mapbox_access_token
    if not cfg.vision.location or token is None:
        return None
    return mapbox_reverser(token.get_secret_value(), cfg.nav.timeout_s)


def _speaker(cfg: AppConfig, args: argparse.Namespace, status: dict[str, str]) -> Speaker:
    if args.no_audio:
        status["audio"] = "off"
        return TextSpeaker()
    try:
        from voice.audio import AudioOutput
        from voice.speaker import AudioSpeaker
        from voice.tts import ClipCache, Tts

        tts = Tts(cfg.voice.tts, cfg.root)
        tts.warmup()
        clips = ClipCache(tts, cfg.resolve_path(cfg.voice.tts.clips_dir))
        clips.prepare(list(FIXED_ALERT_TEXTS))
        output = AudioOutput(cfg.voice.output_device, volume=cfg.voice.output_volume)
        output.start()
        t = cfg.voice.tts
        return AudioSpeaker(tts, output, clips, t.chime_volume, t.listen_chime).start()
    except Exception as exc:  # noqa: BLE001 - no voice file, no audio device...: keep running without sound
        log.error("speech output failed to start: answers will be shown, not spoken", error=str(exc))
        status["audio"] = "off"
        return TextSpeaker()


def make_voice_pipeline(app: DrivePilotApp, cfg: AppConfig) -> Any:
    """Wake word + Whisper (warmed up) feeding this app's question handlers (a `VoicePipeline`)."""
    from voice.pipeline import VoicePipeline
    from voice.stt import Transcriber
    from voice.wakeword import WakeWordDetector

    stt = Transcriber(cfg.voice.stt, cfg.root)
    stt.warmup()
    wake = WakeWordDetector(cfg.voice.wakeword, cfg.root)
    return VoicePipeline(
        cfg.voice,
        wake,
        stt,
        app.on_heard,
        on_wake=app.on_wake,
        on_nothing=app.on_nothing,
        speaking=app.speaking,
    )


def _voice(app: DrivePilotApp, cfg: AppConfig, status: dict[str, str]) -> None:
    try:
        from voice.audio import MicStream

        app.attach_voice(make_voice_pipeline(app, cfg), MicStream(cfg.voice.input_device))
        status["voice"] = "on"
    except Exception as exc:  # noqa: BLE001 - no microphone, model missing...: keep running with typed questions
        log.error("voice input failed to start: type questions on the page instead", error=str(exc))
        status["voice"] = "off"


def build_app(
    cfg: AppConfig, args: argparse.Namespace, recorder: RunRecorder | None, speaker: Speaker | None = None
) -> DrivePilotApp:
    """The app with real components (camera or file, GPU detector, Piper, Whisper, Mapbox, vision model)."""
    status: dict[str, str] = {}
    gpu = GpuMonitor(cfg.perception.device)
    detector = Detector(cfg.perception, cfg.root)
    speaker = speaker if speaker is not None else _speaker(cfg, args, status)
    hub = None if args.no_ui else UiHub(width=cfg.ui.video_width, fps=cfg.ui.video_fps)
    app = DrivePilotApp(
        cfg,
        _source_factory(cfg, hold_at_end=not args.exit_at_end),  # the page stays up: seek back to replay
        detector,
        speaker,
        vision=_vision(cfg, status),
        reading_vision=_reading_vision(cfg, status),
        routing=_routing(cfg, status),
        location=_reverser(cfg),
        gps=_gps_source(cfg),
        recorder=recorder,
        hub=hub,
        gpu=gpu,
        status=status,
        scripted=args.ask,
        exit_at_end=args.exit_at_end,
        lane_model=lane_model_factory(cfg, status),
        sign_model=sign_model_factory(cfg, status),
    )
    if hasattr(speaker, "on_record"):  # the audio speaker reports when each utterance was heard
        speaker.on_record = app.on_speech_record
    if args.no_voice:
        status["voice"] = "off"
    else:
        _voice(app, cfg, status)
    return app


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the config and run. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_cli(load_config(args.config, overrides=args.overrides), args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.logging.level, secrets=cfg.secrets.values())

    if args.print_config:
        print(json.dumps(cfg.redacted_dump(), indent=2))
        return 0

    if cfg.video.source == "file" and not cfg.resolve_path(cfg.video.file.path).is_file():
        print(f"video file not found: {cfg.resolve_path(cfg.video.file.path)} (use --video PATH)", file=sys.stderr)
        return 2
    label = args.label or ("camera" if cfg.video.source == "camera" else cfg.video.file.path.stem)
    with RunRecorder(
        cfg.resolve_path(cfg.logging.runs_dir), label, cfg.redacted_dump(), cfg.logging.frame_log_interval_s
    ) as recorder:
        setup_logging(cfg.logging.level, log_file=recorder.log_file, secrets=cfg.secrets.values())
        app = build_app(cfg, args, recorder)
        if args.destination:
            app.pending_destination = args.destination.strip() or None
        server = None
        if app.hub is not None:
            from ui.server import UiServer, allowed_hosts, create_server

            page = create_server(app.hub, PageActions(app), allowed_hosts(cfg.ui.host), overlays=cfg.ui.overlays)
            server = UiServer(page, cfg.ui.host, cfg.ui.port)
            try:
                server.start()
            except RuntimeError as exc:
                log.error("web page failed", error=str(exc))
                return 2
            print(f"DrivePilot screen: {server.url}   phone GPS page: {server.url}gps", flush=True)
        try:
            return app.run()
        finally:
            if server is not None:
                server.stop()
            print(f"run folder: {recorder.dir}", flush=True)


class PageActions:
    """`ui.server.UiActions` on top of the app."""

    def __init__(self, app: DrivePilotApp) -> None:
        self.app = app

    def ask_typed(self, text: str) -> None:
        self.app.ask_typed(text)

    def toggle_pause(self) -> bool | None:
        return self.app.toggle_pause()

    def seek(self, delta_s: float) -> bool:
        return self.app.seek(delta_s)

    def gps_fix(
        self, lat: float, lon: float, speed: float | None, heading: float | None, accuracy: float | None
    ) -> bool:
        return self.app.gps_fix_from_phone(lat, lon, speed, heading, accuracy)

    def set_wake_mode(self, mode: str) -> bool:
        return self.app.set_wake_mode(mode)

    def set_alert(self, name: str, on: bool) -> None:
        self.app.set_alert(name, on)

    def suggest_places(self, query: str) -> list[dict[str, Any]]:
        return self.app.suggest_places(query)

    def set_destination(self, query: str, place: dict[str, Any] | None) -> bool:
        picked = None
        if place is not None:
            picked = Place(
                str(place["name"]),
                str(place.get("address") or place["name"]),
                (float(place["lat"]), float(place["lon"])),
                str(place.get("kind") or ""),
            )
        return self.app.set_destination(query, picked)

    def cancel_route(self) -> bool:
        return self.app.cancel_route()


if __name__ == "__main__":
    sys.exit(main())
