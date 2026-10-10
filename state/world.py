"""World state: everything the co-pilot knows right now, built from perception results.

`WorldState.on_perception` is registered as a PerceptionWorker listener (it runs on the perception thread, so
it stays cheap). It maintains per-vehicle motion, which approach each vehicle is on, whether we are at a stop
sign, the 4-way-stop queue, the traffic-light state and a short buffer of recent frames. Answer handlers and
prompts read an immutable `WorldSnapshot` (thread-safe); alerts receive `WorldEvent`s through a callback.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.config import AppConfig
from core.logging_setup import get_logger
from core.types import Frame
from perception.detector import VEHICLE_CLASSES, Track
from perception.pipeline import PerceptionResult
from state.approach import classify_approach, is_clipped
from state.colors import color_name, frame_gains
from state.stop_queue import Approach, EgoStop, QueueSnapshot, StopQueue, WaitingVehicle
from state.traffic_light import LightColor, LightReading, TrafficLightTracker
from state.vehicle_motion import VehicleMotionTracker, VehicleState

log = get_logger(__name__)

NARROW_FOV_DEG = 100.0
COLOR_REFRESH_S = 1.0
FRAME_BUFFER_S = 8.0
FRAME_BUFFER_SPACING_S = 0.25


@dataclass(frozen=True, slots=True)
class VehicleView:
    """A tracked vehicle as answers describe it."""

    track_id: int
    cls: str
    color: str
    approach: Approach | None
    is_lead: bool
    stopped: bool
    stopped_since: float | None
    box: tuple[float, float, float, float]
    clipped: bool

    @property
    def label(self) -> str:
        """e.g. "white car", "truck"."""
        return f"{self.color} {self.cls}".strip()


@dataclass(frozen=True, slots=True)
class WorldEvent:
    """Something the alert engine may announce."""

    kind: str  # "your_turn", "light_green", "arrived_at_stop", "left_intersection", ...
    t: float
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Read-only view of the world at one moment."""

    t: float
    frame: Frame | None
    ego_stopped: bool
    ego_stopped_since: float | None
    ego_motion_confident: bool
    stop_sign_visible: bool
    stop_sign_last_seen: float | None
    at_stop_sign: bool  # stopped, a stop sign seen recently, and no car directly in front of us
    lead_vehicle: VehicleView | None
    vehicles: tuple[VehicleView, ...]
    queue: QueueSnapshot
    light: LightReading
    caveats: tuple[str, ...]


def _empty_queue(t: float) -> QueueSnapshot:
    return QueueSnapshot(
        t=t, active=False, order=(), departed=(), ego_position=None, tie_groups=(), deadlock=False, your_turn=False
    )


class WorldState:
    """Thread-safe world model fed by perception."""

    def __init__(self, cfg: AppConfig, on_event: Callable[[WorldEvent], None] | None = None) -> None:
        self.cfg = cfg
        self.on_event = on_event
        self._lock = threading.Lock()
        self.vehicle_motion = VehicleMotionTracker(cfg.vehicle_motion)
        self.queue = StopQueue(cfg.stop_queue)
        self.lights = TrafficLightTracker(cfg.traffic_light)
        self._colors: dict[int, tuple[float, str]] = {}  # track id -> (when named, color)
        self._frames: deque[Frame] = deque()
        self._stop_sign_last_seen: float | None = None
        self._ego_moving_since: float | None = None
        self._snapshot: WorldSnapshot | None = None

    # ---------------------------------------------------------------------------------- updating

    def reset(self) -> None:
        """Forget everything (new video segment)."""
        self.vehicle_motion.reset()
        self.queue.reset()
        self.lights.reset()
        self._colors.clear()
        self._frames.clear()
        self._stop_sign_last_seen = None
        self._ego_moving_since = None

    def on_perception(self, result: PerceptionResult) -> None:
        """PerceptionWorker listener: fold one frame's results into the world state."""
        if result.new_segment:
            self.reset()
        frame, t, ego = result.frame, result.frame.t, result.ego
        cfg = self.cfg.intersection
        events: list[WorldEvent] = []

        vehicles = [tr for tr in result.tracks if tr.cls in VEHICLE_CLASSES]
        states = self.vehicle_motion.update(t, vehicles, ego.homography, ego.stopped)

        signs = [tr for tr in result.tracks if tr.cls == "stop sign" and tr.conf >= cfg.stop_sign_min_conf]
        if signs:
            self._stop_sign_last_seen = t
        sign_recent = self._stop_sign_last_seen is not None and t - self._stop_sign_last_seen <= cfg.stop_sign_memory_s

        views: list[VehicleView] = []
        waiting: list[WaitingVehicle] = []
        lead: VehicleView | None = None
        gains: list[np.ndarray] = []  # this frame's color normalization, computed once if any car needs naming
        for track in vehicles:
            zone = classify_approach(track, frame.width, frame.height, cfg)
            state = states.get(track.track_id)
            view = VehicleView(
                track_id=track.track_id,
                cls=track.cls,
                color=self._color(track, frame, t, gains),
                approach=zone if isinstance(zone, Approach) else None,
                is_lead=zone == "lead",
                stopped=bool(state and state.stopped),
                stopped_since=state.stopped_since if state else None,
                box=track.box,
                clipped=is_clipped(track, frame.width, frame.height),
            )
            views.append(view)
            if view.is_lead and (lead is None or track.height > (lead.box[3] - lead.box[1])):
                lead = view
            if view.approach is not None and state is not None:
                waiting.append(self._waiting(track, view, state))

        if ego.stopped:
            self._ego_moving_since = None
        elif self._ego_moving_since is None:
            self._ego_moving_since = t
        at_stop = ego.stopped and sign_recent and lead is None
        ego_stop = EgoStop(
            at_stop=at_stop,
            stopped_since=ego.stopped_since,
            moving_for_s=0.0 if self._ego_moving_since is None else t - self._ego_moving_since,
        )
        for qe in self.queue.update(t, ego_stop, waiting):
            if qe.kind in ("your_turn", "arrived", "left_intersection", "vehicle_departed"):
                events.append(WorldEvent("arrived_at_stop" if qe.kind == "arrived" else qe.kind, t, dict(qe.detail)))

        light, turned_green = self.lights.update(t, frame.image, result.tracks, ego_stopped=ego.stopped)
        if turned_green:
            events.append(WorldEvent("light_green", t, {"ego_stopped": ego.stopped}))

        caveats = []
        if cfg.camera_hfov_deg < NARROW_FOV_DEG:
            caveats.append("narrow_view")
        caveats += [f"clipped:{v.approach.value}" for v in views if v.clipped and v.approach is not None]
        if not ego.confident:
            caveats.append("ego_motion_unknown")

        self._buffer_frame(frame)
        snapshot = WorldSnapshot(
            t=t,
            frame=frame,
            ego_stopped=ego.stopped,
            ego_stopped_since=ego.stopped_since,
            ego_motion_confident=ego.confident,
            stop_sign_visible=bool(signs),
            stop_sign_last_seen=self._stop_sign_last_seen,
            at_stop_sign=at_stop,
            lead_vehicle=lead,
            vehicles=tuple(views),
            queue=self.queue.snapshot(),
            light=light,
            caveats=tuple(caveats),
        )
        with self._lock:
            self._snapshot = snapshot
        for event in events:
            log.info("world event", kind=event.kind, t=round(t, 2), **event.detail)
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:
                    log.exception("world event handler failed", kind=event.kind)

    def _waiting(self, track: Track, view: VehicleView, state: VehicleState) -> WaitingVehicle:
        assert view.approach is not None
        return WaitingVehicle(
            track_id=track.track_id,
            approach=view.approach,
            stopped=state.stopped,
            stopped_since=state.stopped_since,
            moved_from_stop=state.moved_from_stop,
            box=track.box,
            label=view.label,
            clipped=view.clipped,
            arrival_known=state.arrival_known,
        )

    def _color(self, track: Track, frame: Frame, t: float, gains: list[np.ndarray]) -> str:
        cached = self._colors.get(track.track_id)
        if cached is not None and t - cached[0] < COLOR_REFRESH_S:
            return cached[1]
        if not gains:
            gains.append(frame_gains(frame.image))
        name = color_name(frame.image, track.box, gains[0])
        if name or cached is None:
            self._colors[track.track_id] = (t, name)
            return name
        return cached[1]

    def _buffer_frame(self, frame: Frame) -> None:
        if not self._frames or frame.t - self._frames[-1].t >= FRAME_BUFFER_SPACING_S or frame.t < self._frames[-1].t:
            self._frames.append(frame)
        while self._frames and frame.t - self._frames[0].t > FRAME_BUFFER_S:
            self._frames.popleft()

    # ----------------------------------------------------------------------------------- reading

    def snapshot(self) -> WorldSnapshot | None:
        """The latest world snapshot (None before the first frame)."""
        with self._lock:
            return self._snapshot

    def recent_frames(self, count: int, spacing_s: float) -> list[Frame]:
        """Up to `count` buffered frames, newest last, at least `spacing_s` apart (for the vision model)."""
        with self._lock:
            latest = self._snapshot.frame if self._snapshot is not None else None
        frames = list(self._frames)
        if latest is not None and (not frames or frames[-1] is not latest):
            frames.append(latest)
        picked: list[Frame] = []
        for frame in reversed(frames):
            if not picked or picked[-1].t - frame.t >= spacing_s:
                picked.append(frame)
            if len(picked) == count:
                break
        return list(reversed(picked))

    @staticmethod
    def light_is(snapshot: WorldSnapshot, color: LightColor) -> bool:
        """True if the smoothed light state facing us is `color` with confidence."""
        return snapshot.light.color is color and snapshot.light.confident


def empty_snapshot(t: float = 0.0) -> WorldSnapshot:
    """A snapshot with nothing in it (before the first frame / in tests)."""
    return WorldSnapshot(
        t=t,
        frame=None,
        ego_stopped=False,
        ego_stopped_since=None,
        ego_motion_confident=False,
        stop_sign_visible=False,
        stop_sign_last_seen=None,
        at_stop_sign=False,
        lead_vehicle=None,
        vehicles=(),
        queue=_empty_queue(t),
        light=LightReading(t=t, color=LightColor.UNKNOWN, confident=False, n_lights=0, boxes=()),
        caveats=(),
    )
