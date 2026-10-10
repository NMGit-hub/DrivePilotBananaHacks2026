"""World state: everything the co-pilot knows right now, built from perception results.

`WorldState.on_perception` is registered as a PerceptionWorker listener (it runs on the perception thread, so
it stays cheap). It maintains per-vehicle motion, which approach each vehicle is on, whether we are at a stop
line, the 4-way-stop queue, the traffic-light state and a short buffer of recent frames. Answer handlers and
prompts read an immutable `WorldSnapshot` (thread-safe); alerts receive `WorldEvent`s through a callback, which
runs on the perception thread and must not block (the assistant only queues speech).

"At the stop line" = stopped, a stop sign seen recently, and no car in front of us in our lane during this
stop. If we stopped behind another car, our stop time is when we reached *that car*, not the line, so the queue
waits until we pull up and stop again.
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
from state.approach import LEAD, classify_approach, front_of_approach, is_clipped
from state.colors import color_name, frame_gains
from state.stop_queue import Approach, EgoStop, QueueSnapshot, StopQueue, WaitingVehicle
from state.traffic_light import LightColor, LightReading, TrafficLightTracker
from state.vehicle_motion import VehicleMotionTracker, VehicleState

log = get_logger(__name__)


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

    kind: str  # "your_turn", "light_green", "arrived_at_stop", "left_intersection", "drove_off", ...
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
    at_stop_sign: bool  # stopped at the line: a stop sign seen recently and no car in front of us this stop
    lead_vehicle: VehicleView | None
    vehicles: tuple[VehicleView, ...]
    queue: QueueSnapshot
    light: LightReading
    caveats: tuple[str, ...]
    people_ahead: int = 0  # pedestrians in front of us (answers add "watch for the pedestrian")
    behind_lead: bool = False  # stopped, and a car was in front of us during this stop (not at the line yet)


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
        self._frames: deque[Frame] = deque()  # guarded by _lock
        self._stop_sign_last_seen: float | None = None
        self._ego_moving_since: float | None = None
        self._was_stopped = False  # stopped since the last "drove_off"
        self._lead_this_stop = False  # a car was in front of us at some point during the current stop
        self._ego_stopped_prev = False  # ego.stopped on the previous frame (for the ego_stopped event)
        self._snapshot: WorldSnapshot | None = None

    # ---------------------------------------------------------------------------------- updating

    def reset(self) -> None:
        """Forget everything (new video segment)."""
        self.vehicle_motion.reset()
        self.queue.reset()
        self.lights.reset()
        self._colors.clear()
        with self._lock:
            self._frames.clear()
        self._stop_sign_last_seen = None
        self._ego_moving_since = None
        self._was_stopped = False
        self._lead_this_stop = False
        self._ego_stopped_prev = False

    def on_perception(self, result: PerceptionResult) -> None:
        """PerceptionWorker listener: fold one frame's results into the world state."""
        frame, t, ego = result.frame, result.frame.t, result.ego
        cfg = self.cfg.intersection
        events: list[WorldEvent] = []
        if result.new_segment:
            self.reset()
            events.append(WorldEvent("segment_reset", t))

        vehicles = [tr for tr in result.tracks if tr.cls in VEHICLE_CLASSES]
        states = self.vehicle_motion.update(t, vehicles, ego.homography, ego.stopped)

        signs = [tr for tr in result.tracks if tr.cls == "stop sign" and tr.conf >= cfg.stop_sign_min_conf]
        if signs:
            self._stop_sign_last_seen = t
        sign_recent = self._stop_sign_last_seen is not None and t - self._stop_sign_last_seen <= cfg.stop_sign_memory_s

        views: list[VehicleView] = []
        lead: VehicleView | None = None
        gains: list[np.ndarray] = []  # this frame's color normalization, computed once if any car needs naming
        front: dict[Approach, tuple[float, int]] = {}  # approach -> (nearness, track id) of the car at the line
        for track in vehicles:
            zone = classify_approach(track, frame.width, frame.height, cfg)
            state = states.get(track.track_id)
            view = VehicleView(
                track_id=track.track_id,
                cls=track.cls,
                color=self._color(track, frame, t, gains),
                approach=zone if isinstance(zone, Approach) else None,
                is_lead=zone == LEAD,
                stopped=bool(state and state.stopped),
                stopped_since=state.stopped_since if state else None,
                box=track.box,
                clipped=is_clipped(track, frame.width, frame.height),
            )
            views.append(view)
            if view.is_lead and (lead is None or track.height > (lead.box[3] - lead.box[1])):
                lead = view
            if view.approach is not None:
                nearness = front_of_approach(view.approach, track)
                if view.approach not in front or nearness > front[view.approach][0]:
                    front[view.approach] = (nearness, track.track_id)
        front_ids = {track_id for _, track_id in front.values()}
        waiting = [
            self._waiting(track, view, states[track.track_id], track.track_id in front_ids)
            for track, view in zip(vehicles, views, strict=True)
            if track.track_id in states
        ]

        if ego.stopped:
            if not self._ego_stopped_prev:
                events.append(WorldEvent("ego_stopped", t, {"since": ego.stopped_since}))
            self._ego_moving_since = None
            self._was_stopped = True
            if lead is not None:
                self._lead_this_stop = True
        else:
            self._lead_this_stop = False
            if self._ego_moving_since is None:
                self._ego_moving_since = t
        moving_for = 0.0 if self._ego_moving_since is None else t - self._ego_moving_since
        self._ego_stopped_prev = ego.stopped
        if self._was_stopped and moving_for >= cfg.drive_off_s:
            self._was_stopped = False
            events.append(WorldEvent("drove_off", t, {"moving_since": self._ego_moving_since}))
        at_stop = ego.stopped and sign_recent and not self._lead_this_stop
        ego_stop = EgoStop(at_stop=at_stop, stopped_since=ego.stopped_since, moving_for_s=moving_for)
        for qe in self.queue.update(t, ego_stop, waiting):
            if qe.kind in ("your_turn", "arrived", "left_intersection", "vehicle_departed"):
                events.append(WorldEvent("arrived_at_stop" if qe.kind == "arrived" else qe.kind, t, dict(qe.detail)))

        light, turned_green = self.lights.update(t, frame.image, result.tracks, ego_stopped=ego.stopped)
        if turned_green:
            events.append(WorldEvent("light_green", t, {"ego_stopped": ego.stopped}))

        people_ahead = sum(
            1
            for tr in result.tracks
            if tr.cls == "person"
            and abs(tr.center[0] / frame.width - 0.5) <= cfg.people_center_band
            and tr.height / frame.height >= cfg.people_min_height
        )

        caveats = []
        if cfg.camera_hfov_deg < cfg.narrow_fov_deg:
            caveats.append("narrow_view")
        # A waiting car cut off by the frame edge (cars just driving through don't matter for the order).
        caveats += [f"clipped:{v.approach.value}" for v in views if v.clipped and v.stopped and v.approach is not None]
        if not ego.confident:
            caveats.append("ego_motion_unknown")

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
            people_ahead=people_ahead,
            behind_lead=ego.stopped and self._lead_this_stop,
        )
        with self._lock:
            self._buffer_frame(frame)
            self._snapshot = snapshot
        self._prune_colors(t)
        for event in events:
            log.info("world event", kind=event.kind, t=round(t, 2), **event.detail)
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:
                    log.exception("world event handler failed", kind=event.kind)

    @staticmethod
    def _waiting(track: Track, view: VehicleView, state: VehicleState, front: bool) -> WaitingVehicle:
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
            moving_for=state.moving_for,
            front=front,
        )

    def _color(self, track: Track, frame: Frame, t: float, gains: list[np.ndarray]) -> str:
        cached = self._colors.get(track.track_id)
        if cached is not None and 0.0 <= t - cached[0] < self.cfg.intersection.color_refresh_s:
            return cached[1]
        if not gains:
            gains.append(frame_gains(frame.image))
        name = color_name(frame.image, track.box, gains[0])
        if name or cached is None:
            self._colors[track.track_id] = (t, name)
            return name
        return cached[1]

    def _prune_colors(self, t: float) -> None:
        if len(self._colors) > 200:
            keep = self.cfg.vehicle_motion.max_track_age_s + self.cfg.intersection.color_refresh_s
            self._colors = {k: v for k, v in self._colors.items() if 0.0 <= t - v[0] <= keep}

    def _buffer_frame(self, frame: Frame) -> None:
        """Keep a frame every `frame_buffer_spacing_s` for `frame_buffer_s` (caller holds the lock)."""
        cfg = self.cfg.intersection
        if self._frames and frame.t < self._frames[-1].t:
            self._frames.clear()  # time went backwards (seek): older frames are from another moment
        if not self._frames or frame.t - self._frames[-1].t >= cfg.frame_buffer_spacing_s:
            self._frames.append(frame)
        while self._frames and frame.t - self._frames[0].t > cfg.frame_buffer_s:
            self._frames.popleft()

    # ----------------------------------------------------------------------------------- reading

    def snapshot(self) -> WorldSnapshot | None:
        """The latest world snapshot (None before the first frame)."""
        with self._lock:
            return self._snapshot

    def recent_frames(self, count: int, spacing_s: float) -> list[Frame]:
        """Up to `count` frames, newest last (the latest processed frame), at least `spacing_s` apart."""
        with self._lock:
            latest = self._snapshot.frame if self._snapshot is not None else None
            frames = list(self._frames)
        if latest is not None and (not frames or latest.t > frames[-1].t):
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
