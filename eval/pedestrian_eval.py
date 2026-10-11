"""Replay cached perception (eval/perception_cache.py) through the real `WorldState` and `AlertEngine` and report
the pedestrian and cyclist warnings (V2-M3, state/pedestrians.py).

`replay` feeds each cached frame to a `WorldState` exactly as the app's world thread does (same tracks, ego-motion
verdict and homography; the image is a flat grey stand-in, which only the color names and the traffic-light reader
look at), with the clip's GPS speed as its `speed_source`, and hands its events to an `AlertEngine` like the
assistant does. It records every spoken pedestrian alert, every pedestrian event (spoken or suppressed), the
per-frame pedestrian snapshots, and the time the pedestrian update took on each frame (it runs on the world
thread). `episodes` groups the frames where someone was in (or entering) the path zone but no alert fired, per
track, with the reasons (we were stopped / GPS said we were barely moving / still confirming / already warned).
"""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from answers.alerts import AlertEngine
from answers.speech import Utterance
from core.config import AppConfig
from core.types import Frame
from eval.perception_cache import CachedFrame
from perception.ego_motion import NOT_MEASURED, EgoMotion
from perception.pipeline import PerceptionResult
from state.pedestrians import DrivableArea, PedestrianEvent, PedestrianSnapshot
from state.world import WorldEvent, WorldState


@dataclass(frozen=True, slots=True)
class SpokenAlert:
    """A pedestrian/cyclist warning the alert engine spoke."""

    t: float
    index: int  # frame index it was raised on
    text: str
    event: PedestrianEvent


@dataclass(slots=True)
class Replay:
    """Everything one clip's replay produced."""

    alerts: list[SpokenAlert] = field(default_factory=list)
    events: list[tuple[int, PedestrianEvent]] = field(default_factory=list)  # (frame index, event), spoken or not
    snapshots: list[PedestrianSnapshot] = field(default_factory=list)  # one per frame
    horizons: list[float] = field(default_factory=list)  # the horizon row the update used, per frame
    cost_ms: list[float] = field(default_factory=list)  # pedestrian update time per frame
    world_ms: list[float] = field(default_factory=list)  # whole world-state update per frame


@dataclass(frozen=True, slots=True)
class Episode:
    """Consecutive frames in which one track was in (or entering) the path zone without a warning."""

    track_id: int
    kind: str
    t0: float
    t1: float
    i0: int
    i1: int
    reasons: dict[str, int]  # why no warning -> frames
    min_z: float
    how: str  # "in_path" | "entering" | "both"


def replay(
    frames: list[CachedFrame],
    cfg: AppConfig,
    gps_speed: Callable[[float], float | None] | None = None,
    road: list[DrivableArea] | None = None,
    road_lag_s: float = 0.1,
) -> Replay:
    """Run the clip's cached perception through WorldState + AlertEngine (see the module docstring). `road`: the
    clip's YOLOPv2 drivable areas in time order (each frame gets the newest one at least `road_lag_s` older)."""
    out = Replay()
    if not frames:
        return out
    current = {"index": 0}
    now = {"t": 0.0}

    road_times = [r.t for r in road] if road else []

    def drivable() -> DrivableArea | None:
        # The newest mask at least `road_lag_s` old: in the app the lanes thread finishes a frame ~0.1 s after the
        # world thread saw it (12 fps on the newest frame + the model's time), so the band beside someone's feet
        # looks at a slightly older view.
        if not road:
            return None
        k = bisect.bisect_right(road_times, now["t"] - road_lag_s + 1e-6) - 1
        return road[k] if k >= 0 else None

    world_events: list[WorldEvent] = []
    said: list[Utterance] = []
    alerts = AlertEngine(cfg.alerts, said.append)

    def on_event(event: WorldEvent) -> None:
        world_events.append(event)
        if event.kind == "segment_reset":
            alerts.reset()
            return
        before = len(said)
        alerts.on_world_event(event)
        if event.kind == "pedestrian_in_path" and isinstance(event.payload, PedestrianEvent):
            out.events.append((current["index"], event.payload))
            if len(said) > before:
                out.alerts.append(SpokenAlert(event.t, current["index"], said[-1].text, event.payload))

    world = WorldState(cfg, on_event=on_event, speed_source=gps_speed, drivable_source=drivable)
    tracker = world.pedestrians
    update = tracker.update
    cost: list[float] = []

    def timed(*args: object, **kwargs: object) -> list[PedestrianEvent]:
        start = time.perf_counter()
        result = update(*args, **kwargs)  # type: ignore[arg-type]
        cost.append((time.perf_counter() - start) * 1000.0)
        return result

    tracker.update = timed  # type: ignore[method-assign]
    image = np.full((frames[0].height, frames[0].width, 3), 100, np.uint8)
    stopped_since: float | None = None
    for k, f in enumerate(frames):
        current["index"] = f.index
        now["t"] = f.t
        if f.ego_stopped and stopped_since is None:
            stopped_since = f.t
        if not f.ego_stopped:
            stopped_since = None
        ego = EgoMotion(
            t=f.t,
            short=NOT_MEASURED,
            long=NOT_MEASURED,
            value=0.0 if f.ego_stopped else 5.0,
            stopped=f.ego_stopped,
            stopped_since=stopped_since,
            confident=True,
            homography=f.homography,
            elapsed_ms=0.0,
        )
        result = PerceptionResult(
            frame=Frame(image=image, t=f.t, index=f.index, captured_at=0.0),
            tracks=list(f.tracks),
            ego=ego,
            timings_ms={},
            new_segment=k == 0,
        )
        start = time.perf_counter()
        world.on_perception(result)
        out.world_ms.append((time.perf_counter() - start) * 1000.0)
        out.snapshots.append(tracker.snapshot())
        out.horizons.append(world.parking.snapshot().horizon)
    out.cost_ms = cost
    return out


@dataclass(slots=True)
class _Open:
    kind: str
    t0: float
    i0: int
    how: str
    min_z: float
    t1: float = 0.0
    i1: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


def episodes(run: Replay, frames: list[CachedFrame], confirm_frames: int) -> list[Episode]:
    """Per track, the runs of frames where it was in (or entering) the path but no warning fired on that frame."""
    alert_at = {(i, e.person.track_id) for i, e in run.events}
    open_: dict[int, _Open] = {}
    done: list[Episode] = []

    def close(track_id: int) -> None:
        ep = open_.pop(track_id)
        done.append(Episode(track_id, ep.kind, ep.t0, ep.t1, ep.i0, ep.i1, ep.reasons, ep.min_z, ep.how))

    for f, snap in zip(frames, run.snapshots, strict=True):
        seen = set()
        for p in snap.people:
            if not (p.in_path or p.entering) or (f.index, p.track_id) in alert_at:
                continue
            seen.add(p.track_id)
            if p.ignored is not None:
                reason = f"ignored ({p.ignored})"
            elif p.alerted:
                reason = "already warned"
            elif not snap.moving:
                reason = f"not moving ({snap.still_reason})"
            elif p.streak < confirm_frames:
                reason = "confirming"
            else:
                reason = "?"
            how = "in_path" if p.in_path else "entering"
            ep = open_.get(p.track_id)
            if ep is None:
                ep = open_[p.track_id] = _Open(p.kind, f.t, f.index, how, p.z)
            ep.t1, ep.i1 = f.t, f.index
            ep.reasons[reason] = ep.reasons.get(reason, 0) + 1
            ep.min_z = min(ep.min_z, p.z)
            if ep.how != how:
                ep.how = "both"
        for track_id in [k for k in open_ if k not in seen]:
            close(track_id)
    for track_id in list(open_):
        close(track_id)
    return sorted(done, key=lambda e: e.t0)
