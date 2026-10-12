"""What the V2 page draws on top of the video (V2-M2), built from the world state of the very frame it is drawn on.

`frame_packet` turns one `WorldSnapshot` (plus that frame's tracks) into a small JSON-able dict: boxes with labels,
badges and a color, in coordinates normalized to the frame (0..1), so the page scales them to whatever size it
shows the video at. The backend sends it in the same WebSocket message as the frame's JPEG, so they can't drift.
`hud_packet` is the slower-changing part (navigation banner with lanes, speed, roundabout diagram), sent with
the status line. `lanes_packet` is what lane detection (V2-M5) saw last: our lane's two lines and the others, each
resampled to `LANE_POINTS` points so the page can ease between the lane model's ~12 updates a second, and the
line we're drifting toward (red, with the screen edge on that side pulsing).

Colors mean the same everywhere (PLAN.md "Visualization"): green = where you should be / your turn / allowed,
red = danger, amber = watch this (pedestrians, cyclists, the car that goes next), white/gray = tracked, nothing to
do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from nav.phrasing import in_distance, maneuver_phrase
from nav.progress import Progress
from perception.detector import HOOD_BOTTOM, Track, is_own_hood
from state.lanes import LanesSnapshot
from state.world import WorldSnapshot

GREEN, RED, AMBER, WHITE, GRAY = "green", "red", "amber", "white", "gray"
PEOPLE = {"person": "pedestrian", "bicycle": "cyclist"}
SIGNS = {"stop sign": "STOP"}
_ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
MPS_TO_MPH = 2.23694


def _norm(box: Sequence[float], width: int, height: int) -> list[float]:
    """A pixel box as fractions of the frame, rounded (small packets), clamped to the frame."""
    x1, y1, x2, y2 = box
    return [
        round(min(max(x1 / width, 0.0), 1.0), 4),
        round(min(max(y1 / height, 0.0), 1.0), 4),
        round(min(max(x2 / width, 0.0), 1.0), 4),
        round(min(max(y2 / height, 0.0), 1.0), 4),
    ]


def ordinal(n: int) -> str:
    """1 → "1st"."""
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else f"{n}th"


def resample(points: Sequence[tuple[float, float]], n: int = 8) -> list[list[float]]:
    """A polyline (near to far) as `n` points evenly spaced along its length (the page eases point by point)."""
    if len(points) < 2:
        return [[round(points[0][0], 4), round(points[0][1], 4)]] * n if points else []
    xy = np.asarray(points, dtype=float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    if along[-1] <= 0.0:
        return [[round(float(xy[0, 0]), 4), round(float(xy[0, 1]), 4)]] * n
    at = np.linspace(0.0, along[-1], n)
    xs, ys = np.interp(at, along, xy[:, 0]), np.interp(at, along, xy[:, 1])
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, ys, strict=True)]


LANE_POINTS = 8
TARGET_LANE_M = 2000.0  # the target lane is shown from this far before a maneuver that needs certain lanes


def lane_target(lanes: LanesSnapshot | None, progress: Progress | None) -> dict[str, Any] | None:
    """Where lane guidance wants us (PLAN.md: "the green target lane"): "ours" if our lane works for the next
    maneuver, else the side to move to and by how many lanes. Only when lane detection's count matches the
    route's lane data for that maneuver (the same check the spoken lane advice makes), else None."""
    if lanes is None or progress is None or progress.arrived:
        return None
    step, to_next = progress.next_step, progress.to_next_m
    if step is None or to_next is None or to_next > TARGET_LANE_M or not step.lanes or not step.needs_lanes:
        return None
    ours, count = lanes.lane_index, lanes.lane_count
    valid = [i + 1 for i, lane in enumerate(step.lanes) if lane.valid]
    if not valid:
        return None
    n = len(step.lanes)
    if ours is None or count is None or count != n:
        # Without the whole count: when the lanes that work are the ones at an edge (an exit from the right lanes),
        # the lanes seen on that side are enough ("1 lane to our right, only the rightmost works: move one right").
        contiguous = valid == list(range(valid[0], valid[-1] + 1))
        if contiguous and valid[-1] == n and valid[0] > 1 and lanes.lanes_right is not None:
            from_right, works = (
                lanes.lanes_right + 1,
                n - valid[0] + 1,
            )  # (ours / the good ones, counted from the right)
            return {"lane": "ours"} if from_right <= works else {"lane": "right", "moves": from_right - works}
        if contiguous and valid[0] == 1 and valid[-1] < n and lanes.lanes_left is not None:
            from_left, works = lanes.lanes_left + 1, valid[-1]
            return {"lane": "ours"} if from_left <= works else {"lane": "left", "moves": from_left - works}
        return None
    if ours in valid:
        return {"lane": "ours"}
    nearest = min(valid, key=lambda i: (abs(i - ours), i))
    return {"lane": "left" if nearest < ours else "right", "moves": abs(nearest - ours)}


def lanes_packet(lanes: LanesSnapshot | None, target: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Our lane (its left and right lines), the lines just beyond them (the lanes beside ours), the other lines,
    the drift side and the target lane, or None without a lane."""
    if lanes is None or not lanes.valid:
        return None
    roles = [line.role for line in lanes.lines]
    if "left" not in roles or "right" not in roles:
        return None
    i, j = roles.index("left"), roles.index("right")
    lines = lanes.lines
    return {
        "left": resample(lines[i].points, LANE_POINTS),
        "right": resample(lines[j].points, LANE_POINTS),
        "left_next": resample(lines[i - 1].points, LANE_POINTS) if i > 0 else None,
        "right_next": resample(lines[j + 1].points, LANE_POINTS) if j + 1 < len(lines) else None,
        "others": [resample(line.points, LANE_POINTS) for line in lines if line.role == "other"],
        "drift": lanes.drift_side,
        "target": target,
    }


def frame_packet(
    snap: WorldSnapshot | None,
    tracks: Sequence[Track],
    width: int,
    height: int,
    *,
    parking_active: bool = False,
    highlight: int | None = None,
    lanes: LanesSnapshot | None = None,
    lane_goal: dict[str, Any] | None = None,
    hood_bottom: float = HOOD_BOTTOM,
) -> dict[str, Any]:
    """The overlay for one frame: every tracked object with its meaning, the 4-way badges, the light's state,
    open parking spots (while the parking watch is on), the "YOU" badge and our lane (`lanes`: lane detection's
    latest, if it is recent; `lane_goal`: `lane_target`'s answer). `highlight`: a track id to pulse. `hood_bottom`:
    a wide "vehicle" reaching this far down is our own hood (not drawn)."""
    items: list[dict[str, Any]] = []
    if snap is None:
        return {"items": items, "you": None, "lanes": lanes_packet(lanes, lane_goal)}
    queue = snap.queue
    queued: dict[int, tuple[int, float]] = {}  # track id -> (position, arrived)
    next_id: int | None = None
    if queue.active:
        for position, entry in enumerate(queue.order, start=1):
            if entry.track_id is None:
                continue
            queued[entry.track_id] = (position, entry.arrived)
            if next_id is None and not queue.deadlock and (queue.ego_position is None or position < queue.ego_position):
                next_id = entry.track_id  # the first car ahead of us: it goes next
    parked = {c.track_id for row in snap.parking.rows.values() for c in row}

    for v in snap.vehicles:
        if is_own_hood(v.box, width, height, hood_bottom):
            continue
        item: dict[str, Any] = {"kind": "vehicle", "id": v.track_id, "box": _norm(v.box, width, height)}
        if v.track_id in queued:
            position, arrived = queued[v.track_id]
            item.update(
                color=AMBER if v.track_id == next_id else WHITE,
                label=v.label,
                badge=ordinal(position),
                timer=round(max(0.0, snap.t - arrived), 1),
            )
        elif v.is_lead:
            item.update(color=WHITE, label=v.label)
        else:
            item.update(color=GRAY if v.track_id in parked else WHITE)
        items.append(item)

    people = {p.track_id: p for p in snap.pedestrians.people}
    for tr in tracks:
        if tr.cls in PEOPLE:
            person = people.get(tr.track_id)
            danger = (
                person is not None
                and snap.pedestrians.moving
                and person.ignored is None
                and (person.in_path or person.entering)
            )
            parked_bike = person is not None and person.kind == "bicycle"
            items.append(
                {
                    "kind": "person",
                    "id": tr.track_id,
                    "box": _norm(tr.box, width, height),
                    "color": RED if danger else GRAY if parked_bike else AMBER,  # red: in (or entering) our path
                    "label": "parked bicycle" if parked_bike else PEOPLE[tr.cls],
                }
            )
        elif tr.cls in SIGNS:
            items.append(
                {
                    "kind": "sign",
                    "id": tr.track_id,
                    "box": _norm(tr.box, width, height),
                    "color": WHITE,
                    "label": SIGNS[tr.cls],
                }
            )

    light = snap.light
    for i, box in enumerate(light.boxes):
        item = {"kind": "light", "box": _norm(box, width, height), "color": GRAY}
        if i == 0 and light.color.value != "unknown":
            color = {"red": RED, "yellow": AMBER, "green": GREEN}[light.color.value]
            item.update(color=color, label=light.color.value.upper() + ("" if light.confident else "?"))
        items.append(item)

    if parking_active:
        for space in snap.parking.candidates:
            if space.confirmed:
                label = "Open spot" if space.stalls == 1 else f"{space.stalls} open spots"
                items.append(
                    {"kind": "spot", "box": _norm(space.outline, width, height), "color": GREEN, "label": label}
                )

    if highlight is not None:
        for item in items:
            if item.get("id") == highlight and item["kind"] in ("vehicle", "person", "sign"):
                item["pulse"] = True

    you = None
    if queue.active and queue.ego_position is not None:
        you = {
            "text": "YOU: your turn" if queue.your_turn else f"YOU: {ordinal(queue.ego_position)}",
            "color": GREEN if queue.your_turn else WHITE,
        }
    return {"items": items, "you": you, "lanes": lanes_packet(lanes, lane_goal)}


def hud_packet(
    progress: Progress | None,
    speed_mps: float | None,
    speed_limit: int | None = None,
    school_limit: int | None = None,
) -> dict[str, Any]:
    """The navigation banner (next step, distance, lanes), our speed, the speed limit read last (the badge), and
    the roundabout diagram when one is near."""
    hud: dict[str, Any] = {
        "speed_mph": None if speed_mps is None else round(speed_mps * MPS_TO_MPH),
        "speed_limit": speed_limit,
        "school_limit": school_limit,
    }
    if progress is None or progress.arrived:
        return hud
    step, to_next = progress.next_step, progress.to_next_m
    if step is not None and to_next is not None:
        banner: dict[str, Any] = {"text": maneuver_phrase(step), "distance": in_distance(to_next)}
        if step.needs_lanes and to_next <= 2000:
            banner["lanes"] = [{"valid": lane.valid, "ways": list(lane.indications)} for lane in step.lanes]
        hud["banner"] = banner
    ring = None
    if progress.step.is_roundabout and progress.step.roundabout_exit:
        ring = progress.step  # in the roundabout now: count the exits passed
    elif step is not None and step.is_roundabout and step.roundabout_exit and to_next is not None and to_next <= 400:
        ring = step
    if ring is not None and ring.roundabout_exit is not None:
        # (the exits before ours, as the navigation handler counts them for "not this one")
        passed = sum(1 for m in ring.roundabout_exits_m if m <= progress.along_m) if ring is progress.step else 0
        hud["roundabout"] = {"exit": ring.roundabout_exit, "passed": passed}
    return hud
