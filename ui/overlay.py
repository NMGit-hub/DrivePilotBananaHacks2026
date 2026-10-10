"""What the V2 page draws on top of the video (V2-M2), built from the world state of the very frame it is drawn on.

`frame_packet` turns one `WorldSnapshot` (plus that frame's tracks) into a small JSON-able dict: boxes with labels,
badges and a color, in coordinates normalized to the frame (0..1), so the page scales them to whatever size it
shows the video at. The backend sends it in the same WebSocket message as the frame's JPEG, so they can't drift.
`hud_packet` is the slower-changing part (navigation banner with lanes, speed, roundabout diagram), sent with
the status line.

Colors mean the same everywhere (PLAN.md "Visualization"): green = where you should be / your turn / allowed,
red = danger, amber = watch this (pedestrians, cyclists, the car that goes next), white/gray = tracked, nothing to
do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nav.phrasing import in_distance, maneuver_phrase
from nav.progress import Progress
from perception.detector import Track, is_own_hood
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


def frame_packet(
    snap: WorldSnapshot | None,
    tracks: Sequence[Track],
    width: int,
    height: int,
    *,
    parking_active: bool = False,
    highlight: int | None = None,
) -> dict[str, Any]:
    """The overlay for one frame: every tracked object with its meaning, the 4-way badges, the light's state,
    open parking spots (while the parking watch is on) and the "YOU" badge. `highlight`: a track id to pulse."""
    items: list[dict[str, Any]] = []
    if snap is None:
        return {"items": items, "you": None}
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
        if is_own_hood(v.box, width, height):
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
                    {"kind": "spot", "box": _norm(space.region, width, height), "color": GREEN, "label": label}
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
    return {"items": items, "you": you}


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
