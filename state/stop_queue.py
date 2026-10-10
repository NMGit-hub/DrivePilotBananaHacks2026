"""The 4-way-stop queue: who stopped first, ties to the right, departures and "your turn".

Pure logic (no images): `StopQueue.update()` is fed, once per processed frame, our own stop state and the
tracked vehicles with their motion state and approach zone (from `state.vehicle_motion` + `state.approach`).

- Joining: the *front* car of an approach (the one at the line) joins once it is stopped; its arrival is when
  it came to rest there. Cars queued behind it join only after they pull up and stop at the line themselves.
  A car whose arrival is unknown (already standing still when we started watching) is put ahead of us.
- Order: by arrival time; arrivals within `tie_window_s` of each other are a tie, resolved by the rule that a
  car yields to the car on its right (a 3-way tie has a clear order; a 4-way tie is flagged).
- Departures need real driving, not a creep to the line: continuous motion for `depart_moving_s` and
  `depart_moved` box heights from where it stopped (the car is followed by track id even when it turns out of
  its approach zone), or vanishing from view while clearly driving off. While we ourselves move (creeping
  forward to see) no departures are decided, because our motion is only approximately cancelled. A "departed"
  car that comes to rest again on its approach is reinstated.
- Lost cars (occlusion by crossing traffic, detector flicker) keep their place and are re-linked when a
  stopped track reappears at the same place. A car lost for `lost_timeout_s`, or one that hasn't moved for
  `parked_after_s` since we arrived (probably parked), is dropped, but never silently turned into a "your turn"
  alert: answers say "I lost sight of the car on your left" instead.
- "Your turn": every car ahead of us departed → one `your_turn` event per stop (only if we had to wait).

Approach geometry, from our seat (we approach from the south, facing north):
    left  = car on the west approach, facing east      ahead = north approach, facing south
    right = east approach, facing west                 us    = south approach, facing north
Each driver yields to the car on *their* right:  us → right, right → ahead, ahead → left, left → us.

Snapshots are deep copies, so readers on other threads never see entries change underneath them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from core.config import StopQueueConfig
from core.logging_setup import get_logger

log = get_logger(__name__)


class Approach(StrEnum):
    """Which approach of the intersection a car is waiting on, as seen from our seat."""

    US = "us"
    LEFT = "left"
    AHEAD = "ahead"
    RIGHT = "right"


# The car on each approach's right (whom that driver yields to on a tie).
RIGHT_OF: dict[Approach, Approach] = {
    Approach.US: Approach.RIGHT,
    Approach.RIGHT: Approach.AHEAD,
    Approach.AHEAD: Approach.LEFT,
    Approach.LEFT: Approach.US,
}

EGO_KEY = "ego"


@dataclass(frozen=True, slots=True)
class WaitingVehicle:
    """A tracked vehicle this frame, as the queue sees it.

    Attributes:
        track_id: tracker id.
        approach: left / ahead / right, or None if it is in no approach zone (e.g. driving through the middle).
        stopped: True if it is currently at rest.
        stopped_since: when it came to rest (None while moving).
        moved_from_stop: box heights moved since it came to rest (0 if unmeasurable); used for departures.
        box: (x1, y1, x2, y2) pixels, for re-linking lost tracks.
        label: short description for answers, e.g. "white car".
        clipped: box touches the frame edge (partly out of view).
        arrival_known: False if it hasn't moved since we started watching (it may have been waiting before us).
        moving_for: seconds of continuous motion (0 unless moving).
        front: the car nearest the intersection on its approach (only front cars join the queue).
    """

    track_id: int
    approach: Approach | None
    stopped: bool
    stopped_since: float | None
    moved_from_stop: float
    box: tuple[float, float, float, float]
    label: str = "car"
    clipped: bool = False
    arrival_known: bool = True
    moving_for: float = 0.0
    front: bool = True


@dataclass(frozen=True, slots=True)
class EgoStop:
    """Our own state at the intersection this frame."""

    at_stop: bool  # stopped at the stop line of a stop-controlled intersection
    stopped_since: float | None  # when we came to rest (None while moving)
    moving_for_s: float  # how long we have been moving continuously (0 while stopped)


@dataclass(slots=True)
class QueueEntry:
    """One car in the queue (or us)."""

    key: str  # "ego" or "track:<id>"
    approach: Approach
    arrived: float  # media time it came to rest
    label: str = "car"
    track_id: int | None = None
    departed: bool = False
    departed_at: float | None = None
    last_seen: float = 0.0
    lost: bool = False  # track not visible right now (kept: occlusion is not a departure)
    box: tuple[float, float, float, float] | None = None
    clipped: bool = False
    arrival_known: bool = True
    last_moving_for: float = 0.0  # motion at its last sighting (a car lost while driving off has departed)
    last_moved: float = 0.0  # box heights from its stop at its last sighting
    drop_reason: str | None = None  # set when dropped ("lost from view", "probably parked")

    @property
    def is_ego(self) -> bool:
        """True for our own entry."""
        return self.key == EGO_KEY


@dataclass(frozen=True, slots=True)
class QueueEvent:
    """Something that happened in the queue this frame."""

    kind: str  # "arrived", "vehicle_joined", "vehicle_departed", "vehicle_returned", "vehicle_dropped",
    # "your_turn", "left_intersection"
    t: float
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """Read-only view of the queue for answers and alerts (entries are copies)."""

    t: float
    active: bool  # we are stopped at a stop-controlled intersection
    order: tuple[QueueEntry, ...]  # waiting entries (not departed) in right-of-way order, us included
    departed: tuple[QueueEntry, ...]  # entries that already went, most recent last
    ego_position: int | None  # 1-based position among waiting entries (None if we're not queued)
    tie_groups: tuple[tuple[str, ...], ...]  # keys of entries whose arrivals were within the tie window
    deadlock: bool  # a 4-way tie: no one has a clear right of way
    your_turn: bool
    dropped: tuple[QueueEntry, ...] = ()  # cars that were ahead of us and were dropped (lost / probably parked)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def right_of_way_order(
    entries: Sequence[QueueEntry], tie_window_s: float
) -> tuple[list[QueueEntry], list[list[str]], bool]:
    """Order waiting entries: by arrival time, ties broken by yielding to the car on the right.

    Returns (ordered entries, tie groups as lists of keys (only groups with 2+ members), deadlock flag).
    A tie group is a run of arrivals each within `tie_window_s` of the group's first arrival. Inside a group,
    an entry must wait for any group member on its right; entries are released in topological order, earlier
    arrival first among those free to go. If every remaining member has someone on its right (4-way tie), the
    group is a deadlock and the rest keep arrival order.

    Entries whose arrival is unknown (already standing still when first seen) are assumed to have been there
    first: they come before every known arrival and never tie with one (a tie could hand us a right of way we
    can't justify), and are ordered among themselves by the same rules.
    """
    groups: list[list[QueueEntry]] = []
    for known in (False, True):
        part = sorted((e for e in entries if e.arrival_known is known), key=lambda e: (e.arrived, e.key))
        first_group = len(groups)
        for entry in part:
            if len(groups) > first_group and entry.arrived - groups[-1][0].arrived <= tie_window_s:
                groups[-1].append(entry)
            else:
                groups.append([entry])

    ordered: list[QueueEntry] = []
    tie_groups: list[list[str]] = []
    deadlock = False
    for group in groups:
        if len(group) == 1:
            ordered.extend(group)
            continue
        tie_groups.append([e.key for e in group])
        remaining = list(group)
        while remaining:
            approaches = {e.approach for e in remaining}
            free = [e for e in remaining if RIGHT_OF[e.approach] not in approaches]
            if not free:
                deadlock = True
                ordered.extend(remaining)  # everyone has someone on their right: keep arrival order
                break
            nxt = free[0]  # earliest arrival among those with nobody on their right
            ordered.append(nxt)
            remaining.remove(nxt)
    return ordered, tie_groups, deadlock


def deadlock_blocks_ego(ordered: Sequence[QueueEntry], tie_groups: Sequence[Sequence[str]], deadlock: bool) -> bool:
    """In a 4-way deadlock that includes us, nobody (including us) has a clear right of way."""
    return deadlock and any(EGO_KEY in group and len(group) >= 4 for group in tie_groups)


class StopQueue:
    """Stateful queue for one intersection at a time (resets when we leave the intersection)."""

    def __init__(self, cfg: StopQueueConfig | None = None) -> None:
        self.cfg = cfg or StopQueueConfig()
        self.reset()

    def reset(self) -> None:
        """Forget everything (new segment, or we left the intersection)."""
        self._entries: dict[str, QueueEntry] = {}
        self._aliases: dict[int, str] = {}  # new track id -> entry key it was re-linked to
        self._dropped_ahead: list[QueueEntry] = []
        self._parked: set[str] = set()  # keys written off as parked: they don't rejoin at this intersection
        self._ego_arrived: float | None = None
        self._your_turn_fired = False
        self._had_cars_ahead = False
        self._t = 0.0

    # ----------------------------------------------------------------------------------- updating

    def update(self, t: float, ego: EgoStop, vehicles: Iterable[WaitingVehicle]) -> list[QueueEvent]:
        """Advance the queue to time `t`. Returns the events that happened."""
        self._t = t
        events: list[QueueEvent] = []
        vehicles = list(vehicles)

        events += self._update_ego(t, ego)
        if self._ego_arrived is None:
            # Not at an intersection: track nothing (cars are only queued relative to our own stop).
            if self._entries:
                self._entries.clear()
            return events
        ego_still = ego.moving_for_s <= 0.0  # departures are only judged while we stand still

        seen_keys: set[str] = set()
        for v in vehicles:
            key = self._key_for(v)
            entry = self._entries.get(key)
            if entry is None:
                joinable = v.approach is not None and v.front and v.stopped and v.stopped_since is not None
                if joinable and key not in self._parked:
                    events.append(self._join(key, v, t))
                    seen_keys.add(key)
                continue
            seen_keys.add(key)
            events += self._observe(entry, v, t, ego_still)

        events += self._unseen(seen_keys, t, ego_still)

        _, _, _, ego_pos, your_turn = self._ranked()
        if ego_pos is not None and ego_pos > 1:
            self._had_cars_ahead = True
        if your_turn and self._had_cars_ahead and not self._your_turn_fired:
            # Only after waiting for someone (at an empty intersection there is nothing to announce), and never
            # because a car ahead was dropped (lost from view / maybe parked): that is a guess, not a departure.
            self._your_turn_fired = True
            if self._dropped_ahead:
                log.info(
                    "your turn not announced: a car ahead was dropped", dropped=[e.key for e in self._dropped_ahead]
                )
            else:
                went = sum(1 for e in self._entries.values() if e.departed)
                events.append(QueueEvent("your_turn", t, {"went_before_us": went}))
        for event in events:
            log.info("queue event", kind=event.kind, t=round(t, 2), **event.detail)
        return events

    def _join(self, key: str, v: WaitingVehicle, t: float) -> QueueEvent:
        assert v.approach is not None and v.stopped_since is not None and self._ego_arrived is not None
        arrived = v.stopped_since
        if not v.arrival_known:
            # It hasn't moved since we started watching (e.g. it was there when we arrived, or came out from
            # behind a truck): assume it was there before us (right_of_way_order puts it ahead of every known
            # arrival, without a tie). Wrongly yielding is the safe mistake.
            arrived = min(arrived, self._ego_arrived)
        self._dropped_ahead = [e for e in self._dropped_ahead if e.key != key]  # lost, now back in view
        self._entries[key] = QueueEntry(
            key=key,
            approach=v.approach,
            arrived=arrived,
            label=v.label,
            track_id=v.track_id,
            last_seen=t,
            box=v.box,
            clipped=v.clipped,
            arrival_known=v.arrival_known,
        )
        detail = {"key": key, "approach": v.approach.value, "arrived": round(arrived, 2), "known": v.arrival_known}
        return QueueEvent("vehicle_joined", t, dict(detail))

    def _observe(self, entry: QueueEntry, v: WaitingVehicle, t: float, ego_still: bool) -> list[QueueEvent]:
        """Update an entry from this frame's sighting; decide departures."""
        cfg = self.cfg
        events: list[QueueEvent] = []
        entry.last_seen, entry.lost, entry.box, entry.clipped = t, False, v.box, v.clipped
        entry.track_id = v.track_id
        entry.last_moving_for, entry.last_moved = v.moving_for, v.moved_from_stop
        if entry.departed:
            # Came to rest again on its own approach right after "departing": it was a creep, not a departure.
            if (
                v.stopped
                and v.approach == entry.approach
                and entry.departed_at is not None
                and t - entry.departed_at <= cfg.undepart_s
            ):
                entry.departed, entry.departed_at = False, None
                events.append(QueueEvent("vehicle_returned", t, {"key": entry.key, "approach": entry.approach.value}))
            return events
        # Followed by track id across zones (a turning car leaves its approach zone): departure is judged on
        # sustained motion and distance only, so a creep across a zone boundary is not a departure.
        if (
            ego_still
            and not v.stopped
            and v.moving_for >= cfg.depart_moving_s
            and v.moved_from_stop >= cfg.depart_moved
        ):
            entry.departed, entry.departed_at = True, t
            events.append(
                QueueEvent(
                    "vehicle_departed", t, {"key": entry.key, "approach": entry.approach.value, "how": "drove off"}
                )
            )
        return events

    def _unseen(self, seen_keys: set[str], t: float, ego_still: bool) -> list[QueueEvent]:
        """Entries not seen this frame: lost while driving off = departed; lost too long / parked = dropped."""
        cfg = self.cfg
        events: list[QueueEvent] = []
        assert self._ego_arrived is not None
        ahead = set()
        for entry in self._ranked()[0]:
            if entry.is_ego:
                break
            ahead.add(entry.key)
        for key, entry in list(self._entries.items()):
            if entry.is_ego or entry.departed:
                continue
            if key not in seen_keys and not entry.lost:
                entry.lost = True
                if (
                    ego_still
                    and entry.last_moving_for >= cfg.depart_lost_moving_s
                    and entry.last_moved >= cfg.depart_lost_moved
                ):
                    entry.departed, entry.departed_at = True, t
                    detail = {"key": key, "approach": entry.approach.value, "how": "drove out of view"}
                    events.append(QueueEvent("vehicle_departed", t, dict(detail)))
                    continue
            reason = None
            if entry.lost and t - entry.last_seen > cfg.lost_timeout_s:
                reason = "lost from view"
            elif entry.last_moving_for == 0.0 and t - max(entry.arrived, self._ego_arrived) > cfg.parked_after_s:
                reason = "probably parked"
            if reason is not None:
                del self._entries[key]
                entry.drop_reason = reason
                if reason == "probably parked":
                    self._parked.add(key)
                if key in ahead:
                    self._dropped_ahead.append(replace(entry))
                events.append(QueueEvent("vehicle_dropped", t, {"key": key, "reason": reason}))
        return events

    def _update_ego(self, t: float, ego: EgoStop) -> list[QueueEvent]:
        events: list[QueueEvent] = []
        if self._ego_arrived is None:
            if ego.at_stop and ego.stopped_since is not None:
                self._ego_arrived = ego.stopped_since
                self._your_turn_fired = False
                self._had_cars_ahead = False
                self._dropped_ahead = []
                self._entries[EGO_KEY] = QueueEntry(
                    key=EGO_KEY, approach=Approach.US, arrived=ego.stopped_since, label="you", last_seen=t
                )
                events.append(QueueEvent("arrived", t, {"arrived": round(ego.stopped_since, 2)}))
        elif ego.moving_for_s >= self.cfg.ego_depart_s:
            # Creeping forward to see is not leaving; moving continuously for a while is.
            events.append(QueueEvent("left_intersection", t, {"waited_s": round(t - self._ego_arrived, 1)}))
            self.reset()
            self._t = t
        else:
            self._entries[EGO_KEY].last_seen = t
        return events

    def _key_for(self, v: WaitingVehicle) -> str:
        """Entry key for a vehicle; a new *stopped* track that overlaps a lost entry's box re-links to it (a
        moving track never inherits a waiting car's place, so a crossing car can't make it look departed)."""
        key = self._aliases.get(v.track_id, f"track:{v.track_id}")
        if key in self._entries or not v.stopped:
            return key
        for entry in self._entries.values():
            if (
                entry.lost
                and not entry.departed
                and entry.box is not None
                and entry.approach == v.approach
                and _iou(entry.box, v.box) >= self.cfg.relink_iou
            ):
                log.info("re-linked lost queue entry", entry=entry.key, new_track=v.track_id)
                self._aliases[v.track_id] = entry.key
                return entry.key
        return key

    # ---------------------------------------------------------------------------------- reading

    def _ranked(self) -> tuple[list[QueueEntry], list[list[str]], bool, int | None, bool]:
        """(order, tie groups, deadlock, our position, your turn) on the live entries (no copies)."""
        waiting = [e for e in self._entries.values() if not e.departed]
        ordered, tie_groups, deadlock = right_of_way_order(waiting, self.cfg.tie_window_s)
        ego_pos = next((i + 1 for i, e in enumerate(ordered) if e.is_ego), None)
        active = self._ego_arrived is not None
        your_turn = active and ego_pos == 1 and not deadlock_blocks_ego(ordered, tie_groups, deadlock)
        return ordered, tie_groups, deadlock, ego_pos, your_turn

    def snapshot(self) -> QueueSnapshot:
        """Current queue state (copies: safe to hand to other threads)."""
        active = self._ego_arrived is not None
        departed = sorted((e for e in self._entries.values() if e.departed), key=lambda e: e.departed_at or 0.0)
        ordered, tie_groups, deadlock, ego_pos, your_turn = self._ranked()
        return QueueSnapshot(
            t=self._t,
            active=active,
            order=tuple(replace(e) for e in ordered),
            departed=tuple(replace(e) for e in departed),
            ego_position=ego_pos,
            tie_groups=tuple(tuple(g) for g in tie_groups),
            deadlock=deadlock,
            your_turn=your_turn,
            dropped=tuple(replace(e) for e in self._dropped_ahead),
        )
