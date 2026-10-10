"""Score the parking tracker (state/parking.py) against hand labels of real parking-lot clips (V2-M1).

The labels (`eval/parking_labels.yaml`) list, per clip: when we are in a lot, when we park, the open spots in the
row next to our path (side, first clearly visible and empty, when it leaves the frame, restriction, certainty),
look-alikes that must not be called open spots, and cars pulling out. `replay` runs a `ParkingTracker` over cached
perception results (eval/perception_cache.py) exactly as `WorldState` drives it; `score_clip` ties its events to
the labels:
- **spots**: a "spot" event on side S at time t hits a labeled spot on S when t_visible − `early_s` ≤ t ≤ t_pass +
  `late_s` (t_pass null = the clip's end). Clear, unrestricted spots are the recall targets ("announced before
  we pass them"); ambiguous or restricted ones (the model check skips restricted ones) neither count nor fail.
  An event that hits no labeled spot is a false alarm (a look-alike hit is reported as such).
- **candidates** (the earlier event that starts the model check) are scored the same way, for reference.
- **lot**: a "lot" event inside an `in_lot` window is right, outside one it's a false offer.
- **parked**: a "parked" event from `parked_tol_s` before `parked_at` to `parked_delay_s` + `parked_tol_s` after
  it is right (the tracker waits `parking.parked_after_s` before it calls a stop parked); any other is false.
- **pulling out**: within `pull_tol_s` of a labeled one on the same side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.config import ParkingConfig
from eval.perception_cache import CachedFrame
from state.parking import ParkingEvent, ParkingTracker

RECALL_RESTRICTIONS = ("none", "ev", "compact")  # announced (EV-only and compact as such); the rest are skipped


@dataclass(frozen=True, slots=True)
class LabeledSpot:
    """An open spot (or run of stalls) in the row next to our path."""

    side: str
    t_visible: float
    t_pass: float | None
    restriction: str
    certainty: str
    stalls: int
    neighbor_before: str | None
    note: str = ""

    @property
    def target(self) -> bool:
        """Counts for recall: clearly open and not a restriction the watch skips."""
        return self.certainty == "clear" and self.restriction in RECALL_RESTRICTIONS


@dataclass(frozen=True, slots=True)
class LookAlike:
    """Something that must not be called an open spot."""

    side: str
    t_start: float
    t_end: float
    what: str


@dataclass(frozen=True, slots=True)
class PullOut:
    """Another car backing or pulling out of a stall."""

    t: float
    side: str


@dataclass(frozen=True, slots=True)
class ClipLabels:
    """The hand labels of one clip."""

    id: str
    scene: str
    usable: bool
    in_lot: tuple[tuple[float, float], ...]
    parked_at: float | None
    spots: tuple[LabeledSpot, ...]
    look_alikes: tuple[LookAlike, ...]
    pull_outs: tuple[PullOut, ...]

    @property
    def lot_s(self) -> float:
        """Seconds of lot driving."""
        return sum(b - a for a, b in self.in_lot)


def load_labels(path: Path) -> list[ClipLabels]:
    """Parse the labels YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for c in data["clips"]:
        spots = tuple(
            LabeledSpot(
                side=s["side"],
                t_visible=float(s["t_visible"]),
                t_pass=None if s.get("t_pass") is None else float(s["t_pass"]),
                restriction=s.get("restriction", "unknown"),
                certainty=s.get("certainty", "ambiguous"),
                stalls=int(s.get("stalls") or 1),
                neighbor_before=s.get("neighbor_before"),
                note=s.get("note", ""),
            )
            for s in c.get("open_spots") or []
        )
        looks = tuple(
            LookAlike(a["side"], float(a["t_start"]), float(a["t_end"]), a.get("what", ""))
            for a in c.get("look_alikes") or []
        )
        pulls = tuple(PullOut(float(p["t"]), p["side"]) for p in c.get("pulling_out") or [])
        out.append(
            ClipLabels(
                id=c["id"],
                scene=c.get("scene", "other"),
                usable=bool(c.get("usable", True)),
                in_lot=tuple((float(a), float(b)) for a, b in c.get("in_lot") or []),
                parked_at=None if c.get("parked_at") is None else float(c["parked_at"]),
                spots=spots,
                look_alikes=looks,
                pull_outs=pulls,
            )
        )
    return out


def replay(
    frames: list[CachedFrame],
    cfg: ParkingConfig,
    hfov_deg: float,
    gps_speed: Callable[[float], float | None] | None = None,
) -> list[ParkingEvent]:
    """Every parking event of the clip, frame by frame as WorldState drives the tracker (`gps_speed(t)`: the
    GPS speed at video time t, None without a fix)."""
    tracker = ParkingTracker(cfg, hfov_deg)
    events: list[ParkingEvent] = []
    for f in frames:
        speed = gps_speed(f.t) if gps_speed is not None else None
        events += tracker.update(f.t, f.width, f.height, f.tracks, f.ego_stopped, f.homography, speed)
    return events


@dataclass(slots=True)
class EventMatch:
    """One spot/candidate event and what it matched."""

    event: ParkingEvent
    spot: LabeledSpot | None = None  # the labeled spot it hit
    look_alike: LookAlike | None = None  # a look-alike window it fell in (if it hit no spot)

    @property
    def false_alarm(self) -> bool:
        """Matches no labeled spot."""
        return self.spot is None


@dataclass(slots=True)
class ClipScore:
    """Everything scored on one clip."""

    clip: ClipLabels
    duration: float
    spots: list[EventMatch] = field(default_factory=list)
    candidates: list[EventMatch] = field(default_factory=list)
    hits: dict[int, float] = field(default_factory=dict)  # target spot index -> time of its first spot event
    candidate_hits: dict[int, float] = field(default_factory=dict)
    lot_events: list[tuple[float, bool]] = field(default_factory=list)  # (t, inside an in_lot window)
    parked_events: list[tuple[float, bool]] = field(default_factory=list)  # (t, right)
    pull_events: list[tuple[float, str, bool]] = field(default_factory=list)  # (t, side, right)
    pulls_found: int = 0
    taken_events: int = 0

    @property
    def targets(self) -> list[int]:
        """Indexes of the recall targets."""
        return [i for i, s in enumerate(self.clip.spots) if s.target]

    @property
    def false_spots(self) -> list[EventMatch]:
        """Spot events that hit no labeled spot."""
        return [m for m in self.spots if m.false_alarm]


def match_report(
    t: float, side: str | None, clip: ClipLabels, duration: float, early_s: float = 1.0, late_s: float = 0.5
) -> tuple[LabeledSpot | None, LookAlike | None]:
    """The labeled spot an "open spot on <side>" report at time t is about (the clearest, then the nearest in
    time), or else the look-alike window it falls in (a false alarm on a look-alike), or (None, None)."""
    best: tuple[int, float, LabeledSpot] | None = None
    for spot in clip.spots:
        if spot.side != side:
            continue
        end = duration if spot.t_pass is None else spot.t_pass
        if spot.t_visible - early_s <= t <= end + late_s:
            rank = (0 if spot.target else 1, abs(t - spot.t_visible))
            if best is None or rank < best[:2]:
                best = (rank[0], rank[1], spot)
    if best is not None:
        return best[2], None
    for look in clip.look_alikes:
        if look.side == side and look.t_start - early_s <= t <= look.t_end + late_s:
            return None, look
    return None, None


def _match(event: ParkingEvent, clip: ClipLabels, duration: float, early_s: float, late_s: float) -> EventMatch:
    side = event.space.side if event.space is not None else None
    spot, look = match_report(event.t, side, clip, duration, early_s, late_s)
    return EventMatch(event, spot=spot, look_alike=look)


def score_clip(
    clip: ClipLabels,
    events: list[ParkingEvent],
    duration: float,
    early_s: float = 1.0,
    late_s: float = 0.5,
    parked_tol_s: float = 3.0,
    parked_delay_s: float = 10.0,
    pull_tol_s: float = 3.0,
) -> ClipScore:
    """Tie a clip's parking events to its labels."""
    score = ClipScore(clip, duration)
    index = {id(s): i for i, s in enumerate(clip.spots)}
    pulls_hit: set[int] = set()
    for e in events:
        if e.kind in ("spot", "candidate") and e.space is not None:
            m = _match(e, clip, duration, early_s, late_s)
            (score.spots if e.kind == "spot" else score.candidates).append(m)
            if m.spot is not None and m.spot.target:
                hits = score.hits if e.kind == "spot" else score.candidate_hits
                hits.setdefault(index[id(m.spot)], e.t)
        elif e.kind == "lot":
            score.lot_events.append((e.t, any(a <= e.t <= b for a, b in clip.in_lot)))
        elif e.kind == "parked":
            right = (
                clip.parked_at is not None
                and clip.parked_at - parked_tol_s <= e.t <= clip.parked_at + parked_delay_s + parked_tol_s
            )
            score.parked_events.append((e.t, right))
        elif e.kind == "pulling_out" and e.car is not None:
            hit = next(
                (
                    i
                    for i, p in enumerate(clip.pull_outs)
                    if p.side == e.car.side and abs(p.t - e.t) <= pull_tol_s and i not in pulls_hit
                ),
                None,
            )
            if hit is not None:
                pulls_hit.add(hit)
            score.pull_events.append((e.t, e.car.side, hit is not None))
        elif e.kind == "taken":
            score.taken_events += 1
    score.pulls_found = len(pulls_hit)
    return score


@dataclass(frozen=True, slots=True)
class Summary:
    """Totals over clips."""

    targets: int
    found: int
    found_with_neighbour: int
    targets_with_neighbour: int
    candidate_found: int
    false_spots: int
    look_alike_hits: int
    spot_events: int
    lot_minutes: float
    lot_right: int
    lot_false: int
    lot_clips: int
    lot_clips_offered: int
    parked_right: int
    parked_false: int
    parked_labels: int
    pulls_found: int
    pull_labels: int
    pull_false: int
    delays: tuple[float, ...]  # spot event time − t_visible for each found target

    @property
    def recall(self) -> float:
        """Share of clear, unrestricted spots announced in time."""
        return self.found / self.targets if self.targets else float("nan")

    @property
    def false_per_10min(self) -> float:
        """False 'open spot' events per 10 minutes of lot driving."""
        return 10.0 * self.false_spots / self.lot_minutes if self.lot_minutes else float("nan")

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly."""
        d = {k: getattr(self, k) for k in self.__slots__}
        d["delays"] = list(self.delays)
        d["recall"] = self.recall
        d["false_per_10min"] = self.false_per_10min
        return d


def summarize(scores: list[ClipScore]) -> Summary:
    """Totals over the scored clips."""
    targets = found = with_n = found_n = cand = false = looks = events = 0
    lot_right = lot_false = lot_clips = lot_offered = 0
    parked_right = parked_false = parked_labels = pulls = pull_labels = pull_false = 0
    delays: list[float] = []
    lot_s = 0.0
    for s in scores:
        clip = s.clip
        lot_s += clip.lot_s
        for i in s.targets:
            targets += 1
            spot = clip.spots[i]
            has_n = spot.neighbor_before is not None
            with_n += has_n
            if i in s.hits:
                found += 1
                found_n += has_n
                delays.append(s.hits[i] - spot.t_visible)
            cand += i in s.candidate_hits
        events += len(s.spots)
        false += len(s.false_spots)
        looks += sum(m.look_alike is not None for m in s.false_spots)
        lot_right += sum(ok for _, ok in s.lot_events)
        lot_false += sum(not ok for _, ok in s.lot_events)
        if clip.in_lot:
            lot_clips += 1
            lot_offered += any(ok for _, ok in s.lot_events)
        parked_right += sum(ok for _, ok in s.parked_events)
        parked_false += sum(not ok for _, ok in s.parked_events)
        parked_labels += clip.parked_at is not None
        pulls += s.pulls_found
        pull_labels += len(clip.pull_outs)
        pull_false += sum(not ok for *_, ok in s.pull_events)
    return Summary(
        targets=targets,
        found=found,
        found_with_neighbour=found_n,
        targets_with_neighbour=with_n,
        candidate_found=cand,
        false_spots=false,
        look_alike_hits=looks,
        spot_events=events,
        lot_minutes=lot_s / 60.0,
        lot_right=lot_right,
        lot_false=lot_false,
        lot_clips=lot_clips,
        lot_clips_offered=lot_offered,
        parked_right=parked_right,
        parked_false=parked_false,
        parked_labels=parked_labels,
        pulls_found=pulls,
        pull_labels=pull_labels,
        pull_false=pull_false,
        delays=tuple(delays),
    )
