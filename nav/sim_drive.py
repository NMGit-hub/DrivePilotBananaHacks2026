"""Simulated drives along a route, as a phone's GPS would report them (HANDOFF.md 5B, V2_BUILD_PLAN.md V2-M0).

Navigation only needs GPS, so a realistic drive along a real (saved) Mapbox route tests destinations, lane
reminders, roundabout guidance, arrival, missed turns and rerouting at home. `simulate` drives a `Route`:

- **Speeds.** Each stretch (step) gets a cruise speed from Mapbox's step distance / duration. Those averages
  include waiting at lights and junctions, so they are scaled up by `cruise_factor`, then kept between
  `min_cruise_mps` and `max_cruise_mps`. (The saved responses carry no per-segment `annotation` speeds, checked
  2026-10-10, so step averages are what there is.) Each maneuver caps the speed through it by its kind: left/right
  turns and roundabouts ~15-17 mph, slight turns ~27 mph, ramps/forks/merges ~40 mph, over `turn_span_m`; the
  roundabout ring is driven at roundabout speed. A forward pass (speeding up at `accel_mps2`) and a backward pass
  (braking at `decel_mps2`) over a fine grid turn those caps into the speed profile: the car starts from rest,
  brakes before each maneuver, speeds up after it and stops at the destination.
- **Stops.** Optionally the car stands still for `stop_s`, `stop_line_m` before chosen maneuvers (a stop sign, a
  red light). It also waits `start_wait_s` before driving off (time to say the destination) and `end_wait_s` at
  the destination.
- **A missed turn.** Optionally, at a chosen step, the car goes straight on (the bearing it came in on) at the
  speed of the road it was on instead of turning, until it is at least `missed_turn_m` past the maneuver and
  `missed_turn_clear_m` from the route line. The track ends there: the new route a reroute would bring doesn't
  exist offline.
- **Fixes** every `fix_interval_s`, like a phone reports them: the position with an error that wanders slowly (a
  first-order Gauss-Markov process per axis, std `noise_m`, correlation time `noise_tau_s`; real GPS errors drift
  over tens of seconds rather than jumping every fix), the speed (Doppler-like: the true speed plus a little white
  noise) and the course (the direction of travel plus a little noise; none while nearly standing still, like
  phones).

Deterministic for a seed. `SimDrive.to_gpx` writes the track in the GPX 1.0 form of `nav/bdd_info.py` (speed,
course, a `video_start` waypoint at the first fix), so the app replays it with `--gpx`, with any clip on loop
as the camera (`--set video.file.loop=true`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from nav.geo import LatLon, Polyline, bearing_deg, destination
from nav.gps import GpsFix, TrackPoint, write_gpx
from nav.route import ROUNDABOUT_TYPES, Route, Step

MPH_PER_MPS = 2.23694
DEFAULT_START_TIME = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)  # GPX time of the first fix (9 am in Seattle)
_EXIT_RING = {"exit roundabout", "exit rotary"}
_RAMPS = {"on ramp", "off ramp", "fork", "merge"}
_SLIGHT_OR_STRAIGHT = {"slight left", "slight right", "straight"}


@dataclass(frozen=True, slots=True)
class SimParams:
    """Every number the simulation uses (speeds in m/s, distances in m, times in s)."""

    # Cruise speed per stretch: Mapbox's step average x cruise_factor, kept within [min, max].
    cruise_factor: float = 1.15  # step averages include waits at lights; the speed between them is higher
    min_cruise_mps: float = 3.0  # ~7 mph: parking lots and driveways
    max_cruise_mps: float = 29.0  # ~65 mph
    default_cruise_mps: float = 13.4  # ~30 mph for a step without a duration
    # Speed through a maneuver, by its kind (or its modifier for turns).
    turn_mps: float = 7.5  # ~17 mph: left / right turns, end of road
    sharp_turn_mps: float = 5.5  # ~12 mph
    slight_turn_mps: float = 12.0  # ~27 mph: bearing slightly left / right
    uturn_mps: float = 3.5  # ~8 mph
    roundabout_mps: float = 7.0  # ~16 mph: into, round and out of a roundabout
    ramp_mps: float = 18.0  # ~40 mph: highway ramps, forks and merges
    turn_span_m: float = 12.0  # the maneuver speed holds over this stretch, centred on the maneuver point
    # Speeding up and braking.
    accel_mps2: float = 2.0  # 0-30 mph in ~7 s
    decel_mps2: float = 2.5  # comfortable braking
    # Waiting.
    start_wait_s: float = 15.0  # standing at the start (time to say or type the destination in the app)
    end_wait_s: float = 5.0  # standing at the destination
    stop_s: float = 20.0  # each requested stop (stop sign, red light)
    stop_line_m: float = 5.0  # stop this far before the maneuver point
    # A missed turn: straight on until both are true (or give up after missed_turn_max_m).
    missed_turn_m: float = 400.0  # at least this far past the maneuver...
    missed_turn_clear_m: float = 100.0  # ...and at least this far from the route line (2x nav.off_route_m)
    missed_turn_max_m: float = 3000.0
    missed_turn_check_m: float = 10.0  # how often the distance from the route is checked along the way
    bearing_window_m: float = 20.0  # the incoming bearing is measured over this much route before the maneuver
    # GPS fixes.
    fix_interval_s: float = 1.0  # phones report once a second
    noise_m: float = 3.0  # std of the position error per axis (east, north): ~4 m horizontal RMS
    noise_tau_s: float = 30.0  # correlation time of the position error
    speed_noise_mps: float = 0.2
    course_noise_deg: float = 2.0
    course_min_speed_mps: float = 1.0  # below this the phone reports no course
    grid_m: float = 1.0  # spacing of the speed-profile grid

    def __post_init__(self) -> None:
        positive = ("max_cruise_mps", "min_cruise_mps", "accel_mps2", "decel_mps2", "fix_interval_s", "grid_m")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.min_cruise_mps > self.max_cruise_mps:
            raise ValueError("min_cruise_mps must be <= max_cruise_mps")
        if self.noise_m < 0 or self.speed_noise_mps < 0 or self.course_noise_deg < 0:
            raise ValueError("noise levels can't be negative")


@dataclass(frozen=True, slots=True)
class SimFix:
    """One simulated GPS fix: what the phone reports, and the truth behind it."""

    t: float  # seconds from the first fix (the video's t=0 in a replay)
    lat: float  # reported (with the position error)
    lon: float
    speed_mps: float  # reported
    course_deg: float | None  # reported; None while (nearly) standing still
    true_lat: float
    true_lon: float
    true_speed_mps: float
    true_heading: float
    along_m: float  # true distance driven along the path (the route, until a missed turn)

    @property
    def latlon(self) -> LatLon:
        """Reported (lat, lon)."""
        return self.lat, self.lon

    @property
    def true_latlon(self) -> LatLon:
        """True (lat, lon)."""
        return self.true_lat, self.true_lon

    def gps_fix(self) -> GpsFix:
        """As the app sees it."""
        return GpsFix(self.t, self.lat, self.lon, self.speed_mps, self.course_deg)


@dataclass(frozen=True, slots=True)
class ManeuverPass:
    """When the car reaches a maneuver point."""

    step: Step
    t: float
    speed_mps: float
    missed: bool = False  # went straight on instead


@dataclass(frozen=True, slots=True)
class StopRecord:
    """A requested stop before a maneuver."""

    step: Step
    t_start: float
    t_end: float


@dataclass
class SimDrive:
    """A simulated drive: the fixes plus what happened when."""

    route: Route
    path: Polyline  # what the car drove: the route, or the route up to a missed turn and then straight on
    on_route_m: float  # the path follows the route up to here
    fixes: list[SimFix]
    maneuvers: list[ManeuverPass]
    stops: list[StopRecord]
    missed: Step | None
    params: SimParams
    seed: int
    knots: tuple[np.ndarray, np.ndarray, np.ndarray] = field(repr=False)  # (t, along, speed), exact truth

    @property
    def duration_s(self) -> float:
        """Time of the last fix."""
        return self.fixes[-1].t

    @property
    def distance_m(self) -> float:
        """Length of the path driven."""
        return self.path.length_m

    @property
    def moving_s(self) -> float:
        """Time spent moving (not standing at the start, at stops or at the destination)."""
        t, s, _ = self.knots
        dt, ds = np.diff(t), np.diff(s)
        return float(dt[ds > 0].sum())

    @property
    def max_speed_mps(self) -> float:
        """Highest true speed."""
        return float(self.knots[2].max())

    def along_at(self, t: float) -> float:
        """True distance driven along the path at time `t`."""
        return float(np.interp(t, self.knots[0], self.knots[1]))

    def speed_at(self, t: float) -> float:
        """True speed at time `t`."""
        return float(np.interp(t, self.knots[0], self.knots[2]))

    def time_at(self, along_m: float) -> float:
        """When the car first gets `along_m` along the path."""
        t, s, _ = self.knots
        i = int(np.searchsorted(s, along_m, side="left"))
        if i <= 0:
            return float(t[0])
        if i >= len(s):
            return float(t[-1])
        s0, s1, t0, t1 = s[i - 1], s[i], t[i - 1], t[i]
        return float(t0 if s1 <= s0 else t0 + (t1 - t0) * (along_m - s0) / (s1 - s0))

    def time_off_route(self, threshold_m: float, step_m: float = 1.0) -> float | None:
        """When the car's true position first gets more than `threshold_m` from the route line (anywhere on it),
        or None if it never does (no missed turn)."""
        along = self.on_route_m
        while along <= self.path.length_m:
            if self.route.line.snap(self.path.point_at(along)).offset_m > threshold_m:
                return self.time_at(along)
            along += step_m
        return None

    def to_gpx(self, start_time: datetime = DEFAULT_START_TIME) -> str:
        """The track as GPX 1.0 (speed, course, a `video_start` waypoint at the first fix = video t=0)."""
        points = [
            TrackPoint(
                start_time + timedelta(seconds=f.t),
                round(f.lat, 7),
                round(f.lon, 7),
                round(f.speed_mps, 2),
                None if f.course_deg is None else round(f.course_deg, 1),
            )
            for f in self.fixes
        ]
        stops = ", ".join(str(s.step.index) for s in self.stops) or "none"
        missed = self.missed.index if self.missed is not None else "none"
        description = (
            f"Simulated drive by scripts/route_to_gpx.py: seed {self.seed}, noise {self.params.noise_m} m, "
            f"stops before steps {stops}, missed turn at step {missed}"
        )
        return write_gpx(
            [points], start_time, f"Simulated drive to {self.route.destination}", description, "simulated drive"
        )


# ----------------------------------------------------------------------------------------------- speeds


def cruise_speed(step: Step, params: SimParams) -> float:
    """Cruise speed on a step's stretch of road."""
    if step.distance_m > 0 and step.duration_s > 0:
        speed = step.distance_m / step.duration_s * params.cruise_factor
    else:
        speed = params.default_cruise_mps
    speed = min(max(speed, params.min_cruise_mps), params.max_cruise_mps)
    return min(speed, params.roundabout_mps) if step.is_roundabout else speed


def maneuver_speed(step: Step, params: SimParams) -> float:
    """The most the car drives through a maneuver (inf: no slowdown, e.g. straight on)."""
    kind, modifier = step.maneuver, step.modifier or "straight"
    if kind in ("depart", "arrive"):
        return math.inf
    if kind in ROUNDABOUT_TYPES or kind in _EXIT_RING:
        return params.roundabout_mps
    by_modifier = {
        "uturn": params.uturn_mps,
        "sharp left": params.sharp_turn_mps,
        "sharp right": params.sharp_turn_mps,
        "left": params.turn_mps,
        "right": params.turn_mps,
        "slight left": params.slight_turn_mps,
        "slight right": params.slight_turn_mps,
    }.get(modifier, math.inf)
    if kind in _RAMPS:
        return params.ramp_mps if modifier in _SLIGHT_OR_STRAIGHT else min(params.ramp_mps, by_modifier)
    return by_modifier


def gauss_markov(n: int, sigma: float, tau_s: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """`n` samples, `dt` apart, of a stationary first-order Gauss-Markov process: std `sigma`, correlation time
    `tau_s` (autocorrelation exp(-lag / tau)). A slow, mean-reverting random walk."""
    a = math.exp(-dt / tau_s) if tau_s > 0 else 0.0
    b = sigma * math.sqrt(1.0 - a * a)
    w = rng.standard_normal(n)
    out = np.empty(n)
    x = sigma * w[0] if n else 0.0
    for k in range(n):
        x = x if k == 0 else a * x + b * w[k]
        out[k] = x
    return out


# ------------------------------------------------------------------------------------------------ path


def _incoming_bearing(line: Polyline, along_m: float, window_m: float) -> float:
    return bearing_deg(line.point_at(along_m - window_m), line.point_at(along_m))


def _missed_turn_path(route: Route, step: Step, params: SimParams) -> Polyline:
    """The route up to the maneuver, then straight on until well away from the route."""
    line, corner_m = route.line, step.start_m
    corner = line.point_at(corner_m)
    heading = _incoming_bearing(line, corner_m, params.bearing_window_m)
    driven = 0.0
    while True:
        driven += params.missed_turn_check_m
        point = destination(corner, heading, driven)
        if driven >= params.missed_turn_m and line.snap(point).offset_m >= params.missed_turn_clear_m:
            break
        if driven >= params.missed_turn_max_m:
            raise ValueError(
                f"going straight at step {step.index} ({step.instruction}) stays within "
                f"{params.missed_turn_clear_m:.0f} m of the route for {params.missed_turn_max_m:.0f} m: nothing to miss"
            )
    kept = [p for p, along in zip(line.points, line.cumulative, strict=True) if along < corner_m]
    return Polyline([*kept, corner, destination(corner, heading, driven)])


# --------------------------------------------------------------------------------------------- simulate


def _validate(route: Route, stops_before: Iterable[int], missed_turn: int | None) -> tuple[set[int], Step | None]:
    last = len(route.steps) - 1
    missed = None
    if missed_turn is not None:
        if not 1 <= missed_turn < last:
            raise ValueError(f"missed turn: step {missed_turn} is not a maneuver between the start and the end")
        missed = route.steps[missed_turn]
    stops = set(stops_before)
    for index in stops:
        if not 1 <= index < last:
            raise ValueError(f"stop: step {index} is not a maneuver between the start and the end")
        if missed_turn is not None and index > missed_turn:
            raise ValueError(f"stop: step {index} comes after the missed turn (step {missed_turn})")
    return stops, missed


def simulate(
    route: Route,
    params: SimParams | None = None,
    stops_before: Iterable[int] = (),
    missed_turn: int | None = None,
    seed: int = 0,
) -> SimDrive:
    """Drive `route` (see the module docstring). `stops_before`: step indices to stop before; `missed_turn`: the
    step index where the car goes straight on instead (the track ends past it)."""
    p = params or SimParams()
    stop_steps, missed = _validate(route, stops_before, missed_turn)
    if missed is None:
        path, on_route_m = route.line, route.line.length_m
    else:
        path, on_route_m = _missed_turn_path(route, missed, p), missed.start_m
    length = path.length_m
    n = math.ceil(length / p.grid_m) + 1
    if n < 4:
        raise ValueError("the route is too short to drive")
    s = np.minimum(np.arange(n) * p.grid_m, length)

    # Speed caps: the stretch's cruise speed, then each maneuver's speed around its point.
    starts = np.array([step.start_m for step in route.steps])
    which = np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(route.steps) - 1)
    if missed is not None:
        which[s >= on_route_m] = missed.index - 1  # straight on: still the road we came in on
    cruise = np.array([cruise_speed(step, p) for step in route.steps])
    cap = cruise[which]
    for step in route.steps[1 : (missed.index if missed is not None else None)]:
        limit = maneuver_speed(step, p)
        if math.isfinite(limit):
            near = np.abs(s - step.start_m) <= p.turn_span_m / 2
            near[min(round(step.start_m / p.grid_m), n - 1)] = True
            cap[near] = np.minimum(cap[near], limit)

    # Where the car stands still, and for how long: the start, each stop, the destination.
    dwell: dict[int, float] = {0: p.start_wait_s}
    stop_at: dict[int, int] = {}  # step index -> grid index
    for index in sorted(stop_steps):
        j = round(max(route.steps[index].start_m - p.stop_line_m, 0.0) / p.grid_m)
        j = 0 if j <= 1 else (n - 1 if j >= n - 2 and missed is None else min(j, n - 3))
        j = next((k for k in dwell if abs(k - j) <= 1), j)  # two stand-stills side by side are one
        dwell[j] = dwell.get(j, 0.0) + p.stop_s
        stop_at[index] = j
    if missed is None:
        dwell[n - 1] = dwell.get(n - 1, 0.0) + p.end_wait_s
    for j in dwell:
        cap[j] = 0.0
    if missed is None:
        cap[n - 1] = 0.0

    # Forward (speeding up) and backward (braking) passes.
    v = cap.copy()
    ds = np.diff(s)
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * p.accel_mps2 * ds[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * p.decel_mps2 * ds[i]))

    # Time along the grid (exact for constant acceleration between grid points), with the waits as knots.
    reach = np.empty(n)  # when the car first reaches each grid point
    kt: list[float] = [0.0]
    ks: list[float] = [0.0]
    kv: list[float] = [0.0]
    t = 0.0
    for i in range(n):
        if i:
            pace = v[i - 1] + v[i]
            if pace <= 0:
                raise ValueError("speed profile stands still between two grid points")  # (zeros are kept apart)
            t += 2.0 * ds[i - 1] / pace
            kt.append(t)
            ks.append(float(s[i]))
            kv.append(float(v[i]))
        reach[i] = t
        if dwell.get(i, 0.0) > 0:
            t += dwell[i]
            kt.append(t)
            ks.append(float(s[i]))
            kv.append(0.0)
    knots = (np.array(kt), np.array(ks), np.array(kv))

    # Fixes: truth at each fix time, then the phone's errors.
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, t + 1e-9, p.fix_interval_s)
    along = np.interp(times, knots[0], knots[1])
    speed = np.interp(times, knots[0], knots[2])
    m = len(times)
    east = gauss_markov(m, p.noise_m, p.noise_tau_s, p.fix_interval_s, rng)
    north = gauss_markov(m, p.noise_m, p.noise_tau_s, p.fix_interval_s, rng)
    speed_err = rng.standard_normal(m) * p.speed_noise_mps
    course_err = rng.standard_normal(m) * p.course_noise_deg
    fixes = []
    for k in range(m):
        true = path.point_at(float(along[k]))
        heading = path.bearing_at(float(along[k]))
        seen = destination(destination(true, 90.0, float(east[k])), 0.0, float(north[k]))
        course = (heading + float(course_err[k])) % 360.0 if speed[k] >= p.course_min_speed_mps else None
        fixes.append(
            SimFix(
                t=float(times[k]),
                lat=seen[0],
                lon=seen[1],
                speed_mps=max(0.0, float(speed[k] + speed_err[k])),
                course_deg=course,
                true_lat=true[0],
                true_lon=true[1],
                true_speed_mps=float(speed[k]),
                true_heading=heading,
                along_m=float(along[k]),
            )
        )

    passes = [
        ManeuverPass(
            step,
            float(np.interp(step.start_m, s, reach)),
            float(np.interp(step.start_m, s, v)),
            missed=step is missed,
        )
        for step in route.steps[1:]
        if step.start_m <= on_route_m
    ]
    stops = [
        StopRecord(route.steps[i], float(reach[j]), float(reach[j]) + p.stop_s) for i, j in sorted(stop_at.items())
    ]
    return SimDrive(route, path, on_route_m, fixes, passes, stops, missed, p, seed, knots)


def summary_lines(drive: SimDrive) -> list[str]:
    """A readable summary: distance, time, speeds, and each maneuver with when we reach it."""
    p = drive.params
    average = drive.distance_m / drive.moving_s if drive.moving_s > 0 else 0.0
    lines = [
        f"{drive.route.destination}: {drive.distance_m / 1609.344:.2f} mi ({drive.distance_m:.0f} m) in "
        f"{drive.duration_s / 60:.1f} min ({len(drive.fixes)} fixes, {p.fix_interval_s:g} s apart); "
        f"moving average {average * MPH_PER_MPS:.0f} mph, max {drive.max_speed_mps * MPH_PER_MPS:.0f} mph; "
        f"GPS noise {p.noise_m:g} m per axis (tau {p.noise_tau_s:g} s), seed {drive.seed}",
    ]
    stops = {stop.step.index: stop for stop in drive.stops}
    for maneuver in drive.maneuvers:
        step = maneuver.step
        note = ""
        if step.index in stops:
            stop = stops[step.index]
            note += f"  [stop {stop.t_end - stop.t_start:.0f} s from t={stop.t_start:.0f}]"
        if maneuver.missed:
            note += f"  [MISSED: straight on for {drive.distance_m - drive.on_route_m:.0f} m; the track ends there]"
        lanes = "".join("V" if lane.valid else "x" for lane in step.lanes)
        lanes = f" lanes[{lanes}]" if lanes else ""
        lines.append(
            f"  t={maneuver.t:7.1f} s  {step.start_m:7.0f} m  {maneuver.speed_mps * MPH_PER_MPS:3.0f} mph  "
            f"step {step.index:2d} {step.maneuver} {step.modifier or ''}: {step.instruction}{lanes}{note}"
        )
    return lines
