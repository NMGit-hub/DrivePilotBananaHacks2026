"""Simulated drives through the real navigation handler: does each guidance alert come at the right place?

V2_BUILD_PLAN.md V2-M0: "an automated check confirms the lane reminders (~1 mi, ~0.25 mi), turn and arrival alerts
fire within 10% of their distances and a wrong turn reroutes". `check_drive` replays a `SimDrive` the way the app
does: its GPX through `GpxReplay`, read every `APP_POLL_S` on the video clock, each fix to `NavHandler.on_fix`,
with the destination set at t=0. The route comes from a saved Mapbox response (`SavedRouteClient`, no API call).
Every alert is recorded with the distance to the maneuver it is about, both from the car's true position and as
the handler measured it (from GPS with noise). Then it checks:
- **lane reminders**: one for each distance in `nav.lane_reminders_m` before every maneuver with lane data, within
  10% of that distance, for each distance we drove into on the route. (A maneuver closer than that to where
  navigation started gets its reminder at once, by design: reported, not checked.)
- **roundabouts**: "At the roundabout, take the 2nd exit..." within 10% of `ROUNDABOUT_ANNOUNCE_M` before the
  entry (or of the distance left once the previous maneuver is passed, or when navigation started, if less), and,
  when the exit count is known, one "Not this one" per exit we pass, near it, the last saying "next exit";
- **arrival**: once, near the end of the route;
- **reroutes**: after a missed turn, the reroute request comes `nav.off_route_s` (± `time_s`) after the car truly
  got more than `nav.off_route_m` from the route; never before; and never in a drive without a missed turn.
  (A turn missed within `NEAR_END_ALONG_M` of the destination may instead count as turning in at it, by design
  in nav/progress.py: then arrival near the destination passes too.)
- nothing else: an alert that can't be tied to a maneuver, or comes at the wrong place, fails the drive.
Offline there is no route from where a missed turn leads, so `SavedRouteClient` records the reroute request and
fails it (the handler then says it couldn't find a new route): the request is what is checked.
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import NavConfig
from nav.geo import LatLon, haversine_m
from nav.gps import GpsFix, GpxReplay
from nav.handler import NOT_THIS_EXIT_NEXT, ROUNDABOUT_ANNOUNCE_M, NavHandler
from nav.mapbox import NavError, Place
from nav.progress import NEAR_END_ALONG_M, Progress
from nav.route import Route, Step, parse_directions
from nav.sim_drive import SimDrive

APP_POLL_S = 0.5  # app.GPS_INTERVAL_S: the app reads the GPS (and feeds navigation) twice a second
REROUTE_ALERTS = ("nav_rerouted", "nav_reroute_failed")
REROUTE_WAIT_S = 5.0  # offline, the reroute thread answers at once; this only bounds a hang
NEXT_MANEUVER_AFTER_M = 5.0  # Route.next_maneuver's default: a maneuver counts as "next" until 5 m past it


@dataclass(frozen=True, slots=True)
class Tolerances:
    """How close is close enough."""

    distance_fraction: float = 0.10  # alerts within 10% of their distance (V2-M0)
    position_m: float = 15.0  # floor for alerts tied to a spot (exits, arrival): GPS noise + one poll of driving
    time_s: float = 2.0  # reroute timing: a fix interval plus the noise's effect on when the line is crossed


class SavedRouteClient:
    """A `RoutingClient` serving one saved Mapbox directions response. The first route request (setting the
    destination) gets it; later ones (reroutes from wherever the car went) are recorded and fail, since there is
    no route from there offline."""

    def __init__(self, place: Place, response: dict[str, Any]) -> None:
        self.place = place
        self.response = response
        self.requests: list[tuple[LatLon, float | None]] = []
        self.answered = 0
        self._lock = threading.Lock()

    def geocode(self, query: str, near: LatLon | None = None) -> Place:
        """Whatever was asked for: the saved place."""
        return self.place

    def route(self, origin: LatLon, destination: Place, heading: float | None = None) -> Route:
        """The saved route the first time; a NavError for reroutes."""
        with self._lock:
            self.requests.append((origin, heading))
            first = len(self.requests) == 1
        try:
            if not first:
                raise NavError("offline check: no route from here")
            return parse_directions(self.response, destination.name)
        finally:
            with self._lock:
                self.answered += 1


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """One spoken navigation alert, and where the car was."""

    t: float
    kind: str
    text: str
    step: Step | None = None  # the maneuver it is about
    target_m: float | None = None  # how far before it the alert should come (0: at the spot)
    true_m: float | None = None  # how far before it the alert came, from the car's true position
    nav_m: float | None = None  # ... as the handler measured it (from the GPS)
    ok: bool | None = None  # None: not checked (by design, or just informational)
    note: str = ""


@dataclass(frozen=True, slots=True)
class Check:
    """One expectation."""

    name: str
    ok: bool
    detail: str


@dataclass
class NavCheck:
    """The outcome of one simulated drive through the handler."""

    drive: SimDrive
    destination_answer: str
    alerts: list[AlertRecord]
    checks: list[Check] = field(default_factory=list)
    reroute_requests: list[float] = field(default_factory=list)  # fix time of each reroute request
    off_route_t: float | None = None  # first fix the handler called off route
    left_route_t: float | None = None  # when the car truly got more than nav.off_route_m from the route

    @property
    def ok(self) -> bool:
        """Every check passed."""
        return all(check.ok for check in self.checks)

    @property
    def reroute_t(self) -> float | None:
        """The first reroute request."""
        return self.reroute_requests[0] if self.reroute_requests else None


def _close(value: float, target: float, tol: Tolerances) -> bool:
    return abs(value - target) <= max(tol.distance_fraction * target, tol.position_m)


def roundabout_target_m(route: Route, step: Step) -> float:
    """How far before a roundabout its announcement is due: ROUNDABOUT_ANNOUNCE_M, or less when the previous
    maneuver is closer than that (the handler names the next maneuver only once the previous one is passed)."""
    previous = route.steps[step.index - 1]
    lead = step.start_m if previous.index == 0 else step.start_m - previous.start_m - NEXT_MANEUVER_AFTER_M
    return min(ROUNDABOUT_ANNOUNCE_M, lead)


class _Recorder:
    """Ties each alert to its maneuver (the handler doesn't say which) and measures where it came."""

    def __init__(self, drive: SimDrive, cfg: NavConfig, tol: Tolerances) -> None:
        self.drive, self.route, self.cfg, self.tol = drive, drive.route, cfg, tol
        self.lanes_said: set[tuple[int, float]] = set()
        self.exits_said: dict[int, int] = {}
        self.records: list[AlertRecord] = []

    def lane_windows(self, along: float) -> list[tuple[Step, float]]:
        """The lane reminders due at this position, in the handler's order (mirrors NavHandler._lane_reminders)."""
        out = []
        for step in self.route.upcoming(along, max(self.cfg.lane_reminders_m, default=0.0) + 50.0):
            if not step.needs_lanes:
                continue
            distance = step.start_m - along
            for threshold in sorted(self.cfg.lane_reminders_m):
                if threshold * 0.5 <= distance <= threshold and (step.index, threshold) not in self.lanes_said:
                    out.append((step, threshold))
                    break
        return out

    def _ours(self, step: Step | None) -> Step | None:
        """The drive's own copy of a handler step (the handler parsed the same response itself)."""
        return None if step is None or step.index >= len(self.route.steps) else self.route.steps[step.index]

    def add(self, alerts: list[tuple[str, str]], fix: GpsFix, progress: Progress | None) -> None:
        """The alerts one fix produced."""
        lanes = self.lane_windows(progress.along_m) if progress is not None else []
        true_along = self.drive.along_at(fix.t)
        on_route = true_along <= self.drive.on_route_m + 1e-6
        for kind, text in alerts:
            if kind == "nav_lanes":
                record = self._lanes(text, fix, progress, lanes, true_along)
            elif kind == "nav_roundabout":
                record = self._roundabout(text, fix, progress, true_along)
            elif kind == "nav_not_this_exit":
                record = self._exit(text, fix, progress, true_along)
            elif kind == "nav_arrived":
                record = self._arrived(text, fix, true_along)
            else:
                record = AlertRecord(fix.t, kind, text)  # reroute news: reported, timed from the requests
            if not on_route and kind in ("nav_lanes", "nav_roundabout", "nav_not_this_exit"):
                note = "after the missed turn: the handler had us on the route (not yet off it, or back on it)"
                record = AlertRecord(record.t, kind, text, record.step, record.target_m, None, record.nav_m, None, note)
            self.records.append(record)

    def _lanes(
        self, text: str, fix: GpsFix, progress: Progress | None, due: list[tuple[Step, float]], true_along: float
    ) -> AlertRecord:
        kind = "nav_lanes"
        if not due:
            return AlertRecord(fix.t, kind, text, ok=False, note="no maneuver matches this lane reminder")
        step, threshold = due.pop(0)
        self.lanes_said.add((step.index, threshold))
        true_m = step.start_m - true_along
        nav_m = step.start_m - progress.along_m if progress is not None else None
        if step.start_m - threshold <= 0.0:  # closer than that when navigation started: said at once, by design
            return AlertRecord(
                fix.t, kind, text, step, threshold, true_m, nav_m, None, "closer at the start: said at once"
            )
        return AlertRecord(fix.t, kind, text, step, threshold, true_m, nav_m, _close(true_m, threshold, self.tol))

    def _roundabout(self, text: str, fix: GpsFix, progress: Progress | None, true_along: float) -> AlertRecord:
        kind = "nav_roundabout"
        step = self._ours(progress.next_step if progress is not None else None)
        if progress is None or step is None or not step.is_roundabout:
            return AlertRecord(fix.t, kind, text, ok=False, note="no roundabout ahead")
        target = roundabout_target_m(self.route, step)
        true_m = step.start_m - true_along
        ok = true_m > 0.0 and _close(true_m, target, self.tol)
        return AlertRecord(fix.t, kind, text, step, target, true_m, progress.to_next_m, ok)

    def _exit(self, text: str, fix: GpsFix, progress: Progress | None, true_along: float) -> AlertRecord:
        kind = "nav_not_this_exit"
        step = self._ours(progress.step if progress is not None else None)
        if progress is None or step is None or not step.is_roundabout:
            return AlertRecord(fix.t, kind, text, ok=False, note="not on a roundabout")
        k = self.exits_said.get(step.index, 0)
        self.exits_said[step.index] = k + 1
        if k >= len(step.roundabout_exits_m):
            return AlertRecord(fix.t, kind, text, step, ok=False, note="more exits announced than passed")
        exit_m = step.roundabout_exits_m[k]
        last = k == len(step.roundabout_exits_m) - 1
        true_m, nav_m = exit_m - true_along, exit_m - progress.along_m  # negative: past the exit
        ok = abs(true_m) <= self.tol.position_m and (text == NOT_THIS_EXIT_NEXT) == last
        return AlertRecord(fix.t, kind, text, step, 0.0, true_m, nav_m, ok, f"exit {k + 1} at {exit_m:.0f} m")

    def _arrived(self, text: str, fix: GpsFix, true_along: float) -> AlertRecord:
        kind, step, end = "nav_arrived", self.route.steps[-1], self.route.line.points[-1]
        nav_m = haversine_m(fix.latlon, end)  # the handler's straight-line distance
        if self.drive.missed is None:
            true_m = self.route.line.length_m - true_along
            reach = max(self.cfg.arrival_m, self.cfg.arrival_radius_m) + self.tol.position_m
            ok = -self.tol.position_m <= true_m <= reach
            return AlertRecord(fix.t, kind, text, step, 0.0, true_m, nav_m, ok, "left to the end")
        true_m = haversine_m(self.drive.path.point_at(true_along), end)
        ok = missed_near_end(self.drive) and true_m <= self.cfg.near_end_m + self.tol.position_m
        note = "straight-line; left the route near the end = turned in (by design)" if ok else "after a missed turn"
        return AlertRecord(fix.t, kind, text, step, 0.0, true_m, nav_m, ok, note)


def missed_near_end(drive: SimDrive) -> bool:
    """The missed turn is so close to the destination that leaving the route there may mean turning in at it
    (nav/progress.py: arrival, not a reroute)."""
    return drive.missed is not None and drive.route.line.length_m - drive.missed.start_m <= NEAR_END_ALONG_M


def check_drive(
    drive: SimDrive,
    response: dict[str, Any],
    place: Place,
    cfg: NavConfig,
    poll_s: float = APP_POLL_S,
    tol: Tolerances | None = None,
) -> NavCheck:
    """Replay `drive` through a `NavHandler` and check its alerts (see the module docstring)."""
    tol = tol or Tolerances()
    with tempfile.TemporaryDirectory() as folder:
        gpx = Path(folder) / "drive.gpx"
        gpx.write_text(drive.to_gpx(), encoding="utf-8")
        replay = GpxReplay(gpx)

    client = SavedRouteClient(place, response)
    current: list[GpsFix | None] = [None]
    pending: list[tuple[str, str]] = []
    lock = threading.Lock()

    def on_alert(kind: str, text: str) -> None:
        with lock:
            pending.append((kind, text))

    def take() -> list[tuple[str, str]]:
        with lock:
            out = pending[:]
            pending.clear()
        return out

    def reroute_spoken() -> bool:
        with lock:
            return client.answered == len(client.requests) and any(kind in REROUTE_ALERTS for kind, _ in pending)

    def wait_until(done: Callable[[], bool]) -> None:
        deadline = time.perf_counter() + REROUTE_WAIT_S
        while not done() and time.perf_counter() < deadline:
            time.sleep(0.001)

    nav = NavHandler(cfg, client, lambda: current[0], on_alert)
    recorder = _Recorder(drive, cfg, tol)
    fix_times: dict[LatLon, float] = {}
    answer = ""
    off_route_t: float | None = None
    heard_reroute = False
    steps = int(replay.duration_s / poll_s + 1e-9)
    for k in range(steps + 1):
        fix = replay.fix_at(k * poll_s)
        if fix is None:
            continue
        current[0] = fix
        fix_times.setdefault(fix.latlon, fix.t)
        if not answer:
            answer = nav.set_destination(place.name)
            continue
        asked = len(client.requests)
        nav.on_fix(fix)
        if len(client.requests) > asked and not heard_reroute:  # the first reroute speaks: wait for it
            wait_until(reroute_spoken)
            heard_reroute = True
        progress = nav.progress
        if progress is not None and progress.off_route and off_route_t is None:
            off_route_t = fix.t
        recorder.add(take(), fix, progress)
    wait_until(lambda: client.answered == len(client.requests))
    late = take()
    if late and current[0] is not None:
        recorder.add(late, current[0], nav.progress)

    result = NavCheck(
        drive,
        answer,
        recorder.records,
        reroute_requests=[fix_times.get(origin, float("nan")) for origin, _ in client.requests[1:]],
        off_route_t=off_route_t,
        left_route_t=drive.time_off_route(cfg.off_route_m) if drive.missed is not None else None,
    )
    result.checks = _checks(result, cfg, tol)
    return result


def _checks(result: NavCheck, cfg: NavConfig, tol: Tolerances) -> list[Check]:
    drive, route = result.drive, result.drive.route
    reached = drive.on_route_m  # the car followed the route up to here
    checks = [Check("destination set", result.destination_answer.startswith("Okay"), result.destination_answer)]

    lanes = {(r.step.index, r.target_m): r for r in result.alerts if r.kind == "nav_lanes" and r.step is not None}
    for step in (s for s in route.steps if s.needs_lanes):
        for threshold in sorted(cfg.lane_reminders_m, reverse=True):
            if not 0.0 < step.start_m - threshold < reached:
                continue  # closer than that at the start (said at once, by design), or never got there
            name = f"lanes {threshold:.0f} m before step {step.index}"
            record = lanes.get((step.index, threshold))
            if record is None or record.true_m is None:
                checks.append(Check(name, False, "not said"))
                continue
            detail = f"said {record.true_m:.0f} m before (handler: {_m(record.nav_m)}) at t={record.t:.1f} s"
            checks.append(Check(name, bool(record.ok), detail))

    for step in (s for s in route.steps if s.is_roundabout):
        target = roundabout_target_m(route, step)
        if step.start_m - target >= reached:
            continue
        said = [r for r in result.alerts if r.kind == "nav_roundabout" and r.step is step]
        name = f"roundabout announced before step {step.index} (due {target:.0f} m before)"
        detail = ", ".join(f"{_m(r.true_m)} before at t={r.t:.1f} s" for r in said) or "not said"
        checks.append(Check(name, len(said) == 1 and bool(said[0].ok), detail))
        if step.start_m + step.distance_m > reached:
            continue
        exits = [r for r in result.alerts if r.kind == "nav_not_this_exit" and r.step is step]
        expected = len(step.roundabout_exits_m) if step.exits_counted else 0
        name = f"'not this one' for each of the {expected} exit(s) passed in step {step.index}"
        detail = ", ".join(f"{r.text!r} {_m(r.true_m)} from the exit" for r in exits) or "none said"
        checks.append(Check(name, len(exits) == expected and all(r.ok for r in exits), detail))

    arrivals = [r for r in result.alerts if r.kind == "nav_arrived"]
    if drive.missed is None:
        detail = ", ".join(f"{_m(r.true_m)} before the end at t={r.t:.1f} s" for r in arrivals) or "not said"
        checks.append(Check("arrival announced once at the end", len(arrivals) == 1 and bool(arrivals[0].ok), detail))
        detail = ", ".join(f"t={t:.1f} s" for t in result.reroute_requests) or "none"
        checks.append(Check("no reroute on a drive along the route", not result.reroute_requests, detail))
    else:
        name = (
            f"reroute requested {cfg.off_route_s:g} s (±{tol.time_s:g}) after leaving the route by "
            f"{cfg.off_route_m:g} m"
        )
        rerouted, detail = _reroute_timing(result, cfg, tol)
        if missed_near_end(drive):  # by design, either outcome is right: it depends on where the car went
            turned_in = len(arrivals) == 1 and bool(arrivals[0].ok) and not result.reroute_requests
            if turned_in:
                detail = f"arrived instead (turned in near the destination): {_m(arrivals[0].true_m)} from it"
            checks.append(Check(f"near the end: arrival, or {name}", turned_in or (rerouted and not arrivals), detail))
        else:
            checks.append(Check("no arrival after a missed turn", not arrivals, f"{len(arrivals)} said"))
            checks.append(Check(name, rerouted, detail))

    wrong = [r for r in result.alerts if r.ok is False]
    detail = "; ".join(f"t={r.t:.1f} {r.kind} {r.text!r}: {r.note or _m(r.true_m)}" for r in wrong) or "none"
    checks.append(Check("no alert out of place", not wrong, detail))
    return checks


def _reroute_timing(result: NavCheck, cfg: NavConfig, tol: Tolerances) -> tuple[bool, str]:
    """Was the first reroute requested off_route_s after the car truly left the route?"""
    left, asked = result.left_route_t, result.reroute_t
    if left is None or asked is None:
        return False, "the car never left the route" if left is None else "no reroute requested"
    delay = asked - left
    ok = cfg.off_route_s - tol.time_s <= delay <= cfg.off_route_s + tol.time_s
    return ok, f"left at t={left:.1f} s, requested at t={asked:.1f} s: {delay:.1f} s"


def _m(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} m"


def report_lines(result: NavCheck) -> list[str]:
    """The alert table and the checks, for printing."""
    lines = [f"  {'t (s)':>7}  {'alert':<18} {'step':>4}  {'due':>7}  {'true':>7}  {'handler':>7}  ok   text"]
    for r in result.alerts:
        ok = {True: "yes", False: "NO", None: " - "}[r.ok]
        step = str(r.step.index) if r.step is not None else "-"
        note = f"  ({r.note})" if r.note else ""
        lines.append(
            f"  {r.t:7.1f}  {r.kind:<18} {step:>4}  {_m(r.target_m):>7}  {_m(r.true_m):>7}  {_m(r.nav_m):>7}  "
            f"{ok:<4} {r.text}{note}"
        )
    if result.drive.missed is not None:
        lines.append(
            f"  reroute: car left the route at t={_t(result.left_route_t)}, handler off route at "
            f"t={_t(result.off_route_t)}, requests at {', '.join(_t(t) for t in result.reroute_requests) or 'none'}"
        )
    for check in result.checks:
        lines.append(f"  [{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return lines


def _t(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"
