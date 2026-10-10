"""Navigation for the assistant: destination by voice, route questions, lane guidance, spoken guidance alerts.

`NavHandler.answer` is the assistant's `NavAnswerer` (destination, route and lane questions);
`NavHandler.on_fix` is fed every GPS fix and speaks guidance through `on_alert(kind, text)`:
- lane reminders about 1 mile and 0.25 mile before a junction where only some lanes work (PLAN.md);
- roundabouts: "At the roundabout, take the 2nd exit onto Main Street.", then "Not this one, next exit." as we
  pass exits that aren't ours;
- off the route (> 50 m for 5 s): a new route from where we are, "You're off the route, so I've rerouted.";
- arrival.
Each alert fires once. Lane questions use the route's lane data; if a `lane_reader` is given (the vision model
counting lanes, M7) and its lane count matches the route's, the answer also says which way to move.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable

from core.config import NavConfig
from core.logging_setup import get_logger
from nav.gps import GpsFix
from nav.mapbox import NavError, Place, RoutingClient
from nav.phrasing import describe_lanes, in_distance, lane_change, maneuver_phrase, road_name, spoken_distance
from nav.progress import Progress, RouteTracker
from nav.route import Route, Step
from state.world import WorldSnapshot
from voice.router import Route as Question
from voice.router import RouteKind

log = get_logger(__name__)

NO_ROUTE = "There's no destination set. Say \"I'm going to\" and a place."
NO_GPS = "I don't have a GPS position yet, so I can't plan a route. Try again in a moment."
EXIT_TYPES = {"off ramp", "fork", "roundabout", "rotary", "roundabout turn"}
LaneReader = Callable[[], tuple[int, int] | None]  # -> (lanes counted in our direction, our lane from the left)


def _cap(text: str) -> str:
    """First letter upper-case, the rest untouched (str.capitalize would lower-case "SR 520 East")."""
    return text[:1].upper() + text[1:]


def _minutes(seconds: float) -> str:
    minutes = round(seconds / 60)
    return "less than a minute" if minutes < 1 else "a minute" if minutes == 1 else f"{minutes} minutes"


class NavHandler:
    """Destination, route progress, guidance alerts and route/lane answers."""

    def __init__(
        self,
        cfg: NavConfig,
        client: RoutingClient,
        gps: Callable[[], GpsFix | None],
        on_alert: Callable[[str, str], None] | None = None,
        lane_reader: LaneReader | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self._gps = gps
        self._on_alert = on_alert
        self.lane_reader = lane_reader
        self._lock = threading.RLock()
        self.route: Route | None = None
        self.place: Place | None = None
        self.progress: Progress | None = None
        self._tracker: RouteTracker | None = None
        self._announced: set[tuple[object, ...]] = set()
        self._rerouting = False
        self._last_reroute_t: float | None = None
        self._rerouted_at: float | None = None

    # --------------------------------------------------------------------------------- answers

    def answer(self, question: Question, snapshot: WorldSnapshot) -> str:
        """Answer a DESTINATION / NAV / LANE question."""
        if question.kind is RouteKind.DESTINATION:
            return self.set_destination(question.place or question.text)
        if question.kind is RouteKind.LANE:
            return self._lane_answer(question.text)
        handlers: dict[str, Callable[[], str]] = {
            "cancel_route": self.cancel,
            "missed_turn": self._missed_turn,
            "exit": self._exit_answer,
            "next_step": self._next_step_answer,
            "eta": self._eta_answer,
            "destination_info": self._destination_answer,
        }
        return handlers.get(question.intent, self._next_step_answer)()

    def set_destination(self, query: str) -> str:
        """Find the place and a route to it from where we are."""
        fix = self._gps()
        if fix is None:
            return NO_GPS
        try:
            place = self.client.geocode(query, near=fix.latlon)
            route = self.client.route(fix.latlon, place, fix.heading)
        except NavError as exc:
            log.warning("destination failed", query=query, error=str(exc))
            return (
                f"Sorry, I couldn't find {query}."
                if "no place" in str(exc)
                else f"Sorry, I couldn't get a route to {query}."
            )
        with self._lock:
            self._install(route, place, fix)
            progress = self.progress
        log.info("destination set", name=place.name, address=place.address, miles=round(route.distance_m / 1609.3, 1))
        first = ""
        if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
            first = f" First, {maneuver_phrase(progress.next_step)} {in_distance(progress.to_next_m)}."
        return f"Okay, {place.name}: {_minutes(route.duration_s)}, {spoken_distance(route.distance_m)}.{first}"

    def cancel(self) -> str:
        """Stop navigating."""
        with self._lock:
            had = self.route is not None
            self.route, self.place, self.progress, self._tracker = None, None, None, None
        return "Okay, navigation is off." if had else NO_ROUTE

    def _install(self, route: Route, place: Place, fix: GpsFix | None) -> None:
        self.route, self.place = route, place
        self._tracker = RouteTracker(route, self.cfg)
        self._announced = set()
        self.progress = self._tracker.update(fix) if fix is not None else None

    def _current(self) -> tuple[Route, Progress] | None:
        with self._lock:
            fix = self._gps()
            if self.route is None or self._tracker is None:
                return None
            if fix is not None:
                self.progress = self._tracker.update(fix)
            if self.progress is None:
                return None
            return self.route, self.progress

    def _next_step_answer(self) -> str:
        state = self._current()
        if state is None:
            return NO_ROUTE if self.route is None else NO_GPS
        route, progress = state
        step = progress.next_step
        if step is None or progress.to_next_m is None:
            return "You're almost there."
        text = f"{_cap(maneuver_phrase(step))} {in_distance(progress.to_next_m)}"
        following = route.next_maneuver(step.start_m)
        if following is not None and following.maneuver != "arrive" and following.start_m - step.start_m < 400:
            text += f", then {maneuver_phrase(following)}"
        return text + "."

    def _exit_answer(self) -> str:
        state = self._current()
        if state is None:
            return NO_ROUTE if self.route is None else NO_GPS
        route, progress = state
        current = progress.step
        if current.is_roundabout and current.roundabout_exits_m:
            left = sum(1 for m in current.roundabout_exits_m if m > progress.along_m + 5)
            return (
                "Take the next exit."
                if left == 0
                else f"Not the next one: {left} more exit{'s' if left > 1 else ''} before yours."
            )
        step = next((s for s in route.steps if s.start_m > progress.along_m + 5 and s.maneuver in EXIT_TYPES), None)
        if step is None:
            return f"There's no exit on the rest of the route. {self._next_step_answer()}"
        distance = step.start_m - progress.along_m
        if step.is_roundabout:
            return (
                f"At the roundabout {in_distance(distance)}, take the {maneuver_phrase(step).split('take the ')[-1]}."
            )
        label = f"exit {step.exit_ref}" if step.exit_ref else "your exit"
        toward = f" toward {step.destinations.split(':')[-1].strip()}" if step.destinations else ""
        if distance < 400:
            return f"This one: {label}{toward}, {in_distance(distance)}."
        return f"Not this one. {_cap(label)}{toward} is {in_distance(distance)}."

    def _missed_turn(self) -> str:
        state = self._current()
        if state is None:
            return NO_ROUTE if self.route is None else NO_GPS
        _, progress = state
        rerouted_recently = self._rerouted_at is not None and progress.t - self._rerouted_at < 60.0
        if progress.off_route:
            return "Yes, you're off the route. I'm finding a new one now."
        if rerouted_recently:
            return f"Yes, so I've rerouted. {self._next_step_answer()}"
        return f"No, you're on the route. {self._next_step_answer()}"

    def _eta_answer(self) -> str:
        state = self._current()
        if state is None:
            return NO_ROUTE if self.route is None else NO_GPS
        _, progress = state
        return f"About {_minutes(progress.remaining_s)}, {spoken_distance(progress.remaining_m)} to go."

    def _destination_answer(self) -> str:
        with self._lock:
            place = self.place
        return NO_ROUTE if place is None else f"We're going to {place.name}, {place.address}."

    def _lane_answer(self, question: str) -> str:
        state = self._current()
        if state is None:
            return NO_ROUTE if self.route is None else NO_GPS
        route, progress = state
        ahead = [s for s in route.steps if s.start_m > progress.along_m + 5]
        step = _matching_step(question, [s for s in ahead if s.needs_lanes]) or next(
            (s for s in ahead if s.needs_lanes), None
        )
        if step is None:
            next_step = progress.next_step
            if next_step is None:
                return "Any lane is fine from here."
            when = in_distance(next_step.start_m - progress.along_m)
            return f"Any lane works for the next maneuver: {maneuver_phrase(next_step)} {when}."
        lanes = describe_lanes(step.lanes) or "any lane"
        road = road_name(step) or "the next turn"
        text = f"For {road} {in_distance(step.start_m - progress.along_m)}, use {lanes}."
        reading = self.lane_reader() if self.lane_reader is not None else None
        if reading is not None:
            counted, ours = reading
            if counted == len(step.lanes) and 1 <= ours <= counted:
                text += f" You're in lane {ours} from the left, so {lane_change(step.lanes, ours)}."
            else:
                text += " I can't tell exactly which lane you're in, so check the overhead signs."
        return text

    # ----------------------------------------------------------------------------- guidance alerts

    def on_fix(self, fix: GpsFix) -> None:
        """Update progress with a GPS fix and speak any guidance that is due."""
        alerts: list[tuple[str, str]] = []
        with self._lock:
            if self.route is None or self._tracker is None:
                return
            route = self.route
            progress = self._tracker.update(fix)
            self.progress = progress
            if progress.arrived:
                if self._once("arrived"):
                    alerts.append(
                        ("nav_arrived", f"You've arrived at {self.place.name if self.place else 'your destination'}.")
                    )
            elif progress.off_route:
                cooled = self._last_reroute_t is None or fix.t - self._last_reroute_t >= self.cfg.reroute_cooldown_s
                if not self._rerouting and cooled:
                    self._rerouting, self._last_reroute_t = True, fix.t
                    threading.Thread(target=self._reroute, args=(fix,), name="reroute", daemon=True).start()
            else:
                alerts += self._lane_reminders(route, progress)
                alerts += self._roundabout(route, progress)
        for kind, text in alerts:
            self._emit(kind, text)

    def _lane_reminders(self, route: Route, progress: Progress) -> list[tuple[str, str]]:
        out = []
        horizon = max(self.cfg.lane_reminders_m, default=0.0) + 50.0
        for step in route.upcoming(progress.along_m, horizon):
            if not step.needs_lanes:
                continue
            distance = step.start_m - progress.along_m
            for threshold in sorted(self.cfg.lane_reminders_m):
                # Say it once when we come within the threshold (not if we started much closer than it).
                if threshold * 0.5 <= distance <= threshold and self._once("lanes", step.index, threshold):
                    lanes = describe_lanes(step.lanes)
                    if lanes:
                        road = road_name(step)
                        target = f" for {road}" if road else ""
                        out.append(("nav_lanes", f"In {spoken_distance(distance)}, use {lanes}{target}."))
                    break
        return out

    def _roundabout(self, route: Route, progress: Progress) -> list[tuple[str, str]]:
        out = []
        step = progress.next_step
        near_roundabout = (
            step is not None and step.is_roundabout and progress.to_next_m is not None and progress.to_next_m <= 150
        )
        if near_roundabout and step is not None and self._once("roundabout", step.index):
            out.append(("nav_roundabout", f"{_cap(maneuver_phrase(step))}."))
        current = progress.step
        if current.is_roundabout:
            exits = current.roundabout_exits_m
            for k, position in enumerate(exits):
                if progress.along_m >= position - 3.0 and self._once("exit", current.index, k):
                    out.append(
                        ("nav_not_this_exit", "Not this one, next exit." if k == len(exits) - 1 else "Not this one.")
                    )
        return out

    def _reroute(self, fix: GpsFix) -> None:
        try:
            with self._lock:
                place = self.place
            if place is None:
                return
            route = self.client.route(fix.latlon, place, fix.heading)
        except NavError as exc:
            log.warning("reroute failed", error=str(exc))
            self._emit(
                "nav_reroute_failed", "You're off the route, and I couldn't find a new one yet. I'll keep trying."
            )
            return
        finally:
            with self._lock:
                self._rerouting = False
        with self._lock:
            if self.place is not place:
                return  # cancelled or a new destination meanwhile
            current = self._gps() or fix
            self._install(route, place, current)
            self._rerouted_at = current.t
            progress = self.progress
        first = ""
        if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
            first = f" {_cap(maneuver_phrase(progress.next_step))} {in_distance(progress.to_next_m)}."
        log.info("rerouted", miles=round(route.distance_m / 1609.3, 1))
        self._emit("nav_rerouted", f"You're off the route, so I've rerouted.{first}")

    def _once(self, *key: object) -> bool:
        if key in self._announced:
            return False
        self._announced.add(key)
        return True

    def _emit(self, kind: str, text: str) -> None:
        log.info("nav alert", kind=kind, text=text)
        if self._on_alert is not None:
            try:
                self._on_alert(kind, text)
            except Exception:
                log.exception("nav alert handler failed", kind=kind)

    # ------------------------------------------------------------------------------ model context

    def extra_context(self) -> list[str]:
        """Navigation lines for the vision model's prompt (speed, next maneuver, its lanes)."""
        with self._lock:
            progress = self.progress
        lines = []
        if progress is not None and progress.speed_mps is not None:
            lines.append(f"Speed: {progress.speed_mps * 2.23694:.0f} mph.")
        if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
            step = progress.next_step
            lines.append(f"Navigation: next, {maneuver_phrase(step)} {in_distance(progress.to_next_m)}.")
            lanes = describe_lanes(step.lanes)
            if lanes:
                lines.append(f"For that maneuver, use {lanes} (left to right, {len(step.lanes)} lanes).")
        return lines


def _matching_step(question: str, steps: list[Step]) -> Step | None:
    """The step a lane question names ("which lane for 520 east?"): match its numbers and words."""
    words = set(re.findall(r"[a-z0-9]+", question.lower())) - {
        "which",
        "what",
        "lane",
        "lanes",
        "for",
        "the",
        "to",
        "i",
        "should",
        "be",
        "in",
        "am",
        "get",
        "take",
    }
    if not words:
        return None
    best, best_hits = None, 0
    for step in steps:
        text = " ".join(filter(None, [step.ref, step.name, step.destinations, step.instruction])).lower()
        hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text))
        if hits > best_hits:
            best, best_hits = step, hits
    return best
