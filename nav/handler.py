"""Navigation for the assistant: destination by voice, route questions, lane guidance, spoken guidance alerts.

`NavHandler.answer` is the assistant's `NavAnswerer` (destination, route and lane questions);
`NavHandler.on_fix` is fed every GPS fix and speaks guidance through `on_alert(kind, text)`:
- lane reminders about 1 mile and 0.25 mile before a junction where only some lanes work (PLAN.md);
- turn-by-turn, if `turn_prompts()` says so (`alerts.auto.turns`): "In half a mile, turn right onto Main Street."
  as we come within each of `nav.turn_prompts_m` of a maneuver (not for roundabouts, which have their own, or
  steps that need no action); a maneuver already within reach when the destination is set isn't repeated (the
  answer said it);
- roundabouts: "At the roundabout, take the 2nd exit onto Main Street.", then "Not this one, next exit." as we
  pass exits that aren't ours (only when our exit count agrees with Mapbox's exit number);
- off the route (> 50 m for 5 s): a new route from where we are, "You're off the route, so I've rerouted.";
  repeated reroutes and failures wait longer each time, and a failure is said once, not every attempt;
- arrival, after which navigation ends.
Each alert fires once. A reroute runs on its own thread; it is dropped if navigation was cancelled, arrived or
got a new destination meanwhile (a generation number, bumped by each of those, is checked before it installs
or speaks). Lane answers use the route's lane data for the next maneuver (or the one the question names); with
no lane data they say to check the signs. If a `lane_reader` is given (the vision model counting lanes) and its
count matches the route's, the answer also says which way to move. Without a current GPS fix, route answers say
so instead of quoting stale distances.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from core.config import NavConfig
from core.logging_setup import get_logger
from nav.gps import GpsFix
from nav.mapbox import NavError, Place, PlaceNotFound, RoutingClient
from nav.phrasing import (
    describe_lanes,
    in_distance,
    lane_change,
    maneuver_phrase,
    ordinal,
    road_name,
    spoken_distance,
)
from nav.progress import Progress, RouteTracker
from nav.route import Route, Step
from state.world import WorldSnapshot
from voice.router import Route as Question
from voice.router import RouteKind

log = get_logger(__name__)

NO_ROUTE = "There's no destination set. Say \"I'm going to\" and a place."
NO_GPS = "I don't have a GPS position yet, so I can't plan a route. Try again in a moment."
GPS_LOST = "I've lost the GPS signal, so I can't tell where we are on the route right now."
NO_HOME = "I don't know where home is yet. Say the address instead, or set nav.home in config.yaml."
NOT_THIS_EXIT = "Not this one."
NOT_THIS_EXIT_NEXT = "Not this one, next exit."
REROUTE_FAILED = "You're off the route, and I couldn't find a new one yet. I'll keep trying."
FIXED_ALERTS = (NOT_THIS_EXIT, NOT_THIS_EXIT_NEXT, REROUTE_FAILED)  # pre-rendered at startup (instant playback)
HOME_WORDS = {"home", "my home", "my house", "the house"}
EXIT_TYPES = {"off ramp", "fork", "roundabout", "rotary", "roundabout turn"}
# Steps that aren't a choice of lane by themselves (unless Mapbox gives lane data for them).
NOT_LANE_MANEUVERS = {"depart", "new name", "continue", "notification", "exit roundabout", "exit rotary", "arrive"}
TURN_PROMPT_FLOOR_M = 20.0  # the last turn-by-turn prompt is still said this close to the maneuver
# Steps that need no action from the driver: no turn-by-turn prompt.
NOT_TURN_MANEUVERS = {"depart", "new name", "continue", "notification", "exit roundabout", "exit rotary"}
REROUTED_RECENTLY_S = 60.0  # "did I miss my turn?" → "Yes, so I've rerouted." within this long of a reroute
QUICK_REROUTE_S = 90.0  # another reroute this soon doubles the wait before the next one
MAX_REROUTE_WAIT_S = 120.0
LEFT_ROUTE_SHARE = 0.5  # asked "did I miss my turn?": this share of the off-route distance already counts
ROUNDABOUT_ANNOUNCE_M = 150.0  # "At the roundabout, take the 2nd exit..." when the roundabout is this close
# (the route's lane count at the maneuver) -> (lanes counted in our direction, our lane from the left), or None
LaneReader = Callable[[int], tuple[int, int] | None]
_MATCH_STOPWORDS = {
    "which", "what", "lane", "lanes", "for", "the", "to", "i", "should", "be", "in", "am", "get", "take", "is",
    "are", "a", "an", "my", "our", "we", "this", "that", "it", "on", "at", "of", "do", "need", "go", "right",
    "left", "correct", "wrong", "proper", "next", "turn", "here", "now", "still", "one",
}  # fmt: skip


def _cap(text: str) -> str:
    """First letter upper-case, the rest untouched (str.capitalize would lower-case "SR 520 East")."""
    return text[:1].upper() + text[1:]


def _minutes(seconds: float) -> str:
    minutes = round(seconds / 60)
    return "less than a minute" if minutes < 1 else "a minute" if minutes == 1 else f"{minutes} minutes"


@dataclass(frozen=True, slots=True)
class _State:
    route: Route
    progress: Progress
    fresh: bool  # the progress comes from a current GPS fix


class NavHandler:
    """Destination, route progress, guidance alerts and route/lane answers."""

    def __init__(
        self,
        cfg: NavConfig,
        client: RoutingClient,
        gps: Callable[[], GpsFix | None],
        on_alert: Callable[[str, str], None] | None = None,
        lane_reader: LaneReader | None = None,
        turn_prompts: Callable[[], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self._gps = gps
        self._on_alert = on_alert
        self.lane_reader = lane_reader
        self.turn_prompts = turn_prompts  # turn-by-turn on? (None: off; the page switches it while running)
        self._lock = threading.RLock()
        self.route: Route | None = None
        self.place: Place | None = None
        self.progress: Progress | None = None
        self.arrived_at: Place | None = None  # the destination we reached (navigation has ended)
        self._tracker: RouteTracker | None = None
        self._announced: set[tuple[object, ...]] = set()
        self._generation = 0  # bumped when navigation starts, changes destination, is cancelled or arrives
        self._reset_reroutes()

    def _reset_reroutes(self) -> None:
        self._reroute_gen: int | None = None  # the generation a reroute in flight belongs to
        self._reroute_t: float | None = None  # fix time of the last reroute attempt
        self._reroute_failures = 0
        self._reroute_streak = 0  # reroutes in quick succession
        self._rerouted_at: float | None = None

    # --------------------------------------------------------------------------------- answers

    def answer(self, question: Question, snapshot: WorldSnapshot | None) -> str:
        """Answer a DESTINATION / NAV / LANE question."""
        if question.kind is RouteKind.DESTINATION:
            return self.set_destination(question.place or question.text)
        if question.kind is RouteKind.LANE:
            return self._lane_answer(question.text)
        if question.intent == "cancel_route":
            return self.cancel()
        if question.intent == "destination_info":
            return self._destination_answer()
        state = self._current()
        if state is None:
            return self._no_route()
        if not state.fresh:
            return self._lost(state)
        if question.intent == "missed_turn":
            return self._missed_turn(state)
        if question.intent == "exit":
            return self._exit_answer(state)
        if question.intent == "turn_distance":
            return self._exit_answer(state) if "exit" in question.text else self._turn_distance_answer(state)
        if question.intent == "eta":
            return f"About {_minutes(state.progress.remaining_s)}, {spoken_distance(state.progress.remaining_m)} to go."
        return self._next_step_answer(state)

    def set_destination(self, query: str, place: Place | None = None) -> str:
        """Find the place (unless the page already picked it: `place`) and a route to it from where we are."""
        query = query.strip()
        if place is None and query.lower() in HOME_WORDS:
            if not self.cfg.home:
                return NO_HOME
            query = self.cfg.home
        fix = self._gps()
        if fix is None:
            return NO_GPS
        with self._lock:
            self._generation += 1  # a reroute for the old destination that finishes now must not land
            generation = self._generation
        try:
            if place is None:
                place = self.client.geocode(query, near=fix.latlon)
            route = self.client.route(fix.latlon, place, fix.heading)
        except PlaceNotFound as exc:
            log.warning("destination not found", query=query, error=str(exc))
            return f"Sorry, I couldn't find {query}."
        except NavError as exc:
            log.warning("destination failed", query=query, error=str(exc))
            return f"Sorry, I couldn't get a route to {query}."
        with self._lock:
            if self._generation != generation:
                log.info("destination superseded", query=query)
                return "Okay."
            self._start(route, place, self._gps() or fix)
            progress = self.progress
        log.info("destination set", name=place.name, address=place.address, miles=round(route.distance_m / 1609.3, 1))
        first = ""
        if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
            first = f" First, {maneuver_phrase(progress.next_step)} {in_distance(progress.to_next_m)}."
        return f"Okay, {place.name}: {_minutes(route.duration_s)}, {spoken_distance(route.distance_m)}.{first}"

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        """Places matching what the driver is typing, nearest first among equals (the page's destination box)."""
        search = getattr(self.client, "search", None)
        if not callable(search):
            return []
        fix = self._gps()
        try:
            places: list[Place] = search(query.strip(), fix.latlon if fix is not None else None, limit)
        except NavError as exc:
            log.warning("place search failed", query=query, error=str(exc))
            return []
        return places

    def cancel(self) -> str:
        """Stop navigating."""
        with self._lock:
            had = self.route is not None
            self._end(arrived=None)
        return "Okay, navigation is off." if had else NO_ROUTE

    def _start(self, route: Route, place: Place, fix: GpsFix | None) -> None:
        """A new destination (caller holds the lock)."""
        self._reset_reroutes()
        self.arrived_at = None
        self._install(route, place, fix)
        progress = self.progress
        if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
            for threshold in self.cfg.turn_prompts_m:  # the answer names the first maneuver: don't say it again
                if progress.to_next_m <= threshold:
                    self._once("turn", progress.next_step.index, threshold)

    def _install(self, route: Route, place: Place, fix: GpsFix | None) -> None:
        """Follow a route (new destination or reroute; caller holds the lock)."""
        self.route, self.place = route, place
        self._tracker = RouteTracker(route, self.cfg)
        self._announced = set()
        self.progress = self._tracker.update(fix) if fix is not None else None

    def _end(self, arrived: Place | None) -> None:
        """Stop navigating (cancelled or arrived; caller holds the lock)."""
        self._generation += 1
        self.route, self.place, self.progress, self._tracker = None, None, None, None
        self.arrived_at = arrived
        self._reset_reroutes()

    def _current(self) -> _State | None:
        with self._lock:
            if self.route is None or self._tracker is None:
                return None
            fix = self._gps()
            if fix is not None and _new_fix(fix, self.progress):
                self.progress = self._tracker.update(fix)
            if self.progress is None:
                return None
            return _State(self.route, self.progress, fresh=fix is not None)

    def _no_route(self) -> str:
        with self._lock:
            if self.route is None:
                return f"You've arrived at {self.arrived_at.name}." if self.arrived_at is not None else NO_ROUTE
        return NO_GPS

    def _lost(self, state: _State) -> str:
        step = state.progress.next_step
        return GPS_LOST if step is None else f"{GPS_LOST} Next on the route: {maneuver_phrase(step)}."

    def _next_step_answer(self, state: _State) -> str:
        route, progress = state.route, state.progress
        step = progress.next_step
        if step is None or progress.to_next_m is None:
            return "You're almost there."
        text = f"{_cap(maneuver_phrase(step))} {in_distance(progress.to_next_m)}"
        following = route.next_maneuver(step.start_m)
        if following is not None and following.maneuver != "arrive" and following.start_m - step.start_m < 400:
            text += f", then {maneuver_phrase(following)}"
        return text + "."

    def _turn_distance_answer(self, state: _State) -> str:
        progress = state.progress
        step = progress.next_step
        if step is None or progress.to_next_m is None:
            return "You're almost there."
        if progress.to_next_m < 30:
            return f"Now: {maneuver_phrase(step)}."
        if step.maneuver == "arrive":
            return f"Your destination is {in_distance(progress.to_next_m)}."
        return f"In {spoken_distance(progress.to_next_m)}: {maneuver_phrase(step)}."

    def _exit_answer(self, state: _State) -> str:
        route, progress = state.route, state.progress
        current = progress.step
        if current.is_roundabout and progress.along_m < current.start_m + current.distance_m:
            onto = f" onto {current.name}" if current.name else ""
            if current.exits_counted:
                left = sum(1 for m in current.roundabout_exits_m if m > progress.along_m + 5)
                if left == 0:
                    return f"Take the next exit{onto}."
                return f"Not the next one: {left} more exit{'s' if left > 1 else ''} before yours."
            return f"{_cap(maneuver_phrase(current))}."
        step = next((s for s in route.steps if s.start_m > progress.along_m + 5 and s.maneuver in EXIT_TYPES), None)
        if step is None:
            return f"There's no exit on the rest of the route. {self._next_step_answer(state)}"
        distance = step.start_m - progress.along_m
        if step.is_roundabout:
            if step.roundabout_exit:
                onto = f" onto {step.name}" if step.name else ""
                return (
                    f"At the roundabout {in_distance(distance)}, take the {ordinal(step.roundabout_exit)} exit{onto}."
                )
            return f"{_cap(maneuver_phrase(step))} {in_distance(distance)}."
        label = f"exit {step.exit_ref}" if step.exit_ref else "your exit"
        toward = f" toward {step.destinations.split(':')[-1].strip()}" if step.destinations else ""
        if distance < 400:
            return f"This one: {label}{toward}, {in_distance(distance)}."
        return f"Not this one. {_cap(label)}{toward} is {in_distance(distance)}."

    def _missed_turn(self, state: _State) -> str:
        progress = state.progress
        if progress.off_route:
            return "Yes, you're off the route. I'm finding a new one now."
        with self._lock:
            recently = self._rerouted_at is not None and 0.0 <= progress.t - self._rerouted_at < REROUTED_RECENTLY_S
            due = self._reroute_due(progress.t)
        if progress.joined and progress.offset_m > self.cfg.off_route_m * LEFT_ROUTE_SHARE:
            # Asked right after passing the turn: don't wait the full 5 s to call it (same backoff as alerts).
            fix = self._gps()
            if due and fix is not None:
                self._start_reroute(fix)
            elif recently:
                return f"Yes, so I've rerouted. {self._next_step_answer(state)}"
            return "It looks like you've left the route. I'm finding a new one now."
        if recently:
            return f"Yes, so I've rerouted. {self._next_step_answer(state)}"
        return f"No, you're on the route. {self._next_step_answer(state)}"

    def _destination_answer(self) -> str:
        with self._lock:
            place, arrived = self.place, self.arrived_at
        if place is None:
            return f"You've arrived at {arrived.name}." if arrived is not None else NO_ROUTE
        return f"We're going to {place.name}, {place.address}."

    def _lane_answer(self, question: str) -> str:
        state = self._current()
        if state is None:
            return self._no_route()
        if not state.fresh:
            return self._lost(state)
        route, progress = state.route, state.progress
        ahead = [
            s
            for s in route.steps
            if s.start_m > progress.along_m + 5 and (s.maneuver not in NOT_LANE_MANEUVERS or s.needs_lanes)
        ]
        words = _question_words(question)
        step = _matching_step(words, ahead)
        if step is None and "exit" in words:
            step = next((s for s in ahead if s.maneuver in EXIT_TYPES), None)
        if step is None:
            step = ahead[0] if ahead else None  # the next maneuver: the one the lane matters for now
        if step is None:
            return "Any lane is fine from here."
        when = in_distance(step.start_m - progress.along_m)
        if not step.lanes:
            return f"I don't have lane data for that. Next, {maneuver_phrase(step)} {when}: check the overhead signs."
        if not step.needs_lanes:
            return f"Any lane works for the next maneuver: {maneuver_phrase(step)} {when}."
        lanes = describe_lanes(step.lanes) or "the marked lanes"
        road = road_name(step) or "the next turn"
        text = f"For {road} {when}, use {lanes}."
        reading = self.lane_reader(len(step.lanes)) if self.lane_reader is not None else None
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
        reroute = False
        with self._lock:
            if self.route is None or self._tracker is None:
                return
            if not _new_fix(fix, self.progress):
                return  # already seen (an answer may have pulled the same fix)
            route = self.route
            progress = self._tracker.update(fix)
            self.progress = progress
            if progress.arrived:
                place = self.place
                alerts.append(("nav_arrived", f"You've arrived at {place.name if place else 'your destination'}."))
                self._end(arrived=place)
                log.info("arrived", name=place.name if place else None)
            elif progress.off_route:
                reroute = self._reroute_due(fix.t)
            else:
                alerts += self._turn_prompt(progress)
                alerts += self._lane_reminders(route, progress)
                alerts += self._roundabout(route, progress)
        if reroute:
            self._start_reroute(fix)
        for kind, text in alerts:
            self._emit(kind, text)

    def _reroute_due(self, t: float) -> bool:
        """Off the route: time for (another) reroute? Each quick repeat or failure doubles the wait."""
        if self._reroute_gen == self._generation:
            return False
        if self._reroute_t is None:
            return True
        wait = min(
            self.cfg.reroute_cooldown_s * 2 ** (self._reroute_streak + self._reroute_failures), MAX_REROUTE_WAIT_S
        )
        return t - self._reroute_t >= wait

    def _start_reroute(self, fix: GpsFix) -> None:
        with self._lock:
            if self._reroute_gen == self._generation or self.place is None:
                return
            self._reroute_gen, self._reroute_t = self._generation, fix.t
            generation = self._generation
        threading.Thread(target=self._reroute, args=(fix, generation), name="reroute", daemon=True).start()

    def _turn_prompt(self, progress: Progress) -> list[tuple[str, str]]:
        """Turn-by-turn: "In half a mile, turn right onto Main Street." once per maneuver and distance."""
        if self.turn_prompts is None or not self.turn_prompts():
            return []
        step, distance = progress.next_step, progress.to_next_m
        if step is None or distance is None or step.is_roundabout or step.maneuver in NOT_TURN_MANEUVERS:
            return []
        thresholds = sorted(self.cfg.turn_prompts_m)
        for k, threshold in enumerate(thresholds):
            # Said once as we come within the threshold (not if we were already much closer when it became next),
            # except the last one: a maneuver right after another (or after a GPS gap) still gets it, down to 20 m.
            low = TURN_PROMPT_FLOOR_M if k == 0 else threshold * 0.5
            if low <= distance <= threshold:
                if self._once("turn", step.index, threshold):
                    return [("nav_turn", f"{_cap(in_distance(distance))}, {maneuver_phrase(step)}.")]
                break
        return []

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
            step is not None
            and step.is_roundabout
            and progress.to_next_m is not None
            and progress.to_next_m <= ROUNDABOUT_ANNOUNCE_M
        )
        if near_roundabout and step is not None and self._once("roundabout", step.index):
            out.append(("nav_roundabout", f"{_cap(maneuver_phrase(step))}."))
        current = progress.step
        if current.is_roundabout and current.exits_counted:
            exits = current.roundabout_exits_m
            for k, position in enumerate(exits):
                if progress.along_m >= position - 3.0 and self._once("exit", current.index, k):
                    out.append(("nav_not_this_exit", NOT_THIS_EXIT_NEXT if k == len(exits) - 1 else NOT_THIS_EXIT))
        return out

    def _reroute(self, fix: GpsFix, generation: int) -> None:
        text: tuple[str, str] | None = None
        try:
            with self._lock:
                place = self.place if self._generation == generation else None
            if place is None:
                return
            error: str | None = None
            try:
                route = self.client.route(fix.latlon, place, fix.heading)
            except NavError as exc:
                error = str(exc)
            except Exception as exc:  # never let the thread die silently (or print past the log redaction)
                log.exception("reroute crashed")
                error = type(exc).__name__
            with self._lock:
                if self._generation != generation:
                    log.info("reroute dropped: navigation changed meanwhile")
                    return
                if error is not None:
                    self._reroute_failures += 1
                    log.warning("reroute failed", error=error, failures=self._reroute_failures)
                    if self._reroute_failures == 1:  # say it once; keep trying quietly, waiting longer each time
                        text = ("nav_reroute_failed", REROUTE_FAILED)
                    return
                current = self._gps() or fix
                quick = self._rerouted_at is not None and current.t - self._rerouted_at < QUICK_REROUTE_S
                self._reroute_streak = self._reroute_streak + 1 if quick else 0
                self._reroute_failures = 0
                self._install(route, place, current)
                self._rerouted_at = current.t
                progress = self.progress
            first = ""
            if progress is not None and progress.next_step is not None and progress.to_next_m is not None:
                first = f" {_cap(maneuver_phrase(progress.next_step))} {in_distance(progress.to_next_m)}."
            log.info("rerouted", miles=round(route.distance_m / 1609.3, 1), streak=self._reroute_streak)
            text = ("nav_rerouted", f"You're off the route, so I've rerouted.{first}")
        finally:
            with self._lock:
                if self._reroute_gen == generation:
                    self._reroute_gen = None
                if self._generation != generation:
                    text = None  # cancelled or a new destination while we finished: say nothing
            if text is not None:
                self._emit(*text)

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
        """Lines for the vision model's prompt: our speed (whenever the GPS has a fresh fix, navigating or not) and,
        while navigating, the next maneuver and its lanes."""
        with self._lock:
            progress = self.progress
            navigating = self.route is not None
        fix = self._gps()
        if navigating and fix is None:
            return ["Navigation: the GPS signal is lost."]
        lines = speed_lines(fix)
        if not navigating or progress is None:
            return lines
        if progress.next_step is not None and progress.to_next_m is not None:
            step = progress.next_step
            lines.append(f"Navigation: next, {maneuver_phrase(step)} {in_distance(progress.to_next_m)}.")
            lanes = describe_lanes(step.lanes)
            if lanes:
                lines.append(f"For that maneuver, use {lanes} (left to right, {len(step.lanes)} lanes).")
        return lines


def speed_lines(fix: GpsFix | None) -> list[str]:
    """["Speed: 17 mph."] for the vision model's prompt when the fix has a speed, else []."""
    speed = fix.speed_mps if fix is not None else None
    if speed is None or not math.isfinite(speed):
        return []
    return [f"Speed: {speed * 2.23694:.0f} mph."]


def _new_fix(fix: GpsFix, progress: Progress | None) -> bool:
    """A fix newer than the last one used, or a jump back in time (a replay was rewound)."""
    return progress is None or fix.t > progress.t or fix.t < progress.t - 1.0


def _question_words(question: str) -> set[str]:
    """The words of a lane question that could name a road, a number or a destination."""
    return set(re.findall(r"[a-z0-9]+", question.lower())) - _MATCH_STOPWORDS


def _matching_step(words: set[str], steps: list[Step]) -> Step | None:
    """The step a lane question names ("which lane for 520 east?"): match its numbers and words."""
    words = words - {"exit"}
    if not words:
        return None
    best, best_hits = None, 0
    for step in steps:
        text = " ".join(filter(None, [step.ref, step.name, step.destinations, step.instruction])).lower()
        hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text))
        if hits > best_hits:
            best, best_hits = step, hits
    return best
