"""4-way-stop queue: arrival order, ties to the right, departures, "your turn", lost tracks, parked cars, creeping."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from state.stop_queue import (
    EGO_KEY,
    Approach,
    EgoStop,
    QueueEntry,
    StopQueue,
    StopQueueConfig,
    WaitingVehicle,
    right_of_way_order,
)

L, A, R, US = Approach.LEFT, Approach.AHEAD, Approach.RIGHT, Approach.US
BOXES = {L: (50.0, 300.0, 250.0, 380.0), A: (560.0, 280.0, 640.0, 330.0), R: (1000.0, 300.0, 1200.0, 380.0)}


@dataclass
class Car:
    """A scripted car: stops at `stop_t` on `approach`; departs (drives off) at `depart_t`."""

    track_id: int
    approach: Approach
    stop_t: float
    depart_t: float | None = None
    hidden: list[tuple[float, float]] = field(default_factory=list)  # (from, to): not visible (occluded)
    relink_id: int | None = None  # track id used after the first occlusion (tracker lost it)

    def observe(self, t: float) -> WaitingVehicle | None:
        if t < self.stop_t - 2.0:
            return None  # not arrived yet
        if any(a <= t < b for a, b in self.hidden):
            return None
        track_id = self.track_id
        if self.relink_id is not None and self.hidden and t >= self.hidden[0][1]:
            track_id = self.relink_id
        if t < self.stop_t:
            return WaitingVehicle(track_id, self.approach, False, None, 0.0, BOXES[self.approach])  # approaching
        driving = self.depart_t is not None and t >= self.depart_t
        moving_for = t - self.depart_t if driving and self.depart_t is not None else 0.0
        moved = moving_for * 2.0  # box heights: a normal start from the line
        if moved > 4.0:
            return None  # gone through the intersection, out of view
        return WaitingVehicle(
            track_id,
            self.approach,
            not driving,
            None if driving else self.stop_t,
            moved,
            BOXES[self.approach],
            label=f"car {track_id}",
            moving_for=moving_for,
        )


def run(
    cars: list[Car],
    ego_stop_t: float | None,
    until: float,
    ego_moving: list[tuple[float, float]] = (),
    cfg: StopQueueConfig | None = None,
    dt: float = 0.1,
) -> tuple[StopQueue, list]:
    """Simulate at 10 Hz. `ego_moving` = intervals where we move after arriving (creep or leave)."""
    q = StopQueue(cfg)
    events = []
    t = 0.0
    moving_since: float | None = None
    while t <= until + 1e-9:
        in_motion = ego_stop_t is None or t < ego_stop_t or any(a <= t < b for a, b in ego_moving)
        moving_since = (t if moving_since is None else moving_since) if in_motion else None
        ego = EgoStop(
            at_stop=not in_motion,
            stopped_since=None if in_motion else ego_stop_t,
            moving_for_s=0.0 if moving_since is None else t - moving_since,
        )
        vehicles = [v for c in cars if (v := c.observe(t)) is not None]
        events += q.update(round(t, 3), ego, vehicles)
        t += dt
    return q, events


def order_of(q: StopQueue) -> list[str]:
    return [e.approach.value for e in q.snapshot().order]


def kinds(events: list) -> list[str]:
    return [e.kind for e in events]


# -------------------------------------------------------------------------------------- arrival order


def test_queue_follows_arrival_order() -> None:
    cars = [Car(1, L, stop_t=10.0), Car(2, A, stop_t=11.5)]
    q, events = run(cars, ego_stop_t=13.0, until=14.0)
    snap = q.snapshot()
    assert order_of(q) == ["left", "ahead", "us"]
    assert snap.ego_position == 3 and not snap.your_turn
    assert "arrived" in kinds(events)


def test_cars_already_waiting_when_we_arrive_are_ahead_of_us() -> None:
    q, _ = run([Car(1, R, stop_t=2.0)], ego_stop_t=9.0, until=10.0)
    assert order_of(q) == ["right", "us"]
    assert q.snapshot().order[0].arrived == pytest.approx(2.0)


def test_car_of_unknown_arrival_is_ahead_of_us_even_on_our_left() -> None:
    # Already standing still when first seen (it may have waited long before we came): it must not count as a
    # tie with us, because the tie rule would give us the right of way over a car on our left.
    q = StopQueue()
    ego = EgoStop(at_stop=True, stopped_since=10.0, moving_for_s=0.0)
    q.update(10.0, ego, [])
    events = q.update(10.6, ego, [WaitingVehicle(1, L, True, 10.1, 0.0, BOXES[L], arrival_known=False)])
    snap = q.snapshot()
    assert order_of(q) == ["left", "us"] and snap.tie_groups == () and not snap.your_turn
    assert kinds(events) == ["vehicle_joined"]


def test_unknown_arrivals_come_first_and_only_tie_among_themselves() -> None:
    entries = [
        QueueEntry("k", R, 0.0),
        QueueEntry("u1", A, 5.0, arrival_known=False),
        QueueEntry("u2", L, 5.5, arrival_known=False),
        QueueEntry(EGO_KEY, US, 5.2),
    ]
    ordered, groups, deadlock = right_of_way_order(entries, tie_window_s=1.0)
    assert [e.key for e in ordered] == ["u2", "u1", "k", EGO_KEY]  # u1 (ahead) yields to u2 (on its right)
    assert groups == [["u1", "u2"]] and not deadlock


def test_car_arriving_after_us_goes_behind_us() -> None:
    q, _ = run([Car(1, L, stop_t=12.0)], ego_stop_t=10.0, until=13.0)
    assert order_of(q) == ["us", "left"] and q.snapshot().ego_position == 1


def test_no_queue_when_not_at_a_stop() -> None:
    q, events = run([Car(1, L, stop_t=1.0)], ego_stop_t=None, until=5.0)
    snap = q.snapshot()
    assert not snap.active and snap.order == () and events == []


# ------------------------------------------------------------------------------------------------ ties


def test_tie_with_car_on_our_right_they_go_first() -> None:
    q, _ = run([Car(1, R, stop_t=10.4)], ego_stop_t=10.0, until=11.0)
    snap = q.snapshot()
    assert order_of(q) == ["right", "us"]
    assert snap.tie_groups == ((EGO_KEY, "track:1"),) and not snap.deadlock


def test_tie_with_car_on_our_left_we_go_first() -> None:
    q, _ = run([Car(1, L, stop_t=9.6)], ego_stop_t=10.0, until=11.0)
    assert order_of(q) == ["us", "left"]


def test_tie_between_two_other_cars_uses_their_right() -> None:
    # Left and ahead tie. The car across from us faces south; the car on its right is our left car, so the
    # ahead car yields: left goes first even though it arrived later.
    q, _ = run([Car(1, L, stop_t=5.5), Car(2, A, stop_t=5.0)], ego_stop_t=9.0, until=10.0)
    assert order_of(q) == ["left", "ahead", "us"]
    # Ahead and right tie: the right car's right is the ahead car, so the right car yields.
    q, _ = run([Car(1, A, stop_t=5.5), Car(2, R, stop_t=5.0)], ego_stop_t=9.0, until=10.0)
    assert order_of(q) == ["ahead", "right", "us"]


def test_three_way_tie_has_a_clear_order() -> None:
    # Us, left and ahead within 1 s: nobody is on our right, so we go; then left (its right is us), then ahead.
    q, _ = run([Car(1, L, stop_t=10.2), Car(2, A, stop_t=10.5)], ego_stop_t=10.0, until=11.0)
    snap = q.snapshot()
    assert order_of(q) == ["us", "left", "ahead"] and not snap.deadlock


def test_opposite_cars_tied_go_by_arrival() -> None:
    q, _ = run([Car(1, L, stop_t=5.3), Car(2, R, stop_t=5.0)], ego_stop_t=9.0, until=10.0)
    assert order_of(q) == ["right", "left", "us"]


def test_four_way_tie_is_a_deadlock_and_never_our_clear_turn() -> None:
    cars = [Car(1, L, stop_t=10.1), Car(2, A, stop_t=10.2), Car(3, R, stop_t=10.3)]
    q, events = run(cars, ego_stop_t=10.0, until=11.0)
    snap = q.snapshot()
    assert snap.deadlock and not snap.your_turn
    assert "your_turn" not in kinds(events)


def test_ties_are_measured_from_the_first_arrival_of_a_group() -> None:
    entries = [QueueEntry("a", A, 0.0), QueueEntry("b", L, 0.9), QueueEntry("c", R, 1.8)]
    ordered, groups, deadlock = right_of_way_order(entries, tie_window_s=1.0)
    assert groups == [["a", "b"]]  # c is 1.8 s after a: not part of the tie even though it is 0.9 s after b
    assert [e.key for e in ordered] == ["b", "a", "c"] and not deadlock  # b (left) is on a's (ahead) right


# ---------------------------------------------------------------------------------- departures / turn


def test_departures_then_your_turn_fires_once() -> None:
    cars = [Car(1, L, stop_t=10.0, depart_t=15.0), Car(2, A, stop_t=11.0, depart_t=18.0)]
    q, events = run(cars, ego_stop_t=12.0, until=25.0)
    snap = q.snapshot()
    assert [e.approach.value for e in snap.departed] == ["left", "ahead"]
    assert order_of(q) == ["us"] and snap.your_turn
    assert kinds(events).count("your_turn") == 1
    your_turn = next(e for e in events if e.kind == "your_turn")
    assert 18.9 <= your_turn.t <= 19.2  # once the second car has kept driving for 1 s (a creep is not a departure)
    assert kinds(events).count("vehicle_departed") == 2


def test_your_turn_not_announced_at_an_empty_intersection() -> None:
    q, events = run([], ego_stop_t=10.0, until=15.0)
    assert q.snapshot().your_turn  # we may go (answers say so when asked)…
    assert "your_turn" not in kinds(events)  # …but there was nobody to wait for, so no alert


def test_creeping_forward_keeps_our_place() -> None:
    cars = [Car(1, L, stop_t=9.0)]
    q, events = run(cars, ego_stop_t=11.0, until=16.0, ego_moving=[(13.0, 13.8)])  # 0.8 s creep to see
    assert "left_intersection" not in kinds(events)
    assert order_of(q) == ["left", "us"] and q.snapshot().order[1].arrived == pytest.approx(11.0)


def test_driving_off_leaves_the_intersection_and_resets() -> None:
    q, events = run([Car(1, R, stop_t=8.0, depart_t=12.0)], ego_stop_t=10.0, until=20.0, ego_moving=[(15.0, 99.0)])
    assert kinds(events)[-1] == "left_intersection"
    assert not q.snapshot().active and q.snapshot().order == ()


# --------------------------------------------------------------------------- lost tracks, parked cars


def test_occluded_car_is_not_a_departure_and_relinks_with_a_new_id() -> None:
    # A truck crossing in front hides the waiting car for 3 s; the tracker gives it a new id afterwards.
    car = Car(1, R, stop_t=5.0, hidden=[(12.0, 15.0)], relink_id=7)
    q, events = run([car], ego_stop_t=10.0, until=16.0)
    snap = q.snapshot()
    assert order_of(q) == ["right", "us"]
    assert snap.order[0].key == "track:1" and snap.order[0].track_id == 7 and not snap.order[0].lost
    assert "vehicle_departed" not in kinds(events) and "vehicle_dropped" not in kinds(events)


def test_long_lost_car_is_dropped() -> None:
    car = Car(1, R, stop_t=5.0, hidden=[(12.0, 99.0)])
    q, events = run([car], ego_stop_t=10.0, until=25.0)
    assert "vehicle_dropped" in kinds(events) and order_of(q) == ["us"]


def test_car_that_never_leaves_is_treated_as_parked_without_an_alert() -> None:
    cfg = StopQueueConfig(parked_after_s=15.0)
    q, events = run([Car(1, L, stop_t=1.0)], ego_stop_t=10.0, until=30.0, cfg=cfg)
    dropped = [e for e in events if e.kind == "vehicle_dropped"]
    assert dropped and dropped[0].detail["reason"] == "probably parked" and 25.0 <= dropped[0].t <= 25.2
    snap = q.snapshot()
    assert order_of(q) == ["us"] and snap.your_turn  # answers may say nobody visible is ahead...
    assert [e.key for e in snap.dropped] == ["track:1"] and snap.dropped[0].drop_reason == "probably parked"
    assert "your_turn" not in kinds(events)  # ...but a guess never becomes a spoken "you're up"


def test_moving_cars_do_not_join() -> None:
    q = StopQueue()
    ego = EgoStop(at_stop=True, stopped_since=1.0, moving_for_s=0.0)
    q.update(1.5, ego, [WaitingVehicle(1, R, False, None, 0.0, BOXES[R])])  # rolling through, never stopped
    assert order_of(q) == ["us"]


def test_reset_clears_everything() -> None:
    q, _ = run([Car(1, L, stop_t=1.0)], ego_stop_t=3.0, until=4.0)
    q.reset()
    assert not q.snapshot().active and q.snapshot().order == ()
