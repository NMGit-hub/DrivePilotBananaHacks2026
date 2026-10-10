"""The periodic model scan for parking spots (eval/parking_scan.py): parsing, the announcement logic and scoring.

No network: replies are written by hand in the shapes the prompts ask for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.config import VisionConfig
from eval.parking_eval import ClipLabels, LabeledSpot, LookAlike
from eval.parking_scan import (
    PROMPTS,
    Announcement,
    ScanDecision,
    ScanReport,
    ScanResult,
    SpotAnnouncer,
    announce,
    colors,
    describe_report,
    different_cars,
    parse_rows,
    parse_scan,
    parse_spots,
    score_scan,
    summarize_scan,
)


def report(
    t: float,
    side: str = "left",
    cars: int | None = 1,
    before: str | None = "white SUV",
    restriction: str = "none",
    confidence: str = "high",
    stalls: int = 1,
    closed: bool | None = None,
) -> ScanReport:
    return ScanReport(t, side, cars, before, restriction, confidence, stalls, closed)


def scan(t: float, *reports: ScanReport, in_lot: bool | None = True, latency: float | None = 1.0) -> ScanResult:
    return ScanResult(t, in_lot, tuple(reports), latency)


# ------------------------------------------------------------------------------------------- parsing: spots


def test_parse_spots_reads_a_well_formed_reply() -> None:
    data = {
        "in_lot": True,
        "spots": [
            {
                "side": "left",
                "cars_ahead": 2,
                "before": "white SUV",
                "stalls": 3,
                "restriction": "none",
                "confidence": "high",
            },
            {
                "side": "Right",
                "cars_ahead": "1",
                "before": None,
                "stalls": 1,
                "restriction": "EV",
                "confidence": "Medium",
            },
        ],
    }
    result = parse_spots(data, 12.0)
    assert result.t == 12.0 and result.in_lot is True
    a, b = result.reports
    assert (a.side, a.cars_ahead, a.before, a.stalls, a.restriction, a.confidence) == (
        "left",
        2,
        "white SUV",
        3,
        "none",
        "high",
    )
    assert (b.side, b.cars_ahead, b.before, b.restriction, b.confidence) == ("right", 1, None, "ev", "medium")


def test_parse_spots_drops_entries_without_a_side_or_confidence_and_cleans_the_rest() -> None:
    data = {
        "in_lot": "yes",  # not a bool: unknown
        "spots": [
            {"side": "ahead", "confidence": "high"},
            {"side": "left", "confidence": "sure"},
            "garbage",
            {
                "side": "left",
                "confidence": "low",
                "cars_ahead": True,
                "before": "x" * 60,
                "stalls": 0,
                "restriction": "Fire lane",
            },
            {"side": "right", "confidence": "high", "cars_ahead": -1, "before": "null", "restriction": "disabled"},
            {"side": "right", "confidence": "high", "restriction": "a mystery"},
        ],
    }
    result = parse_spots(data, 3.0)
    assert result.in_lot is None
    low, disabled, mystery = result.reports
    assert (low.cars_ahead, low.before, low.stalls, low.restriction) == (None, None, 1, "fire_lane")
    assert (disabled.cars_ahead, disabled.before, disabled.restriction) == (None, None, "handicap")
    assert mystery.restriction == "unknown" and mystery.cars_ahead is None


def test_parse_spots_survives_a_reply_without_a_list() -> None:
    assert parse_spots({"in_lot": False, "spots": "none"}, 1.0).reports == ()
    assert parse_spots({}, 1.0) == ScanResult(1.0, None, ())


# -------------------------------------------------------------------------------------------- parsing: rows


def test_parse_rows_finds_the_first_gap_on_each_side() -> None:
    data = {
        "in_lot": True,
        "left": {"stalls": ["gray sedan", "white SUV", "empty", "empty", "black pickup"], "confidence": "high"},
        "right": {"stalls": ["empty", "red car"], "confidence": "medium"},
    }
    left, right = parse_rows(data, 4.0).reports
    assert (left.side, left.cars_ahead, left.before, left.stalls, left.closed) == ("left", 2, "white SUV", 2, True)
    assert (right.side, right.cars_ahead, right.before, right.stalls, right.closed) == ("right", 0, None, 1, True)
    assert left.confidence == "high" and right.confidence == "medium"


def test_parse_rows_marks_an_open_row_end_and_reads_restrictions() -> None:
    data = {"left": {"stalls": ["white SUV", "Empty (EV)", "empty: ev"], "confidence": "high"}}
    (spot,) = parse_rows(data, 0.0).reports
    assert (spot.restriction, spot.stalls, spot.closed, spot.cars_ahead) == ("ev", 2, False, 1)


def test_parse_rows_reports_restricted_runs_up_to_the_first_unrestricted_one() -> None:
    items = ["black SUV", "empty:handicap", "empty", "gray sedan", "empty"]
    handicap, free = parse_rows({"right": {"stalls": items, "confidence": "high"}}, 0.0).reports
    assert (handicap.restriction, handicap.stalls, handicap.restricted) == ("handicap", 1, True)
    assert (free.restriction, free.stalls, free.cars_ahead, free.before) == ("none", 1, 1, "black SUV")


def test_parse_rows_skips_sides_without_a_row_of_stalls_or_confidence() -> None:
    data = {
        "left": {"beside": "curb", "stalls": ["empty"], "confidence": "high"},
        "right": {"beside": "stalls", "stalls": ["empty"]},
    }
    assert parse_rows(data, 0.0).reports == ()
    ok = {"right": {"beside": "Stalls", "stalls": ["empty"], "confidence": "high"}}
    assert len(parse_rows(ok, 0.0).reports) == 1


def test_parse_rows_trusts_an_open_row_end_only_when_a_boundary_is_named() -> None:
    def row(end: str) -> dict[str, object]:
        return {"left": {"beside": "stalls", "stalls": ["white SUV", "empty"], "end": end, "confidence": "high"}}

    assert parse_rows(row("out_of_view"), 0.0).reports == ()
    assert parse_rows(row("car"), 0.0).reports == ()
    assert len(parse_rows(row("Cross aisle"), 0.0).reports) == 1
    assert len(parse_rows(row("island"), 0.0).reports) == 1
    gap = {"left": {"beside": "stalls", "stalls": ["empty", "red car"], "end": "out_of_view", "confidence": "high"}}
    assert len(parse_rows(gap, 0.0).reports) == 1  # bounded by the red car
    facing = {"left": {"beside": "facing", "stalls": ["empty", "red car"], "confidence": "high"}}
    assert parse_rows(facing, 0.0).reports == ()


def test_parse_rows_stops_at_a_malformed_item() -> None:
    data = {"left": {"stalls": ["white SUV", 3, "empty"], "confidence": "high"}}
    assert parse_rows(data, 0.0).reports == ()


def test_parse_scan_dispatches_on_the_prompt_kind_and_keeps_the_latency() -> None:
    rows = {"left": {"stalls": ["empty"], "confidence": "high"}}
    assert len(parse_scan(rows, 2.0, "rows", 1.25).reports) == 1
    assert parse_scan(rows, 2.0, "rows", 1.25).latency_s == 1.25
    assert parse_scan(rows, 2.0, "spots").reports == ()


# ------------------------------------------------------------------------------------------------- deciding


def test_decision_accepts_by_confidence_restriction_and_shape() -> None:
    d = ScanDecision()
    assert d.accepts(report(0.0, confidence="medium"))
    assert not d.accepts(report(0.0, confidence="low"))
    assert not d.accepts(report(0.0, restriction="handicap"))
    assert d.accepts(report(0.0, restriction="ev"))  # announced as such
    assert d.accepts(report(0.0, restriction="unknown"))
    assert not ScanDecision(min_confidence="high").accepts(report(0.0, confidence="medium"))
    closed = ScanDecision(require_closed=True)
    assert not closed.accepts(report(0.0, closed=False))
    assert closed.accepts(report(0.0, closed=True)) and closed.accepts(report(0.0, closed=None))
    neighbour = ScanDecision(require_neighbour=True)
    assert not neighbour.accepts(report(0.0, cars=0)) and neighbour.accepts(report(0.0, cars=1))


def test_decision_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        ScanDecision(min_confidence="sure")
    with pytest.raises(ValueError):
        ScanDecision(confirm_scans=0)


def test_a_spot_is_announced_once_as_we_approach_it() -> None:
    results = [scan(0.0, report(0.0, cars=3)), scan(2.0, report(2.0, cars=2)), scan(4.0, report(4.0, cars=0))]
    (a,) = announce(results)
    assert (a.t, a.first_t, a.report.cars_ahead, a.spoken_t) == (0.0, 0.0, 3, 1.0)


def test_a_spot_further_ahead_than_the_known_one_is_a_new_spot() -> None:
    results = [scan(0.0, report(0.0, cars=1)), scan(2.0, report(2.0, cars=4))]
    assert [a.t for a in announce(results)] == [0.0, 2.0]


def test_a_spot_seen_again_after_the_gap_is_announced_again() -> None:
    results = [scan(0.0, report(0.0)), scan(2.0), scan(4.0), scan(6.0, report(6.0))]
    assert [a.t for a in announce(results)] == [0.0, 6.0]
    assert [a.t for a in announce(results, ScanDecision(track_gap_s=8.0))] == [0.0]


def test_the_count_cannot_drop_faster_than_we_drive() -> None:
    d = ScanDecision(count_slack=0, max_drop_per_s=1.0)
    results = [scan(0.0, report(0.0, cars=6)), scan(2.0, report(2.0, cars=1))]  # 5 cars in 2 s: another spot
    assert len(announce(results, d)) == 2
    results = [scan(0.0, report(0.0, cars=6)), scan(2.0, report(2.0, cars=4))]
    assert len(announce(results, d)) == 1


def test_sides_and_colours_keep_spots_apart() -> None:
    results = [
        scan(0.0, report(0.0, side="left"), report(0.0, side="right")),
        scan(2.0, report(2.0, side="left", before="silver sedan")),  # white SUV before: not the same gap
    ]
    assert [(a.t, a.report.side) for a in announce(results)] == [(0.0, "left"), (0.0, "right"), (2.0, "left")]
    assert len(announce(results, ScanDecision(use_before=False))) == 2
    gray = [scan(0.0, report(0.0, before="gray sedan")), scan(2.0, report(2.0, before="silver sedan"))]
    assert len(announce(gray)) == 1  # gray and silver: one colour group


def test_two_reports_on_one_side_in_one_scan_are_two_spots() -> None:
    results = [scan(0.0, report(0.0, cars=1), report(0.0, cars=4))]
    assert len(announce(results)) == 2


def test_confirmation_needs_agreeing_scans() -> None:
    d = ScanDecision(confirm_scans=2)
    assert announce([scan(0.0, report(0.0))], d) == []
    (a,) = announce([scan(0.0, report(0.0, cars=2)), scan(1.0, report(1.0, cars=2))], d)
    assert (a.t, a.first_t) == (1.0, 0.0)


def test_low_confidence_restricted_and_off_lot_reports_are_ignored() -> None:
    assert announce([scan(0.0, report(0.0, confidence="low"), report(0.0, side="right", restriction="reserved"))]) == []
    off_lot = [scan(0.0, report(0.0), in_lot=False)]
    assert len(announce(off_lot)) == 1
    assert announce(off_lot, ScanDecision(require_lot=True)) == []
    assert len(announce([scan(0.0, report(0.0), in_lot=None)], ScanDecision(require_lot=True))) == 1


def test_the_announcer_sorts_by_time_and_resets() -> None:
    announcer = SpotAnnouncer()
    assert len(announcer.update(scan(0.0, report(0.0)))) == 1
    assert announcer.update(scan(2.0, report(2.0))) == []
    announcer.reset()
    assert len(announcer.update(scan(4.0, report(4.0)))) == 1
    assert [a.t for a in announce([scan(2.0, report(2.0, cars=5)), scan(0.0, report(0.0, cars=1))])] == [0.0, 2.0]


def test_colours() -> None:
    assert colors("Silver Toyota sedan") == colors("gray car") == frozenset({"gray"})
    assert colors(None) == frozenset() and colors("dark SUV") == frozenset()
    assert different_cars("white SUV", "black sedan")
    assert not different_cars("white SUV", "dark sedan")  # "dark": no colour named
    assert not different_cars("white SUV", None)


@pytest.mark.parametrize(
    ("r", "text"),
    [
        (report(0.0, cars=2, before="white SUV"), "Open spot on your left, two cars ahead, just after the white SUV."),
        (report(0.0, side="right", cars=1, before="gray sedan"), "Open spot on your right, just after the gray sedan."),
        (report(0.0, cars=0, before=None, stalls=3), "Three open spots on your left."),
        (report(0.0, cars=None, before=None, restriction="ev"), "Open EV charging spot on your left."),
        (
            report(0.0, cars=12, before=None, restriction="compact", stalls=2),
            "Two open compact spots on your left, 12 cars ahead.",
        ),
    ],
)
def test_describe_report(r: ScanReport, text: str) -> None:
    assert describe_report(r) == text
    assert Announcement(0.0, 0.0, r).text == text


# --------------------------------------------------------------------------------------------------- prompts


def test_prompt_requests_carry_the_frames_the_version_asks_for() -> None:
    vision = VisionConfig()
    now = np.zeros((720, 1280, 3), np.uint8)
    plain = PROMPTS["v3"].request(now, now, vision)
    assert [i.label for i in plain.images] == ["Now"] and plain.temperature == vision.temperature
    motion = PROMPTS["v3e"].request(now, now, vision)
    assert [i.label for i in motion.images] == ["0.7 s earlier", "Now"]
    assert PROMPTS["v3e"].request(now, None, vision).images[-1].label == "Now"
    assert all(p.version == name for name, p in PROMPTS.items())
    assert all(p.kind in ("spots", "rows") for p in PROMPTS.values())


# --------------------------------------------------------------------------------------------------- scoring


def clip() -> ClipLabels:
    spots = (
        LabeledSpot("left", 5.0, 8.0, "none", "clear", 2, "white SUV"),  # target with a neighbour
        LabeledSpot("right", 10.0, None, "none", "clear", 3, None),  # target at a row start, never passed
        LabeledSpot("right", 20.0, 22.0, "handicap", "clear", 1, "red car"),  # not a target
    )
    looks = (LookAlike("left", 30.0, 32.0, "island"),)
    return ClipLabels("c", "lot_aisles", True, ((0.0, 40.0),), None, spots, looks, ())


def test_score_scan_ties_announcements_to_the_labels() -> None:
    results = [
        scan(4.0, report(4.0, cars=2)),  # left spot, 1 s early: a hit
        scan(6.0, report(6.0, cars=4)),  # left again, further ahead: a new announcement, a repeat of the same spot
        scan(14.0, report(14.0, side="right", cars=0, before=None), latency=2.0),  # right target
        # the open run again (a spot that never leaves the frame matches for 8 s after it became visible)
        scan(17.0, report(17.0, side="right", cars=3, before="red car")),
        scan(31.0, report(31.0, cars=1, before="blue van")),  # look-alike
        scan(36.0, report(36.0, side="right", cars=5, before="black SUV")),  # long after: no window, false alarm
        scan(37.0, report(37.0, cars=1, before="green car")),  # nothing labeled: false alarm
    ]
    s = score_scan(clip(), 40.0, results)
    assert s.targets == [0, 1]
    assert s.hits[0].t == 4.0 and s.hits[1].t == 14.0
    assert s.matches[3].spot == clip().spots[1]
    assert [m.repeat for m in s.matches] == [False, True, False, True, False, False, False]
    assert [m.false_alarm for m in s.matches] == [False, False, False, False, True, True, True]
    assert s.matches[4].look_alike is not None and s.matches[6].look_alike is None

    summary = summarize_scan([s])
    assert (summary.targets, summary.found, summary.targets_with_neighbour, summary.found_with_neighbour) == (
        2,
        2,
        1,
        1,
    )
    assert (summary.false_alarms, summary.look_alike_hits, summary.repeats, summary.announcements) == (3, 1, 2, 7)
    assert summary.recall == 1.0 and summary.recall_with_neighbour == 1.0 and summary.recall_without_neighbour == 1.0
    assert summary.false_per_10min == pytest.approx(10.0 * 3 / (40.0 / 60.0))
    assert summary.delays == (-1.0, 4.0) and summary.spoken_delays == (0.0, 6.0)
    assert summary.late == 0 and summary.restricted == 0
    d = summary.as_dict()
    assert d["delay_median"] == 1.5 and d["recall"] == 1.0 and d["scans"] == 7


def test_announcing_a_restricted_labeled_spot_is_counted() -> None:
    handicap = [scan(21.0, report(21.0, side="right", cars=0, before=None, restriction="ev"))]  # misread as EV
    s = score_scan(clip(), 40.0, handicap)
    assert s.matches[0].spot == clip().spots[2]  # (the open run's window ended at 18 s)
    narrow = ClipLabels("n", "lot_aisles", True, ((0.0, 40.0),), None, clip().spots[2:], (), ())
    summary = summarize_scan([score_scan(narrow, 40.0, handicap)])
    assert summary.restricted == 1 and summary.false_alarms == 0 and summary.targets == 0


def test_late_announcements_and_empty_summaries() -> None:
    late = [scan(7.9, report(7.9, cars=1), latency=1.5)]  # matched at 7.9 s but spoken at 9.4 s, after t_pass 8.0
    summary = summarize_scan([score_scan(clip(), 40.0, late)])
    assert summary.found == 1 and summary.late == 1
    empty = summarize_scan([])
    assert math.isnan(empty.recall) and math.isnan(empty.false_per_10min)
    assert math.isnan(empty.as_dict()["delay_median"])
