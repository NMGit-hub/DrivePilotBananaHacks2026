"""The showcase renderer's timeline (ui/timeline.py): placing spoken lines like the real speaker, caption cards,
and the audio mix."""

from __future__ import annotations

import numpy as np
import pytest

from ui.timeline import (
    ALERT,
    ANSWER,
    QUESTION,
    SAFETY,
    Caption,
    FramePicker,
    Line,
    Window,
    active,
    after_answers,
    answer_delay,
    build_captions,
    mix,
    question_caption,
    schedule,
    visible_captions,
)


def starts(placed: list) -> dict[str, tuple[float, float, bool]]:
    return {p.line.text: (round(p.start, 3), round(p.end, 3), p.cut) for p in placed}


def test_answer_delay_local_and_model() -> None:
    assert answer_delay("local", 1.0) == pytest.approx(0.45)
    assert answer_delay("control", None) == pytest.approx(0.45)
    assert answer_delay("model", 1200.0, 0.3) == pytest.approx(1.5)
    assert answer_delay("model", 100.0, 0.1) == pytest.approx(0.8)  # never instant
    assert answer_delay("model", 9000.0, 0.3) == pytest.approx(3.0)  # a slow call doesn't leave the video silent
    assert answer_delay("model", None) == pytest.approx(0.45)


def test_after_answers_moves_events_the_answer_caused() -> None:
    answering = [(10.0, 11.2), (20.0, 21.0)]
    assert after_answers(10.5, answering) == 11.2  # route guidance while "take me to ..." was being answered
    assert after_answers(9.9, answering) == 9.9
    assert after_answers(11.2, answering) == 11.2
    assert after_answers(20.0, answering) == 21.0
    assert after_answers(15.0, []) == 15.0


def test_question_caption() -> None:
    assert question_caption("tell me when it's green") == "Tell me when it's green"
    assert question_caption("what does that sign say") == "What does that sign say?"
    assert question_caption("is it my turn?") == "Is it my turn?"
    assert question_caption("  how   far ") == "How far?"
    assert question_caption("") == ""


def test_sentences_follow_each_other() -> None:
    lines = [
        Line(ANSWER, "One.", 10.0, 1.0, group=1, uid=1),
        Line(ANSWER, "Two.", 10.0, 2.0, group=1, uid=2),
    ]
    placed = starts(schedule(lines, gap=0.1))
    assert placed["One."] == (10.0, 11.0, False)
    assert placed["Two."] == (11.1, 13.1, False)


def test_alert_interrupts_answer_which_is_said_again() -> None:
    lines = [
        Line(ANSWER, "Long answer.", 10.0, 3.0, group=1, uid=1),
        Line(ALERT, "Stop sign ahead.", 11.0, 1.0, priority=0, uid=2),
    ]
    placed = schedule(lines, gap=0.1)
    answer = [p for p in placed if p.line.text == "Long answer."]
    assert [(round(p.start, 3), round(p.end, 3), p.cut) for p in answer] == [(10.0, 11.0, True), (12.1, 15.1, False)]
    alert = next(p for p in placed if p.line.kind == ALERT)
    assert (alert.start, alert.end, alert.cut) == (11.0, 12.0, False)


def test_alert_that_makes_answers_stale_drops_them() -> None:
    lines = [
        Line(ANSWER, "First.", 10.0, 3.0, group=1, uid=1),
        Line(ANSWER, "Second.", 10.0, 3.0, group=1, uid=2),
        Line(ALERT, "Light's green.", 11.0, 1.0, priority=2, drops_answers=True, uid=3),
    ]
    placed = starts(schedule(lines))
    assert placed["First."] == (10.0, 11.0, True)
    assert "Second." not in placed  # queued, stale: never said
    assert placed["Light's green."] == (11.0, 12.0, False)


def test_a_stale_making_alert_also_drops_answers_still_on_their_way() -> None:
    """Asked at 10.0, "Light's green." at 10.2, the answer due at 10.45: in the app it would have been cut while
    playing; it must not be said after the alert. A question asked after the alert is answered."""
    lines = [
        Line(ANSWER, "The light is red, wait.", 10.45, 1.5, group=1, uid=1, asked_t=10.0),
        Line(ALERT, "Light's green.", 10.2, 1.0, priority=2, drops_answers=True, uid=2),
        Line(ANSWER, "It's green now.", 12.45, 1.0, group=2, uid=3, asked_t=12.0),
        Line(ANSWER, "Unrelated.", 10.5, 1.0, uid=4),  # (not an answer to a question: kept)
    ]
    placed = starts(schedule(lines))
    assert "The light is red, wait." not in placed
    assert placed["Light's green."] == (10.2, 11.2, False)
    assert "It's green now." in placed and "Unrelated." in placed


def test_safety_alert_cuts_off_a_lesser_alert() -> None:
    lines = [
        Line(ALERT, "School zone ahead.", 10.0, 2.0, priority=0, uid=1),
        Line(ALERT, "Pedestrian ahead!", 10.5, 1.0, priority=SAFETY, uid=2),
        Line(ALERT, "Speed limit 30.", 10.6, 1.0, priority=0, uid=3),
    ]
    placed = schedule(lines, gap=0.1)
    order = [(p.line.text, round(p.start, 2), p.cut) for p in placed]
    assert order == [
        ("School zone ahead.", 10.0, True),
        ("Pedestrian ahead!", 10.5, False),
        ("School zone ahead.", 11.6, False),  # said again right after
        ("Speed limit 30.", 13.7, False),
    ]


def test_question_stops_the_previous_answer_and_drops_late_ones() -> None:
    lines = [
        Line(ANSWER, "Old answer.", 10.0, 5.0, group=1, uid=1),
        Line(QUESTION, "New question", 12.0, 1.5, group=2, uid=2),
        Line(ANSWER, "Old answer, part two.", 12.5, 1.0, group=1, uid=3),  # arrives after the new question
        Line(ANSWER, "New answer.", 14.0, 1.0, group=2, uid=4),
    ]
    placed = starts(schedule(lines))
    assert placed["Old answer."] == (10.0, 12.0, True)
    assert placed["New question"] == (12.0, 13.5, False)
    assert "Old answer, part two." not in placed
    assert placed["New answer."] == (14.0, 15.0, False)


def test_captions_group_question_and_answer() -> None:
    lines = [
        Line(QUESTION, "What's that?", 8.0, 1.5, group=1, uid=1),
        Line(ANSWER, "A stop sign.", 10.0, 1.0, group=1, uid=2),
        Line(ANSWER, "Stop there.", 10.0, 1.0, group=1, uid=3),
        Line(ALERT, "Stop sign ahead.", 10.5, 1.0, priority=0, uid=4),
    ]
    captions = build_captions(schedule(lines, gap=0.0))
    kinds = [c.kind for c in captions]
    assert kinds == [QUESTION, ANSWER, ALERT]
    question, answer, _ = captions
    assert question.end == pytest.approx(answer.end)  # the question stays up while its answer plays
    assert answer.text_at(10.2) == "A stop sign."  # (the interrupted sentence appears once)
    assert answer.text_at(answer.end) == "A stop sign. Stop there."


def test_visible_captions_fade_and_limit() -> None:
    cards = [
        Caption(QUESTION, [(0.0, "q")], 0.0, 2.0),
        Caption(ANSWER, [(1.0, "a")], 1.0, 3.0),
        Caption(ALERT, [(2.0, "x")], 2.0, 2.5),
    ]
    shown = visible_captions(cards, 2.1, hold_s=1.0, fade_s=0.5, fade_in_s=0.2, max_shown=2)
    assert [s.text for s in shown] == ["a", "x"]
    assert shown[1].alpha == pytest.approx(0.5)  # fading in
    late = visible_captions(cards, 3.75, hold_s=1.0, fade_s=0.5)
    assert [(s.text, round(s.alpha, 3)) for s in late] == [("a", 1.0), ("x", 0.5)]  # x fading out, q gone
    assert visible_captions(cards, 10.0) == []
    assert visible_captions(cards, -1.0) == []


def test_active_window() -> None:
    windows = [Window(1.0, 3.0, 7), Window(2.0, 4.0, 9)]
    assert active(windows, 0.5) is None
    assert active(windows, 1.5).value == 7
    assert active(windows, 2.5).value == 9  # the latest one
    assert active(windows, 4.0) is None


def test_frame_picker_repeats_and_drops_to_a_constant_rate() -> None:
    # irregular source: three frames with one timestamp (a BDD .mov's start), then a gap, then a burst
    source = [(0.0, "a"), (0.0333, "b"), (0.0333, "c"), (0.0333, "d"), (0.1, "e"), (0.2, "f"), (0.21, "g"), (0.3, "h")]
    picker = FramePicker(iter(source), half_slot=0.05)
    shown = [picker.at(k * 0.1) for k in range(5)]
    assert [s[1] if s else None for s in shown] == ["d", "e", "g", "h", "h"]
    assert picker.exhausted
    empty = FramePicker(iter([]), 0.05)
    assert empty.at(0.0) is None and empty.exhausted


def test_mix_places_and_cuts() -> None:
    rate = 100
    lines = [
        Line(ANSWER, "a", 1.0, 1.0, group=1, uid=1),
        Line(ALERT, "b", 1.5, 0.5, priority=0, uid=2),
    ]
    placed = schedule(lines, gap=0.0)
    audio = {1: np.full(100, 0.25, np.float32), 2: np.full(50, 0.5, np.float32)}
    track = mix(placed, lambda line: audio[line.uid], rate, t0=0.0, t1=4.0)
    s = track.samples
    assert len(s) == 400
    assert np.all(s[:100] == 0)
    assert np.allclose(s[100:150], 0.25)  # the answer until the alert
    assert np.allclose(s[150:200], 0.5)  # the alert (the answer was cut, not mixed under it)
    assert np.allclose(s[200:300], 0.25)  # the answer again
    assert np.all(s[300:] == 0)
    assert np.allclose(track.chunk(1.4, 1.6), [0.25] * 10 + [0.5] * 10)
    assert len(track.chunk(3.9, 4.2)) == 30 and np.all(track.chunk(3.9, 4.2) == 0)


def test_mix_keeps_within_range_and_clips_before_start() -> None:
    lines = [Line(QUESTION, "q", -0.5, 1.0, uid=1), Line(ALERT, "x", 0.0, 1.0, priority=0, uid=2)]
    placed = schedule(lines)
    track = mix(placed, lambda line: np.full(100, 0.8, np.float32), 100, t0=0.0, t1=1.0)
    assert len(track.samples) == 100
    assert float(np.max(np.abs(track.samples))) <= 1.0 + 1e-6
