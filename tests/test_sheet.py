"""M8: the test-sheet runner's parsing, scoring and report (the clip runs themselves need the GPU and a key)."""

from __future__ import annotations

import csv
from pathlib import Path

from answers.vision import MockVision
from eval.run_sheet import DEFAULT_SHEET, Row, judge, load_sheet, parse_selection, summarize, write_outputs


def make_sheet(tmp_path: Path) -> Path:
    path = tmp_path / "sheet.csv"
    path.write_text(
        "clip,time_s,question,expected,type,notes\n"
        "red_light,28,tell me when it's green,\"ALERT@36.4: agrees, then says it's green\",lights,\n"
        'red_light,30,is the light green?,"No, it\'s red.",lights,a note\n'
        "school_bus,18,can I pass the school bus?,No: its STOP arm is out.,safety,\n",
        encoding="utf-8",
    )
    return path


def test_sheet_rows_and_alert_expectations(tmp_path: Path) -> None:
    rows = load_sheet(make_sheet(tmp_path))
    assert [r.index for r in rows] == [1, 2, 3]
    assert rows[0].alert_at == 36.4 and rows[0].expected == "agrees, then says it's green"
    assert rows[1].alert_at is None and rows[1].notes == "a note" and rows[1].type == "lights"


def test_the_starter_sheet_is_well_formed() -> None:
    rows = load_sheet(DEFAULT_SHEET)
    assert len(rows) >= 25
    assert {r.type for r in rows} >= {"right of way", "lights", "signs", "parking", "road", "lanes", "safety"}
    assert all(r.question and r.expected and r.time_s >= 0 for r in rows)


def test_row_selection() -> None:
    assert parse_selection("1-3,7") == {1, 2, 3, 7}
    assert parse_selection("5") == {5}


def row(**kwargs: object) -> Row:
    base: dict[str, object] = {
        "index": 1,
        "clip": "c",
        "time_s": 1.0,
        "question": "q?",
        "expected": "e",
        "type": "lights",
    }
    return Row(**{**base, **kwargs})  # type: ignore[arg-type]


def test_alert_rows_need_the_alert_at_the_right_time() -> None:
    never = row(answer="Okay, I'll tell you.", alert_at=36.4)
    judge(None, never)
    assert never.judge == "wrong" and "never spoken" in never.judge_reason
    late = row(answer="Okay, I'll tell you.", alert_at=36.4, alert_said_at=39.0)
    judge(None, late)
    assert late.judge == "wrong"
    on_time = row(answer="Okay, I'll tell you.", alert_at=36.4, alert_said_at=36.9)
    judge(MockVision('{"verdict": "right", "reason": "ok"}'), on_time)
    assert on_time.judge == "right"


def test_unanswered_rows_are_wrong_and_unjudged_rows_say_why() -> None:
    silent = row(answer=None, source="not heard")
    judge(MockVision('{"verdict": "right"}'), silent)
    assert silent.judge == "wrong" and "not heard" in silent.judge_reason
    nobody = row(answer="Red.")
    judge(None, nobody)
    assert nobody.judge == "" and "no judge" in nobody.judge_reason


def test_manual_verdicts_win_and_the_summary_counts_them(tmp_path: Path) -> None:
    rows = [
        row(index=1, judge="right", first_word_s=0.4, answer_s=0.2, answer="a"),
        row(index=2, judge="wrong", manual="partly", first_word_s=0.6, answer_s=0.3, answer="b"),
        row(index=3, type="signs", judge="wrong", first_word_s=2.0, answer_s=1.5, answer="c"),
    ]
    summary = summarize(rows)
    lights = summary["by_type"]["lights"]
    assert (lights["right"], lights["partly"], lights["wrong"]) == (1, 1, 0)
    assert lights["accuracy"] == 0.5 and lights["accuracy_with_partly"] == 0.75
    assert summary["overall"]["median_first_word_s"] == 0.6 and summary["overall"]["median_answer_s"] == 0.3
    write_outputs(tmp_path, rows, summary, {"sheet": "s", "mode": "typed", "judge": "mock", "when": "now"})
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    assert [r["verdict"] for r in written] == ["right", "partly", "wrong"] and written[1]["manual"] == "partly"
    assert "| lights | 2 | 1 | 1 | 0 | 0.5 |" in (tmp_path / "report.md").read_text(encoding="utf-8")
