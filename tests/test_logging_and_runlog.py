"""Structured logging (fields, JSON lines, secret redaction) and the per-run review folder."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.logging_setup import get_logger, redact, register_secrets, setup_logging, to_jsonable
from core.runlog import RunRecorder

SECRET = "sk-ant-supersecret-value-123"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    setup_logging("WARNING", secrets=[])


def _flush_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------------------- logging


def test_json_log_has_fields_and_redacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_file = tmp_path / "log.jsonl"
    setup_logging("INFO", log_file=log_file, secrets=[SECRET])
    log = get_logger("test.module")
    log.info("frame processed", fps=29.5, gpu_mb=812, note=f"key={SECRET}")
    log.debug("hidden at INFO")
    try:
        raise RuntimeError(f"boom {SECRET}")
    except RuntimeError:
        log.exception("failure")
    _flush_handlers()

    lines = _read_jsonl(log_file)
    assert len(lines) == 2
    first = lines[0]
    assert first["msg"] == "frame processed" and first["level"] == "INFO" and first["logger"] == "test.module"
    assert first["fps"] == 29.5 and first["gpu_mb"] == 812
    assert first["note"] == "key=***"
    assert "exc" in lines[1] and SECRET not in lines[1]["exc"]
    console = capsys.readouterr().err
    assert "frame processed fps=29.5 gpu_mb=812" in console
    assert SECRET not in console and SECRET not in log_file.read_text(encoding="utf-8")


def test_any_field_name_works_and_reserved_names_are_kept(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    setup_logging("INFO", log_file=log_file)
    log = get_logger("mic")
    log.info("mic level", level=0.8, msg="inner", ts=1, logger="x", exc="y", name="z")
    _flush_handlers()
    record = _read_jsonl(log_file)[0]
    assert record["level"] == "INFO" and record["msg"] == "mic level"
    assert record["f_level"] == 0.8 and record["f_msg"] == "inner" and record["f_ts"] == 1
    assert record["f_logger"] == "x" and record["f_exc"] == "y" and record["name"] == "z"


def test_numpy_and_nan_fields_become_valid_json(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    setup_logging("INFO", log_file=log_file)
    get_logger("t").info("np", gpu_mb=np.int64(812), fps=np.float32(29.5), bad=float("nan"), arr=np.arange(3))
    _flush_handlers()
    text = log_file.read_text(encoding="utf-8")
    assert "NaN" not in text
    record = json.loads(text)
    assert record["gpu_mb"] == 812 and record["fps"] == 29.5 and record["bad"] is None and record["arr"] == [0, 1, 2]


def test_resetup_without_secrets_keeps_redaction(tmp_path: Path) -> None:
    setup_logging("INFO", secrets=[SECRET])
    setup_logging("INFO", log_file=tmp_path / "log.jsonl")  # e.g. once the run folder exists
    assert redact(f"x {SECRET}") == "x ***"
    setup_logging("INFO", secrets=[])  # explicit empty list does clear
    assert redact(SECRET) == SECRET


def test_redact_ignores_short_values() -> None:
    register_secrets(["abc", SECRET])
    assert redact("abc " + SECRET) == "abc ***"


def test_to_jsonable() -> None:
    value = {"a": (1, 2.5, float("inf")), "b": np.float64(1.5), Path("k"): Path("v"), "c": {np.int32(3)}}
    assert to_jsonable(value) == {"a": [1, 2.5, None], "b": 1.5, "k": "v", "c": [3]}


# ---------------------------------------------------------------------------------------- run folder


def test_run_recorder_writes_all_streams(tmp_path: Path) -> None:
    register_secrets([SECRET])
    fixed = datetime(2026, 10, 10, 12, 0, 0)
    image = np.full((90, 160, 3), 127, np.uint8)
    with RunRecorder(
        tmp_path, "file:dev clip", {"a": 1, "leak": SECRET}, frame_interval_s=1.0, now=lambda: fixed
    ) as rec:
        assert rec.dir.name == "20261010-120000_file_dev_clip"
        rec.event("question", t=12.5, text="who got here first?", source="voice", score=float("nan"))
        rec.perf(fps=30.1, gpu_mb=np.int64(900))
        rec.detections(12.5, 375, [{"id": 4, "cls": "car", "box": [1, 2, 3, 4]}])
        saved = [rec.maybe_save_frame(image, t) for t in (0.0, 0.4, 0.99, 1.0, 1.5, 2.2)]
        assert saved == [True, False, False, True, False, True]
        assert rec.save_frame(image, "model q1/frame 0")
        rec.flush()
        out_dir = rec.dir

    events = _read_jsonl(out_dir / "events.jsonl")
    assert events[0]["kind"] == "question" and events[0]["t"] == 12.5 and events[0]["text"] == "who got here first?"
    assert events[0]["score"] is None
    assert _read_jsonl(out_dir / "perf.jsonl")[0]["gpu_mb"] == 900
    assert _read_jsonl(out_dir / "detections.jsonl")[0]["tracks"][0]["id"] == 4
    frames = sorted(p.name for p in (out_dir / "frames").iterdir())
    assert frames == [
        "000001_t00000.00.jpg",
        "000002_t00001.00.jpg",
        "000003_t00002.20.jpg",
        "000004_model_q1_frame_0.jpg",
    ]
    decoded = cv2.imread(str(out_dir / "frames" / "000002_t00001.00.jpg"))
    assert decoded is not None and decoded.shape == (90, 160, 3)
    config_text = (out_dir / "config.json").read_text("utf-8")
    assert SECRET not in config_text and '"a": 1' in config_text


def test_records_are_snapshotted_at_call_time(tmp_path: Path) -> None:
    with RunRecorder(tmp_path, "mutation") as rec:
        tracks = [{"id": 1}]
        rec.detections(1.0, 30, tracks)
        tracks.append({"id": 2})  # the perception loop reuses its list for the next frame
        tracks[0]["id"] = 99
        rec.flush()
        record = _read_jsonl(rec.dir / "detections.jsonl")[0]
    assert record["tracks"] == [{"id": 1}]


def test_frames_are_dropped_not_blocking_when_writer_is_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    real_imencode = cv2.imencode

    def slow_imencode(*args, **kwargs):  # type: ignore[no-untyped-def]
        gate.wait(5)
        return real_imencode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", slow_imencode)
    image = np.zeros((8, 8, 3), np.uint8)
    with RunRecorder(tmp_path, "stall", max_pending_frames=2) as rec:
        start = time.monotonic()
        results = [rec.save_frame(image, f"f{i}") for i in range(10)]
        for i in range(200):
            rec.event("tick", i=i)  # text must never block, even with the frame writer stalled
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        assert results.count(False) >= 7 and rec.dropped_frames == results.count(False)
        gate.set()
        rec.flush()
        assert len(_read_jsonl(rec.dir / "events.jsonl")) == 200


def test_close_drains_everything_and_later_records_are_ignored(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "drain")
    for i in range(500):
        rec.event("e", i=i)
    rec.close()
    assert len(_read_jsonl(rec.dir / "events.jsonl")) == 500
    rec.close()  # idempotent
    rec.event("late")  # ignored, must not raise
    rec.flush()  # must not hang after close
    assert not rec.save_frame(np.zeros((4, 4, 3), np.uint8), "late")
    assert len(_read_jsonl(rec.dir / "events.jsonl")) == 500


def test_concurrent_writers_and_close(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "threads")

    def write(k: int) -> None:
        for i in range(2000):
            rec.event("e", k=k, i=i)

    threads = [threading.Thread(target=write, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.01)
    start = time.monotonic()
    rec.close()  # concurrently with the writers: must not hang or raise
    assert time.monotonic() - start < 10.0
    for t in threads:
        t.join(10)
    lines = (rec.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines and all(json.loads(line)["kind"] == "e" for line in lines)  # every written line is complete
    for k in range(4):  # each writer's records that made it in are in order (no interleaving corruption)
        idx = [json.loads(line)["i"] for line in lines if json.loads(line)["k"] == k]
        assert idx == sorted(idx)


def test_unique_dirs_and_ascii_labels(tmp_path: Path) -> None:
    fixed = datetime(2026, 1, 1)
    a = RunRecorder(tmp_path, "café run", now=lambda: fixed)
    b = RunRecorder(tmp_path, "café run", now=lambda: fixed)
    assert a.dir != b.dir and b.dir.name.endswith("-2")
    assert a.dir.name.isascii()
    assert a.save_frame(np.zeros((4, 4, 3), np.uint8), "ünïcode tag")
    a.close()
    b.close()
    assert len(list((a.dir / "frames").glob("*.jpg"))) == 1


def test_seek_backwards_saves_again_without_overwriting(tmp_path: Path) -> None:
    with RunRecorder(tmp_path, "seek") as rec:
        image = np.zeros((8, 8, 3), np.uint8)
        assert rec.maybe_save_frame(image, 10.0)
        assert rec.maybe_save_frame(image, 4.0)  # after a seek back, a frame is saved immediately
        assert rec.maybe_save_frame(image, 10.0)  # same media time again (loop): a new file, not an overwrite
        rec.flush()
        assert len(list((rec.dir / "frames").glob("*.jpg"))) == 3
