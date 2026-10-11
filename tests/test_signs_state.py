"""V2-M6 integration: sign sightings, the current speed limit and once-only alerts (state/signs.py), and the sign
detector's thread (perception/sign_worker.py), with fake detections (no GPU)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import SignAlertsConfig
from core.types import Frame
from perception.sign_worker import SignWorker
from state.signs import SCHOOL_ZONE_TEXT, SignTracker, alert_texts, limit_text

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Det:
    kind: str
    value: int | None
    conf: float
    box: Box


def approach(kind: str, values: list[int | None], start: float = 10.0, fps: float = 4.0) -> list[tuple[float, Det]]:
    """A sign growing and sliding right as we approach it, one detection per sign-thread frame."""
    out = []
    for k, value in enumerate(values):
        size = 14.0 + 6.0 * k
        x = 900.0 + 25.0 * k
        out.append((start + k / fps, Det(kind, value, 0.8, (x, 300.0 - 3 * k, x + size, 300.0 - 3 * k + size))))
    return out


def run(tracker: SignTracker, frames: list[tuple[float, Det]]) -> list[str]:
    said = []
    for t, det in frames:
        said += [e.text for e in tracker.update(t, [det])]
    return said


def test_a_limit_is_announced_once_after_two_agreeing_frames() -> None:
    tracker = SignTracker(SignAlertsConfig())
    said = run(tracker, approach("speed_limit", [None, 35, None, 35, 35, 35]))
    assert said == ["Speed limit is now 35."]
    snap = tracker.snapshot()
    assert snap.limit == 35 and snap.limit_t == 10.75  # (said when the second reading agreed)
    (view,) = tracker.last_frame  # this frame's sign, for the memory close-up
    assert view.kind == "speed_limit" and view.value == 35 and view.id == snap.recent[0].id


def test_one_frame_or_disagreeing_frames_are_never_announced() -> None:
    tracker = SignTracker(SignAlertsConfig())
    assert run(tracker, approach("speed_limit", [None, 35, None, None])) == []  # one reading: could be a look-alike
    tracker = SignTracker(SignAlertsConfig())
    assert run(tracker, approach("speed_limit", [35, 30, 35, 30])) == []  # the readings disagree: never say either
    assert tracker.snapshot().limit is None


def test_the_same_limit_again_is_quiet_and_a_new_one_is_said() -> None:
    tracker = SignTracker(SignAlertsConfig())
    said = run(tracker, approach("speed_limit", [35, 35]))
    said += run(tracker, approach("speed_limit", [35, 35], start=40.0))  # the next 35 sign: no repeat
    said += run(tracker, approach("speed_limit", [25, 25], start=70.0))
    assert said == ["Speed limit is now 35.", "Speed limit is now 25."]


def test_two_signs_in_one_frame_are_two_sightings() -> None:
    tracker = SignTracker(SignAlertsConfig())
    events = []
    for k in range(3):
        t = 10.0 + k * 0.25
        events += tracker.update(
            t,
            [
                Det("speed_limit", 45, 0.9, (100.0 + k, 300.0, 130.0 + k, 340.0)),
                Det("speed_limit", 45, 0.85, (1100.0 + k, 300.0, 1130.0 + k, 340.0)),
            ],
        )
    assert [e.text for e in events] == ["Speed limit is now 45."]  # two signs, same limit: said once
    assert len(tracker.snapshot().recent) == 2


def test_school_zone_and_school_limits() -> None:
    tracker = SignTracker(SignAlertsConfig())
    said = run(tracker, approach("school_zone", [None, None, None]))
    said += run(tracker, approach("school_speed_limit", [20, 20], start=12.0))
    assert said == [SCHOOL_ZONE_TEXT, "School speed limit 20."]
    assert tracker.snapshot().limit is None and tracker.snapshot().school_limit == 20
    assert run(SignTracker(SignAlertsConfig()), approach("school_zone", [None])) == []  # one frame: not yet


def test_a_school_zone_is_said_once_per_zone() -> None:
    """Regression (school-zone dev clip, 2026-10-10: said 4 times): a sign on each side of the road in the same
    frame, the right one's plaque detected as a second box, the right sign lost for a frame and found again as a new
    sighting, then a look-alike (a worker's hi-vis vest) for two frames 8 s later: one alert."""
    tracker = SignTracker(SignAlertsConfig())
    said = []
    for k in range(18):  # 0-4.5 s at 4 fps, as the detector saw them on the clip
        t = k * 0.25
        dets = [Det("school_zone", None, 0.75, (250.0 - 12 * k, 300.0, 280.0 - 12 * k, 335.0))]  # left sign
        if k != 7:  # the right sign is missed in one frame...
            x = 640.0 + 20 * k + (200.0 if k > 7 else 0.0)  # ...and jumps farther than "near" allows: a new sighting
            dets.append(Det("school_zone", None, 0.85, (x, 290.0 - 8 * k, x + 50 + 4 * k, 355.0 - 8 * k)))
            if k in (6, 9):  # its AHEAD plaque as a second box
                dets.append(Det("school_zone", None, 0.66, (x, 356.0 - 8 * k, x + 50 + 4 * k, 380.0 - 8 * k)))
        said += [e.text for e in tracker.update(t, dets)]
    for t in (12.2, 12.45):  # the vest
        said += [e.text for e in tracker.update(t, [Det("school_zone", None, 0.72, (586.0, 414.0, 618.0, 453.0))])]
    assert said == [SCHOOL_ZONE_TEXT]
    assert len({v.id for v in tracker.snapshot().recent}) >= 3  # (several sightings, one zone)


def test_a_new_school_zone_after_repeat_s_without_one_is_said_again() -> None:
    cfg = SignAlertsConfig(repeat_s=60.0)
    tracker = SignTracker(cfg)
    said = run(tracker, approach("school_zone", [None, None, None]))
    said += run(tracker, approach("school_zone", [None, None], start=50.0))  # the same zone: 40 s since the last one
    said += run(tracker, approach("school_zone", [None, None], start=105.0))  # 55 s since: still the same zone
    assert said == [SCHOOL_ZONE_TEXT]
    said += run(tracker, approach("school_zone", [None, None], start=170.0))  # 65 s without one: a new zone
    assert said == [SCHOOL_ZONE_TEXT, SCHOOL_ZONE_TEXT]
    # One-frame look-alikes keep an announced zone alive but never start one.
    tracker = SignTracker(cfg)
    assert run(tracker, approach("school_zone", [None], start=0.0)) == []
    assert run(tracker, approach("school_zone", [None, None], start=30.0)) == [SCHOOL_ZONE_TEXT]
    for t in (80.0, 130.0, 180.0):  # one frame each, 50 s apart
        assert run(tracker, approach("school_zone", [None], start=t)) == []
    assert run(tracker, approach("school_zone", [None, None], start=235.0)) == []  # 55 s after the last look-alike
    tracker.update(10.0, [])  # a seek back forgets the zone
    assert run(tracker, approach("school_zone", [None, None], start=12.0)) == [SCHOOL_ZONE_TEXT]


def test_a_speed_limit_seen_twice_in_one_frame_or_lost_and_found_is_said_once() -> None:
    """The SPEED LIMIT 30 sign on ego_holdout_long_stop (read at 8.74-10.33 s on the clip), here also seen by a
    second detection in one frame and lost for 2 s mid-approach: one "Speed limit is now 30."."""
    tracker = SignTracker(SignAlertsConfig())
    said = run(tracker, approach("speed_limit", [None, None, None, 30, 30, 30], start=7.95))
    said += [e.text for e in tracker.update(9.5, [Det("speed_limit", 30, 0.8, (860.0, 280.0, 897.0, 314.0))] * 2)]
    said += run(tracker, approach("speed_limit", [30, 30, 30], start=11.6))
    assert said == ["Speed limit is now 30."]


def test_weak_detections_are_ignored_and_a_seek_forgets() -> None:
    tracker = SignTracker(SignAlertsConfig())
    for k in range(4):
        tracker.update(10.0 + k * 0.25, [Det("speed_limit", 35, 0.3, (900.0, 300.0, 920.0, 320.0))])
    assert tracker.snapshot().limit is None
    run(tracker, approach("speed_limit", [55, 55], start=20.0))
    assert tracker.snapshot().limit == 55
    tracker.update(5.0, [])  # time went backwards
    assert tracker.snapshot().limit is None


def test_context_lines_for_the_vision_model() -> None:
    tracker = SignTracker(SignAlertsConfig())
    run(tracker, approach("speed_limit", [30, 30]))
    run(tracker, approach("no_turn_on_red", [None, None], start=14.0))
    lines = tracker.context(now=20.0)
    assert lines[0] == "The sign detector read a speed limit of 30 mph 10 s ago."
    assert any("no turn on red" in line for line in lines)


def test_every_alert_text_is_prerenderable() -> None:
    texts = alert_texts(list(range(5, 90, 5)))
    assert "Speed limit is now 85." in texts and "School speed limit 15." in texts and SCHOOL_ZONE_TEXT in texts
    assert len(texts) == len(set(texts)) and limit_text("speed_limit", 35) in texts


# ------------------------------------------------------------------------------------------- worker


def frame(t: float) -> Frame:
    return Frame(image=np.zeros((72, 128, 3), np.uint8), t=t, index=round(t * 30), captured_at=time.perf_counter())


class SlowDetector:
    def __init__(self) -> None:
        self.thread: str | None = None
        self.seen: list[float] = []

    def detect(self, f: Frame) -> list[Det]:
        self.thread = threading.current_thread().name
        self.seen.append(f.t)
        time.sleep(0.02)
        return [Det("stop", None, 0.9, (1.0, 1.0, 5.0, 5.0))]


def test_the_worker_builds_the_detector_on_its_thread_and_keeps_only_the_newest_frame() -> None:
    detector = SlowDetector()
    results: list[tuple[float, int]] = []
    worker = SignWorker(lambda: detector, lambda f, dets: results.append((f.t, len(dets))), fps=10.0).start()
    assert worker.ready.wait(2.0) and worker.error is None
    for k in range(60):  # 30 fps for 2 s
        worker.offer(frame(k / 30))
        time.sleep(1 / 30)
    time.sleep(0.3)
    worker.stop()
    assert detector.thread == "signs"
    assert 12 <= len(results) <= 24  # ~10 per second, not 30
    assert results[-1][0] >= 1.8  # it kept up with the newest frames
    assert all(n == 1 for _, n in results)


def test_a_detector_that_cannot_start_reports_why() -> None:
    def broken() -> SlowDetector:
        raise RuntimeError("CUDA is not available")

    worker = SignWorker(broken, lambda f, d: None, fps=4.0).start()
    assert worker.ready.wait(2.0)
    time.sleep(0.05)
    assert worker.error == "RuntimeError: CUDA is not available" and not worker.running


def test_sign_alerts_are_muted_in_quiet_mode(tmp_path: Path) -> None:
    """Sign alerts are INFO, like "you're up": quiet mode keeps only requested and safety alerts."""
    from state.signs import SignEvent
    from tests.test_lanes import _app

    app, speaker = _app(tmp_path)
    event = SignEvent("speed_limit", 3.0, limit_text("speed_limit", 35), 35, 1, (0.0, 0.0, 10.0, 10.0))
    app.signs.update = lambda t, dets: [event]  # type: ignore[method-assign]
    frame = Frame(np.zeros((8, 8, 3), np.uint8), t=3.0, index=90, captured_at=time.perf_counter())
    app.assistant.alerts.set_quiet(True)
    app._on_signs(frame, [])
    assert speaker.said == []
    app.assistant.alerts.set_quiet(False)
    app._on_signs(frame, [])
    assert [u.text for u in speaker.said] == ["Speed limit is now 35."]
