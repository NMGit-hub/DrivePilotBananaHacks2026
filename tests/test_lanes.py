"""Lanes (V2-M5): the lane tracker on rendered lane masks, the drift warning on simulated drives, the lane model's
pre/post-processing and its worker thread.

Masks are drawn the way YOLOPv2 outputs them (360x640, stretched over a 1280x720 frame): lane lines as thin runs
and the drivable area as a band, projected through a pinhole dashcam (60° field of view, 1.3 m high, horizon at
half height) from a flat road with straight or gently curved parallel lines. The tracker gets nothing but the
masks and our GPS speed, like in the app.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core.config import LanesConfig, load_config
from core.types import Frame
from perception.lanes import MASK_SIZE, PAD_TOP, PAD_VALUE, LaneMasks, LaneWorker, decode, preprocess
from state.lanes import DRIFT_TEXTS, EMPTY, LanesSnapshot, LaneTracker, drift_text, vanishing_row

W, H = 1280, 720
HFOV = 60.0
F = (W / 2) / math.tan(math.radians(HFOV) / 2)
CAM_H = 1.3
LANE = 3.6
FPS = 12.0
FAST = 27.0  # m/s (60 mph)


def cfg_lanes(**changes: object) -> LanesConfig:
    """The repo's lanes settings, with changes."""
    return load_config().lanes.model_copy(update=changes)


@dataclass
class Road:
    """A flat road seen from the camera: line offsets beside the car (m, + = right of the camera), a shared heading
    and curvature, the drivable area's edges, and which lines are dashed."""

    lines: Sequence[float]
    drivable: tuple[float, float]
    heading: float = 0.0
    curvature: float = 0.0
    dashed: Sequence[bool] = field(default_factory=tuple)
    horizon: float = 0.5  # true horizon row / frame height
    camera_h: float = CAM_H  # true camera height
    phase: float = 0.0  # dash pattern offset (m): moves as we drive


def render(road: Road) -> tuple[np.ndarray, np.ndarray]:
    """(lane lines, drivable area) masks (360x640 uint8) of the road."""
    mw, mh = MASK_SIZE
    sx, sy = W / mw, H / mh
    lines = np.zeros((mh, mw), np.uint8)
    drivable = np.zeros((mh, mw), np.uint8)
    y_h = road.horizon * H
    for r in range(mh):
        y = (r + 0.5) * sy
        dy = y - y_h
        if dy <= 1.0:
            continue
        z = F * road.camera_h / dy
        if z > 80.0:
            continue
        shift = road.heading * z + road.curvature * z * z

        def col(x_m: float, dy: float = dy) -> float:
            return (W / 2 + x_m * dy / road.camera_h) / sx

        a, b = col(road.drivable[0] + shift), col(road.drivable[1] + shift)
        lo, hi = max(0, math.floor(a)), min(mw, math.ceil(b))
        if hi > lo:
            drivable[r, lo:hi] = 1
        half = max(0.5, 0.075 * dy / road.camera_h / sx)  # a 15 cm wide line, at least one mask pixel
        for i, x0 in enumerate(road.lines):
            if i < len(road.dashed) and road.dashed[i] and ((z + road.phase) % 12.0) > 3.0:
                continue  # 3 m dashes every 12 m
            c = col(x0 + shift)
            lo, hi = max(0, round(c - half)), min(mw, round(c + half) + 1)
            if hi > lo:
                lines[r, lo:hi] = 1
    return lines, drivable


@dataclass(frozen=True)
class Masks:
    """What LaneTracker reads (perception.lanes.LaneMasks has these too)."""

    lines: np.ndarray
    drivable: np.ndarray
    frame_size: tuple[int, int] = (W, H)


def three_lanes(offset: float = 0.0, **kw: object) -> Road:
    """A 3-lane road, us in the middle lane `offset` m right of its center; drivable area = the 3 lanes + 0.5 m."""
    edges = [-1.5 * LANE - offset, -0.5 * LANE - offset, 0.5 * LANE - offset, 1.5 * LANE - offset]
    return Road(
        lines=edges,
        drivable=(edges[0] - 0.5, edges[-1] + 0.5),
        dashed=(False, True, True, False),
        **kw,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------------------------------ geometry


def test_measure_finds_our_lane_its_width_and_our_offset() -> None:
    tracker = LaneTracker(cfg_lanes(), HFOV)
    m = None
    for _ in range(3):
        m = tracker.measure(Masks(*render(three_lanes(offset=0.4))))
    assert m is not None and m.left is not None and m.right is not None
    assert m.width_m == pytest.approx(LANE, abs=0.15)
    assert m.offset_m == pytest.approx(0.4, abs=0.12)
    assert [round(line.offset_m, 1) for line in m.lines] == pytest.approx([-5.8, -2.2, 1.4, 5.0], abs=0.25)
    assert (m.lanes_left, m.lanes_right) == (1, 1)


def test_lane_count_and_index_need_the_road_edges() -> None:
    """Middle of three: lane 2 of 3. When the drivable area reaches the frame's edge (the road may go on out of
    view), that side's count is unknown, not a guess."""
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    road = three_lanes()
    t = 0.0
    for i in range(int(2 * FPS)):
        t = i / FPS
        road.phase = FAST * t
        tracker.update(t, Masks(*render(road)), FAST)
    snap = tracker.snapshot()
    assert (snap.lane_index, snap.lane_count) == (2, 3)
    assert snap.confidence >= cfg.vote_share

    wide = LaneTracker(cfg, HFOV)
    road = three_lanes()
    road.drivable = (road.drivable[0], 60.0)  # paved out of view on the right
    for i in range(int(2 * FPS)):
        road.phase = FAST * i / FPS
        wide.update(i / FPS, Masks(*render(road)), FAST)
    snap = wide.snapshot()
    assert snap.lanes_left == 1 and snap.lane_index == 2
    assert snap.lanes_right is None and snap.lane_count is None


def test_leftmost_lane_of_two_on_a_freeway_curve() -> None:
    road = Road(lines=[-1.8, 1.8, 5.4], drivable=(-2.3, 5.9), dashed=(False, True, False), curvature=0.0008)
    tracker = LaneTracker(cfg_lanes(), HFOV)  # (X = b·Z²: b = 0.0008 is a 625 m radius)
    for i in range(int(2 * FPS)):
        road.phase = FAST * i / FPS
        tracker.update(i / FPS, Masks(*render(road)), FAST)
    snap = tracker.snapshot()
    assert (snap.lane_index, snap.lane_count) == (1, 2)
    assert snap.offset_m == pytest.approx(0.0, abs=0.15)
    assert tracker.last is not None and tracker.last.curvature == pytest.approx(0.0008, abs=0.0003)


def test_a_sharp_curve_at_the_start_does_not_lock_in_a_wrong_horizon() -> None:
    """A ramp (250 m radius) for 3 s, where the lines' crossing points pile up at a wrong row, then a straight
    road: the horizon comes back to the truth and the lane is tracked."""
    tracker = LaneTracker(cfg_lanes(), HFOV)
    for i in range(int(13 * FPS)):
        t = i / FPS
        road = three_lanes(horizon=0.48)
        road.curvature = 0.002 if t < 3.0 else 0.0
        road.phase = FAST * t
        tracker.update(t, Masks(*render(road)), FAST)
    assert tracker.horizon == pytest.approx(0.48, abs=0.01)
    assert tracker.snapshot().lane_index == 2


def test_horizon_and_scale_are_learned() -> None:
    """True horizon 0.46 (prior 0.5) and camera 1.5 m high (prior 1.3): the vanishing row brings the horizon in,
    and our lane's width sets the scale, so offsets come out in true metres."""
    road = three_lanes(offset=-0.5, horizon=0.46, camera_h=1.5)
    tracker = LaneTracker(cfg_lanes(), HFOV)
    for i in range(int(4 * FPS)):
        road.phase = FAST * i / FPS
        tracker.update(i / FPS, Masks(*render(road)), FAST)
    assert tracker.horizon == pytest.approx(0.46, abs=0.01)
    snap = tracker.snapshot()
    assert snap.lane_width_m == pytest.approx(LANE, abs=0.1)
    assert snap.offset_m == pytest.approx(-0.5, abs=0.12)


def test_vanishing_row_needs_lines_leaning_both_ways() -> None:
    cfg = cfg_lanes()
    lines, _ = render(three_lanes(horizon=0.45))
    row = vanishing_row(lines, (W, H), cfg.horizon_min, cfg.horizon_max, cfg.hood_cutoff)
    assert row == pytest.approx(0.45, abs=0.01)
    one_side, _ = render(Road(lines=[1.8, 5.4], drivable=(-2.0, 6.0), horizon=0.45))
    assert vanishing_row(one_side, (W, H), cfg.horizon_min, cfg.horizon_max, cfg.hood_cutoff) is None
    assert vanishing_row(np.zeros((360, 640), np.uint8), (W, H), 0.3, 0.75, 0.92) is None


def test_no_lines_no_lane() -> None:
    tracker = LaneTracker(cfg_lanes(), HFOV)
    tracker.update(0.0, Masks(np.zeros((360, 640), np.uint8), np.ones((360, 640), np.uint8)), FAST)
    snap = tracker.snapshot()
    assert not snap.valid and snap.lane_polygon == () and snap.lane_index is None
    assert tracker.context() == []


def test_snapshot_polygon_and_context() -> None:
    tracker = LaneTracker(cfg_lanes(), HFOV)
    road = three_lanes()
    for i in range(int(2 * FPS)):
        road.phase = FAST * i / FPS
        tracker.update(i / FPS, Masks(*render(road)), FAST)
    snap = tracker.snapshot()
    assert snap.valid and len(snap.lane_polygon) >= 4
    xs = [p[0] for p in snap.lane_polygon]
    ys = [p[1] for p in snap.lane_polygon]
    assert min(xs) >= 0.0 and max(xs) <= 1.0 and min(ys) > 0.5 and max(ys) <= 1.0  # below the horizon
    roles = [line.role for line in snap.lines]
    assert roles.count("left") == 1 and roles.count("right") == 1
    assert tracker.context() == [
        "Lane detection: we are in lane 2 counted from the left edge of our direction (1 lane(s) to our left). "
        "It can't count the lanes to our right: count those in the image."
    ]  # (not the count: lanes to our right are often undercounted)


# --------------------------------------------------------------------------------------------- drift


@dataclass
class Drive:
    """A simulated drive down the middle lane of a 3-lane road: our offset in the lane over time."""

    offset_at: Callable[[float], float]  # true offset of the camera from the lane center (m, + = right)
    seconds: float
    speed: float | None = FAST
    noise_m: float = 0.0
    seed: int = 0


def run(drive: Drive, cfg: LanesConfig | None = None, tracker: LaneTracker | None = None) -> list[tuple[float, str]]:
    """Feed the drive's masks to a tracker at FPS; returns the drift alerts (t, text)."""
    rng = np.random.default_rng(drive.seed)
    tracker = tracker if tracker is not None else LaneTracker(cfg or cfg_lanes(), HFOV)
    alerts: list[tuple[float, str]] = []
    for i in range(int(drive.seconds * FPS)):
        t = i / FPS
        lateral = drive.offset_at(t) + (rng.normal(0.0, drive.noise_m) if drive.noise_m else 0.0)
        road = three_lanes(offset=lateral)  # (the road stays put: past a line we're in the next lane)
        road.phase = FAST * t
        for event in tracker.update(t, Masks(*render(road)), drive.speed):
            alerts.append((t, event.text))
    return alerts


def slide(start: float, rate: float, base: float = 0.0, stop: float | None = None) -> Callable[[float], float]:
    """Offset `base` until `start`, then moving at `rate` m/s (until `stop`, then held)."""

    def offset(t: float) -> float:
        end = t if stop is None else min(t, stop)
        return base + rate * max(0.0, end - start)

    return offset


def threshold_time(start: float, rate: float, base: float, habit: float, cfg: LanesConfig) -> float:
    """When the car's side, counted from our usual place, is drift_gap_m from the line."""
    reach = LANE / 2 - cfg.car_width_m / 2 - cfg.drift_gap_m
    return start + (reach - abs(base - habit)) / abs(rate)


def test_straight_driving_is_quiet() -> None:
    assert run(Drive(lambda t: 0.1, 20.0, noise_m=0.05)) == []


def test_slow_drift_alerts_once_on_time() -> None:
    cfg = cfg_lanes()
    alerts = run(Drive(slide(12.0, 0.15), 22.0), cfg)
    assert [text for _, text in alerts] == [drift_text("right")]
    expected = threshold_time(12.0, 0.15, 0.0, 0.0, cfg)
    assert alerts[0][0] == pytest.approx(expected, abs=0.6)


def test_drift_left_and_the_drift_line() -> None:
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    alerts = run(Drive(slide(12.0, -0.2), 18.0), cfg, tracker)
    assert [text for _, text in alerts] == [drift_text("left")]
    snap = tracker.snapshot()
    assert snap.drift_side == "left" and snap.drift_line is not None
    left_line = next(line for line in snap.lines if line.role == "left")
    assert snap.drift_line == left_line.points


def test_drift_is_measured_from_our_usual_place() -> None:
    """The camera sits 0.6 m left of the car's center (or the driver keeps left): steady driving there is quiet,
    and a drift is measured from there (the right line is far, so a drift right alerts later)."""
    cfg = cfg_lanes()
    assert run(Drive(lambda t: -0.6, 25.0, noise_m=0.05), cfg) == []
    alerts = run(Drive(slide(12.0, 0.15, base=-0.6), 24.0), cfg)
    assert [text for _, text in alerts] == [drift_text("right")]
    assert alerts[0][0] == pytest.approx(threshold_time(12.0, 0.15, -0.6, -0.6, cfg), abs=0.6)


def test_no_alert_before_our_usual_place_is_known() -> None:
    cfg = cfg_lanes()
    assert run(Drive(slide(1.0, 0.2, stop=6.0), 8.0), cfg) == []  # habit needs habit_min_s of driving at speed


def test_lane_change_is_quiet_and_logged() -> None:
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    alerts = run(Drive(slide(12.0, 0.9, stop=12.0 + LANE / 0.9), 22.0), cfg, tracker)  # one lane right at 0.9 m/s
    assert alerts == []
    assert [k for _, k in tracker.lane_changes] == [1]
    assert tracker.snapshot().lane_index == 3  # the rightmost lane now


def test_start_of_a_fast_move_near_the_line_is_not_a_drift() -> None:
    """Already close to the line, then a quick move across it: the speed fit lags at first, but the move wasn't
    steady before, so it's a lane change, not a drift."""
    cfg = cfg_lanes()
    alerts = run(Drive(slide(12.0, 0.8, base=0.35, stop=17.0), 20.0), cfg)
    assert alerts == []


def test_drift_needs_speed_and_gps() -> None:
    cfg = cfg_lanes()
    assert run(Drive(slide(12.0, 0.15), 22.0, speed=cfg.min_speed_mps - 1.0), cfg) == []
    assert run(Drive(slide(12.0, 0.15), 22.0, speed=None), cfg) == []


def test_rearmed_after_coming_back() -> None:
    """Drift right, back to our place, drift right again after the cooldown: two alerts. Staying near the line:
    one."""
    cfg = cfg_lanes()

    def there_and_back(t: float) -> float:
        if t < 12.0:
            return 0.0
        if t < 18.0:
            return min(0.15 * (t - 12.0), 0.75)
        if t < 22.0:
            return max(0.75 - 0.4 * (t - 18.0), 0.0)
        return min(0.15 * (t - 26.0), 0.75) if t >= 26.0 else 0.0

    alerts = run(Drive(there_and_back, 34.0), cfg)
    assert [text for _, text in alerts] == [drift_text("right"), drift_text("right")]
    assert alerts[1][0] - alerts[0][0] >= cfg.cooldown_s
    assert len(run(Drive(slide(12.0, 0.15, stop=17.0), 30.0), cfg)) == 1


def test_reset_from_another_thread() -> None:
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    run(Drive(lambda t: 0.0, 10.0), cfg, tracker)
    assert tracker.snapshot().valid
    habit = tracker.snapshot().habit_m
    threading.Thread(target=tracker.reset).start()
    time.sleep(0.05)
    assert tracker.snapshot() is EMPTY  # at once: the page shows no stale lane
    tracker.update(0.0, Masks(*render(three_lanes())), FAST)
    snap = tracker.snapshot()
    assert snap.valid and snap.habit_m == habit  # our usual place belongs to the mount: kept
    assert snap.lane_index is None  # the votes start over


def test_drift_texts() -> None:
    assert DRIFT_TEXTS == ("Careful, you're drifting left.", "Careful, you're drifting right.")


# ------------------------------------------------------------------------------------- lane model I/O


def test_preprocess_is_the_official_letterbox() -> None:
    image = np.full((720, 1280, 3), 50, np.uint8)
    out = preprocess(image)
    assert out.shape == (384, 640, 3)
    assert (out[:PAD_TOP] == PAD_VALUE).all() and (out[-PAD_TOP:] == PAD_VALUE).all()
    assert (out[PAD_TOP:-PAD_TOP] == 50).all()
    assert preprocess(np.zeros((1080, 1920, 3), np.uint8)).shape == (384, 640, 3)  # any size is stretched


def test_decode_crops_the_padding() -> None:
    seg = torch.zeros((1, 2, 384, 640))
    seg[0, 1, 100:200] = 1.0  # drivable rows 100..199 of the padded input
    ll = torch.zeros((1, 1, 384, 640))
    ll[0, 0, 50, 10] = 0.6
    ll[0, 0, 60, 10] = 0.5  # exactly 0.5 is not a line (official: round)
    lines, drivable = decode(seg, ll)
    assert lines.shape == (360, 640) and drivable.shape == (360, 640)
    assert lines.dtype == torch.uint8
    assert lines[50 - PAD_TOP, 10] == 1 and lines[60 - PAD_TOP, 10] == 0 and int(lines.sum()) == 1
    assert drivable[100 - PAD_TOP : 200 - PAD_TOP].all() and int(drivable.sum()) == 100 * 640


def _frame(i: int) -> Frame:
    return Frame(image=np.zeros((8, 8, 3), np.uint8), t=i / 30.0, index=i, captured_at=time.perf_counter())


def test_worker_runs_on_the_newest_frame_at_its_rate() -> None:
    seen: list[int] = []
    built_on: list[str] = []

    def factory() -> Callable[[Frame], LaneMasks]:
        built_on.append(threading.current_thread().name)

        def model(frame: Frame) -> LaneMasks:
            z = np.zeros((360, 640), np.uint8)
            return LaneMasks(frame.t, frame.index, (8, 8), z, z, {})

        return model

    worker = LaneWorker(factory, lambda f, m: seen.append(m.frame_index), fps=10.0).start()
    assert worker.ready.wait(2.0) and worker.error is None
    for i in range(30):  # 1 s of frames at 30 fps
        worker.offer(_frame(i))
        time.sleep(1 / 30)
    time.sleep(0.3)
    worker.stop()
    assert built_on == ["lanes"]
    assert 6 <= len(seen) <= 13  # ~10 a second
    assert seen == sorted(seen) and seen[-1] >= 26  # newest frames, in order


def test_worker_survives_a_bad_frame_and_reports_a_failed_start() -> None:
    seen: list[int] = []

    def model(frame: Frame) -> LaneMasks:
        if frame.index == 0:
            raise RuntimeError("bad frame")
        z = np.zeros((360, 640), np.uint8)
        return LaneMasks(frame.t, frame.index, (8, 8), z, z, {})

    worker = LaneWorker(lambda: model, lambda f, m: seen.append(m.frame_index), fps=100.0).start()
    worker.ready.wait(2.0)
    worker.offer(_frame(0))
    time.sleep(0.1)
    worker.offer(_frame(1))
    time.sleep(0.1)
    worker.stop()
    assert seen == [1]

    def broken() -> Callable[[Frame], LaneMasks]:
        raise FileNotFoundError("no weights")

    failed = LaneWorker(broken, lambda f, m: None, fps=10.0).start()
    assert failed.ready.wait(2.0)
    time.sleep(0.05)
    assert failed.error == "FileNotFoundError: no weights" and not failed.running


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        LanesConfig(horizon_prior=0.8)  # above horizon_max
    with pytest.raises(ValueError):
        LanesConfig(lane_change_mps=0.05)  # below drift_min_mps
    with pytest.raises(ValueError):
        LanesConfig(near_m=40.0)  # beyond far_m


# --------------------------------------------------------------------------------------------- GPU


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
def test_lane_model_on_the_gpu() -> None:
    from perception.lanes import LaneModel

    cfg = load_config()
    weights = cfg.resolve_path(cfg.lanes.weights)
    if not weights.is_file():
        pytest.skip(f"no weights at {weights}")
    clip = Path("data/videos/highway/7dc2914d-a20a969a.mov")
    if not clip.exists():
        pytest.skip("highway clip missing")
    import cv2

    model = LaneModel(cfg.lanes, cfg.root)
    params = list(model.model.parameters())
    assert all(p.device.type == "cuda" and p.dtype == torch.float16 for p in params)
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, 10_000)
    ok, image = cap.read()
    cap.release()
    assert ok
    lines, drivable, timings = model.infer(image)
    assert lines.shape == (360, 640) and drivable.shape == (360, 640)
    assert 0 < lines.mean() < 0.1 and 0.05 < drivable.mean() < 0.6  # some lines, a road
    tracker = LaneTracker(cfg.lanes, 60.0)
    m = tracker.measure(Masks(lines, drivable))
    assert m.left is not None and m.right is not None  # our lane on a California freeway
    assert timings["gpu"] < 200.0


# ------------------------------------------------------------------------------- app, page, guidance


def _lanes_snapshot(**changes: object) -> LanesSnapshot:
    """A snapshot as the tracker makes it on the middle lane of three."""
    tracker = LaneTracker(cfg_lanes(), HFOV)
    road = three_lanes()
    for i in range(int(2 * FPS)):
        road.phase = FAST * i / FPS
        tracker.update(i / FPS, Masks(*render(road)), FAST)
    return replace(tracker.snapshot(), **changes)  # type: ignore[arg-type]


def test_lanes_packet_for_the_page() -> None:
    from ui.overlay import LANE_POINTS, frame_packet, lanes_packet, resample

    assert resample([(0.0, 1.0), (0.0, 0.5)], 3) == [[0.0, 1.0], [0.0, 0.75], [0.0, 0.5]]
    assert lanes_packet(None) is None and lanes_packet(EMPTY) is None
    snap = _lanes_snapshot()
    packet = lanes_packet(snap)
    assert packet is not None and packet["drift"] is None
    assert len(packet["left"]) == len(packet["right"]) == LANE_POINTS
    assert len(packet["others"]) == 2 and all(len(line) == LANE_POINTS for line in packet["others"])
    assert packet["left"][0][0] < packet["right"][0][0]  # left of right, near the car
    drifting = lanes_packet(replace(snap, drift_side="right"))
    assert drifting is not None and drifting["drift"] == "right"
    assert frame_packet(None, [], W, H, lanes=snap)["lanes"] == packet  # (drawn even before the world state)


def test_lane_guidance_uses_lane_detection_when_it_is_sure() -> None:
    from answers.lanes import CameraLaneReader

    asked: list[int | None] = []

    def model(expected: int | None) -> tuple[int, int] | None:
        asked.append(expected)
        return 4, 3

    snap = _lanes_snapshot()
    assert CameraLaneReader(lambda: snap, model)(3) == (3, 2) and asked == []
    assert CameraLaneReader(lambda: replace(snap, lane_count=None), model)(4) == (4, 3)  # not sure: the model
    assert CameraLaneReader(lambda: None, model)(4) == (4, 3)  # stale or no lanes
    # Review finding: the route says 4 lanes, lane detection counted 3: the model is asked (it was "I can't tell").
    assert CameraLaneReader(lambda: snap, model)(4) == (4, 3) and asked[-1] == 4
    assert CameraLaneReader(lambda: None, None)(3) is None


def _app(tmp_path: Path, **kwargs: Any) -> Any:
    import cv2

    from answers.speech import TextSpeaker
    from app import DrivePilotApp
    from capture.file_source import FileSource
    from tests.test_app import FakeDetector

    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W // 4, H // 4))
    for _ in range(10):
        writer.write(np.zeros((H // 4, W // 4, 3), np.uint8))
    writer.release()
    speaker = TextSpeaker()
    app = DrivePilotApp(load_config(), lambda: FileSource(clip), FakeDetector(), speaker, **kwargs)  # type: ignore[arg-type]
    return app, speaker


def test_app_speaks_a_drift_as_a_safety_alert_and_shares_the_drivable_area(tmp_path: Path) -> None:
    from answers.speech import Priority
    from app import FIXED_ALERT_TEXTS
    from state.lanes import DriftEvent

    assert set(DRIFT_TEXTS) <= set(FIXED_ALERT_TEXTS)  # pre-rendered: played at once
    app, speaker = _app(tmp_path)
    lines, drivable = render(three_lanes())
    masks = LaneMasks(5.0, 150, (W, H), lines, drivable, {})
    frame = Frame(np.zeros((H, W, 3), np.uint8), t=5.0, index=150, captured_at=time.perf_counter())
    app._on_lanes(frame, masks)
    assert app._drivable is not None and app._drivable.t == 5.0 and app._drivable.mask is drivable
    assert app.lanes_view(5.2) is not None  # our lane, fresh
    assert app.lanes_view(6.0) is None  # too old for the frame shown at 6 s
    assert speaker.said == []

    event = DriftEvent(5.1, "right", drift_text("right"), 0.05, 0.2, 0.6, 0.0, 30.0)
    app.lanes.update = lambda t, m, speed: [event]  # type: ignore[method-assign]
    app._on_lanes(frame, masks)
    said = speaker.said[-1]
    assert said.text == "Careful, you're drifting right." and said.priority == Priority.SAFETY and said.clip == "drift"


def test_lanes_need_the_model(tmp_path: Path) -> None:
    from app import lane_model_factory

    cfg = load_config()
    status: dict[str, str] = {}
    off = cfg.lanes.model_copy(update={"enabled": False})
    assert lane_model_factory(cfg.model_copy(update={"lanes": off}), status) is None
    assert status["lanes"] == "off"
    missing = cfg.lanes.model_copy(update={"weights": tmp_path / "nothing.pt"})
    assert lane_model_factory(cfg.model_copy(update={"lanes": missing}), status) is None
    assert status["lanes"] == "off (no model)"
    app, _ = _app(tmp_path)  # no lane model given: lanes off, nothing started
    app._start_lanes()
    assert app.lane_worker is None and app.status["lanes"] == "off"


def test_app_runs_the_lanes_thread_on_its_frames(tmp_path: Path) -> None:
    def factory() -> Callable[[Frame], LaneMasks]:
        lines, drivable = render(three_lanes())
        return lambda f: LaneMasks(f.t, f.index, (W, H), lines, drivable, {})

    app, _ = _app(tmp_path, lane_model=factory)
    app.start()
    try:
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and (app.lane_worker is None or app.lane_worker.processed < 2):
            time.sleep(0.05)
        assert app.status["lanes"] == "on" and app.lane_worker is not None and app.lane_worker.processed >= 2
        assert app.lanes.snapshot().valid and app._drivable is not None
    finally:
        app.stop()


def test_the_green_target_lane() -> None:
    """Lane guidance on the page: our lane green if it works for the next maneuver, else the side to move to,
    only when lane detection's count matches the route's lanes (else nothing: no guess on screen)."""
    from types import SimpleNamespace

    from nav.route import Lane
    from ui.overlay import lane_target, lanes_packet

    def progress(valid: list[bool], to_next: float = 800.0, needs: bool = True) -> Any:
        step = SimpleNamespace(lanes=tuple(Lane(("straight",), v) for v in valid), needs_lanes=needs)
        return SimpleNamespace(arrived=False, next_step=step, to_next_m=to_next)

    snap = _lanes_snapshot()  # lane 2 of 3
    assert lane_target(snap, progress([False, True, True])) == {"lane": "ours"}
    assert lane_target(snap, progress([False, False, True])) == {"lane": "right", "moves": 1}
    assert lane_target(snap, progress([True, False, False])) == {"lane": "left", "moves": 1}
    assert lane_target(snap, progress([True, True])) is None  # the counts disagree
    assert lane_target(snap, progress([False, False, True], to_next=3000.0)) is None  # too far yet
    assert lane_target(snap, progress([False, False, True], needs=False)) is None
    assert lane_target(None, progress([False, False, True])) is None
    packet = lanes_packet(snap, {"lane": "right", "moves": 1})
    assert packet is not None and packet["target"] == {"lane": "right", "moves": 1}
    assert packet["left_next"] is not None and packet["right_next"] is not None  # the lanes beside ours
    assert packet["right_next"][0][0] > packet["right"][0][0]


# ----------------------------------------------------------------------------- review findings (V2-M5)


def test_a_wide_first_lane_does_not_lock_out_normal_lanes() -> None:
    """Review: a new width had to be within 0.72-1.28x the median of the widths seen so far, and rejected widths
    never joined them: 1 s of a 5.4 m lane first, and normal 3.6 m lanes were rejected for the whole session."""
    tracker = LaneTracker(cfg_lanes(), HFOV)
    wide = Road(lines=[-2.7, 2.7], drivable=(-3.2, 3.2))
    for i in range(int(1.0 * FPS)):
        tracker.update(i / FPS, Masks(*render(wide)), FAST)
    seen = []
    for i in range(int(1.0 * FPS), int(9.0 * FPS)):
        road = three_lanes()
        road.phase = FAST * i / FPS
        tracker.update(i / FPS, Masks(*render(road)), FAST)
        seen.append(tracker.snapshot().valid)
    assert all(seen[int(3.0 * FPS) :])  # the width history relearned within ~2 s; tracked ever since
    assert tracker.snapshot().lane_width_m == pytest.approx(LANE, abs=0.15)


def test_lane_changes_do_not_amplify_width_noise() -> None:
    """Review: the position was d + k * width_now, so after k lane changes each frame's width wobble moved it by k
    times as much. Two lane changes, then a straight stretch: the lateral speed stays as calm as before them."""
    tracker = LaneTracker(cfg_lanes(), HFOV)
    rng = np.random.default_rng(4)
    speeds = []
    for i in range(int(20 * FPS)):
        t = i / FPS
        lateral = -LANE + min(max(0.0, t - 3.0), 2 * LANE)  # left lane, then two lanes right at 1 m/s from 3 s
        road = three_lanes(offset=lateral)
        road.lines = [x * (1.0 + rng.normal(0.0, 0.03)) for x in road.lines]  # the measured width wobbles ~0.1 m
        road.phase = FAST * t
        tracker.update(t, Masks(*render(road)), FAST)
        if t > 14.0 and tracker.snapshot().lateral_speed_mps is not None:
            speeds.append(tracker.snapshot().lateral_speed_mps)
    assert [k for _, k in tracker.lane_changes] == [1, 1]
    assert speeds and float(np.std(speeds)) < 0.04  # (the old k * width: 0.069 m/s)


def test_the_lane_index_follows_a_lane_change_at_once() -> None:
    """Review: the votes kept the old lane for ~0.35 s after crossing (lane 2 shown while in lane 3)."""
    tracker = LaneTracker(cfg_lanes(), HFOV)
    crossed = None
    for i in range(int(10 * FPS)):
        t = i / FPS
        lateral = min(max(0.0, t - 4.0), LANE)  # middle lane, then one lane right at 1 m/s
        road = three_lanes(offset=lateral)
        road.phase = FAST * t
        tracker.update(t, Masks(*render(road)), FAST)
        snap = tracker.snapshot()
        if crossed is None and lateral > LANE / 2 + 0.3:
            crossed = t
        if crossed is not None and t >= crossed and snap.lane_index is not None:
            assert snap.lane_index == 3, f"t={t:.2f}: lane {snap.lane_index}"


def test_the_drift_shows_on_the_page_for_a_few_seconds() -> None:
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    alerts = run(Drive(slide(12.0, 0.15, stop=17.0), 26.0), cfg, tracker)  # drifts, then stays near the line
    assert len(alerts) == 1
    assert 26.0 - alerts[0][0] > cfg.drift_show_s
    assert tracker.snapshot().drift_side is None and tracker.snapshot().drift_line is None


def test_more_config_validation() -> None:
    with pytest.raises(ValueError):
        LanesConfig(rearm_gap_m=0.1, drift_gap_m=0.2)
    with pytest.raises(ValueError):
        LanesConfig(quick_window_s=2.0, velocity_window_s=1.5)


def test_lane_results_from_before_a_new_segment_are_dropped(tmp_path: Path) -> None:
    """Review: after a seek or loop restart, the lanes thread's frame from the old segment still set the drivable
    area and our lane, and the new segment's first pedestrian check used the old road."""
    from perception.ego_motion import NOT_MEASURED, EgoMotion
    from perception.pipeline import PerceptionResult

    app, _ = _app(tmp_path)
    lines, drivable = render(three_lanes())
    old = Frame(np.zeros((H, W, 3), np.uint8), t=39.9, index=1197, captured_at=time.perf_counter(), segment=0)
    app._on_lanes(old, LaneMasks(39.9, 1197, (W, H), lines, drivable, {}))
    assert app._current_drivable() is not None
    new = Frame(np.zeros((H, W, 3), np.uint8), t=40.0, index=0, captured_at=time.perf_counter(), segment=1)
    seen: list[object] = []
    app.world.on_perception = lambda result: seen.append(app._current_drivable())  # type: ignore[method-assign]
    ego = EgoMotion(40.0, NOT_MEASURED, NOT_MEASURED, 5.0, False, None, True, None, 0.0)
    app._world_update(PerceptionResult(new, [], ego, {}, new_segment=True))
    assert seen == [None]  # the new segment's first frame: no road from the old one
    app._on_lanes(old, LaneMasks(39.9, 1197, (W, H), lines, drivable, {}))  # the old frame finishing late
    assert app._current_drivable() is None and app.lanes_view(40.0) is None


def test_a_lane_narrowing_beside_us_is_not_a_drift() -> None:
    """A highway clip (ae3733cf) where a lane splits off: the right line curves in toward us while we hold our line.
    Our lane narrows, its center moves left, so our offset grows: not a drift (the lane's width isn't steady)."""
    cfg = cfg_lanes()
    tracker = LaneTracker(cfg, HFOV)
    alerts = []
    for i in range(int(22 * FPS)):
        t = i / FPS
        squeeze = min(max(0.0, t - 12.0) * 0.3, 0.9)  # the right line moves 0.9 m in over 3 s
        road = Road(lines=[-1.5 * LANE, -0.5 * LANE, 0.5 * LANE - squeeze, 1.5 * LANE], drivable=(-5.9, 5.9))
        road.phase = FAST * t
        alerts += tracker.update(t, Masks(*render(road)), FAST)
    assert alerts == []
