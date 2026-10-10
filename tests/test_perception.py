"""Perception: stop state machine, ego motion on synthetic camera motion, the worker loop, and the GPU detector."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from capture.file_source import FileSource
from core.config import EgoMotionConfig, PerceptionConfig, load_config
from core.types import Frame
from perception.detector import DetectionResult, Track
from perception.ego_motion import EgoMotionEstimator, StopStateMachine
from perception.pipeline import PerceptionResult, PerceptionWorker, StageStats

FPS = 30.0


# ------------------------------------------------------------------------------------ state machine


def feed(machine: StopStateMachine, values: list[float | None], t0: float = 0.0) -> list[bool]:
    states = []
    for i, v in enumerate(values):
        machine.update(t0 + i / FPS, v)
        states.append(machine.stopped)
    return states


def test_stops_after_dwell_and_reports_when_stillness_began() -> None:
    m = StopStateMachine(stopped_below=1.0, moving_above=2.0, stopped_after_s=0.5, smoothing_s=0.0)
    states = feed(m, [5.0] * 30 + [0.1] * 30)
    first_stopped = states.index(True)
    assert first_stopped == 30 + 15  # 0.5 s of stillness at 30 fps
    assert m.stopped_since == pytest.approx(30 / FPS)  # arrival time = when stillness began, not when confirmed


def test_single_frame_glitch_does_not_flip_state() -> None:
    m = StopStateMachine(stopped_below=1.0, moving_above=2.0, stopped_after_s=0.5, smoothing_s=0.2)
    feed(m, [0.1] * 60)
    assert m.stopped
    since = m.stopped_since
    states = feed(m, [500.0] + [0.1] * 10, t0=2.0)  # e.g. a near-zero time step makes one huge rate
    assert all(states) and m.stopped_since == since


def test_hysteresis_band_and_moving() -> None:
    m = StopStateMachine(stopped_below=1.0, moving_above=2.0, stopped_after_s=0.5, smoothing_s=0.0)
    feed(m, [0.1] * 30)
    assert m.stopped
    assert all(feed(m, [1.5] * 30, t0=1.0))  # inside the band: stays stopped
    assert not any(feed(m, [3.0] * 3, t0=2.0))  # above moving_above: moving
    assert m.stopped_since is None
    assert not any(feed(m, [1.5] * 60, t0=3.0))  # inside the band while moving: stays moving


def test_unmeasurable_frames_hold_state() -> None:
    m = StopStateMachine(stopped_below=1.0, moving_above=2.0, stopped_after_s=0.5, smoothing_s=0.0)
    feed(m, [0.1] * 30)
    assert all(feed(m, [None] * 30, t0=1.0))
    m.reset()
    assert not m.stopped and m.stopped_since is None


# ------------------------------------------------------------------------- ego motion on synthetic video


def texture(width: int = 1280, height: int = 720, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(base, (width, height), interpolation=cv2.INTER_NEAREST), (5, 5), 0)


def warp(image: np.ndarray, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def run_sequence(cfg: EgoMotionConfig, frames: list[np.ndarray], boxes: list[list] | None = None) -> list:
    est = EgoMotionEstimator(cfg)
    out = []
    for i, image in enumerate(frames):
        frame = Frame(image=image, t=i / FPS, index=i, captured_at=0.0)
        out.append(est.update(frame, boxes[i] if boxes else []))
    return out


@pytest.fixture
def ego_cfg() -> EgoMotionConfig:
    return load_config().ego_motion


def test_static_scene_is_stopped(ego_cfg: EgoMotionConfig) -> None:
    base = texture()
    results = run_sequence(ego_cfg, [base] * 30)
    assert results[-1].stopped and results[-1].confident
    assert results[-1].stopped_since == pytest.approx(0.0, abs=0.05)
    assert results[-1].long.zoom is not None and results[-1].long.zoom < 1e-3
    assert results[-1].long.baseline_s == pytest.approx(ego_cfg.baseline_s, abs=1 / FPS)


def test_driving_forward_is_moving(ego_cfg: EgoMotionConfig) -> None:
    base = texture()
    frames = [warp(base, scale=1.004**i) for i in range(30)]  # ~0.4 % expansion per frame ≈ slow city driving
    results = run_sequence(ego_cfg, frames)
    assert not any(r.stopped for r in results)
    zoom = results[-1].long.zoom
    assert zoom is not None and zoom == pytest.approx(math.log(1.004) * FPS, rel=0.2)


def test_pitch_rocking_alone_is_not_motion_for_zoom_metric(ego_cfg: EgoMotionConfig) -> None:
    base = texture()
    frames = [warp(base, dy=3.0 * math.sin(i / 3)) for i in range(45)]  # suspension settling after braking
    results = run_sequence(ego_cfg, frames)
    assert ego_cfg.metric == "combined"
    assert results[-1].stopped
    magnitudes = [r.short.magnitude for r in results[1:] if r.short.magnitude is not None]
    assert max(magnitudes) > 5  # the pixels do move (vertically), yet the zoom metric reports no travel


def test_crossing_car_is_masked_out(ego_cfg: EgoMotionConfig) -> None:
    base = texture(seed=1)
    car = texture(width=300, height=160, seed=2)
    frames, boxes = [], []
    for i in range(45):
        x = 100 + i * 20  # crosses quickly from left to right
        image = base.copy()
        image[300:460, x : x + 300] = car[:, : max(0, min(300, 1280 - x))]
        frames.append(image)
        boxes.append([(float(x), 300.0, float(x + 300), 460.0)])
    results = run_sequence(ego_cfg, frames, boxes)
    assert results[-1].stopped


def test_duplicate_timestamps_do_not_spike(ego_cfg: EgoMotionConfig) -> None:
    base = texture()
    est = EgoMotionEstimator(ego_cfg)
    for i in range(30):
        est.update(Frame(image=warp(base, dx=0.2 * i), t=i / FPS, index=i, captured_at=0.0))
    # same scene shifted a little but stamped 0.1 ms later: the rate uses the min_dt floor
    result = est.update(Frame(image=warp(base, dx=0.2 * 30 + 0.2), t=29 / FPS + 1e-4, index=30, captured_at=0.0))
    assert result.short.magnitude is not None and result.short.magnitude < 0.5 / ego_cfg.min_dt_s


def test_featureless_frames_are_not_confident(ego_cfg: EgoMotionConfig) -> None:
    flat = np.full((720, 1280, 3), 90, np.uint8)
    results = run_sequence(ego_cfg, [flat] * 10)
    assert not results[-1].confident and not results[-1].stopped


def test_homography_maps_full_resolution_points(ego_cfg: EgoMotionConfig) -> None:
    base = texture()
    results = run_sequence(ego_cfg, [base, warp(base, dx=8.0)])
    h = results[-1].homography
    assert h is not None
    point = h @ np.array([640.0, 400.0, 1.0])
    assert point[0] / point[2] == pytest.approx(648.0, abs=1.5)


# ------------------------------------------------------------------------------------ worker loop


class FakeDetector:
    """Stands in for the GPU detector: one fixed track per frame, counts resets."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, frame: Frame) -> DetectionResult:
        return DetectionResult([Track(1, "car", 0.9, (10.0, 10.0, 50.0, 40.0))], {"detect_track_total": 1.0})


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    base = texture(160, 120)
    for i in range(45):
        writer.write(warp(base, dx=float(i)))
    writer.release()
    return path


def test_worker_delivers_results_and_survives_bad_listener(short_clip: Path, ego_cfg: EgoMotionConfig) -> None:
    source = FileSource(short_clip, realtime=False)
    detector = FakeDetector()
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(ego_cfg))  # type: ignore[arg-type]
    results: list[PerceptionResult] = []

    def bad_listener(_: PerceptionResult) -> None:
        raise ValueError("listener bug")

    worker.add_listener(bad_listener)
    worker.add_listener(results.append)
    source.start()
    worker.start()
    worker.join(timeout=20)
    source.stop()
    assert worker.error is None and not worker.running
    assert len(results) == 45 == worker.processed
    assert results[0].new_segment and not any(r.new_segment for r in results[1:])
    assert results[0].tracks[0].cls == "car"
    assert {"detect_track_total", "ego_motion", "latency", "wait_for_frame"} <= set(results[-1].timings_ms)
    summary = worker.run_stats.summary()
    assert summary["detect_track_total"]["n"] == 45


def test_worker_resets_on_seek(short_clip: Path, ego_cfg: EgoMotionConfig) -> None:
    source = FileSource(short_clip, realtime=False)
    detector = FakeDetector()
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(ego_cfg))  # type: ignore[arg-type]
    segments: list[int] = []
    worker.add_listener(lambda r: segments.append(r.frame.segment))
    source.start()
    worker.start()
    deadline = time.monotonic() + 10
    while len(segments) < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    source.seek_to(0.2)
    worker.join(timeout=20)
    source.stop()
    assert 1 in segments and detector.resets >= 1


def test_stage_stats_summary() -> None:
    stats = StageStats()
    for v in range(1, 101):
        stats.add({"a": float(v), "b": float("nan")})
    summary = stats.summary()
    assert summary["a"]["p50"] == pytest.approx(50.5) and summary["a"]["max"] == 100.0
    assert "b" not in summary  # all-NaN stages are omitted


# --------------------------------------------------------------------------------- GPU detector


DEV_CLIP = Path("data/videos/dev/stop_sign_turn_lanes_red_light__c4130097-887da34f.mov")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
@pytest.mark.skipif(not DEV_CLIP.exists(), reason="dev clip missing (scripts/fetch_dev_clips.py)")
def test_detector_runs_on_gpu_and_tracks_cars() -> None:
    from perception.detector import Detector

    cfg = load_config()
    detector = Detector(cfg.perception, cfg.root)
    assert detector.model.predictor is not None
    device, precision = detector._verify_gpu(detector.model.predictor.model)  # raises if not on the GPU
    assert device.startswith("cuda") and precision in ("fp16", "torch.float16")

    source = FileSource(DEV_CLIP, realtime=False, start_s=6.0)  # the pickup crossing after the stop line
    source.start()
    ids_per_frame = []
    for _ in range(30):
        frame = source.read(timeout=10)
        assert frame is not None
        result = detector(frame)
        ids_per_frame.append({t.track_id for t in result.tracks if t.cls in ("car", "truck")})
        for track in result.tracks:
            x1, y1, x2, y2 = track.box
            assert 0 <= x1 < x2 <= frame.width + 1 and 0 <= y1 < y2 <= frame.height + 1
    source.stop()
    # Vehicles come and go in this window; the car crossing first must keep one id for >= 15 frames.
    runs: dict[int, int] = {}
    best = 0
    for ids in ids_per_frame:
        runs = {i: runs.get(i, 0) + 1 for i in ids}
        best = max([best, *runs.values()])
    assert best >= 15
    detector.reset()
    assert detector.tracker.frame_id == 0


def test_vehicle_duplicates_are_merged_across_classes() -> None:
    from perception.detector import vehicle_duplicate_mask

    xyxy = np.array([[0, 0, 100, 50], [2, 1, 101, 52], [300, 0, 400, 50], [1, 0, 99, 50]], dtype=float)
    conf = np.array([0.6, 0.8, 0.9, 0.95])
    is_vehicle = np.array([True, True, True, False])  # the last box is e.g. a person: never merged
    keep = vehicle_duplicate_mask(xyxy, conf, is_vehicle, 0.7)
    assert keep.tolist() == [False, True, True, True]


def test_perception_config_rejects_unknown_class_and_bad_size() -> None:
    with pytest.raises(ValueError):
        PerceptionConfig(classes=("car", "dragon"))
    with pytest.raises(ValueError):
        PerceptionConfig(imgsz=650)


# ------------------------------------------------------------------------- review regressions (M1)


class FlakyDetector(FakeDetector):
    """Raises on chosen calls (e.g. a transient CUDA error)."""

    def __init__(self, fail_on: set[int] | None = None, always: bool = False) -> None:
        super().__init__()
        self.calls, self.fail_on, self.always, self.warmed = 0, fail_on or set(), always, False

    def warmup(self) -> None:
        self.warmed = True

    def __call__(self, frame: Frame) -> DetectionResult:
        self.calls += 1
        if self.always or self.calls in self.fail_on:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return super().__call__(frame)


def test_worker_survives_a_failing_frame(short_clip: Path, ego_cfg: EgoMotionConfig) -> None:
    source = FileSource(short_clip, realtime=False)
    detector = FlakyDetector(fail_on={5})
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(ego_cfg))  # type: ignore[arg-type]
    source.start()
    worker.start()
    worker.join(timeout=20)
    source.stop()
    assert worker.error is None and detector.warmed
    assert worker.frame_errors == 1 and worker.processed == 44
    assert not worker.stale(max_age_s=5.0)


def test_worker_gives_up_on_persistent_errors(short_clip: Path, ego_cfg: EgoMotionConfig) -> None:
    source = FileSource(short_clip, realtime=False, loop=True)
    worker = PerceptionWorker(source, FlakyDetector(always=True), EgoMotionEstimator(ego_cfg))  # type: ignore[arg-type]
    source.start()
    worker.start()
    worker.join(timeout=30)
    source.stop()
    assert isinstance(worker.error, RuntimeError) and "in a row" in str(worker.error)
    assert worker.stale(max_age_s=0.0)


def test_ego_motion_survives_a_frame_size_change(ego_cfg: EgoMotionConfig) -> None:
    est = EgoMotionEstimator(ego_cfg)
    for i in range(10):
        est.update(Frame(image=texture(1280, 720), t=i / FPS, index=i, captured_at=0.0))
    result = est.update(Frame(image=texture(960, 720), t=10 / FPS, index=10, captured_at=0.0))  # new session size
    assert result.long.baseline_s == 0.0  # history was reset instead of crashing


def test_vehicle_dedup_is_fast_in_crowded_scenes() -> None:
    from perception.detector import vehicle_duplicate_mask

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1800, (150, 2))
    xyxy = np.hstack([xy, xy + rng.uniform(20, 120, (150, 2))])
    conf = rng.uniform(0.1, 0.9, 150)
    start = time.perf_counter()
    keep = vehicle_duplicate_mask(xyxy, conf, np.ones(150, bool), 0.7)
    assert (time.perf_counter() - start) * 1000 < 10.0
    assert keep.sum() >= 100


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA GPU")
@pytest.mark.skipif(not DEV_CLIP.exists(), reason="dev clip missing (scripts/fetch_dev_clips.py)")
def test_track_ids_stay_unique_across_resets() -> None:
    from perception.detector import Detector

    cfg = load_config()
    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(DEV_CLIP, realtime=False, start_s=6.0)
    source.start()
    first: set[int] = set()
    second: set[int] = set()
    for i in range(40):
        frame = source.read(timeout=10)
        assert frame is not None
        if i == 20:
            detector.reset()
        ids = {t.track_id for t in detector(frame).tracks}
        (first if i < 20 else second).update(ids)
    source.stop()
    assert first and second and not (first & second)
