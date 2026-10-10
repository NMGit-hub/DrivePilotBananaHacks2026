"""The perception worker: video source → YOLO26s + ByteTrack → ego motion → listeners, on its own thread.

Detection (GPU, TensorRT) and ego motion (CPU, OpenCV) are independent and both release the GIL, so they run
concurrently: ego motion on a helper thread, masking moving objects with the previous frame's detections (they
move a few pixels in one frame period; the mask has a margin). Processing time is max(detect, ego), not the sum.

Listeners (world state, UI, recorder hooks) are called synchronously on the perception thread with each
`PerceptionResult`; they must be fast and must not block. Exceptions in a listener are logged and do not stop
perception. Performance (fps, per-stage latency percentiles, GPU memory) is logged and recorded periodically.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from capture.base import VideoSource
from core.logging_setup import get_logger
from core.runlog import RunRecorder
from core.types import Frame
from perception.detector import Detector, Track
from perception.ego_motion import EgoMotion, EgoMotionEstimator
from perception.gpu_monitor import GpuMonitor

log = get_logger(__name__)

_MAX_CONSECUTIVE_ERRORS = 30  # ~1 s of frames: then something is persistently wrong


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Everything perception knows about one frame."""

    frame: Frame
    tracks: list[Track]
    ego: EgoMotion
    timings_ms: dict[str, float]
    new_segment: bool  # first result after a seek/loop/restart: downstream state must reset

    @property
    def latency_ms(self) -> float:
        """From frame decode to this result being ready (stamped when listeners are called)."""
        return self.timings_ms.get("latency", float("nan"))


Listener = Callable[[PerceptionResult], None]


@dataclass(slots=True)
class StageStats:
    """Accumulates per-stage timings; summarizes as percentiles."""

    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, timings: dict[str, float]) -> None:
        """Add one frame's timings."""
        for key, value in timings.items():
            self.samples[key].append(value)

    def summary(self) -> dict[str, dict[str, float]]:
        """p50/p95/max/mean per stage (ms)."""
        out = {}
        for key, values in self.samples.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[key] = {
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "n": int(arr.size),
                }
        return out


class PerceptionWorker:
    """Owns the perception thread."""

    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        ego: EgoMotionEstimator,
        recorder: RunRecorder | None = None,
        perf_interval_s: float = 2.0,
        gpu: GpuMonitor | None = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.ego = ego
        self.recorder = recorder
        self.perf_interval_s = perf_interval_s
        self.gpu = gpu
        self._listeners: list[Listener] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed = 0
        self.run_stats = StageStats()  # whole run (for the benchmark report)
        self.gpu_samples: list[dict[str, float]] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: BaseException | None = None
        self.frame_errors = 0  # frames that failed (detector/ego exceptions) and were skipped
        self.last_result_at: float | None = None  # time.perf_counter() of the last delivered result

    def add_listener(self, listener: Listener) -> None:
        """Register a callback for every result (call before start)."""
        self._listeners.append(listener)

    def start(self) -> None:
        """Start the worker thread (the source must already be started)."""
        self._thread = threading.Thread(target=self._main, name="perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the worker to stop and wait for it."""
        self._stop.set()
        self.join()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to end (e.g. at the end of a file)."""
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def stale(self, max_age_s: float = 1.0) -> bool:
        """True if no result was delivered in the last `max_age_s` (camera stalled, worker stuck or dead).

        While stale, the world state is out of date: answers and alerts must say so instead of using it.
        """
        last = self.last_result_at
        return last is None or time.perf_counter() - last > max_age_s

    # ------------------------------------------------------------------------------------- the loop

    def _main(self) -> None:
        try:
            # CUDA state is per thread: warm up here so the first real frame doesn't pay seconds of start-up.
            warmup = getattr(self.detector, "warmup", None)
            if callable(warmup):
                warmup()
            self._loop()
        except BaseException as exc:
            self.error = exc
            log.exception("perception thread crashed")
        finally:
            self.finished_at = time.perf_counter()

    def _loop(self) -> None:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ego-motion") as ego_pool:
            self._loop_with(ego_pool)

    def _loop_with(self, ego_pool: ThreadPoolExecutor) -> None:
        last_segment: int | None = None
        consecutive_errors = 0
        prev_boxes: list[tuple[float, float, float, float]] = []
        window = StageStats()
        window_start = time.perf_counter()
        window_frames = 0
        self.started_at = time.perf_counter()
        while not self._stop.is_set():
            wait_start = time.perf_counter()
            frame = self.source.read(timeout=0.5)  # raises if the capture thread failed (camera unplugged...)
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if frame is None:
                if self.source.finished:
                    log.info("video source finished", processed=self.processed)
                    return
                continue

            new_segment = frame.segment != last_segment
            if new_segment:  # also on the first frame: the detector/estimator may carry state from a previous run
                self.detector.reset()
                self.ego.reset()
                prev_boxes = []
            last_segment = frame.segment

            try:
                started = time.perf_counter()
                ego_future = ego_pool.submit(self.ego.update, frame, prev_boxes)
                detection = self.detector(frame)
                ego = ego_future.result()
                processing_ms = (time.perf_counter() - started) * 1000
                prev_boxes = detection.raw_boxes or [t.box for t in detection.tracks]
            except Exception:
                self.frame_errors += 1
                consecutive_errors += 1
                if consecutive_errors in (1, 10) or self.frame_errors % 100 == 0:
                    log.exception("perception failed on a frame; skipping it", errors=self.frame_errors)
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(f"perception failed on {consecutive_errors} frames in a row") from None
                time.sleep(min(0.5, 0.01 * consecutive_errors))  # back off a little
                last_segment = None  # resync detector/ego state on the next good frame
                continue
            consecutive_errors = 0
            timings = dict(detection.timings_ms)
            timings["ego_motion"] = ego.elapsed_ms
            timings["processing"] = processing_ms  # wall time of detect + ego running in parallel
            timings["wait_for_frame"] = wait_ms
            timings["latency"] = (time.perf_counter() - frame.captured_at) * 1000
            result = PerceptionResult(
                frame=frame, tracks=detection.tracks, ego=ego, timings_ms=timings, new_segment=new_segment
            )

            self.last_result_at = time.perf_counter()
            for listener in self._listeners:
                try:
                    listener(result)
                except Exception:
                    log.exception("perception listener failed", listener=getattr(listener, "__qualname__", "?"))

            if self.recorder is not None:
                self.recorder.detections(frame.t, frame.index, [t.to_record() for t in detection.tracks])
                self.recorder.maybe_save_frame(frame.image, frame.t)

            self.processed += 1
            window_frames += 1
            self.run_stats.add(timings)
            window.add(timings)
            elapsed = time.perf_counter() - window_start
            if elapsed >= self.perf_interval_s:
                self._report(window, window_frames / elapsed)
                window, window_start, window_frames = StageStats(), time.perf_counter(), 0

    def _report(self, window: StageStats, fps: float) -> None:
        summary = window.summary()
        gpu = self.gpu.sample() if self.gpu is not None else {}
        if gpu:
            self.gpu_samples.append(gpu)
        stats = self.source.stats
        fields: dict[str, Any] = {
            "fps": round(fps, 1),
            "detect_p50_ms": round(summary.get("detect_track_total", {}).get("p50", float("nan")), 1),
            "detect_p95_ms": round(summary.get("detect_track_total", {}).get("p95", float("nan")), 1),
            "ego_p50_ms": round(summary.get("ego_motion", {}).get("p50", float("nan")), 1),
            "latency_p95_ms": round(summary.get("latency", {}).get("p95", float("nan")), 1),
            "source_dropped": stats.dropped,
            **{k: round(v) for k, v in gpu.items() if k in ("torch_allocated_mb", "device_used_mb", "gpu_util_pct")},
        }
        log.info("perception perf", **fields)
        if self.recorder is not None:
            self.recorder.perf(stage="perception", **fields, stages=summary)


class AsyncListener:
    """Runs a slower listener (the world state) on its own thread, so it never holds up perception.

    The perception thread only hands results over. At most `max_pending` results wait; if the consumer falls
    behind, the oldest waiting result is dropped (counted in `dropped`). Downstream logic is time-based, so a
    skipped frame is harmless, but a dropped result's `new_segment` flag is carried to the next one so a seek is
    never missed. Exceptions in the listener are logged and don't stop the thread.
    """

    def __init__(self, listener: Listener, name: str = "listener", max_pending: int = 2) -> None:
        self._listener = listener
        self.name = name
        self.max_pending = max_pending
        self._pending: deque[PerceptionResult] = deque()
        self._cond = threading.Condition()
        self._stopping = False
        self.dropped = 0
        self.handled = 0
        self._thread = threading.Thread(target=self._run, name=f"async-{name}", daemon=True)

    def start(self) -> AsyncListener:
        """Start the consumer thread (returns self for chaining)."""
        self._thread.start()
        return self

    def __call__(self, result: PerceptionResult) -> None:
        """Hand a result over (perception thread; never blocks)."""
        with self._cond:
            if len(self._pending) >= self.max_pending:
                old = self._pending.popleft()
                self.dropped += 1
                if old.new_segment:
                    result = replace(result, new_segment=True)
            self._pending.append(result)
            self._cond.notify()

    def stop(self, timeout: float | None = 2.0) -> None:
        """Finish what is pending, then stop the thread."""
        with self._cond:
            self._stopping = True
            self._cond.notify()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if not self._pending:
                    return
                result = self._pending.popleft()
            try:
                self._listener(result)
            except Exception:
                log.exception("async listener failed", listener=self.name)
            self.handled += 1
