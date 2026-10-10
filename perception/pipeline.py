"""The perception worker: video source → YOLO26s + ByteTrack → ego motion → listeners, on its own thread.

Listeners (world state, UI, recorder hooks) are called synchronously on the perception thread with each
`PerceptionResult`; they must be fast and must not block. Exceptions in a listener are logged and do not stop
perception. Performance (fps, per-stage latency percentiles, GPU memory) is logged and recorded periodically.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
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

    # ------------------------------------------------------------------------------------- the loop

    def _main(self) -> None:
        try:
            self._loop()
        except BaseException as exc:
            self.error = exc
            log.exception("perception thread crashed")
        finally:
            self.finished_at = time.monotonic()

    def _loop(self) -> None:
        last_segment: int | None = None
        window = StageStats()
        window_start = time.monotonic()
        window_frames = 0
        self.started_at = time.monotonic()
        while not self._stop.is_set():
            wait_start = time.perf_counter()
            frame = self.source.read(timeout=0.5)
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if frame is None:
                if self.source.finished:
                    log.info("video source finished", processed=self.processed)
                    return
                continue

            new_segment = frame.segment != last_segment
            if new_segment and last_segment is not None:
                self.detector.reset()
                self.ego.reset()
            last_segment = frame.segment

            detection = self.detector(frame)
            ego = self.ego.update(frame, [t.box for t in detection.tracks])
            timings = dict(detection.timings_ms)
            timings["ego_motion"] = ego.elapsed_ms
            timings["wait_for_frame"] = wait_ms
            timings["latency"] = (time.monotonic() - frame.captured_at) * 1000
            result = PerceptionResult(
                frame=frame, tracks=detection.tracks, ego=ego, timings_ms=timings, new_segment=new_segment
            )

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
            elapsed = time.monotonic() - window_start
            if elapsed >= self.perf_interval_s:
                self._report(window, window_frames / elapsed)
                window, window_start, window_frames = StageStats(), time.monotonic(), 0

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
