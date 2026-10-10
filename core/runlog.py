"""Per-run review folder: frames, detections, questions, answers, alerts and timings.

Each run gets `runs/<YYYYmmdd-HHMMSS>_<label>/` containing:
    config.json       the resolved config (secrets reduced to set/missing)
    log.jsonl         structured application log (written by core.logging_setup)
    events.jsonl      questions, answers, alerts, state changes, timings
    detections.jsonl  per processed frame: tracks with boxes
    perf.jsonl        periodic performance samples (fps, GPU memory, stage latency)
    frames/           one JPEG per `frame_interval_s`, plus every frame sent to the vision model

Callers never block on disk:
- text records are serialized on the caller's thread (so later mutation of the caller's objects can't leak in)
  and appended to an unbounded queue drained by a text-writer thread; they are never dropped;
- frames go to a small bounded queue drained by a separate frame-writer thread; when it is full the frame is
  dropped and counted in `dropped_frames`.
`close()` drains both queues before closing files, and is also registered with `atexit`.
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.logging_setup import dumps_json_line, get_logger, redact, to_jsonable

log = get_logger(__name__)

_STOP = object()


def _ascii_slug(text: str, max_len: int, extra: str = "-_") -> str:
    """Keep ASCII letters/digits and `extra`; replace everything else with '_' (cv2 on Windows needs ASCII paths)."""
    return "".join(c if (c.isascii() and c.isalnum()) or c in extra else "_" for c in text)[:max_len]


class RunRecorder:
    """Writes the review folder for one run. Thread-safe; call `close()` (or use as a context manager)."""

    def __init__(
        self,
        runs_dir: Path,
        label: str,
        config_dump: dict[str, Any] | None = None,
        frame_interval_s: float = 1.0,
        jpeg_quality: int = 85,
        max_pending_frames: int = 16,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        stamp = now().strftime("%Y%m%d-%H%M%S")
        self.dir = self._unique_dir(Path(runs_dir) / f"{stamp}_{_ascii_slug(label, 40)}")
        (self.dir / "frames").mkdir(parents=True)
        self.frame_interval_s = frame_interval_s
        self.jpeg_quality = jpeg_quality
        self.dropped_frames = 0
        self.failed_writes = 0
        self._last_frame_t: float | None = None
        self._frame_seq = 0
        self._lock = threading.Lock()  # guards _closed + enqueue, so nothing is queued after the stop marker
        self._closed = False
        self._text_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max_pending_frames)
        self._files = {
            name: open(self.dir / f"{name}.jsonl", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
            for name in ("events", "detections", "perf")
        }
        self._text_thread = threading.Thread(target=self._text_writer, name="run-text-writer", daemon=True)
        self._frame_thread = threading.Thread(target=self._frame_writer, name="run-frame-writer", daemon=True)
        self._text_thread.start()
        self._frame_thread.start()
        atexit.register(self.close)
        if config_dump is not None:
            text = redact(json.dumps(to_jsonable(config_dump), indent=2, allow_nan=False))
            (self.dir / "config.json").write_text(text, encoding="utf-8")
        log.info("run folder created", path=str(self.dir))

    @staticmethod
    def _unique_dir(path: Path) -> Path:
        candidate, n = path, 1
        while candidate.exists():
            n += 1
            candidate = path.with_name(f"{path.name}-{n}")
        return candidate

    @property
    def log_file(self) -> Path:
        """Path for the structured application log of this run."""
        return self.dir / "log.jsonl"

    # ------------------------------------------------------------------------------------- records

    def event(self, kind: str, t: float | None = None, **data: Any) -> None:
        """Record an event (question, answer, alert, state change...) with media time `t` and wall time."""
        self._put_text("events", {"kind": kind, "t": t, "wall": time.time(), **data})

    def perf(self, **data: Any) -> None:
        """Record a performance sample."""
        self._put_text("perf", {"wall": time.time(), **data})

    def detections(self, t: float, index: int, tracks: list[dict[str, Any]]) -> None:
        """Record the tracks of one processed frame."""
        self._put_text("detections", {"t": round(t, 3), "i": index, "tracks": tracks})

    def maybe_save_frame(self, image: np.ndarray, t: float) -> bool:
        """Save a frame if `frame_interval_s` of media time has passed since the last one. True if queued."""
        if self._last_frame_t is not None and 0.0 <= t - self._last_frame_t < self.frame_interval_s:
            return False
        self._last_frame_t = t
        return self.save_frame(image, f"t{t:08.2f}")

    def save_frame(self, image: np.ndarray, tag: str) -> bool:
        """Queue a JPEG of `image` as frames/<seq>_<tag>.jpg. Returns False if dropped (writer behind or closed)."""
        with self._lock:
            if self._closed:
                return False
            self._frame_seq += 1
            path = self.dir / "frames" / f"{self._frame_seq:06d}_{_ascii_slug(tag, 60, '-_.')}.jpg"
            try:
                self._frame_queue.put_nowait((path, image.copy()))
                return True
            except queue.Full:
                self.dropped_frames += 1
                return False

    def _put_text(self, stream: str, record: dict[str, Any]) -> None:
        line = dumps_json_line(record)  # serialize now: the caller may mutate its objects afterwards
        with self._lock:
            if not self._closed:
                self._text_queue.put((stream, line))

    # ------------------------------------------------------------------------------------- writers

    def _text_writer(self) -> None:
        while True:
            item = self._text_queue.get()
            if item is _STOP:
                return
            stream, line = item
            if stream == "__flush__":
                line.set()  # a threading.Event placed by flush(): everything before it is written
                continue
            try:
                self._files[stream].write(line + "\n")
            except Exception:
                self.failed_writes += 1
                log.exception("run recorder text write failed", stream=stream)

    def _frame_writer(self) -> None:
        while True:
            item = self._frame_queue.get()
            try:
                if item is _STOP:
                    return
                path, image = item
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                path.write_bytes(encoded.tobytes())
            except Exception:
                self.failed_writes += 1
                log.exception("run recorder frame write failed")
            finally:
                self._frame_queue.task_done()

    def flush(self, timeout: float = 30.0) -> None:
        """Block until everything queued so far is written (or `timeout` passes)."""
        marker = threading.Event()
        with self._lock:
            if self._closed:
                return
            self._text_queue.put(("__flush__", marker))
        self._frame_queue.join()
        marker.wait(timeout)

    def close(self) -> None:
        """Drain both queues, stop the writers and close the files. Idempotent; later records are ignored."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._text_queue.put(_STOP)
        self._frame_queue.put(_STOP)  # blocks only until the frame writer frees a slot
        self._text_thread.join()
        self._frame_thread.join()
        for handle in self._files.values():
            handle.close()
        atexit.unregister(self.close)
        if self.dropped_frames or self.failed_writes:
            log.warning("run recorder lost data", dropped_frames=self.dropped_frames, failed_writes=self.failed_writes)

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
