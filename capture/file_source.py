"""Recorded-drive input: plays a video file through the same pipeline as the live camera.

Real-time mode (default) paces frames on the video's own timestamps via a `MediaClock`; if decoding or the
consumer falls behind, frames are skipped so the pipeline always sees "now" (like a live camera). Lock-step
mode (`realtime=False`) delivers every frame as fast as the consumer takes them, and the clock follows the
frame timestamps, so benchmarks and evals are deterministic.

Pause / skip work in both modes; every seek or loop restart starts a new `segment` so trackers reset.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2

from capture.base import VideoSource
from core.clock import MediaClock
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)

_CONTROL_POLL_S = 0.05


class FileSource(VideoSource):
    """Video file source (`video.source: file`)."""

    def __init__(
        self,
        path: Path,
        clock: MediaClock | None = None,
        realtime: bool = True,
        start_s: float = 0.0,
        loop: bool = False,
    ) -> None:
        super().__init__(clock if clock is not None else MediaClock(start=start_s), name=f"file:{Path(path).name}")
        self.path = Path(path)
        self.realtime = realtime
        self.start_s = start_s
        self.loop = loop
        self._lockstep = not realtime
        self._media_clock: MediaClock = self.clock  # type: ignore[assignment]
        self._cap: cv2.VideoCapture | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._ctrl_lock = threading.Lock()
        self._wake = threading.Event()
        self._paused = False
        self._pending_seek: float | None = None

    # ---------------------------------------------------------------------------------- properties

    @property
    def is_live(self) -> bool:
        """Always False: files can pause and seek."""
        return False

    @property
    def fps(self) -> float:
        """Nominal frame rate from the container."""
        return self._fps

    @property
    def duration(self) -> float:
        """Clip length in seconds (from frame count / fps)."""
        return self._frame_count / self._fps if self._fps else 0.0

    @property
    def paused(self) -> bool:
        """True while playback is paused."""
        with self._ctrl_lock:
            return self._paused

    # ------------------------------------------------------------------------------------ controls

    def toggle_pause(self) -> bool:
        """Pause or resume. Returns the new paused state."""
        with self._ctrl_lock:
            self._paused = not self._paused
            paused = self._paused
        if self.realtime:
            if paused:
                self._media_clock.pause()
            else:
                self._media_clock.play()
        self._wake.set()
        log.info("playback paused" if paused else "playback resumed", t=round(self._media_clock.now(), 2))
        return paused

    def seek_relative(self, delta_s: float) -> None:
        """Skip by `delta_s` seconds from the current playback time (clamped to the clip)."""
        self.seek_to(self._media_clock.now() + delta_s)

    def seek_to(self, t: float) -> None:
        """Jump to media time `t` seconds (clamped to the clip)."""
        upper = max(0.0, self.duration - 1.0 / max(self._fps, 1.0))
        target = min(max(0.0, t), upper)
        with self._ctrl_lock:
            self._pending_seek = target
        self._wake.set()

    # --------------------------------------------------------------------------------- capture loop

    def _open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            "video opened",
            path=str(self.path),
            fps=round(self._fps, 3),
            frames=self._frame_count,
            size=f"{width}x{height}",
            realtime=self.realtime,
        )
        if self.start_s > 0:
            with self._ctrl_lock:
                self._pending_seek = self.start_s

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None, "_open() must run first"
        period = 1.0 / self._fps
        segment = 0
        last_t = -math.inf
        resync_clock = True  # anchor the clock to the first frame of every segment

        while not self._stop.is_set():
            with self._ctrl_lock:
                seek, self._pending_seek = self._pending_seek, None
                paused = self._paused
            if seek is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
                segment += 1
                last_t = -math.inf
                resync_clock = True
                self._media_clock.seek(seek)
                log.info("seek", target=round(seek, 2), segment=segment)
            if paused:
                self._wake.wait(_CONTROL_POLL_S)
                self._wake.clear()
                continue

            if not cap.grab():
                if self.loop and self._frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    segment += 1
                    last_t = -math.inf
                    resync_clock = True
                    continue
                log.info("end of video", path=str(self.path))
                return

            index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if math.isfinite(last_t) and pts < last_t + 0.5 * period:
                # Phone recordings (variable frame rate) occasionally stamp two frames with the same time, and
                # some containers repeat early timestamps. A near-zero time step breaks per-second rates
                # downstream (optical flow, stop timers), so such a frame is skipped.
                self._count_skipped()
                continue
            t = max(pts, 0.0)
            last_t = t

            if self.realtime and not resync_clock:
                if self._media_clock.now() - t > period:  # behind real time: skip decoding this frame
                    self._count_skipped()
                    continue
                if not self._wait_until(t):  # a control (pause/seek/stop) arrived while waiting
                    continue

            ok, image = cap.retrieve()
            if not ok or image is None:
                self._count_skipped()
                continue
            if resync_clock:
                # Anchor the clock once the first frame of a segment is actually decoded, so a slow first
                # decode (codec start-up) doesn't make the whole segment start "late".
                self._media_clock.seek(t)
                if self.realtime:
                    self._media_clock.play()
                resync_clock = False
            elif not self.realtime:
                self._media_clock.seek(t)  # lock-step: the (paused) clock follows the frames
            self._publish(Frame(image=image, t=t, index=index, captured_at=time.monotonic(), segment=segment))

    def _wait_until(self, t: float) -> bool:
        """Sleep until media time `t`. Returns False if interrupted by a control change or stop."""
        while not self._stop.is_set():
            remaining = t - self._media_clock.now()
            if remaining <= 0:
                return True
            if self._wake.wait(min(remaining, _CONTROL_POLL_S)):
                self._wake.clear()
                with self._ctrl_lock:
                    interrupted = self._paused or self._pending_seek is not None
                if interrupted:
                    return False
        return False
