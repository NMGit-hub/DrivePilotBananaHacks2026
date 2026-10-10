"""Recorded-drive input: plays a video file through the same pipeline as the live camera.

Real-time mode (default) paces frames on the video's own timestamps via a `MediaClock`; if decoding or the
consumer falls behind, frames are skipped so the pipeline always sees "now" (like a live camera). Lock-step
mode (`realtime=False`) delivers every frame as fast as the consumer takes them, and the clock follows the
frame timestamps, so benchmarks and evals are deterministic.

Pause / skip work in both modes; every seek or loop restart starts a new `segment` so trackers reset. With
`hold_at_end` the source doesn't finish at the end of the file: the clock stops on the last frame (so GPS replay
and timers stop too) and a seek back plays on from there.
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
_DISCONTINUITY_S = 1.0  # a timestamp jump bigger than this (either way) starts a new segment
_MAX_LAG_S = 0.5  # further behind real time than this: re-anchor (slow down) instead of skipping


class FileSource(VideoSource):
    """Video file source (`video.source: file`)."""

    def __init__(
        self,
        path: Path,
        clock: MediaClock | None = None,
        realtime: bool = True,
        start_s: float = 0.0,
        loop: bool = False,
        hold_at_end: bool = False,
    ) -> None:
        super().__init__(clock if clock is not None else MediaClock(start=start_s), name=f"file:{Path(path).name}")
        self.path = Path(path)
        self.realtime = realtime
        self.start_s = start_s
        self.loop = loop
        self.hold_at_end = hold_at_end
        self._at_end = False
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
    def at_end(self) -> bool:
        """True while holding at the end of the file (`hold_at_end`)."""
        with self._ctrl_lock:
            return self._at_end

    @property
    def paused(self) -> bool:
        """True while playback is paused."""
        with self._ctrl_lock:
            return self._paused

    # ------------------------------------------------------------------------------------ controls

    def toggle_pause(self) -> bool:
        """Pause or resume. Returns the new paused state."""
        with self._ctrl_lock:  # same lock as the capture thread's clock re-anchoring
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
        late_frames = 0
        grab_ms = 0.0  # running average of the decode cost per frame
        preview = False  # a seek while paused shows one frame at the new position

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
                preview = paused
                log.info("seek", target=round(seek, 2), segment=segment)
            if paused:
                if preview:
                    preview = False
                    frame = self._decode_one(cap, segment)
                    if frame is not None:
                        last_t = frame.t
                        with self._ctrl_lock:
                            self._media_clock.seek(frame.t)
                        self._publish(frame)
                    continue
                self._wake.wait(_CONTROL_POLL_S)
                self._wake.clear()
                continue

            grab_start = time.perf_counter()
            grabbed = cap.grab()
            grab_ms = 0.9 * grab_ms + 0.1 * (time.perf_counter() - grab_start) * 1000.0
            if not grabbed:
                if self.loop and self._frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    segment += 1
                    last_t = -math.inf
                    resync_clock = True
                    continue
                if self.hold_at_end:
                    with self._ctrl_lock:
                        self._at_end = True
                        self._media_clock.pause()
                    log.info("end of video: holding the last frame", path=str(self.path), late_frames=late_frames)
                    self._wait_for_seek()
                    with self._ctrl_lock:
                        self._at_end = False
                    continue
                log.info("end of video", path=str(self.path), late_frames=late_frames)
                return

            index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            pts = max(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 0.0)
            if math.isfinite(last_t):
                step = pts - last_t
                if abs(step) > _DISCONTINUITY_S:
                    # Joined recordings or a timestamp reset: a new segment on a re-anchored clock (instead of
                    # dropping everything until the timestamps catch up, or sleeping through a forward gap).
                    segment += 1
                    resync_clock = True
                    log.warning(
                        "timestamp jump in video; starting a new segment", jump_s=round(step, 2), segment=segment
                    )
                elif step < 0.5 * period:
                    # Phone recordings (variable frame rate) occasionally stamp two frames with the same time,
                    # and some containers repeat early timestamps. A near-zero time step breaks per-second rates
                    # downstream (optical flow, stop timers), so such a frame is skipped.
                    self._count_skipped()
                    continue
            t = pts
            last_t = t

            if self.realtime and not resync_clock:
                lag = self._media_clock.now() - t
                decode_bound = grab_ms > 0.8 * period * 1000.0
                if lag > _MAX_LAG_S or (decode_bound and lag > period):
                    # Behind real time and skipping can't help: grab() itself decodes, so when decoding is the
                    # bottleneck (e.g. 4K HEVC on a busy CPU) skipping would starve the pipeline. Play on from
                    # this frame instead: playback runs slower than real time.
                    late_frames += 1
                    if late_frames in (1, 10, 100) or late_frames % 1000 == 0:
                        log.warning(
                            "decoding slower than real time; playback slowed",
                            lag_s=round(lag, 2),
                            decode_ms=round(grab_ms, 1),
                            count=late_frames,
                        )
                    resync_clock = True
                elif lag > period:  # slightly behind (the consumer was slow): skip this frame to catch up
                    self._count_skipped()
                    continue
                elif not self._wait_until(t):  # a control (pause/seek/stop) arrived while waiting
                    continue

            ok, image = cap.retrieve()
            if not ok or image is None:
                self._count_skipped()
                continue
            if resync_clock:
                # Anchor the clock once the first frame of a segment is actually decoded, so a slow first
                # decode (codec start-up) doesn't make the whole segment start "late".
                with self._ctrl_lock:
                    self._media_clock.seek(t)
                    if self.realtime and not self._paused:  # a pause may have arrived since the loop top
                        self._media_clock.play()
                resync_clock = False
            elif not self.realtime:
                self._media_clock.seek(t)  # lock-step: the (paused) clock follows the frames
            self._publish(Frame(image=image, t=t, index=index, captured_at=time.perf_counter(), segment=segment))

    def _wait_for_seek(self) -> None:
        """At the end of the file: wait until a seek (or stop) arrives."""
        while not self._stop.is_set():
            with self._ctrl_lock:
                if self._pending_seek is not None:
                    return
            self._wake.wait(_CONTROL_POLL_S)
            self._wake.clear()

    @staticmethod
    def _decode_one(cap: cv2.VideoCapture, segment: int) -> Frame | None:
        """The next frame, decoded (or None at the end)."""
        if not cap.grab():
            return None
        ok, image = cap.retrieve()
        if not ok or image is None:
            return None
        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t = max(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 0.0)
        return Frame(image=image, t=t, index=index, captured_at=time.perf_counter(), segment=segment)

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
