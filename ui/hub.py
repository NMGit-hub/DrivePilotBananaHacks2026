"""What the page shows, shared between the app's threads and the web server.

- Video: the latest processed frame as a JPEG, encoded on the hub's own thread at most `fps` times a second and
  scaled to `width` (the perception thread only hands over a reference, so it never waits on encoding). With its
  overlay packet (V2, ui/overlay.py) the frame also becomes one binary message: `MAGIC`, a 4-byte big-endian
  header length, the JSON header (time, frame size, capture wall time, overlay) and the JPEG, so the page draws
  each overlay on the very frame it was built from.
- Messages: questions, answers, alerts and state changes, pushed to every open page over WebSocket. The last
  `history` messages are kept so a page opened (or reloaded) later shows the conversation so far. The latest
  status (time, pause, fps, GPS, next maneuver) is kept separately and replaced, not appended.
Producers never block: frames use a one-slot hand-off, messages a bounded deque, and delivery to the asyncio
server goes through `loop.call_soon_threadsafe` into per-page queues that drop the oldest message if a page
stops reading.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import struct
import threading
import time
import uuid
from collections import deque
from typing import Any

import cv2

from core.logging_setup import get_logger, redact, to_jsonable
from core.types import Frame

log = get_logger(__name__)

PAGE_QUEUE = 500  # messages buffered per open page before the oldest are dropped
CLOSE = "_close"  # internal: the server closes the page's WebSocket
MAGIC = b"DPV1"  # the frame + overlay messages of the V2 page start with this
IMAGES = 30  # thumbnails kept for the page


def frame_message(header: dict[str, Any], jpeg: bytes) -> bytes:
    """One binary video message: MAGIC, header length (4 bytes, big-endian), JSON header, JPEG."""
    body = json.dumps(header, separators=(",", ":")).encode()
    return MAGIC + struct.pack(">I", len(body)) + body + jpeg


def parse_frame_message(message: bytes) -> tuple[dict[str, Any], bytes]:
    """(header, JPEG) of a binary video message (tests, tools)."""
    if message[:4] != MAGIC:
        raise ValueError("not a DrivePilot video message")
    (size,) = struct.unpack(">I", message[4:8])
    return json.loads(message[8 : 8 + size]), message[8 + size :]


class UiHub:
    """Frames and messages for the V1 page. Thread-safe."""

    def __init__(self, width: int = 960, fps: float = 20.0, jpeg_quality: int = 75, history: int = 200) -> None:
        self.session = uuid.uuid4().hex[:12]  # a page reconnecting to a new app run starts over
        self.width = width
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._frame_cond = threading.Condition()
        self._pending: tuple[Frame, dict[str, Any] | None] | None = None
        self._jpeg: bytes | None = None
        self._message: bytes | None = None  # the newest frame + overlay as one binary message
        self._jpeg_seq = 0
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._status: dict[str, Any] = {}
        self._images: dict[int, bytes] = {}  # thumbnails shown in the conversation (newest IMAGES kept)
        self._ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pages: set[asyncio.Queue[dict[str, Any]]] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.encoded = 0

    # ---------------------------------------------------------------------------------- video

    def start(self) -> UiHub:
        """Start the JPEG encoder thread (returns self)."""
        self._thread = threading.Thread(target=self._encode_loop, name="ui-encoder", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the encoder thread, and tell open pages' connections to close."""
        self._stop.set()
        with self._lock:
            self._push_locked({"type": CLOSE})
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def stopped(self) -> bool:
        """True once the app is shutting down (open video streams end)."""
        return self._stop.is_set()

    def offer_frame(self, frame: Frame, overlay: dict[str, Any] | None = None) -> None:
        """A new processed frame and, for the V2 page, its overlay (any thread): keep only the newest, never wait."""
        with self._frame_cond:
            self._pending = (frame, overlay)
            self._frame_cond.notify()

    def latest_jpeg(self) -> tuple[int, bytes | None]:
        """(sequence number, JPEG) of the newest encoded frame; the number grows with every new frame."""
        with self._frame_cond:
            return self._jpeg_seq, self._jpeg

    def latest_message(self) -> tuple[int, bytes | None]:
        """(sequence number, binary frame + overlay message) of the newest encoded frame."""
        with self._frame_cond:
            return self._jpeg_seq, self._message

    def _encode_loop(self) -> None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        next_at = 0.0
        while not self._stop.is_set():
            with self._frame_cond:
                while self._pending is None and not self._stop.is_set():
                    self._frame_cond.wait(0.5)
                pending, self._pending = self._pending, None
            if pending is None:
                continue
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
                with self._frame_cond:  # a newer frame arrived while we waited: show that one
                    if self._pending is not None:
                        pending, self._pending = self._pending, None
            next_at = time.perf_counter() + interval
            frame, overlay = pending
            try:
                jpeg = self._encode(frame)
                header = {
                    "t": round(frame.t, 3),
                    "w": frame.width,
                    "h": frame.height,
                    # when the frame was captured, on the wall clock (the page measures frame-to-screen latency)
                    "captured": round(time.time() - (time.perf_counter() - frame.captured_at), 4),
                    "overlay": overlay,
                }
                message = frame_message(header, jpeg)
            except Exception:
                log.exception("frame encoding failed")
                continue
            with self._frame_cond:
                self._jpeg, self._message, self._jpeg_seq = jpeg, message, self._jpeg_seq + 1
            self.encoded += 1

    def _encode(self, frame: Frame) -> bytes:
        image = frame.image
        h, w = image.shape[:2]
        if w > self.width:
            image = cv2.resize(image, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    # ------------------------------------------------------------------------------- messages

    def publish(self, kind: str, **data: Any) -> dict[str, Any]:
        """Add a message to the conversation feed and push it to every open page (any thread). Ids, history
        order and delivery order agree (all under one lock), so a page never sees message N+1 before N."""
        body = _clean(data)
        with self._lock:
            message = {"type": kind, "id": next(self._ids), "wall": round(time.time(), 3), **body}
            self._history.append(message)
            self._push_locked(message)
        return message

    def set_status(self, **status: Any) -> None:
        """Replace the status line (any thread)."""
        message = {"type": "status", **_clean(status)}
        with self._lock:
            self._status = message
            self._push_locked(message)

    def put_image(self, image_id: int, jpeg: bytes) -> None:
        """Keep a JPEG the page can show by id (memory thumbnails); only the newest `IMAGES` are kept."""
        with self._lock:
            self._images[image_id] = jpeg
            while len(self._images) > IMAGES:
                del self._images[min(self._images)]

    def image(self, image_id: int) -> bytes | None:
        """A kept JPEG, or None."""
        with self._lock:
            return self._images.get(image_id)

    def history(self) -> list[dict[str, Any]]:
        """The conversation so far (oldest first)."""
        with self._lock:
            return list(self._history)

    def status(self) -> dict[str, Any]:
        """The latest status message."""
        with self._lock:
            return dict(self._status)

    # ------------------------------------------------------------------- server side (asyncio)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The server's event loop (called once it runs)."""
        with self._lock:
            self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """A queue of new messages for one page (call on the server loop)."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=PAGE_QUEUE)
        with self._lock:
            self._pages.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Forget a page's queue."""
        with self._lock:
            self._pages.discard(queue)

    def _push_locked(self, message: dict[str, Any]) -> None:
        """Deliver to every page (caller holds the lock; call_soon_threadsafe doesn't block)."""
        if self._loop is None or not self._pages:
            return
        with contextlib.suppress(RuntimeError):  # the loop is closed (shutting down)
            self._loop.call_soon_threadsafe(_deliver, list(self._pages), message)


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    """Plain JSON values, with any secret redacted (messages reach the page and could carry error text)."""
    return dict(json.loads(redact(json.dumps(to_jsonable(data)))))


def _deliver(pages: list[asyncio.Queue[dict[str, Any]]], message: dict[str, Any]) -> None:
    """Put a message on each page's queue (server loop); a page that stopped reading loses its oldest."""
    for queue in pages:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(message)
