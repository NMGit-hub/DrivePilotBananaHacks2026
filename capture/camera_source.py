"""Live phone camera through scrcpy-server, decoded in-process (no scrcpy window, no screen capture).

Flow: find adb + scrcpy-server (the winget scrcpy install or `video.camera.scrcpy_dir`), push the server to the
phone, open an adb forward tunnel, start the server in camera mode, then read the H.264 stream from the socket
and decode it with PyAV. Frames are stamped with the pipeline's wall clock on arrival.
"""

from __future__ import annotations

import contextlib
import os
import random
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel

from capture.base import VideoSource
from capture.scrcpy_protocol import MediaPacket, ProtocolError, SessionHeader, read_next, read_stream_info
from core.clock import Clock
from core.config import CameraVideoConfig
from core.logging_setup import get_logger
from core.types import Frame

log = get_logger(__name__)

DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
CONNECT_TIMEOUT_S = 8.0
_READ_POLL_S = 0.5  # socket timeout: how often a blocked read checks for stop / stalls


class CameraError(RuntimeError):
    """The phone camera could not be started (message says what the user should do)."""


# Measured on a recorded 1080p30 phone stream (RTX 4050 laptop): software decode + conversion ~42 ms/frame
# (too slow for 30 fps), D3D11VA ~12.5 ms, CUDA ~14 ms. Hardware decoding also keeps the CPU free.
_DECODER_ORDER = {
    "auto": ("d3d11va", "cuda", "software"),
    "d3d11va": ("d3d11va",),
    "cuda": ("cuda",),
    "software": ("software",),
}


def create_h264_decoder(preference: str) -> tuple[Any, str]:
    """Create an H.264 decoder, trying hardware backends in order. Returns (decoder, backend name)."""
    errors = []
    for backend in _DECODER_ORDER[preference]:
        try:
            if backend == "software":
                decoder = av.CodecContext.create("h264", "r")
                decoder.thread_type = "FRAME"
            else:
                hwaccel = HWAccel(device_type=backend, allow_software_fallback=False)
                decoder = av.CodecContext.create("h264", "r", hwaccel=hwaccel)
            decoder.flags |= av.codec.context.Flags.low_delay
            return decoder, backend
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{backend}: {exc}")
    raise CameraError(f"no usable H.264 decoder ({'; '.join(errors)})")


def frame_to_bgr(frame: av.VideoFrame) -> np.ndarray:
    """Convert a decoded frame to BGR. OpenCV's NV12/I420 conversion is ~2x faster than PyAV's swscale here."""
    fmt = frame.format.name
    if fmt == "nv12":
        return cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_NV12)
    if fmt in ("yuv420p", "yuvj420p"):
        return cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_I420)
    return frame.to_ndarray(format="bgr24")


def find_scrcpy_dir(configured: Path | None = None) -> Path:
    """Return the folder holding adb.exe and scrcpy-server (configured path, PATH, or the winget install)."""
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    on_path = shutil.which("scrcpy")
    if on_path:
        candidates.append(Path(on_path).resolve().parent)
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.is_dir():
        candidates += sorted(winget.glob("Genymobile.scrcpy*/scrcpy-win64-v*"), reverse=True)
    for folder in candidates:
        if (folder / "scrcpy-server").is_file() and (folder / "adb.exe").is_file():
            return folder
    raise CameraError(
        "scrcpy not found: install it with `winget install Genymobile.scrcpy` or set video.camera.scrcpy_dir"
    )


def scrcpy_version(folder: Path) -> str:
    """The scrcpy version (the server refuses to start unless the client passes exactly this)."""
    out = subprocess.run([str(folder / "scrcpy.exe"), "--version"], capture_output=True, text=True, timeout=15)
    match = re.search(r"scrcpy (\d+\.\d+(?:\.\d+)?)", out.stdout + out.stderr)
    if not match:
        raise CameraError(f"cannot read the scrcpy version from {folder}")
    return match.group(1)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ScrcpyCameraSource(VideoSource):
    """Live camera source (`video.source: camera`)."""

    def __init__(self, cfg: CameraVideoConfig, clock: Clock) -> None:
        super().__init__(clock, name="camera")
        self.cfg = cfg
        self._dir: Path | None = None
        self._adb: str = "adb"
        self._serial: str | None = None
        self._port: int | None = None
        self._server: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self.device_name = ""
        self.frame_size: tuple[int, int] | None = None
        self.bad_packets = 0
        self.decoder_backend = ""
        self._server_log: deque[str] = deque(maxlen=50)  # last lines of scrcpy-server output (for errors)

    @property
    def is_live(self) -> bool:
        """Always True."""
        return True

    @property
    def fps(self) -> float:
        """Requested camera frame rate."""
        return float(self.cfg.fps)

    # ----------------------------------------------------------------------------------- adb helpers

    def _adb_run(self, *args: str, timeout: float = 30.0) -> str:
        cmd = [self._adb] + (["-s", self._serial] if self._serial else []) + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise CameraError(f"adb {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
        return result.stdout

    def _pick_device(self) -> str:
        lines = self._adb_run("devices").splitlines()[1:]
        states = [line.split() for line in lines if line.strip()]
        ready = [parts[0] for parts in states if len(parts) >= 2 and parts[1] == "device"]
        if ready:
            if len(ready) > 1:
                log.warning("several Android devices connected; using the first", devices=ready)
            return ready[0]
        if any(len(p) >= 2 and p[1] == "unauthorized" for p in states):
            raise CameraError("phone is connected but not authorized: unlock it and accept 'Allow USB debugging'")
        raise CameraError("no phone found: connect it by USB and enable USB debugging (Developer options)")

    # ------------------------------------------------------------------------------------ lifecycle

    def _open(self) -> None:
        self._dir = find_scrcpy_dir(self.cfg.scrcpy_dir)
        self._adb = str(self._dir / "adb.exe")
        version = scrcpy_version(self._dir)
        self._serial = self._pick_device()
        self._adb_run("push", str(self._dir / "scrcpy-server"), DEVICE_SERVER_PATH, timeout=60)

        scid = random.randrange(0, 0x7FFFFFFF)
        self._port = _free_port()
        self._adb_run("forward", f"tcp:{self._port}", f"localabstract:scrcpy_{scid:08x}")
        options = {
            "scid": f"{scid:08x}",
            "log_level": "info",
            "tunnel_forward": "true",
            "audio": "false",
            "control": "false",
            "cleanup": "true",
            "video_source": "camera",
            "video_codec": "h264",
            "camera_id": str(self.cfg.camera_id),
            "camera_size": f"{self.cfg.width}x{self.cfg.height}",
            "camera_fps": str(self.cfg.fps),
            "camera_zoom": str(self.cfg.camera_zoom),
            "video_bit_rate": str(int(self.cfg.bitrate_mbps * 1_000_000)),
            "send_device_meta": "true",
            "send_stream_meta": "true",
            "send_frame_meta": "true",
            "send_dummy_byte": "true",
        }
        cmd = [
            self._adb,
            "-s",
            self._serial,
            "shell",
            f"CLASSPATH={DEVICE_SERVER_PATH}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            version,
        ] + [f"{k}={v}" for k, v in options.items()]
        log.info(
            "starting scrcpy-server",
            serial=self._serial,
            version=version,
            camera_id=self.cfg.camera_id,
            size=options["camera_size"],
            zoom=self.cfg.camera_zoom,
            fps=self.cfg.fps,
        )
        self._server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Drain the server's output continuously: an unread pipe fills up and can stall adb and the stream.
        threading.Thread(target=self._drain_server_log, args=(self._server,), name="scrcpy-log", daemon=True).start()
        self._sock = self._connect()
        info = read_stream_info(self._read_exactly, expect_dummy_byte=False)  # dummy byte consumed by _connect
        self.device_name = info.device_name
        if info.codec != "h264":
            raise CameraError(f"unexpected codec {info.codec}")
        log.info("camera stream connected", device=info.device_name, codec=info.codec)

    def _connect(self) -> socket.socket:
        """Connect to the forwarded port; the dummy byte proves the server (not just adb) is listening."""
        assert self._port is not None
        deadline = time.perf_counter() + CONNECT_TIMEOUT_S
        last_error = "timeout"
        while time.perf_counter() < deadline:
            if self._server is not None and self._server.poll() is not None:
                time.sleep(0.2)  # let the drain thread collect the last lines
                raise CameraError(f"scrcpy-server exited: {' | '.join(self._server_log)[-500:]}")
            sock = socket.create_connection(("127.0.0.1", self._port), timeout=2.0)
            try:
                if sock.recv(1):
                    sock.settimeout(_READ_POLL_S)
                    return sock
                last_error = "connection closed before the server was ready"
            except OSError as exc:
                last_error = str(exc)
            sock.close()
            time.sleep(0.1)
        raise CameraError(f"could not connect to scrcpy-server: {last_error}")

    def _drain_server_log(self, server: subprocess.Popen[bytes]) -> None:
        assert server.stderr is not None
        for raw in iter(server.stderr.readline, b""):
            line = raw.decode(errors="replace").strip()
            if line:
                self._server_log.append(line)
                if "ERROR" in line or "WARN" in line:
                    log.warning("scrcpy-server", line=line)

    def _read_exactly(self, n: int) -> bytes:
        sock = self._sock
        if sock is None:
            raise ConnectionAbortedError("camera closed")
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        last_data = time.perf_counter()
        while got < n:
            if self._stop.is_set():
                raise ConnectionAbortedError("stopping")
            try:
                k = sock.recv_into(view[got:], n - got)
            except TimeoutError:
                if time.perf_counter() - last_data > self.cfg.stall_timeout_s:
                    raise ConnectionError(f"no video from the phone for {self.cfg.stall_timeout_s:.0f} s") from None
                continue
            if k == 0:
                raise ConnectionError("scrcpy stream closed")
            got += k
            last_data = time.perf_counter()
        return bytes(buf)

    def stop(self) -> None:
        """Stop promptly: shut the socket down first so a blocked read returns immediately."""
        self._stop.set()
        sock = self._sock
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        super().stop()

    def _run(self) -> None:
        decoder, backend = create_h264_decoder(self.cfg.decoder)
        self.decoder_backend = backend
        log.info("camera decoder", backend=backend)
        pending_config = b""
        index = 0
        segment = -1
        try:
            while not self._stop.is_set():
                item = read_next(self._read_exactly)
                if isinstance(item, SessionHeader):
                    # A new capture session (start, rotation, resolution change): downstream state must reset.
                    segment += 1
                    self.frame_size = (item.width, item.height)
                    log.info("camera session", width=item.width, height=item.height, segment=segment)
                    continue
                assert isinstance(item, MediaPacket)
                if item.config:
                    pending_config += item.data  # SPS/PPS: prepend to the next frame packet
                    continue
                packet = av.Packet(pending_config + item.data)
                pending_config = b""
                packet.pts = item.pts_us
                try:
                    decoded_frames = decoder.decode(packet)
                except av.error.InvalidDataError:
                    # A corrupt packet must not kill a live drive: skip it; the decoder recovers by the next
                    # key frame at the latest.
                    self.bad_packets += 1
                    if self.bad_packets in (1, 10, 100) or self.bad_packets % 1000 == 0:
                        log.warning("undecodable camera packet skipped", count=self.bad_packets)
                    continue
                for decoded in decoded_frames:
                    image = frame_to_bgr(decoded)
                    self._publish(
                        Frame(
                            image=image,
                            t=self.clock.now(),
                            index=index,
                            captured_at=time.perf_counter(),
                            segment=max(segment, 0),
                        )
                    )
                    index += 1
        except (OSError, ProtocolError) as exc:  # OSError includes ConnectionError and a socket closed by stop()
            if not self._stop.is_set():
                log.error("camera stream ended", error=str(exc))
                raise

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()
            self._server = None
        if self._port is not None and self._serial is not None:
            with contextlib.suppress(CameraError, subprocess.TimeoutExpired, OSError):  # the device may be gone
                self._adb_run("forward", "--remove", f"tcp:{self._port}", timeout=10)
            self._port = None
