"""Video sources: file pacing/lock-step/pause/seek/loop, the scrcpy protocol, and camera decoding over a fake socket."""

from __future__ import annotations

import io
import socket
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from capture.camera_source import ScrcpyCameraSource
from capture.file_source import FileSource
from capture.scrcpy_protocol import (
    MediaPacket,
    ProtocolError,
    SessionHeader,
    encode_packet,
    encode_preamble,
    encode_session,
    read_next,
    read_stream_info,
)
from core.clock import WallClock
from core.config import CameraVideoConfig

FPS = 30
N_FRAMES = 90  # 3 s


def frame_value(i: int) -> int:
    """Gray level for flat test frames (camera test: steps of 20 survive H.264)."""
    return (i * 2) % 256


def barcode_image(i: int, width: int = 64, height: int = 48) -> np.ndarray:
    """Frame whose index is written as 8 black/white vertical bars (robust to lossy compression)."""
    image = np.zeros((height, width, 3), np.uint8)
    bar = width // 8
    for bit in range(8):
        if (i >> bit) & 1:
            image[:, bit * bar : (bit + 1) * bar] = 255
    return image


def decoded_index(image: np.ndarray) -> int:
    bar = image.shape[1] // 8
    return sum(1 << bit for bit in range(8) if image[:, bit * bar + 2 : (bit + 1) * bar - 2].mean() > 127)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(barcode_image(i))
    writer.release()
    return path


def drain(source: FileSource, timeout: float = 5.0) -> list:
    frames = []
    while (frame := source.read(timeout=timeout)) is not None:
        frames.append(frame)
    return frames


# ------------------------------------------------------------------------------------- file source


def test_lockstep_delivers_every_frame_in_order(clip: Path) -> None:
    source = FileSource(clip, realtime=False)
    source.start()
    frames = drain(source)
    source.stop()
    assert len(frames) == N_FRAMES
    assert [f.index for f in frames] == list(range(N_FRAMES))
    assert [decoded_index(f.image) for f in frames] == list(range(N_FRAMES))
    times = np.array([f.t for f in frames])
    assert np.all(np.diff(times) > 0)
    assert np.allclose(np.diff(times), 1 / FPS, atol=2e-3)
    assert source.clock.now() == pytest.approx(times[-1])  # the paused clock follows the frames
    assert source.finished and source.stats.dropped == 0 and source.stats.delivered == N_FRAMES


def test_realtime_pacing_follows_the_clock(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    source.start()
    start = time.monotonic()
    frames = []
    while time.monotonic() - start < 1.0:
        frame = source.read(timeout=1.0)
        assert frame is not None
        frames.append(frame)
        # a frame is never delivered before its time, and never much later
        assert frame.t <= source.clock.now() + 1e-3
        assert source.clock.now() - frame.t < 0.1
    source.stop()
    assert 25 <= len(frames) <= 33  # ~30 fps


def test_slow_consumer_gets_the_newest_frames(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    source.start()
    frames = []
    for _ in range(8):
        frame = source.read(timeout=1.0)
        assert frame is not None
        frames.append(frame)
        assert source.clock.now() - frame.t < 0.1  # stays near real time despite the slow consumer
        time.sleep(0.1)  # 3 frame periods of "processing"
    source.stop()
    assert source.stats.dropped > 0
    # ~100 ms between reads on a 33 ms frame grid: each read skips 2-3 frames
    assert np.all(np.diff([f.t for f in frames]) > 0.06)


def test_pause_and_resume(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    source.start()
    assert source.read(timeout=1.0) is not None
    assert source.toggle_pause() is True
    time.sleep(0.05)
    source.read(timeout=0.05)  # drain a frame published just before the pause took effect
    frozen = source.clock.now()
    assert source.read(timeout=0.4) is None
    assert source.clock.now() == pytest.approx(frozen)
    assert source.toggle_pause() is False
    frame = source.read(timeout=1.0)
    source.stop()
    assert frame is not None and frame.t >= frozen - 0.05


def test_seek_starts_a_new_segment(clip: Path) -> None:
    source = FileSource(clip, realtime=False)
    source.start()
    first = source.read(timeout=2.0)
    assert first is not None and first.segment == 0
    source.seek_to(2.0)
    after = None
    for _ in range(10):  # a frame grabbed before the seek may still be in flight
        after = source.read(timeout=2.0)
        assert after is not None
        if after.segment == 1:
            break
    source.stop()
    assert after is not None and after.segment == 1
    assert after.t == pytest.approx(2.0, abs=0.05)
    assert decoded_index(after.image) in (59, 60, 61)


def test_seek_relative_is_clamped(clip: Path) -> None:
    source = FileSource(clip, realtime=False)
    source.start()
    assert source.read(timeout=2.0) is not None
    source.seek_relative(-30.0)
    frames = drain(source)
    source.stop()
    seg1 = [f for f in frames if f.segment == 1]
    assert seg1 and seg1[0].t == pytest.approx(0.0, abs=0.05)


def test_start_offset(clip: Path) -> None:
    source = FileSource(clip, realtime=False, start_s=1.0)
    source.start()
    frames = drain(source)
    source.stop()
    assert frames[0].t == pytest.approx(1.0, abs=0.05)
    assert len(frames) == pytest.approx(N_FRAMES - FPS, abs=2)


def test_loop_restarts_with_a_new_segment(clip: Path) -> None:
    source = FileSource(clip, realtime=False, loop=True)
    source.start()
    frames = [source.read(timeout=2.0) for _ in range(N_FRAMES + 10)]
    source.stop()
    assert all(f is not None for f in frames)
    assert frames[N_FRAMES - 1].segment == 0 and frames[N_FRAMES].segment == 1  # type: ignore[union-attr]
    assert frames[N_FRAMES].t == pytest.approx(0.0, abs=0.05)  # type: ignore[union-attr]


def test_missing_file_fails_at_start(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.mp4").start()


def test_stop_is_prompt_and_idempotent(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    source.start()
    assert source.read(timeout=1.0) is not None
    start = time.monotonic()
    source.stop()
    source.stop()
    assert time.monotonic() - start < 1.0
    assert source.read(timeout=0.1) is None


# --------------------------------------------------------------------------------- scrcpy protocol


def test_protocol_roundtrip() -> None:
    stream = io.BytesIO(
        encode_preamble("OnePlus CPH2583", 0x68323634)
        + encode_session(1920, 1080)
        + encode_packet(b"\x00\x00\x00\x01config", None)
        + encode_packet(b"frame-1", 33_333, key_frame=True)
        + encode_packet(b"frame-2", 66_666)
    )

    def read_exactly(n: int) -> bytes:
        data = stream.read(n)
        if len(data) < n:
            raise ConnectionError("eof")
        return data

    info = read_stream_info(read_exactly)
    assert info.device_name == "OnePlus CPH2583" and info.codec == "h264"
    assert read_next(read_exactly) == SessionHeader(1920, 1080)
    assert read_next(read_exactly) == MediaPacket(b"\x00\x00\x00\x01config", None, config=True, key_frame=False)
    assert read_next(read_exactly) == MediaPacket(b"frame-1", 33_333, config=False, key_frame=True)
    assert read_next(read_exactly) == MediaPacket(b"frame-2", 66_666, config=False, key_frame=False)
    with pytest.raises(ConnectionError):
        read_next(read_exactly)


@pytest.mark.parametrize(("codec_id", "message"), [(0, "disabled"), (1, "configuration error"), (0x1234, "unknown")])
def test_protocol_server_errors(codec_id: int, message: str) -> None:
    stream = io.BytesIO(encode_preamble("x", codec_id))
    with pytest.raises(ProtocolError, match=message):
        read_stream_info(stream.read)


# ------------------------------------------------------------------- camera decode over a fake server


def h264_packets(n: int, width: int = 128, height: int = 96) -> tuple[bytes, list[tuple[bytes, bool]]]:
    """Encode n frames with libx264 (Annex B): returns (config bytes, [(packet bytes, keyframe)])."""
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base = Fraction(1, FPS)
    # In-band SPS/PPS (like Android MediaCodec), Annex B start codes.
    enc.options = {"tune": "zerolatency", "preset": "ultrafast", "x264-params": "repeat-headers=1:annexb=1"}
    packets: list[tuple[bytes, bool]] = []
    for i in range(n):
        frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), frame_value(i * 10), np.uint8), "bgr24")
        frame.pts = i
        packets += [(bytes(p), p.is_keyframe) for p in enc.encode(frame)]
    packets += [(bytes(p), p.is_keyframe) for p in enc.encode(None)]
    # Split everything before the first coded slice (SPS, PPS, SEI) into a separate config packet, the way
    # MediaCodec emits BUFFER_FLAG_CODEC_CONFIG before the first frame.
    first, key = packets[0]
    pos = 0
    while (start := first.find(b"\x00\x00\x01", pos)) >= 0:
        if first[start + 3] & 0x1F in (1, 5):  # coded slice (non-IDR / IDR)
            break
        pos = start + 3
    assert start > 0, "no parameter sets before the first slice"
    cut = start - 1 if first[start - 1] == 0 else start  # include a 4-byte start code's leading zero
    return first[:cut], [(first[cut:], key), *packets[1:]]


class FakeCamera(ScrcpyCameraSource):
    """ScrcpyCameraSource with adb replaced by a local TCP server that speaks the scrcpy protocol."""

    def __init__(self, payload: bytes, decoder: str = "auto", hold_s: float = 0.5) -> None:
        super().__init__(CameraVideoConfig(decoder=decoder), WallClock())
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._port = self._listener.getsockname()[1]
        threading.Thread(target=self._serve, args=(payload, hold_s), daemon=True).start()

    def _serve(self, payload: bytes, hold_s: float) -> None:
        conn, _ = self._listener.accept()
        conn.sendall(payload)
        time.sleep(hold_s)  # then hang up (or stay silent for a long hold: a stalled phone)
        conn.close()

    def _open(self) -> None:
        self._sock = self._connect()
        read_stream_info(self._read_exactly, expect_dummy_byte=False)

    def _close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        self._listener.close()


@pytest.mark.parametrize("decoder", ["auto", "software"])
def test_camera_source_decodes_scrcpy_stream(decoder: str) -> None:
    config, packets = h264_packets(12)
    payload = encode_preamble("Fake Phone", 0x68323634) + encode_session(128, 96) + encode_packet(config, None)
    payload += b"".join(encode_packet(data, (i + 1) * 33_333, key) for i, (data, key) in enumerate(packets))
    camera = FakeCamera(payload, decoder)
    camera._lockstep = True  # deliver every frame so the test can check them all
    camera.start()
    frames = []
    with pytest.raises(RuntimeError, match="capture failed"):  # the stream closing unexpectedly is surfaced
        while (frame := camera.read(timeout=3.0)) is not None:
            frames.append(frame)
    camera.stop()
    assert len(frames) == 12
    assert frames[0].image.shape == (96, 128, 3)
    assert [round(float(f.image.mean()) / 20) for f in frames] == list(range(12))
    assert [f.index for f in frames] == list(range(12))
    assert camera.frame_size == (128, 96)
    assert camera.decoder_backend in (("d3d11va", "cuda", "software") if decoder == "auto" else ("software",))


def test_camera_source_survives_a_corrupt_packet() -> None:
    config, packets = h264_packets(12)
    stream = [encode_packet(data, (i + 1) * 33_333, key) for i, (data, key) in enumerate(packets)]
    stream.insert(6, encode_packet(b"\x00\x00\x01\x65" + b"\xff" * 64, 6 * 33_333))  # garbage "IDR" slice
    payload = encode_preamble("Fake Phone", 0x68323634) + encode_session(128, 96) + encode_packet(config, None)
    camera = FakeCamera(payload + b"".join(stream))
    camera._lockstep = True
    camera.start()
    frames = []
    with pytest.raises(RuntimeError, match="capture failed"):
        while (frame := camera.read(timeout=3.0)) is not None:
            frames.append(frame)
    camera.stop()
    assert camera.bad_packets >= 1 or len(frames) >= 12  # skipped, or tolerated by the decoder itself
    assert len(frames) >= 11  # decoding continued after the corrupt packet


# ------------------------------------------------------------------------- review regressions (M1)


class CapProxy:
    """Wraps cv2.VideoCapture to inject slow decoding or rewritten timestamps."""

    def __init__(
        self, cap: cv2.VideoCapture, grab_delay_s: float = 0.0, pts_offset_after: tuple[int, float] | None = None
    ) -> None:
        self._cap, self._delay, self._offset = cap, grab_delay_s, pts_offset_after
        self._n = 0

    def grab(self) -> bool:
        time.sleep(self._delay)
        ok = bool(self._cap.grab())
        self._n += ok
        return ok

    def get(self, prop: int) -> float:
        value = float(self._cap.get(prop))
        if prop == cv2.CAP_PROP_POS_MSEC and self._offset is not None and self._n > self._offset[0]:
            value += self._offset[1] * 1000.0
        return value

    def __getattr__(self, name: str) -> object:  # retrieve, set, release, ...
        return getattr(self._cap, name)


def start_with_proxy(source: FileSource, **proxy_kwargs: object) -> None:
    source._open()
    source._cap = CapProxy(source._cap, **proxy_kwargs)  # type: ignore[arg-type, assignment]
    source._thread = threading.Thread(target=source._thread_main, daemon=True)
    source._thread.start()


def test_slow_decoding_slows_playback_instead_of_starving(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    start_with_proxy(source, grab_delay_s=0.045)  # slower than the 33 ms frame period
    frames = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = source.read(timeout=0.5)
        if frame is not None:
            frames.append(frame)
    source.stop()
    assert len(frames) >= 15  # keeps flowing at the decoder's pace (~22 fps here) instead of collapsing
    assert np.all(np.diff([f.t for f in frames]) > 0)


def test_pause_right_after_seek_freezes_the_clock(clip: Path) -> None:
    for _ in range(5):
        source = FileSource(clip, realtime=True)
        source.start()
        assert source.read(timeout=1.0) is not None
        source.seek_to(1.0)
        time.sleep(0.01)
        assert source.toggle_pause() is True
        time.sleep(0.15)
        frozen = source.clock.now()
        time.sleep(0.3)
        assert source.clock.now() == pytest.approx(frozen, abs=1e-6)
        source.stop()


def test_backward_timestamp_jump_starts_a_new_segment(clip: Path) -> None:
    source = FileSource(clip, realtime=False)
    start_with_proxy(source, pts_offset_after=(45, -1.5))  # joined recordings: time jumps back 1.5 s
    frames = drain(source)
    source.stop()
    assert len(frames) == N_FRAMES  # nothing dropped
    assert {f.segment for f in frames} == {0, 1}
    assert np.all(np.diff([f.t for f in frames if f.segment == 1]) > 0)


def test_forward_timestamp_gap_does_not_sleep_through_it(clip: Path) -> None:
    source = FileSource(clip, realtime=True)
    start_with_proxy(source, pts_offset_after=(10, 30.0))  # a 30 s recording gap after frame 10
    start = time.monotonic()
    frames = drain(source, timeout=2.0)
    source.stop()
    assert time.monotonic() - start < 6.0  # ~3 s of video, not 33 s
    assert len(frames) >= 60 and frames[-1].segment == 1


def test_camera_stop_is_fast_and_stalls_are_detected() -> None:
    payload = encode_preamble("Fake Phone", 0x68323634) + encode_session(128, 96)  # then silence
    camera = FakeCamera(payload, "software", hold_s=10.0)
    camera.cfg = camera.cfg.model_copy(update={"stall_timeout_s": 1.0})
    camera.start()
    with pytest.raises(RuntimeError, match="capture failed"):
        camera.read(timeout=4.0)  # the stall watchdog fires after ~1 s without data
    assert isinstance(camera._error, ConnectionError) and "no video" in str(camera._error)
    camera.stop()

    camera = FakeCamera(payload, "software", hold_s=10.0)
    camera.start()
    time.sleep(0.2)
    start = time.monotonic()
    camera.stop()  # blocked in recv: must return promptly
    assert time.monotonic() - start < 1.0
