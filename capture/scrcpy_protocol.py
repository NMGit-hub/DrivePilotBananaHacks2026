"""Parser for the scrcpy 4.x video socket protocol (no I/O, so it is unit-testable).

Byte layout sent by scrcpy-server on the video socket (tunnel_forward=true, send_dummy_byte, send_device_meta,
send_stream_meta and send_frame_meta all enabled):

    1 byte    dummy byte (forward tunnel only: lets the client detect a connection error)
    64 bytes  device name, UTF-8, NUL-padded
    4 bytes   codec id, big-endian ("h264" = 0x68323634); 0 = stream disabled, 1 = configuration error
    then a sequence of 12-byte headers, each either:
      session header  (MSB of byte 0 set): u32 flags | u32 width | u32 height   (new capture session)
      packet header   (MSB clear):         u64 pts_and_flags | u32 size, followed by `size` payload bytes
          pts_and_flags bit 62 = config packet (codec config such as SPS/PPS; no pts)
          bit 61 = key frame; low 61 bits = presentation time in microseconds
Reference: scrcpy v4.1 server/src/main/java/com/genymobile/scrcpy/device/Streamer.java.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

DEVICE_NAME_LENGTH = 64
HEADER_LENGTH = 12
PACKET_FLAG_SESSION = 1 << 63
PACKET_FLAG_CONFIG = 1 << 62
PACKET_FLAG_KEY_FRAME = 1 << 61
PTS_MASK = PACKET_FLAG_KEY_FRAME - 1

CODEC_IDS = {0x68323634: "h264", 0x68323635: "h265", 0x00617631: "av1"}


class ProtocolError(Exception):
    """The byte stream does not follow the scrcpy protocol (or the server reported an error)."""


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """What the server announces before the first packet."""

    device_name: str
    codec: str


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """Start of a capture session (initially and after rotation/resizing)."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class MediaPacket:
    """One encoded packet. Config packets carry codec parameters and must be fed to the decoder too."""

    data: bytes
    pts_us: int | None
    config: bool
    key_frame: bool


ReadExactly = Callable[[int], bytes]


def read_stream_info(read_exactly: ReadExactly, expect_dummy_byte: bool = True) -> StreamInfo:
    """Consume the stream preamble (dummy byte, device name, codec id)."""
    if expect_dummy_byte:
        read_exactly(1)
    raw_name = read_exactly(DEVICE_NAME_LENGTH)
    device_name = raw_name.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    (codec_id,) = struct.unpack(">I", read_exactly(4))
    if codec_id == 0:
        raise ProtocolError("the device disabled the video stream")
    if codec_id == 1:
        raise ProtocolError("scrcpy-server reported a configuration error (check camera id / size / zoom)")
    codec = CODEC_IDS.get(codec_id)
    if codec is None:
        raise ProtocolError(f"unknown codec id 0x{codec_id:08x}")
    return StreamInfo(device_name=device_name, codec=codec)


def read_next(read_exactly: ReadExactly) -> SessionHeader | MediaPacket:
    """Read the next session header or media packet."""
    header = read_exactly(HEADER_LENGTH)
    if header[0] & 0x80:
        _flags, width, height = struct.unpack(">III", header)
        return SessionHeader(width=width, height=height)
    pts_and_flags, size = struct.unpack(">QI", header)
    if size > 64 * 1024 * 1024:
        raise ProtocolError(f"implausible packet size {size}")
    data = read_exactly(size)
    if pts_and_flags & PACKET_FLAG_CONFIG:
        return MediaPacket(data=data, pts_us=None, config=True, key_frame=False)
    return MediaPacket(
        data=data,
        pts_us=pts_and_flags & PTS_MASK,
        config=False,
        key_frame=bool(pts_and_flags & PACKET_FLAG_KEY_FRAME),
    )


def encode_preamble(device_name: str, codec_id: int, dummy_byte: bool = True) -> bytes:
    """Build a stream preamble (used by tests and fakes)."""
    name = device_name.encode("utf-8")[: DEVICE_NAME_LENGTH - 1].ljust(DEVICE_NAME_LENGTH, b"\x00")
    return (b"\x00" if dummy_byte else b"") + name + struct.pack(">I", codec_id)


def encode_session(width: int, height: int) -> bytes:
    """Build a session header (used by tests and fakes)."""
    return struct.pack(">III", PACKET_FLAG_SESSION >> 32, width, height)


def encode_packet(data: bytes, pts_us: int | None, key_frame: bool = False) -> bytes:
    """Build a packet header + payload; `pts_us=None` marks a config packet (used by tests and fakes)."""
    key_flag = PACKET_FLAG_KEY_FRAME if key_frame else 0
    flags = PACKET_FLAG_CONFIG if pts_us is None else (pts_us & PTS_MASK) | key_flag
    return struct.pack(">QI", flags, len(data)) + data
