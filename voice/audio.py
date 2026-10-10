"""Microphone in and speaker out (sounddevice / PortAudio), plus small audio helpers.

- `MicStream`: 16 kHz mono int16 audio in 80 ms chunks (openWakeWord's frame size). The PortAudio callback
  only copies into a bounded queue; if the device can't record at 16 kHz it records at its own rate and the
  chunks are resampled.
- `AudioOutput`: one persistent output stream (opening a stream per sound adds ~50-100 ms on Windows) that
  plays one sound at a time, can be interrupted, and reports when the first sample of a sound reached the
  device (plus the device's own output latency), which is how answer and alert latency are measured.
"""

from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from core.logging_setup import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms at 16 kHz


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling of mono float32 audio."""
    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    ratio = Fraction(target_rate, source_rate).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def to_float(audio: np.ndarray) -> np.ndarray:
    """int16 → float32 in [-1, 1] (float input is returned as float32)."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] → int16 (int16 input is returned unchanged)."""
    if audio.dtype == np.int16:
        return audio
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def tone(rate: int, freqs: tuple[float, ...], durations: tuple[float, ...], volume: float) -> np.ndarray:
    """Consecutive sine tones with 8 ms fades (chimes and earcons), float32."""
    parts = []
    for freq, duration in zip(freqs, durations, strict=True):
        t = np.arange(round(rate * duration)) / rate
        part = np.sin(2 * np.pi * freq * t) * volume
        fade = min(len(part) // 2, round(rate * 0.008))
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            part[:fade] *= ramp
            part[-fade:] *= ramp[::-1]
        parts.append(part)
    return np.concatenate(parts).astype(np.float32)


class MicStream:
    """Microphone audio as 16 kHz mono int16 chunks of `CHUNK` samples."""

    def __init__(self, device: str | int | None = None, max_queue_s: float = 5.0) -> None:
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, round(max_queue_s * SAMPLE_RATE / CHUNK)))
        self._stream: object | None = None
        self._rate = SAMPLE_RATE
        self._pending = np.zeros(0, np.float32)
        self.overflows = 0
        self.dropped = 0

    def start(self) -> None:
        """Open the device (at 16 kHz if it can, otherwise at its default rate with resampling)."""
        import sounddevice as sd

        try:
            sd.check_input_settings(device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            self._rate = SAMPLE_RATE
        except sd.PortAudioError:
            info = sd.query_devices(self.device, "input")
            self._rate = int(info["default_samplerate"])
            log.info("microphone can't record at 16 kHz: resampling", device_rate=self._rate)
        blocksize = round(CHUNK * self._rate / SAMPLE_RATE)
        stream = sd.InputStream(
            device=self.device,
            samplerate=self._rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=self._callback,
        )
        stream.start()
        self._stream = stream
        log.info("microphone open", device=str(self.device), rate=self._rate, latency=round(stream.latency, 3))

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if status:
            self.overflows += 1
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped += 1  # the consumer is stuck; keep the newest audio flowing

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """The next 80 ms chunk (int16, 16 kHz), or None on timeout."""
        while True:
            try:
                block = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None
            if self._rate == SAMPLE_RATE:
                return block
            self._pending = np.concatenate((self._pending, resample(to_float(block), self._rate, SAMPLE_RATE)))
            if len(self._pending) >= CHUNK:
                chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
                return to_int16(chunk)

    def stop(self) -> None:
        """Close the device."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()  # type: ignore[attr-defined]
            stream.close()  # type: ignore[attr-defined]


class AudioOutput:
    """Plays one sound at a time on a persistent output stream; interruptible; measures when sound starts."""

    def __init__(self, device: str | int | None = None, volume: float = 1.0) -> None:
        self.device = device
        self.volume = volume
        self.rate = 48000
        self._lock = threading.Lock()
        self._buffer = np.zeros(0, np.float32)
        self._position = 0
        self._started_at: float | None = None
        self._done = threading.Event()
        self._done.set()
        self._stream: object | None = None
        self.latency_s = 0.0

    def start(self) -> None:
        """Open the device at its default rate."""
        import sounddevice as sd

        info = sd.query_devices(self.device, "output")
        self.rate = int(info["default_samplerate"])
        stream = sd.OutputStream(
            device=self.device,
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )
        stream.start()
        self._stream = stream
        self.latency_s = float(stream.latency)
        log.info("speaker open", device=str(self.device), rate=self.rate, latency=round(self.latency_s, 3))

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        with self._lock:
            remaining = len(self._buffer) - self._position
            n = min(frames, max(remaining, 0))
            if n > 0:
                if self._started_at is None:
                    self._started_at = time.perf_counter()
                outdata[:n, 0] = self._buffer[self._position : self._position + n]
                self._position += n
            outdata[n:, 0] = 0.0
            if n < frames and remaining <= frames and not self._done.is_set():
                self._done.set()

    def play(self, audio: np.ndarray, rate: int, timeout: float | None = None) -> float | None:
        """Play mono audio and wait until it has finished or was interrupted. Returns the time (perf_counter)
        its first sample reached the speaker (incl. device latency), or None if nothing was played."""
        samples = (resample(to_float(audio), rate, self.rate) * self.volume).astype(np.float32)
        with self._lock:
            self._buffer, self._position, self._started_at = samples, 0, None
            self._done.clear()
        if self._stream is None:
            raise RuntimeError("AudioOutput.start() was not called")
        self._done.wait(timeout if timeout is not None else len(samples) / self.rate + 2.0)
        with self._lock:
            started = self._started_at
        return None if started is None else started + self.latency_s

    def stop(self) -> None:
        """Interrupt the current sound now."""
        with self._lock:
            self._position = len(self._buffer)
            self._done.set()

    @property
    def busy(self) -> bool:
        """True while a sound is playing."""
        return not self._done.is_set()

    def close(self) -> None:
        """Close the device."""
        self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()  # type: ignore[attr-defined]
            stream.close()  # type: ignore[attr-defined]
