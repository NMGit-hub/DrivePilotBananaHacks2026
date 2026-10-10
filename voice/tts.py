"""Text to speech: Piper (local, CPU), pre-rendered alert clips, and the chimes.

Piper synthesizes about 10x faster than real time on this laptop (a 3.6 s sentence in ~0.35 s) once warmed up;
the very first synthesis takes seconds, so `warmup()` runs at startup. Alerts must start within 0.3 s, so their
fixed texts are rendered once into WAV files (`ClipCache`) and played straight from memory.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from core.config import TtsConfig
from core.logging_setup import get_logger
from voice.audio import tone

log = get_logger(__name__)


class Tts:
    """Piper voice."""

    def __init__(self, cfg: TtsConfig, root: Path) -> None:
        from piper import PiperVoice, SynthesisConfig  # deferred: loads onnxruntime + espeak data

        path = cfg.voice if cfg.voice.is_absolute() else root / cfg.voice
        if not path.is_file():
            raise FileNotFoundError(f"Piper voice not found: {path} (see README: voice setup)")
        start = time.perf_counter()
        self.voice = PiperVoice.load(str(path))
        self.rate = int(self.voice.config.sample_rate)
        self.name = path.stem
        self.cfg = cfg
        self._syn = SynthesisConfig(length_scale=cfg.length_scale)
        self._lock = threading.Lock()  # one synthesis at a time (shared phonemizer / session)
        log.info("piper voice loaded", voice=self.name, rate=self.rate, s=round(time.perf_counter() - start, 1))

    def synthesize(self, text: str) -> np.ndarray:
        """float32 mono audio at `self.rate`."""
        with self._lock:
            chunks = [chunk.audio_float_array for chunk in self.voice.synthesize(text, self._syn)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

    def warmup(self) -> None:
        """The first synthesis initializes the model (seconds); do it before anyone is waiting."""
        start = time.perf_counter()
        self.synthesize("Ready when you are.")
        log.info("piper warmed up", s=round(time.perf_counter() - start, 1))


class ClipCache:
    """Fixed texts (alerts) rendered once to WAV, then played from memory."""

    def __init__(self, tts: Tts, clips_dir: Path) -> None:
        self.tts = tts
        self.dir = clips_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, np.ndarray] = {}

    def _path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.tts.name}|{self.tts.cfg.length_scale}|{text}".encode()).hexdigest()[:12]
        return self.dir / f"{key}.wav"

    def get(self, text: str) -> np.ndarray:
        """The clip for `text` (rendered and saved on first use)."""
        cached = self._memory.get(text)
        if cached is not None:
            return cached
        path = self._path(text)
        if path.is_file():
            audio, rate = sf.read(path, dtype="float32")
            if rate != self.tts.rate:
                audio = self.tts.synthesize(text)  # a clip from another voice: re-render
        else:
            audio = self.tts.synthesize(text)
            sf.write(path, audio, self.tts.rate)
        self._memory[text] = audio
        return audio

    def cached(self, text: str) -> np.ndarray | None:
        """The clip for `text` if it was rendered already (`get` / `prepare`); never renders."""
        return self._memory.get(text)

    def prepare(self, texts: list[str]) -> None:
        """Render every alert up front (startup)."""
        for text in texts:
            self.get(text)


def alert_chime(rate: int, volume: float) -> np.ndarray:
    """The rising two-tone chime played before every alert (PLAN.md)."""
    return tone(rate, (880.0, 1320.0), (0.07, 0.09), volume)


def listen_tone(rate: int, volume: float) -> np.ndarray:
    """A short soft blip: "I heard the wake word, go ahead"."""
    return tone(rate, (660.0,), (0.06,), volume * 0.6)
