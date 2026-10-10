"""Speech to text: faster-whisper on the GPU (float16), verified to really run on CUDA.

The question audio starts with a little pre-roll from just before the wake word fired, so transcripts can
begin with the tail of the wake phrase ("...pilot, who got here first?"); `strip_wake_fragment` removes it.
The vocabulary prompt biases Whisper toward driving words (lanes, exits, merging) and the product name.
"""

from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path

import numpy as np

from core.config import SttConfig
from core.logging_setup import get_logger

log = get_logger(__name__)

# The wake phrase, or Whisper's misheard version of its tail ("It's a driver's pilot, who got here first?"),
# within the first few words.
_WAKE_FRAGMENT = re.compile(r"^\W*(?:\S+\s+){0,4}?\S*?(?:pilot|pylot|pilots|pilate)\b[\s,.!?:;-]*", re.IGNORECASE)
_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "thank you for watching", "bye", "."}


def _use_torch_cuda_libraries() -> None:
    """CTranslate2 loads cuBLAS 12 (cublas64_12.dll) at its first GPU call; PyTorch's CUDA 12.6 build ships it
    in torch/lib. Make that folder visible on Windows instead of installing a second copy of CUDA."""
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return
    lib = Path(spec.origin).parent / "lib"
    if lib.is_dir() and str(lib) not in os.environ.get("PATH", ""):
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")  # CTranslate2 uses LoadLibrary


def strip_wake_fragment(text: str) -> str:
    """Remove a leading (partial) wake phrase picked up from the pre-roll."""
    return _WAKE_FRAGMENT.sub("", text.strip(), count=1).strip()


class Transcriber:
    """Whisper speech-to-text for one short question at a time."""

    def __init__(self, cfg: SttConfig, root: Path | None = None) -> None:
        _use_torch_cuda_libraries()
        from faster_whisper import WhisperModel  # deferred: loads CTranslate2 + CUDA

        model = cfg.model
        if root is not None and (root / model).exists():
            model = str(root / model)
        self.cfg = cfg
        start = time.perf_counter()
        self._model = WhisperModel(model, device=cfg.device, compute_type=cfg.compute_type)
        self.device = str(self._model.model.device)
        log.info(
            "whisper loaded",
            model=cfg.model,
            device=self.device,
            compute_type=cfg.compute_type,
            s=round(time.perf_counter() - start, 1),
        )
        if cfg.device == "cuda" and self.device != "cuda":
            raise RuntimeError(f"whisper is running on {self.device}, not the GPU")

    def warmup(self) -> None:
        """One throwaway transcription so the first real question doesn't pay CUDA initialization."""
        self.transcribe(np.zeros(16000, np.float32))

    def transcribe_raw(self, audio: np.ndarray, prompt: str | None = None) -> tuple[str, float]:
        """Whisper's text as is (no vocabulary prompt unless given, nothing removed) and the time it took (ms)."""
        start = time.perf_counter()
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.cfg.beam_size,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, (time.perf_counter() - start) * 1000.0

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Text of a float32 16 kHz utterance (wake fragment removed) and the time it took (ms)."""
        text, ms = self.transcribe_raw(audio, self.cfg.initial_prompt)
        cleaned = strip_wake_fragment(text)
        if cleaned.lower().strip(" .!?") in _HALLUCINATIONS:
            cleaned = ""  # Whisper's stock outputs for silence/noise
        return cleaned, ms
