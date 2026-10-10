"""Wake word: the custom openWakeWord "Hey DrivePilot" model (trained in scripts/wakeword/), on the CPU.

Fed 80 ms int16 chunks at 16 kHz. A detection is a score at or above `threshold`, followed by `debounce_s` of
deafness (openWakeWord 0.6.0's own `patience` option never fires once a frame is suppressed; see
scripts/wakeword/README.md, so debouncing is done here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import WakeWordConfig
from core.logging_setup import get_logger

log = get_logger(__name__)


class WakeWordDetector:
    """Streaming "Hey DrivePilot" detector."""

    def __init__(self, cfg: WakeWordConfig, root: Path) -> None:
        from openwakeword.model import Model  # deferred: loads onnxruntime sessions

        path = cfg.model if cfg.model.is_absolute() else root / cfg.model
        if not path.is_file():
            raise FileNotFoundError(f"wake-word model not found: {path} (see scripts/wakeword/README.md)")
        self.cfg = cfg
        self.name = path.stem
        self._model = Model(wakeword_models=[str(path)], inference_framework="onnx")
        self._last_trigger = -1e9
        self.last_score = 0.0

    def process(self, chunk: np.ndarray, t: float) -> float | None:
        """Feed one 80 ms int16 chunk ending at stream time `t` (s). Returns the score if the wake word fired."""
        scores = self._model.predict(chunk)
        score = float(scores[self.name])
        self.last_score = score
        if score >= self.cfg.threshold and t - self._last_trigger >= self.cfg.debounce_s:
            self._last_trigger = t
            log.info("wake word", score=round(score, 3), t=round(t, 2))
            return score
        return None

    def reset(self) -> None:
        """Clear the model's audio buffer (e.g. after our own speech or a long pause)."""
        self._model.reset()
