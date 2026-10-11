"""End of the question: Silero VAD after the wake word, with a hint for speculative transcription.

`StreamingVad` runs Silero VAD v6 (the ONNX model bundled with faster-whisper) on 32 ms chunks, carrying the
recurrent state and a 64-sample context from chunk to chunk (faster-whisper's own wrapper is batch-only).

`Endpointer` collects the question after the wake word fires (plus a little pre-roll, because the wake-word
model fires a few hundred ms after "pilot" and the question may already have started). It reports:
- `speculate`: the speaker has been silent for `speculative_silence_s` after speech: transcribe what we have now,
  so the transcript is ready when the silence is confirmed (it is kept only if no speech follows);
- `end`: `end_silence_s` of silence after speech (PLAN.md: ~0.6 s), or the maximum length;
- `timeout`: nothing was said after the wake word.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from core.config import EndpointConfig
from voice.audio import SAMPLE_RATE, to_float

VAD_CHUNK = 512  # 32 ms
VAD_CONTEXT = 64


class StreamingVad:
    """Silero VAD v6, one 32 ms chunk at a time."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from faster_whisper.vad import get_assets_path

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=options)
        self.reset()

    def reset(self) -> None:
        """Forget the recurrent state (a new utterance)."""
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(VAD_CONTEXT, np.float32)
        self._pending = np.zeros(0, np.float32)

    def process(self, audio: np.ndarray) -> list[float]:
        """Speech probability for every complete 32 ms chunk in `audio` (float or int16 at 16 kHz)."""
        self._pending = np.concatenate((self._pending, to_float(audio)))
        probs = []
        while len(self._pending) >= VAD_CHUNK:
            chunk, self._pending = self._pending[:VAD_CHUNK], self._pending[VAD_CHUNK:]
            x = np.concatenate((self._context, chunk))[None, :]
            out, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
            self._context = chunk[-VAD_CONTEXT:]
            probs.append(float(out[0]))
        return probs


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    """What the endpointer decided after a chunk."""

    kind: str  # "speculate" | "end" | "timeout"
    audio: np.ndarray  # float32 16 kHz: the question so far (pre-roll included)
    speech_end_t: float | None  # stream time the speaker stopped talking (None if they never started)
    reason: str = ""


class Endpointer:
    """Collects one question after the wake word and decides when it is over."""

    def __init__(self, cfg: EndpointConfig, vad: StreamingVad) -> None:
        self.cfg = cfg
        self.vad = vad
        self.active = False

    def start(self, pre_roll: np.ndarray, t: float) -> None:
        """The wake word fired at stream time `t`; `pre_roll` is the audio just before (float or int16)."""
        self.vad.reset()
        self._audio: list[np.ndarray] = [to_float(pre_roll)]
        self._samples = len(pre_roll)
        self._t0 = t - len(pre_roll) / SAMPLE_RATE  # stream time of the first collected sample
        self._vad_t = self._t0  # stream time covered by VAD decisions
        self._heard_speech = False
        self._in_wake_tail = True  # speech right after the wake fires may be just the end of "...pilot"
        self._question_speech = False  # speech that is certainly the question (after the wake phrase)
        self._speech_end: float | None = None  # end of the last speech (stream time)
        self._speculated_at: float | None = None
        self._started_at = t
        self._timeout_s = self.cfg.no_speech_timeout_s
        self.vad.process(pre_roll)  # warm the VAD state on the pre-roll (may contain the end of "pilot")
        self._vad_t = self._t0 + (len(pre_roll) // 512) * 512 / SAMPLE_RATE
        self.active = True

    def start_open(self, pre_roll: np.ndarray, t: float) -> None:
        """Speech started at stream time `t` without the full wake word (`voice.wake_mode` "hey" / "none"):
        everything said is the question (no wake-phrase tail to wait out), and it ends after `end_silence_s`."""
        self.start(pre_roll, t)
        self._in_wake_tail = False

    def resume(self, t: float, wake_t: float, timeout_s: float | None = None) -> None:
        """Listen again from stream time `t`: what ended before was only the wake phrase (its transcript was empty),
        or DrivePilot asked a yes/no question and waits for the reply (no wake word).

        The wake phrase is over, so any speech now is the question and ends after `end_silence_s`; the
        no-speech timeout (`timeout_s`, default `no_speech_timeout_s`) counts from `wake_t`, so a resume never
        extends it.
        """
        self.start(np.zeros(0, np.float32), t)
        self._in_wake_tail = False
        self._started_at = wake_t
        if timeout_s is not None:
            self._timeout_s = timeout_s

    def rewoke(self, t: float) -> None:
        """The wake word fired again while collecting (the driver repeated it): the no-speech timeout now counts
        from `t`. (The wake-tail rule is long over: the wake word can't fire twice within its debounce.)"""
        self._started_at = t

    def feed(self, chunk: np.ndarray, t: float) -> EndpointEvent | None:
        """Add a chunk ending at stream time `t`; returns an event when something should happen."""
        if not self.active:
            return None
        self._audio.append(to_float(chunk))
        self._samples += len(chunk)
        cfg = self.cfg
        for prob in self.vad.process(chunk):
            self._vad_t += VAD_CHUNK / SAMPLE_RATE
            if prob >= cfg.speech_threshold:
                if not (self._in_wake_tail and self._vad_t - self._started_at < cfg.wake_tail_s):
                    self._in_wake_tail = False  # (speech running on without a pause: said in one breath)
                    self._question_speech = True
                self._heard_speech = True
                self._speech_end = None
                self._speculated_at = None  # they kept talking: any speculation is void
            else:
                self._in_wake_tail = False
                if self._heard_speech and self._speech_end is None:
                    self._speech_end = self._vad_t - VAD_CHUNK / SAMPLE_RATE
        if not self._heard_speech:
            if t - self._started_at >= self._timeout_s:
                return self._finish("timeout", "nothing said")
            return None
        if self._samples / SAMPLE_RATE >= cfg.max_utterance_s:
            return self._finish("end", "maximum length")
        if self._speech_end is not None:
            silence = self._vad_t - self._speech_end
            # Only the end of the wake phrase so far ("Hey DrivePilot, ... is it safe"): the pause before the
            # question is longer than a pause within it. (A one-word command said right away, "Hey DrivePilot,
            # stop", ends this way too: speech was already stopped by the wake word itself.)
            needed = cfg.end_silence_s if self._question_speech else cfg.wake_pause_s
            if silence >= needed:
                return self._finish("end", "silence")
            if self._question_speech and silence >= cfg.speculative_silence_s and self._speculated_at is None:
                self._speculated_at = self._speech_end
                return EndpointEvent("speculate", self.audio(), self._speech_end, "short silence")
        return None

    def speculation_still_valid(self) -> bool:
        """True if the last `speculate` covered everything said so far (no speech since)."""
        return self._speculated_at is not None and self._speculated_at == self._speech_end

    def audio(self) -> np.ndarray:
        """The question so far, float32 16 kHz."""
        return np.concatenate(self._audio) if self._audio else np.zeros(0, np.float32)

    def _finish(self, kind: str, reason: str) -> EndpointEvent:
        self.active = False
        return EndpointEvent(kind, self.audio(), self._speech_end if self._heard_speech else None, reason)
