"""Augment the synthesized clips and compute openWakeWord features for training.

This mirrors ``openwakeword.data.augment_clips`` + ``openwakeword.utils.compute_features_from_generator``
(the official training pipeline) with these deliberate changes:

* Background noise is mixed **per example** (not per batch) and the SNR is defined against the RMS of
  the speech itself, not of the whole zero-padded window, so it matches the evaluation's SNR definition.
* A dedicated in-car noise stage: real car-interior recordings (QUT-NOISE CAR) at 0-20 dB SNR, general
  background (DEMAND environments + AudioSet) at -3-15 dB SNR, or both (car + radio/passengers).
* The room impulse response is applied to the speech *before* noise is added (noise recordings already
  contain their own acoustics), with p = 0.5 as in the original.
* Gain range -24..0 dB (original -18..0) to cover quieter far-field laptop-mic input; output is
  peak-limited instead of overflowing int16.
* Noise-only windows (no speech) are added as an extra negative set.
* Features are computed on the GPU with TF32 disabled so they match CPU inference to ~1e-4.

The kept parts: right-aligned placement with 0-200 ms end jitter in a fixed window (2 s), seven-band EQ
and tanh distortion (p = 0.25 each), pitch shift +-3 semitones, band-stop filter, coloured noise
10-30 dB (p = 0.25 each), MIT impulse responses, and the frozen melspectrogram + Google speech-embedding
feature models shipped with openWakeWord.

Usage (training venv, from the repo root):

    .venv-wakeword/Scripts/python scripts/wakeword/augment_features.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from common import (
    CLIPS_DIR,
    NOISE_DIR,
    OWW_FEATURES_DIR,
    POSITIVE_EXCLUDED_VOICES,
    RIR_DIR,
    SAMPLE_RATE,
    setup_logging,
    stable_seed,
    write_json,
)

log = logging.getLogger("augment_features")

AUGMENTATION_ROUNDS = 2
N_BACKGROUND_ONLY = 20000
BATCH_SIZE = 128  # clips per feature-extraction batch
# torch-audiomentations runs on sub-batches of 16 (openWakeWord's augmentation_batch_size). Larger
# sub-batches make PitchShift's STFT (hop = 7 samples) exceed the 6 GB GPU, which on Windows spills
# into host RAM and exhausted the machine's commit limit.
GPU_SUBBATCH = 16
END_JITTER_S = 0.2
P_RIR = 0.5
# noise mode -> probability; SNR ranges in dB (speech RMS vs noise RMS)
NOISE_MODES = {"car": 0.40, "general": 0.30, "car+general": 0.10, "none": 0.20}
CAR_SNR = (0.0, 20.0)
GENERAL_SNR = (-3.0, 15.0)
MIXED_SNR = (5.0, 20.0)
GAIN_DB = (-24.0, 0.0)
# Positives longer than this (~2% of clips: long pauses after "hey," or TTS glitches) are not used for
# training: in a 2 s window their start ("hey") would be cut off, which would look like the negative
# "drive pilot".
MAX_POSITIVE_S = 2.2


# ---------------------------------------------------------------------------------------------
# Building blocks (also imported by evaluate.py and the unit tests)
# ---------------------------------------------------------------------------------------------
class NoisePool:
    """Random excerpts from a pool written by prepare_noise.py."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.data = np.load(NOISE_DIR / f"{name}.npy", mmap_mode="r")
        index = json.loads((NOISE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        self.starts = np.array([s["start"] for s in index["segments"]])
        self.lengths = np.array([s["length"] for s in index["segments"]])
        self.weights = self.lengths / self.lengths.sum()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``n`` samples (float32 in [-1, 1]) from one randomly chosen recording."""
        k = rng.choice(len(self.starts), p=self.weights)
        start, length = self.starts[k], self.lengths[k]
        if length >= n:
            off = start + rng.integers(0, length - n + 1)
            seg = np.asarray(self.data[off : off + n])
        else:
            seg = np.resize(np.asarray(self.data[start : start + length]), n)
        return seg.astype(np.float32) / 32768.0


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude."""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, speech_span: tuple[int, int]) -> np.ndarray:
    """Add ``noise`` to ``speech`` so that RMS(speech span) / RMS(noise) equals ``snr_db``."""
    s0, s1 = speech_span
    target_noise_rms = rms(speech[s0:s1]) / (10 ** (snr_db / 20))
    return speech + noise * (target_noise_rms / rms(noise))


def load_rirs(split: str) -> list[np.ndarray]:
    """Load MIT impulse responses, peak-normalised and aligned to the direct path."""
    import soundfile as sf

    rirs = []
    for p in sorted((RIR_DIR / split).glob("*.wav")):
        rir, sr = sf.read(str(p), dtype="float32", always_2d=True)
        rir = rir[:, 0]
        if sr != SAMPLE_RATE:
            raise ValueError(f"{p} is {sr} Hz")
        peak = int(np.argmax(np.abs(rir)))
        rir = rir[peak:]
        rirs.append(rir / (np.abs(rir).max() + 1e-9))
    return rirs


def apply_rir(x: np.ndarray, rir: np.ndarray, max_tail_s: float = 0.3) -> np.ndarray:
    """Convolve with an impulse response, keep up to ``max_tail_s`` of tail, match mean abs amplitude."""
    from scipy.signal import fftconvolve

    y = fftconvolve(x, rir)[: len(x) + int(max_tail_s * SAMPLE_RATE)]
    return (y * (np.mean(np.abs(x)) / (np.mean(np.abs(y)) + 1e-9))).astype(np.float32)


def place_in_window(
    x: np.ndarray, total: int, rng: np.random.Generator, keep_end: bool
) -> tuple[np.ndarray, tuple[int, int]]:
    """Right-align ``x`` in a ``total``-sample window with 0-200 ms end jitter (as create_fixed_size_clip).

    Clips longer than the window keep their end (positives: the end of the phrase must be visible) or a
    random excerpt (negatives).
    """
    jitter = int(rng.uniform(0, END_JITTER_S) * SAMPLE_RATE)
    room = total - jitter
    if len(x) > room:
        if keep_end:
            x = x[-room:]
        else:
            off = rng.integers(0, len(x) - room + 1)
            x = x[off : off + room]
    out = np.zeros(total, dtype=np.float32)
    start = room - len(x)
    out[start:room] = x
    return out, (start, room)


class Augmenter:
    """Per-example CPU augmentation + per-batch GPU augmentation, producing int16 windows."""

    def __init__(self, total_length: int, seed: int, device: str = "cuda") -> None:
        import audiomentations
        import torch
        import torch_audiomentations as ta

        self.total = total_length
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = device
        self.car = NoisePool("train_car")
        self.general = NoisePool("train_general")
        self.rirs = load_rirs("train")
        self.augment1 = audiomentations.Compose(
            [
                audiomentations.SevenBandParametricEQ(min_gain_db=-6, max_gain_db=6, p=0.25),
                audiomentations.TanhDistortion(min_distortion=0.0001, max_distortion=0.10, p=0.25),
            ]
        )
        self.augment2 = ta.Compose(
            [
                ta.PitchShift(
                    min_transpose_semitones=-3,
                    max_transpose_semitones=3,
                    p=0.25,
                    sample_rate=SAMPLE_RATE,
                    mode="per_batch",
                    output_type="dict",
                ),
                ta.BandStopFilter(p=0.25, mode="per_batch", sample_rate=SAMPLE_RATE, output_type="dict"),
                ta.AddColoredNoise(
                    min_snr_in_db=10,
                    max_snr_in_db=30,
                    min_f_decay=-1,
                    max_f_decay=2,
                    p=0.25,
                    mode="per_batch",
                    sample_rate=SAMPLE_RATE,
                    output_type="dict",
                ),
                ta.Gain(
                    min_gain_in_db=GAIN_DB[0],
                    max_gain_in_db=GAIN_DB[1],
                    p=1.0,
                    sample_rate=SAMPLE_RATE,
                    output_type="dict",
                ),
            ],
            output_type="dict",
        )

    def _add_noise(self, canvas: np.ndarray, span: tuple[int, int]) -> np.ndarray:
        mode = self.rng.choice(list(NOISE_MODES), p=list(NOISE_MODES.values()))
        if mode == "car":
            return mix_at_snr(canvas, self.car.sample(self.total, self.rng), self.rng.uniform(*CAR_SNR), span)
        if mode == "general":
            return mix_at_snr(canvas, self.general.sample(self.total, self.rng), self.rng.uniform(*GENERAL_SNR), span)
        if mode == "car+general":
            noise_c = self.car.sample(self.total, self.rng)
            noise_g = self.general.sample(self.total, self.rng)
            out = mix_at_snr(canvas, noise_c, self.rng.uniform(*MIXED_SNR), span)
            return out + (mix_at_snr(canvas, noise_g, self.rng.uniform(*MIXED_SNR), span) - canvas)
        return canvas

    def _gpu_stage(self, batch: list[np.ndarray]) -> np.ndarray:
        import torch

        parts = []
        for i in range(0, len(batch), GPU_SUBBATCH):
            x = torch.from_numpy(np.stack(batch[i : i + GPU_SUBBATCH])).unsqueeze(1).to(self.device)
            with torch.no_grad():
                parts.append(self.augment2(samples=x, sample_rate=SAMPLE_RATE).samples.squeeze(1).cpu().numpy())
        y = np.concatenate(parts)
        peaks = np.abs(y).max(axis=1, keepdims=True)
        y = np.where(peaks > 0.99, y * (0.99 / np.maximum(peaks, 1e-9)), y)
        return (y * 32767).astype(np.int16)

    def speech_batch(self, clips: list[np.ndarray], keep_end: bool) -> np.ndarray:
        """Augment a batch of speech clips (float32, 16 kHz)."""
        out = []
        for x in clips:
            if self.rng.random() < P_RIR:
                x = apply_rir(x, self.rirs[self.rng.integers(len(self.rirs))])
            canvas, span = place_in_window(x, self.total, self.rng, keep_end)
            canvas = self.augment1(samples=canvas, sample_rate=SAMPLE_RATE)
            out.append(self._add_noise(canvas, span).astype(np.float32))
        return self._gpu_stage(out)

    def noise_only_batch(self, n: int) -> np.ndarray:
        """Background-only windows (car, general, both, or coloured noise) at random levels."""
        out = []
        for _ in range(n):
            r = self.rng.random()
            if r < 0.45:
                x = self.car.sample(self.total, self.rng)
            elif r < 0.85:
                x = self.general.sample(self.total, self.rng)
            else:
                x = self.car.sample(self.total, self.rng) + self.general.sample(
                    self.total, self.rng
                ) * self.rng.uniform(0.1, 1.0)
            x = x / (np.abs(x).max() + 1e-9) * self.rng.uniform(0.05, 0.9)
            out.append(x.astype(np.float32))
        return self._gpu_stage(out)


def make_feature_extractor():
    """openWakeWord AudioFeatures on the GPU with TF32 disabled (matches CPU features to ~1e-4)."""
    import onnxruntime as ort
    import torch  # noqa: F401  (loads the CUDA/cuDNN DLLs that onnxruntime-gpu needs)
    from openwakeword.utils import AudioFeatures

    feats = AudioFeatures(device="gpu")
    model_dir = Path(__import__("openwakeword").__file__).parent / "resources" / "models"
    prov = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    feats.melspec_model = ort.InferenceSession(str(model_dir / "melspectrogram.onnx"), providers=prov)
    feats.embedding_model = ort.InferenceSession(str(model_dir / "embedding_model.onnx"), providers=prov)
    feats.melspec_model_predict = lambda x: feats.melspec_model.run(None, {"input": x})
    feats.embedding_model_predict = lambda x: feats.embedding_model.run(None, {"input_1": x})[0].squeeze()
    feats.onnx_execution_provider = feats.melspec_model.get_providers()[0]
    if "CUDA" not in feats.onnx_execution_provider:
        log.warning("feature extraction is running on %s, not CUDA", feats.onnx_execution_provider)
    return feats


# ---------------------------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------------------------
def read_manifest(set_name: str) -> list[dict]:
    """Rows of ``clips/<set>/manifest.jsonl`` whose WAV exists, deduplicated, sorted by path.

    Positive sets skip voices in ``POSITIVE_EXCLUDED_VOICES``.
    """
    rows = {}
    for line in (CLIPS_DIR / set_name / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "positive" in set_name and row["voice"] in POSITIVE_EXCLUDED_VOICES:
            continue  # see common.POSITIVE_EXCLUDED_VOICES
        if (CLIPS_DIR / row["path"]).exists():
            rows[row["path"]] = row
    return [rows[k] for k in sorted(rows)]


def load_clip(rel_path: str) -> np.ndarray:
    """Load a 16-bit clip as float32."""
    import soundfile as sf

    x, sr = sf.read(str(CLIPS_DIR / rel_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{rel_path}: {sr} Hz")
    return x


def compute_total_length() -> int:
    """openWakeWord's rule (train.py): median positive duration rounded + 750 ms, min/snap to 2 s."""
    durs = [r["duration_s"] for r in read_manifest("train/positive")]
    total = int(round(np.median(durs) * SAMPLE_RATE / 1000) * 1000) + 12000
    if total < 32000 or abs(total - 32000) <= 4000:
        total = 32000
    return total


def build_set(name: str, total_length: int, feats, overwrite: bool) -> None:
    """Augment one clip set (or noise-only windows) and write ``oww_features/<name>.npy``."""
    out_path = OWW_FEATURES_DIR / f"{name}.npy"
    if out_path.exists() and not overwrite:
        log.info("%s exists, skipping (use --overwrite)", out_path.name)
        return
    spec = {
        "positive_train": ("train/positive", AUGMENTATION_ROUNDS, True),
        "negative_train": ("train/negative", AUGMENTATION_ROUNDS, False),
        "positive_val": ("val/positive", 1, True),
        "negative_val": ("val/negative", 1, False),
        "background_train": (None, 1, False),
    }[name]
    clip_set, rounds, keep_end = spec
    aug = Augmenter(total_length, seed=stable_seed("augment", name))
    if clip_set is None:
        n_total = N_BACKGROUND_ONLY
        paths = None
    else:
        rows = read_manifest(clip_set)
        if keep_end:
            n_before = len(rows)
            rows = [r for r in rows if r["duration_s"] <= MAX_POSITIVE_S]
            log.info(
                "%s: dropped %d/%d positives longer than %.1f s", name, n_before - len(rows), n_before, MAX_POSITIVE_S
            )
        paths = []
        for r in range(rounds):
            order = np.random.default_rng(stable_seed(name, r)).permutation(len(rows))
            paths.extend(rows[i]["path"] for i in order)
        n_total = len(paths)

    shape = feats.get_embedding_shape(total_length / SAMPLE_RATE)
    OWW_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".partial.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n_total, *shape))
    t0 = time.time()
    for i in range(0, n_total, BATCH_SIZE):
        n = min(BATCH_SIZE, n_total - i)
        if paths is None:
            audio = aug.noise_only_batch(n)
        else:
            audio = aug.speech_batch([load_clip(p) for p in paths[i : i + n]], keep_end=keep_end)
        out[i : i + n] = feats.embed_clips(audio, batch_size=n)
        if (i // BATCH_SIZE) % 50 == 0:
            rate = (i + n) / (time.time() - t0)
            log.info("%s: %d/%d (%.0f clips/s)", name, i + n, n_total, rate)
    out.flush()
    del out
    os.replace(tmp_path, out_path)
    log.info("%s: wrote %s with %d rows in %.1f min", name, out_path.name, n_total, (time.time() - t0) / 60)


def save_examples(total_length: int, n: int = 6) -> None:
    """Write a few augmented examples to oww_features/examples/ for listening checks."""
    import soundfile as sf

    ex_dir = OWW_FEATURES_DIR / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    aug = Augmenter(total_length, seed=7)
    rows = read_manifest("train/positive")[:n]
    audio = aug.speech_batch([load_clip(r["path"]) for r in rows], keep_end=True)
    for r, a in zip(rows, audio, strict=True):
        sf.write(str(ex_dir / Path(r["path"]).name), a, SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(ex_dir / "noise_only_0.wav"), aug.noise_only_batch(1)[0], SAMPLE_RATE, subtype="PCM_16")


SETS = ["positive_train", "negative_train", "background_train", "positive_val", "negative_val"]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", nargs="+", choices=SETS, default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--examples", action="store_true", help="also write a few augmented example WAVs")
    args = parser.parse_args()
    setup_logging("augment_features")
    total_length = compute_total_length()
    log.info("window length: %d samples (%.2f s)", total_length, total_length / SAMPLE_RATE)
    feats = make_feature_extractor()
    write_json(
        OWW_FEATURES_DIR / "feature_config.json",
        {
            "total_length": total_length,
            "input_shape": list(feats.get_embedding_shape(total_length / SAMPLE_RATE)),
            "augmentation_rounds": AUGMENTATION_ROUNDS,
            "noise_modes": NOISE_MODES,
            "car_snr_db": CAR_SNR,
            "general_snr_db": GENERAL_SNR,
            "mixed_snr_db": MIXED_SNR,
            "gain_db": GAIN_DB,
            "p_rir": P_RIR,
            "max_positive_s": MAX_POSITIVE_S,
        },
    )
    if args.examples:
        save_examples(total_length)
    for name in SETS if args.all else args.only:
        build_set(name, total_length, feats, args.overwrite)


if __name__ == "__main__":
    main()
