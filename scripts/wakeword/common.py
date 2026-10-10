"""Shared paths, constants and data-split definitions for the "Hey DrivePilot" wake-word pipeline.

Every other script in ``scripts/wakeword/`` imports from here so that the voice/speaker splits,
directory layout and audio format are defined in exactly one place.

Directory layout (all generated data is gitignored):

    data/wakeword/
        features/        openWakeWord precomputed negative features (ACAV100M, validation set)
        piper_voices/    Piper TTS voices (.onnx + .onnx.json) from HF rhasspy/piper-voices
        downloads/       raw archives (DEMAND, QUT-NOISE, AudioSet parquet, LibriSpeech)
        rirs/{train,test}/            MIT environmental impulse responses (16 kHz), split by file
        noise/{train,test}/{car,general}/   16 kHz mono background-noise segments
        librispeech/     LibriSpeech test-clean + test-other as 16 kHz wav (false-accept test)
        clips/<split>/<kind>/         synthesized 16 kHz mono 16-bit clips + manifest.jsonl
        oww_features/    augmented openWakeWord features (.npy) used for training
    models/wakeword/     final ONNX model, metrics.json, training history
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "wakeword"
FEATURES_DIR = DATA_DIR / "features"
VOICES_DIR = DATA_DIR / "piper_voices"
DOWNLOADS_DIR = DATA_DIR / "downloads"
RIR_DIR = DATA_DIR / "rirs"
NOISE_DIR = DATA_DIR / "noise"
LIBRISPEECH_DIR = DATA_DIR / "librispeech"
CLIPS_DIR = DATA_DIR / "clips"
OWW_FEATURES_DIR = DATA_DIR / "oww_features"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = REPO_ROOT / "models" / "wakeword"

MODEL_NAME = "hey_drivepilot"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms, the openWakeWord frame size

ACAV_FEATURES = FEATURES_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FEATURES = FEATURES_DIR / "validation_set_features.npy"

# ---------------------------------------------------------------------------------------------
# Piper voices and the train / val / test speaker split.
#
# * Single-speaker voices are assigned wholesale to one split. Voices that exist in several
#   qualities (alan, amy, lessac, ryan, ljspeech, cori) are the same person, so all qualities of a
#   held-out person are excluded from training.
# * Multi-speaker voices are split by speaker index with a fixed seed. en_US-libritts-high uses the
#   same speaker order as en_US-libritts_r-medium (index i is the same LibriTTS reader), so both
#   use the same index split and a held-out LibriTTS reader is unseen in both models.
# ---------------------------------------------------------------------------------------------
HF_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

SINGLE_SPEAKER_VOICES: dict[str, list[str]] = {
    "train": [
        "en_US-lessac-medium",
        "en_US-amy-medium",
        "en_US-ryan-medium",
        "en_US-bryce-medium",
        "en_US-danny-low",
        "en_US-hfc_male-medium",
        "en_US-kathleen-low",
        "en_US-kusal-medium",
        "en_US-ljspeech-medium",
        "en_US-mike-medium",
        "en_US-norman-medium",
        "en_US-sam-medium",
        "en_GB-alba-medium",
        "en_GB-cori-medium",
        "en_GB-southern_english_female-low",
    ],
    "val": [
        "en_US-john-medium",
        "en_GB-jenny_dioco-medium",
    ],
    "test": [
        "en_GB-alan-medium",
        "en_US-joe-medium",
        "en_US-kristin-medium",
        "en_GB-northern_english_male-medium",
        "en_US-hfc_female-medium",
        "en_US-reza_ibrahim-medium",
    ],
}

# voice name -> number of speakers
MULTI_SPEAKER_VOICES: dict[str, int] = {
    "en_US-libritts_r-medium": 904,
    "en_US-libritts-high": 904,
    "en_GB-vctk-medium": 109,
    "en_US-l2arctic-medium": 24,
    "en_US-arctic-medium": 18,
    "en_GB-aru-medium": 12,
    "en_GB-semaine-medium": 4,
}
# Voices whose speaker index refers to the same person share a split "group".
SPEAKER_SPLIT_GROUP = {"en_US-libritts-high": "en_US-libritts_r-medium"}
TEST_SPEAKER_FRACTION = 0.12
VAL_SPEAKER_FRACTION = 0.06
SPLIT_SEED = 1234

# Voices excluded from all *positive* sets (still used for negatives). A Whisper small.en check
# (check_tts.py) recognised "drive ... pilot" in only 7.5% of en_US-l2arctic-medium renditions
# ("he tried pilot", "hey trade partner", ...) vs 73-100% for every other voice, so its "positives"
# are mostly mispronunciations that would teach the model to accept near-miss phrases.
POSITIVE_EXCLUDED_VOICES = {"en_US-l2arctic-medium"}

# Relative share of each multi-speaker voice / of the single-speaker pool when sampling clips.
TRAIN_VOICE_WEIGHTS: dict[str, float] = {
    "en_US-libritts_r-medium": 0.46,
    "en_US-libritts-high": 0.10,
    "en_GB-vctk-medium": 0.12,
    "en_US-l2arctic-medium": 0.08,
    "en_US-arctic-medium": 0.05,
    "en_GB-aru-medium": 0.04,
    "en_GB-semaine-medium": 0.02,
    "__single__": 0.13,
}


def all_voice_names() -> list[str]:
    """Return every Piper voice used anywhere in the pipeline."""
    names = list(MULTI_SPEAKER_VOICES)
    for split_voices in SINGLE_SPEAKER_VOICES.values():
        names.extend(split_voices)
    return names


def voice_hf_path(name: str) -> str:
    """Return the path of a voice inside the rhasspy/piper-voices HF repo (without extension)."""
    lang, speaker, quality = name.split("-")
    family = lang.split("_")[0]
    return f"{family}/{lang}/{speaker}/{quality}/{name}"


def speaker_splits() -> dict[str, dict[str, list[int]]]:
    """Deterministically split the speakers of every multi-speaker voice into train/val/test.

    Returns:
        ``{voice: {"train": [...], "val": [...], "test": [...]}}`` with speaker indices.
    """
    groups: dict[str, dict[str, list[int]]] = {}
    for voice, n in MULTI_SPEAKER_VOICES.items():
        if voice in SPEAKER_SPLIT_GROUP:
            continue
        rng = random.Random(f"{SPLIT_SEED}-{voice}")
        ids = list(range(n))
        rng.shuffle(ids)
        n_test = max(1, round(n * TEST_SPEAKER_FRACTION))
        n_val = max(1, round(n * VAL_SPEAKER_FRACTION)) if n >= 8 else 0
        groups[voice] = {
            "test": sorted(ids[:n_test]),
            "val": sorted(ids[n_test : n_test + n_val]),
            "train": sorted(ids[n_test + n_val :]),
        }
    for voice, parent in SPEAKER_SPLIT_GROUP.items():
        groups[voice] = groups[parent]
    return {voice: groups[voice] for voice in MULTI_SPEAKER_VOICES}


# ---------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------
def setup_logging(name: str) -> logging.Logger:
    """Configure root logging to stderr and to ``data/wakeword/logs/<name>.log``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")],
        force=True,
    )
    return logging.getLogger(name)


def stable_seed(*parts: Any) -> int:
    """Return a reproducible 32-bit seed derived from arbitrary values."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
