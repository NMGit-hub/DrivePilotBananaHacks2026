"""Synthesize positive and negative "Hey DrivePilot" clips with Piper TTS.

Each clip is synthesized by a randomly chosen voice/speaker *from its split only* (see the speaker
split in common.py), with randomized Piper ``length_scale`` (relative to the voice default),
``noise_scale`` and ``noise_w``. Output is trimmed of leading/trailing silence, resampled to 16 kHz
and written as mono 16-bit WAV, plus a ``manifest.jsonl`` with every generation parameter.

Usage (training venv, from the repo root):

    .venv-wakeword/Scripts/python scripts/wakeword/generate_clips.py --all
    .venv-wakeword/Scripts/python scripts/wakeword/generate_clips.py --set train/positive --n 30000

The script is resumable: clips that already exist are skipped (their jobs are regenerated
deterministically, so a re-run produces the same parameters).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from common import (
    CLIPS_DIR,
    MULTI_SPEAKER_VOICES,
    POSITIVE_EXCLUDED_VOICES,
    SAMPLE_RATE,
    SINGLE_SPEAKER_VOICES,
    TRAIN_VOICE_WEIGHTS,
    VOICES_DIR,
    setup_logging,
    speaker_splits,
    stable_seed,
    write_json,
)

log = logging.getLogger("generate_clips")

# Default number of clips per (split, kind). Training clips are augmented several times later.
DEFAULT_COUNTS: dict[str, int] = {
    "train/positive": 30000,
    "train/negative": 24000,
    "val/positive": 2000,
    "val/negative": 2000,
    "test/positive": 3000,
    "test/positive_command": 600,
    "test/negative": 3000,
}
# Mix of negative text sources.
NEGATIVE_SOURCE_WEIGHTS = {"near_miss": 0.35, "in_car": 0.25, "adversarial": 0.40}

LENGTH_FACTOR_RANGE = (0.75, 1.30)  # multiplies the voice's default length_scale
NOISE_SCALE_RANGE = (0.333, 0.98)
NOISE_W_RANGE = (0.333, 0.98)
TRIM_DB = -45.0  # frames quieter than peak + TRIM_DB are trimmed from the ends
TRIM_MARGIN_S = 0.08


@dataclass
class Job:
    """One clip to synthesize."""

    path: str
    split: str
    kind: str
    text: str
    text_source: str
    voice: str
    speaker_id: int | None
    length_scale: float
    noise_scale: float
    noise_w: float


# ---------------------------------------------------------------------------------------------
# Job planning
# ---------------------------------------------------------------------------------------------
def _voice_pool(split: str, kind: str) -> tuple[list[str], list[float], dict[str, list[int]]]:
    """Voices usable for ``split``/``kind`` with sampling weights, and the allowed speakers per voice."""
    splits = speaker_splits()
    voices, weights, speakers = [], [], {}
    for voice in MULTI_SPEAKER_VOICES:
        ids = splits[voice][split]
        if kind.startswith("positive") and voice in POSITIVE_EXCLUDED_VOICES:
            continue
        if ids:
            voices.append(voice)
            weights.append(TRAIN_VOICE_WEIGHTS[voice])
            speakers[voice] = ids
    singles = SINGLE_SPEAKER_VOICES[split]
    for voice in singles:
        voices.append(voice)
        weights.append(TRAIN_VOICE_WEIGHTS["__single__"] / len(singles))
        speakers[voice] = []
    total = sum(weights)
    return voices, [w / total for w in weights], speakers


def _voice_defaults(voice: str) -> dict:
    cfg = json.loads((VOICES_DIR / f"{voice}.onnx.json").read_text(encoding="utf-8"))
    return cfg.get("inference", {})


def load_text_pools() -> dict:
    """Load (or build and cache) the negative text pools — adversarial generation takes minutes."""
    cache = CLIPS_DIR / "negative_text_pools.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from phrases import negative_text_pools

    pools = negative_text_pools()
    write_json(cache, pools)
    return pools


def plan_jobs(set_name: str, n: int) -> list[Job]:
    """Deterministically create ``n`` jobs for e.g. ``train/positive``."""
    from phrases import POSITIVE_TEXTS, POSITIVE_WITH_COMMAND_TEXTS

    split, kind = set_name.split("/")
    rng = random.Random(stable_seed("jobs", set_name))
    voices, weights, speakers = _voice_pool(split, kind)
    defaults = {v: _voice_defaults(v) for v in voices}
    out_dir = CLIPS_DIR / split / kind

    # val negatives use the training text pools (val speakers are still unseen), so the held-out
    # test phrases never influence training or checkpoint selection
    pools = load_text_pools()["test" if split == "test" else "train"] if kind == "negative" else None
    pos_texts, pos_weights = zip(*POSITIVE_TEXTS, strict=True)

    jobs = []
    for i in range(n):
        voice = rng.choices(voices, weights)[0]
        speaker = rng.choice(speakers[voice]) if speakers[voice] else None
        if kind == "positive":
            text, source = rng.choices(pos_texts, pos_weights)[0], "positive"
        elif kind == "positive_command":
            text, source = rng.choice(POSITIVE_WITH_COMMAND_TEXTS), "positive_command"
        else:
            source = rng.choices(list(NEGATIVE_SOURCE_WEIGHTS), list(NEGATIVE_SOURCE_WEIGHTS.values()))[0]
            text = rng.choice(pools[source])
        base_ls = float(defaults[voice].get("length_scale", 1.0))
        jobs.append(
            Job(
                path=str(out_dir / f"{split}_{kind}_{i:06d}.wav"),
                split=split,
                kind=kind,
                text=text,
                text_source=source,
                voice=voice,
                speaker_id=speaker,
                length_scale=round(base_ls * rng.uniform(*LENGTH_FACTOR_RANGE), 3),
                noise_scale=round(rng.uniform(*NOISE_SCALE_RANGE), 3),
                noise_w=round(rng.uniform(*NOISE_W_RANGE), 3),
            )
        )
    return jobs


# ---------------------------------------------------------------------------------------------
# Synthesis worker
# ---------------------------------------------------------------------------------------------
_VOICE_CACHE: OrderedDict[str, object] = OrderedDict()
_THREADS = 2


def _init_worker(threads: int) -> None:
    global _THREADS
    _THREADS = threads


def _get_voice(name: str):
    import onnxruntime as ort
    from piper import PiperVoice
    from piper.config import PiperConfig

    if name in _VOICE_CACHE:
        _VOICE_CACHE.move_to_end(name)
        return _VOICE_CACHE[name]
    while len(_VOICE_CACHE) >= 1:  # one loaded voice per worker keeps RAM at ~270 MB/worker
        _VOICE_CACHE.popitem(last=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _THREADS
    opts.inter_op_num_threads = 1
    cfg = json.loads((VOICES_DIR / f"{name}.onnx.json").read_text(encoding="utf-8"))
    voice = PiperVoice(
        config=PiperConfig.from_dict(cfg),
        session=ort.InferenceSession(
            str(VOICES_DIR / f"{name}.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        ),
    )
    _VOICE_CACHE[name] = voice
    return voice


def trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading/trailing low-energy audio (relative to the loudest 10 ms frame)."""
    frame = int(0.01 * sr)
    n = len(audio) // frame
    if n < 3:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    active = np.where(db > TRIM_DB)[0]
    margin = int(TRIM_MARGIN_S * sr)
    start = max(0, active[0] * frame - margin)
    end = min(len(audio), (active[-1] + 1) * frame + margin)
    return audio[start:end]


def synthesize(job: Job) -> dict:
    """Synthesize one job, write the WAV and return its manifest row."""
    from math import gcd

    import soundfile as sf
    from piper.config import SynthesisConfig
    from scipy.signal import resample_poly

    voice = _get_voice(job.voice)
    syn = SynthesisConfig(
        speaker_id=job.speaker_id,
        length_scale=job.length_scale,
        noise_scale=job.noise_scale,
        noise_w_scale=job.noise_w,
        normalize_audio=True,
    )
    chunks = [c.audio_float_array for c in voice.synthesize(job.text, syn_config=syn)]
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sr = voice.config.sample_rate
    audio = trim_silence(audio.astype(np.float64), sr)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g)
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.9
    Path(job.path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(job.path, (audio * 32767).astype(np.int16), SAMPLE_RATE, subtype="PCM_16")
    row = asdict(job)
    row["path"] = str(Path(job.path).relative_to(CLIPS_DIR)).replace("\\", "/")
    row["duration_s"] = round(len(audio) / SAMPLE_RATE, 3)
    return row


def _run_chunk(jobs: list[Job]) -> list[dict]:
    return [synthesize(j) for j in jobs]


def generate(set_name: str, n: int, workers: int, threads: int) -> None:
    """Plan, synthesize and write the manifest for one clip set."""
    jobs = plan_jobs(set_name, n)
    out_dir = CLIPS_DIR / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    done: dict[str, dict] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (CLIPS_DIR / row["path"]).exists():
                done[row["path"]] = row
    todo = [j for j in jobs if str(Path(j.path).relative_to(CLIPS_DIR)).replace("\\", "/") not in done]
    log.info("%s: %d planned, %d already done, %d to synthesize", set_name, len(jobs), len(done), len(todo))
    # group by voice so each worker reuses a loaded voice for a whole chunk
    todo.sort(key=lambda j: (j.voice, j.path))
    chunks = [todo[i : i + 100] for i in range(0, len(todo), 100)]
    t0 = time.time()
    with (
        open(manifest_path, "a", encoding="utf-8") as mf,
        mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(threads,)) as pool,
    ):
        n_done = 0
        for rows in pool.imap_unordered(_run_chunk, chunks):
            for row in rows:
                mf.write(json.dumps(row) + "\n")
            mf.flush()
            n_done += len(rows)
            if n_done % 2000 < 100:
                rate = n_done / (time.time() - t0)
                log.info("%s: %d/%d clips (%.1f clips/s)", set_name, n_done, len(todo), rate)
    log.info("%s: finished in %.1f min", set_name, (time.time() - t0) / 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="generate every set with default counts")
    parser.add_argument("--set", dest="sets", nargs="+", default=[], choices=sorted(DEFAULT_COUNTS))
    parser.add_argument("--n", type=int, default=None, help="override the clip count (with --set)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1, help="onnxruntime threads per worker")
    args = parser.parse_args()
    setup_logging("generate_clips")
    write_json(CLIPS_DIR / "speaker_splits.json", speaker_splits())
    sets = list(DEFAULT_COUNTS) if args.all else args.sets
    if not sets:
        parser.error("pass --all or --set ...")
    load_text_pools()
    for s in sets:
        generate(s, args.n or DEFAULT_COUNTS[s], args.workers, args.threads)


if __name__ == "__main__":
    main()
