"""Convert downloaded background-noise datasets into 16 kHz mono int16 pools for augmentation/evaluation.

Pools (each is ``data/wakeword/noise/<pool>.npy`` — one long int16 array — plus ``<pool>.json`` with
the start/length/source of every segment so random excerpts never straddle two recordings):

    train_car      QUT-NOISE CAR: real recordings inside a moving car (windows up and down), all sessions
    train_general  DEMAND bus/metro/street-traffic/public-square/cafeteria/living-room/office/field
                   (channels 1 and 9) + AudioSet balanced-train clips (music, speech, vehicles, ...)
    test_car       DEMAND TCAR: a *different* car, microphones and country; held out from training and
                   used only for the car-noise recall conditions and the car-noise false-accept test

Usage (training venv, from the repo root):

    .venv-wakeword/Scripts/python scripts/wakeword/prepare_noise.py
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from common import DOWNLOADS_DIR, NOISE_DIR, SAMPLE_RATE, setup_logging, write_json

log = logging.getLogger("prepare_noise")

DEMAND_TRAIN_ENVS = ["TBUS", "TMETRO", "STRAFFIC", "SPSQUARE", "PCAFETER", "DLIVING", "OOFFICE", "NFIELD"]
DEMAND_TRAIN_CHANNELS = ["ch01", "ch09"]
MIN_RMS_DBFS = -65.0  # drop (near-)silent segments
BLOCK_S = 60


class PoolWriter:
    """Accumulates int16 segments into a single .npy file plus a JSON index."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.raw_path = NOISE_DIR / f"{name}.raw"
        NOISE_DIR.mkdir(parents=True, exist_ok=True)
        self.f = open(self.raw_path, "wb")  # noqa: SIM115 - stays open across add() calls, closed in close()
        self.segments: list[dict] = []
        self.pos = 0
        self.skipped = 0

    def add(self, audio: np.ndarray, source: str, **meta) -> None:
        """Add one float audio segment (16 kHz mono, roughly in [-1, 1])."""
        if len(audio) < SAMPLE_RATE:  # < 1 s is useless for 2 s windows with offsets
            self.skipped += 1
            return
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
        if 20 * np.log10(rms) < MIN_RMS_DBFS:
            self.skipped += 1
            return
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        self.f.write(pcm.tobytes())
        self.segments.append({"start": self.pos, "length": len(pcm), "source": source, **meta})
        self.pos += len(pcm)

    def close(self) -> None:
        """Convert the raw stream to .npy and write the index."""
        self.f.close()
        data = np.memmap(self.raw_path, dtype=np.int16, mode="r")
        out = np.lib.format.open_memmap(NOISE_DIR / f"{self.name}.npy", mode="w+", dtype=np.int16, shape=(len(data),))
        for i in range(0, len(data), 1 << 24):
            out[i : i + (1 << 24)] = data[i : i + (1 << 24)]
        out.flush()
        del out, data
        self.raw_path.unlink()
        hours = self.pos / SAMPLE_RATE / 3600
        write_json(NOISE_DIR / f"{self.name}.json", {"total_hours": round(hours, 3), "segments": self.segments})
        log.info(
            "%s: %d segments, %.2f h (skipped %d silent/short)", self.name, len(self.segments), hours, self.skipped
        )


def read_resampled(path: Path, segment_s: float = 30.0):
    """Yield consecutive ``segment_s`` chunks of a long audio file as 16 kHz mono float32."""
    info = sf.info(str(path))
    rs = (
        soxr.ResampleStream(info.samplerate, SAMPLE_RATE, 1, dtype="float32")
        if info.samplerate != SAMPLE_RATE
        else None
    )
    buf = np.zeros(0, dtype=np.float32)
    seg = int(segment_s * SAMPLE_RATE)
    with sf.SoundFile(str(path)) as f:
        while True:
            block = f.read(int(BLOCK_S * info.samplerate), dtype="float32", always_2d=True)
            last = len(block) == 0 or f.tell() >= info.frames
            mono = block.mean(axis=1) if len(block) else np.zeros(0, dtype=np.float32)
            if rs is not None:
                mono = rs.resample_chunk(mono, last=last)
            buf = np.concatenate([buf, mono])
            while len(buf) >= seg:
                yield buf[:seg]
                buf = buf[seg:]
            if last:
                break
    if len(buf):
        yield buf


def build_train_car() -> None:
    """QUT-NOISE CAR recordings."""
    pool = PoolWriter("train_car")
    files = sorted((DOWNLOADS_DIR / "qut" / "extracted").rglob("*.wav"))
    files = [f for f in files if "CAR" in f.name.upper()]
    if not files:
        raise FileNotFoundError("QUT-NOISE CAR wavs not found; run download_data.py --only qut_car")
    for f in files:
        for i, seg in enumerate(read_resampled(f)):
            pool.add(seg, source=f"QUT-NOISE/{f.name}", index=i)
        log.info("added %s", f.name)
    pool.close()


def build_train_general(audioset_max_clips: int | None = None) -> None:
    """DEMAND (non-car environments) + AudioSet balanced-train clips."""
    import pyarrow.parquet as pq

    pool = PoolWriter("train_general")
    for env in DEMAND_TRAIN_ENVS:
        for ch in DEMAND_TRAIN_CHANNELS:
            path = DOWNLOADS_DIR / "demand" / env / f"{ch}.wav"
            for i, seg in enumerate(read_resampled(path)):
                pool.add(seg, source=f"DEMAND/{env}/{ch}", index=i)
    n_as = 0
    for shard in sorted((DOWNLOADS_DIR / "audioset").glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=32, columns=["video_id", "audio", "human_labels"]):
            for row in batch.to_pylist():
                try:
                    audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
                except Exception as exc:  # noqa: BLE001 - a few clips may be corrupt; skip them
                    log.warning("skip AudioSet %s: %s", row["video_id"], exc)
                    continue
                mono = audio.mean(axis=1)
                if sr != SAMPLE_RATE:
                    mono = soxr.resample(mono, sr, SAMPLE_RATE)
                pool.add(mono, source=f"AudioSet/{row['video_id']}", labels=row["human_labels"])
                n_as += 1
                if audioset_max_clips and n_as >= audioset_max_clips:
                    break
        log.info("AudioSet %s done (%d clips so far)", shard.name, n_as)
    pool.close()


def build_test_car() -> None:
    """DEMAND TCAR, all 16 channels (held out from training)."""
    pool = PoolWriter("test_car")
    for path in sorted((DOWNLOADS_DIR / "demand" / "TCAR").glob("ch*.wav")):
        for i, seg in enumerate(read_resampled(path)):
            pool.add(seg, source=f"DEMAND/TCAR/{path.stem}", index=i)
    pool.close()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["train_car", "train_general", "test_car"],
        default=["train_car", "train_general", "test_car"],
    )
    args = parser.parse_args()
    setup_logging("prepare_noise")
    builders = {"train_car": build_train_car, "train_general": build_train_general, "test_car": build_test_car}
    for name in args.only:
        builders[name]()


if __name__ == "__main__":
    main()
