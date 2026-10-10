"""Download every external dataset/model used to train and evaluate the "Hey DrivePilot" wake word.

Run with the training venv from the repo root, e.g.:

    .venv-wakeword/Scripts/python scripts/wakeword/download_data.py --all
    .venv-wakeword/Scripts/python scripts/wakeword/download_data.py --only voices rirs

Sources (see scripts/wakeword/README.md for licenses):
    features    HF davidscripka/openwakeword_features (ACAV100M 2000 h features + validation set)
    voices      HF rhasspy/piper-voices (Piper TTS voices listed in common.py)
    rirs        HF davidscripka/MIT_environmental_impulse_responses (16 kHz)
    demand      Zenodo record 1227121, DEMAND noise database (16 kHz zips)
    qut_car     QUT-NOISE CAR scenario (real car-interior noise, windows up/down)
    audioset    HF agkphysics/AudioSet balanced-train parquet shards (general background audio)
    librispeech OpenSLR 12 test-clean + test-other (real speech for the streaming false-accept test)

Downloads resume after interruption and are verified against published hashes where available.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from common import (
    DOWNLOADS_DIR,
    FEATURES_DIR,
    HF_VOICES_BASE,
    LIBRISPEECH_DIR,
    RIR_DIR,
    VOICES_DIR,
    all_voice_names,
    setup_logging,
    stable_seed,
    voice_hf_path,
)

log = logging.getLogger("download_data")

HF_FEATURES = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
ACAV_SHA256 = "721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f"
FEATURE_FILES = {  # file -> sha256 (git-lfs oid on HF)
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": ACAV_SHA256,
    "validation_set_features.npy": "a56a8a0f8e0efb91900acc6de4c0cdf4c564842e8475a7d49b36c039e17a690f",
}
RIR_API = "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz"
RIR_BASE = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main"
RIR_TEST_FRACTION = 0.1
# Some hosts (QUT) reject the default python-requests User-Agent, so identify the script explicitly.
HEADERS = {"User-Agent": "DrivePilot-wakeword-downloader/1.0"}

DEMAND_BASE = "https://zenodo.org/api/records/1227121/files"
# TCAR (car interior) is held out entirely for the car-noise *test* conditions; the other
# environments are general training background noise (buses, metro, traffic, cafes, homes...).
DEMAND_ENVS = ["TCAR", "TBUS", "TMETRO", "STRAFFIC", "SPSQUARE", "PCAFETER", "DLIVING", "OOFFICE", "NFIELD"]

QUT_CAR_URL = (
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip"
)
QUT_CAR_MD5 = "d680118b4517e1257a9263b99d1ac401"

AUDIOSET_BASE = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train"
AUDIOSET_SHARDS = ["00", "01", "02"]

LIBRISPEECH_BASE = "https://www.openslr.org/resources/12"
LIBRISPEECH_SETS = {  # archive -> md5 (from openslr md5sum.txt)
    "test-clean.tar.gz": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other.tar.gz": "fb5a50374b501bb3bac4815ee91d3135",
}


def file_hash(path: Path, algo: str) -> str:
    """Return the hex digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_size: int | None = None, retries: int = 50) -> Path:
    """Download ``url`` to ``dest`` with resume support. Skips files that are already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and expected_size is None:
        return dest
    for attempt in range(1, retries + 1):
        try:
            pos = tmp.stat().st_size if tmp.exists() else 0
            headers = dict(HEADERS, Range=f"bytes={pos}-") if pos else dict(HEADERS)
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if pos and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except (requests.RequestException, OSError) as exc:
            log.warning("download %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            time.sleep(min(30, 3 * attempt))
    else:
        raise RuntimeError(f"giving up on {url}")
    tmp.replace(dest)
    if expected_size is not None and dest.stat().st_size != expected_size:
        raise RuntimeError(f"{dest} has size {dest.stat().st_size}, expected {expected_size}")
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def get_features() -> None:
    """openWakeWord precomputed negative features (17.3 GB + 185 MB)."""
    for name, sha in FEATURE_FILES.items():
        dest = FEATURES_DIR / name
        download(f"{HF_FEATURES}/{name}", dest)
        digest = file_hash(dest, "sha256")
        if digest != sha:
            raise RuntimeError(f"sha256 mismatch for {dest}: {digest}")
        log.info("verified %s", name)


def get_voices() -> None:
    """Piper TTS voices (.onnx + .onnx.json)."""
    voices_json = download(f"{HF_VOICES_BASE}/voices.json", VOICES_DIR / "voices.json")
    import json

    catalog = json.loads(voices_json.read_text(encoding="utf-8"))
    for name in all_voice_names():
        files = catalog[name]["files"]
        rel = voice_hf_path(name)
        for ext in (".onnx", ".onnx.json"):
            size = files[rel + ext]["size_bytes"]
            download(f"{HF_VOICES_BASE}/{rel}{ext}", VOICES_DIR / f"{name}{ext}", expected_size=size)
        md5 = files[rel + ".onnx"]["md5_digest"]
        if file_hash(VOICES_DIR / f"{name}.onnx", "md5") != md5:
            raise RuntimeError(f"md5 mismatch for voice {name}")
    log.info("all %d voices present and verified", len(all_voice_names()))


def get_rirs() -> None:
    """MIT environmental impulse responses, split by file into train (90%) / test (10%)."""
    listing = requests.get(RIR_API, timeout=60, headers=HEADERS).json()
    wavs = sorted(item["path"] for item in listing if item["path"].endswith(".wav"))
    for rel in wavs:
        name = Path(rel).name
        split = "test" if (stable_seed("rir", name) % 1000) < RIR_TEST_FRACTION * 1000 else "train"
        download(f"{RIR_BASE}/{rel}", RIR_DIR / split / name)
    n_train = len(list((RIR_DIR / "train").glob("*.wav")))
    n_test = len(list((RIR_DIR / "test").glob("*.wav")))
    log.info("RIRs: %d train, %d test", n_train, n_test)


def get_demand() -> None:
    """DEMAND noise environments (16 kHz, 16 channels, ~5 min each)."""
    for env in DEMAND_ENVS:
        zpath = download(f"{DEMAND_BASE}/{env}_16k.zip/content", DOWNLOADS_DIR / "demand" / f"{env}_16k.zip")
        out = DOWNLOADS_DIR / "demand" / env
        if not out.exists():
            with zipfile.ZipFile(zpath) as z:
                z.extractall(DOWNLOADS_DIR / "demand")
        log.info("DEMAND %s ready", env)


def get_qut_car() -> None:
    """QUT-NOISE CAR scenario (real recordings inside a moving car)."""
    zpath = download(QUT_CAR_URL, DOWNLOADS_DIR / "qut" / "qutnoisecar.zip", expected_size=1823708518)
    if file_hash(zpath, "md5") != QUT_CAR_MD5:
        raise RuntimeError("QUT-NOISE CAR md5 mismatch")
    out = DOWNLOADS_DIR / "qut" / "extracted"
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    log.info("QUT-NOISE CAR ready")


def get_audioset() -> None:
    """A few AudioSet balanced-train parquet shards (10 s YouTube clips with labels)."""
    for shard in AUDIOSET_SHARDS:
        download(f"{AUDIOSET_BASE}/{shard}.parquet", DOWNLOADS_DIR / "audioset" / f"bal_train_{shard}.parquet")
    log.info("AudioSet shards ready")


def get_librispeech() -> None:
    """LibriSpeech test-clean and test-other (~10.8 h of read English speech)."""
    for archive, md5 in LIBRISPEECH_SETS.items():
        apath = download(f"{LIBRISPEECH_BASE}/{archive}", DOWNLOADS_DIR / "librispeech" / archive)
        if file_hash(apath, "md5") != md5:
            raise RuntimeError(f"md5 mismatch for {archive}")
        marker = LIBRISPEECH_DIR / f".{archive}.extracted"
        if not marker.exists():
            with tarfile.open(apath) as t:
                t.extractall(LIBRISPEECH_DIR, filter="data")
            marker.touch()
    log.info("LibriSpeech ready")


STEPS = {
    "features": get_features,
    "voices": get_voices,
    "rirs": get_rirs,
    "demand": get_demand,
    "qut_car": get_qut_car,
    "audioset": get_audioset,
    "librispeech": get_librispeech,
}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="download everything")
    parser.add_argument("--only", nargs="+", choices=sorted(STEPS), default=[], help="download only these")
    args = parser.parse_args()
    setup_logging("download_data")
    steps = list(STEPS) if args.all else args.only
    if not steps:
        parser.error("pass --all or --only ...")
    for step in steps:
        log.info("=== %s ===", step)
        STEPS[step]()


if __name__ == "__main__":
    main()
