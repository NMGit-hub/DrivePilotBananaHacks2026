"""Check the wake word and Whisper on REAL voices (the model was trained on synthetic speech only).

    .venv/Scripts/python scripts/wakeword/check_real.py --speaker alex            # record 20 prompted questions
    .venv/Scripts/python scripts/wakeword/check_real.py --speaker sam --count 30 --seconds 5
    .venv/Scripts/python scripts/wakeword/check_real.py --evaluate data/wakeword/real    # score saved recordings

Recording: for each prompt the script prints a question; say "Hey DrivePilot, <question>" in a normal voice,
sitting where the driver sits, with the laptop where it will be in the car (engine running is best). Each take
is saved as data/wakeword/real/<speaker>_<n>.wav (16 kHz, gitignored).

Evaluation (also run after recording): every take goes through the real voice pipeline (wake word, end of
question, Whisper) with the configured threshold, and the wake-word score is tracked so the report shows the
wake rate at other thresholds too, plus what Whisper heard. Pick `voice.wakeword.threshold` from it: the lowest
threshold with ~100% here, as long as the 10-minute false-wake test (scripts/voice_eval.py) stays at 0.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from voice.audio import CHUNK, SAMPLE_RATE  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "wakeword" / "real"
PROMPTS = [
    "who got here first?",
    "can I go?",
    "tell me when it's green",
    "which lane for 520 east?",
    "what does that sign say?",
    "can I park here?",
    "I'm going to Redmond Town Center",
    "did I miss my turn?",
    "which exit do I take?",
    "repeat that",
    "quiet mode",
    "is the road closed ahead?",
    "how far to my turn?",
    "stop",
    "what's the speed limit here?",
]
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def save_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.astype(np.int16).tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(f.readframes(f.getnframes()), np.int16)


def record(speaker: str, count: int, seconds: float, device: str | int | None) -> list[Path]:
    """Prompted takes from the microphone."""
    from voice.audio import MicStream

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mic = MicStream(device)
    mic.start()
    saved = []
    try:
        for n in range(count):
            prompt = PROMPTS[n % len(PROMPTS)]
            print(f'\n[{n + 1}/{count}] Get ready to say:  "Hey DrivePilot, {prompt}"')
            for k in (3, 2, 1):
                print(f"  {k}...", flush=True)
                time.sleep(0.7)
            while mic.read(timeout=0.01) is not None:  # drop what was heard during the countdown
                pass
            print("  >>> SPEAK NOW", flush=True)
            chunks: list[np.ndarray] = []
            needed = round(seconds * SAMPLE_RATE / CHUNK)
            while len(chunks) < needed:
                chunk = mic.read(timeout=1.0)
                if chunk is not None:
                    chunks.append(chunk)
            path = OUT_DIR / f"{speaker}_{n + 1:03d}.wav"
            save_wav(path, np.concatenate(chunks))
            saved.append(path)
            print(f"  saved {path.name}")
    finally:
        mic.stop()
    return saved


def evaluate(paths: list[Path]) -> int:
    """Run every take through the voice pipeline; report wake rates by threshold and transcripts."""
    from voice.pipeline import HeardQuestion, VoicePipeline
    from voice.router import route
    from voice.stt import Transcriber
    from voice.wakeword import WakeWordDetector

    cfg = load_config(REPO_ROOT / "config.yaml")
    stt = Transcriber(cfg.voice.stt, cfg.root)
    stt.warmup()
    wake = WakeWordDetector(cfg.voice.wakeword, cfg.root)
    heard: list[HeardQuestion] = []
    pipeline = VoicePipeline(cfg.voice, wake, stt, heard.append, answer_in_thread=False)
    rows = []
    for path in paths:
        pcm = load_wav(path)
        pcm = np.concatenate((pcm, np.zeros(2 * SAMPLE_RATE + (-len(pcm)) % CHUNK, np.int16)))  # room to end
        before_wakes, before = pipeline.wakes, len(heard)
        best = 0.0
        for k in range(0, len(pcm), CHUNK):
            pipeline.feed(pcm[k : k + CHUNK])
            best = max(best, wake.last_score)
        pipeline.flush(timeout=120.0)
        wake.reset()
        text = heard[before].text if len(heard) > before else None
        routed = route(text) if text else None
        rows.append((path.name, best, pipeline.wakes > before_wakes, text, routed))
        kind = f"{routed.kind.value}/{routed.intent}" if routed else "-"
        print(f"{path.name:24} score {best:.2f}  {'WOKE ' if rows[-1][2] else 'MISSED'}  {text!r} -> {kind}")
    if not rows:
        print("no recordings")
        return 1
    print(f"\n{len(rows)} takes; configured threshold {cfg.voice.wakeword.threshold}")
    for threshold in THRESHOLDS:
        rate = sum(1 for _, best, _, _, _ in rows if best >= threshold) / len(rows)
        mark = "  <- configured" if abs(threshold - cfg.voice.wakeword.threshold) < 1e-9 else ""
        print(f"  wake rate at {threshold:.2f}: {rate:6.1%}{mark}")
    transcribed = [r for r in rows if r[3]]
    print(f"heard a question on {len(transcribed)}/{len(rows)} takes (see the transcripts above)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker", default="me", help="A name for the files (one per person).")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.5, help="Recording length per take.")
    parser.add_argument("--device", default=None, help="Microphone (sounddevice name or index).")
    parser.add_argument("--evaluate", type=Path, default=None, help="Only score the WAVs in this folder.")
    args = parser.parse_args(argv)
    setup_logging("WARNING")
    if args.evaluate is not None:
        return evaluate(sorted(args.evaluate.glob("*.wav")))
    device = int(args.device) if args.device is not None and args.device.isdigit() else args.device
    paths = record(args.speaker, args.count, args.seconds, device)
    print("\nScoring the takes...")
    return evaluate(paths)


if __name__ == "__main__":
    raise SystemExit(main())
