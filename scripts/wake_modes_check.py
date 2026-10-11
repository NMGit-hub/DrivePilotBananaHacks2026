"""The wake modes (`voice.wake_mode`) with the real models: Silero VAD, the "Hey DrivePilot" model and Whisper on the
GPU, fed synthetic speech from six Piper test voices (the wake-word evaluation's, scripts/voice_eval.py), clean and
mixed with car noise. Each case says what should happen: heard (and as what) or ignored.

    .venv/Scripts/python scripts/wake_modes_check.py [--data DIR] [--snr 10] [--noise-minutes 2]

Also feeds car noise alone to the "none" mode (no wake word): nothing should be answered. Needs the test voices and
car noise from scripts/wakeword (under data/wakeword/).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from voice.audio import CHUNK, SAMPLE_RATE, resample, to_int16  # noqa: E402
from voice.pipeline import HeardQuestion, VoicePipeline  # noqa: E402
from voice.router import normalize  # noqa: E402
from voice.stt import Transcriber  # noqa: E402
from voice.wakeword import WakeWordDetector  # noqa: E402

TEST_VOICES = [
    "en_GB-alan-medium",
    "en_US-joe-medium",
    "en_US-kristin-medium",
    "en_GB-northern_english_male-medium",
    "en_US-hfc_female-medium",
    "en_US-reza_ibrahim-medium",
]


@dataclass(frozen=True)
class Case:
    mode: str
    said: tuple[str, ...]  # phrases, with a 1 s pause between them
    expect: str | None  # the normalized question that should be heard, or None: ignored


CASES = [
    Case("hey", ("Hey, what does that sign say?",), "what does that sign say"),
    Case("hey", ("Hey, who got here first?",), "who got here first"),
    Case("hey", ("Hey.", "Is it my turn?"), "is it my turn"),
    Case("hey", ("Hey Drive Pilot, can I go?",), "can i go"),
    Case("hey", ("What does that sign say?",), None),
    Case("hey", ("We're almost there, I think.",), None),
    Case("hey", ("Hey, I'm so hungry.",), None),
    Case("hey", ("Hey there, which lane should I be in?",), "which lane should i be in"),
    Case("none", ("What does that sign say?",), "what does that sign say"),
    Case("none", ("Is it my turn?",), "is it my turn"),
    Case("none", ("Tell me when the light is green.",), "tell me when the light is green"),
    Case("none", ("I love this song.",), None),
    Case("none", ("We're almost there, I think.",), None),
    Case("none", ("We're going to grandma's house later.",), None),
    Case("none", ("Never mind.",), None),
    Case("hey_drivepilot", ("What does that sign say?",), None),
    Case("hey_drivepilot", ("Hey Drive Pilot, what does that sign say?",), "what does that sign say"),
]


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Speech plus noise scaled to the SNR (over the speech's active part)."""
    active = speech[np.abs(speech) > 0.01]
    p_speech = float(np.mean(active**2)) if active.size else 1e-6
    p_noise = float(np.mean(noise**2)) + 1e-12
    return speech + noise * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data", help="folder with wakeword/ test data")
    parser.add_argument("--snr", type=float, default=10.0, help="car-noise SNR (dB) for the noisy condition")
    parser.add_argument("--noise-minutes", type=float, default=2.0, help='car noise alone, "none" mode')
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    cfg = load_config(REPO_ROOT / "config.yaml")
    from piper import PiperVoice  # deferred: heavy

    voices = [PiperVoice.load(str(args.data / "wakeword" / "piper_voices" / f"{n}.onnx")) for n in TEST_VOICES]
    noise = np.load(args.data / "wakeword" / "noise" / "test_car.npy", mmap_mode="r")
    stt = Transcriber(cfg.voice.stt, cfg.root)
    stt.warmup()
    heard: list[HeardQuestion] = []
    pipe = VoicePipeline(
        cfg.voice, WakeWordDetector(cfg.voice.wakeword, cfg.root), stt, heard.append, answer_in_thread=False
    )
    rng = np.random.default_rng(7)

    def speak(voice: object, text: str) -> np.ndarray:
        chunks = [c.audio_float_array for c in voice.synthesize(text)]  # type: ignore[attr-defined]
        return resample(np.concatenate(chunks), voice.config.sample_rate, SAMPLE_RATE)  # type: ignore[attr-defined]

    def feed(stream: np.ndarray) -> None:
        pcm = to_int16(stream)
        pcm = np.concatenate((pcm, np.zeros((-len(pcm)) % CHUNK, np.int16)))
        for k in range(0, len(pcm), CHUNK):
            pipe.feed(pcm[k : k + CHUNK])
        pipe.flush(timeout=120.0)
        while pipe.listening:
            pipe.feed(np.zeros(CHUNK, np.int16))
            pipe.flush(timeout=120.0)

    failures = 0
    total = 0
    for condition in ("clean", f"car {args.snr:g} dB"):
        print(f"\n== {condition}")
        for case in CASES:
            pipe.set_mode(case.mode)
            outcomes = []
            for v, voice in enumerate(voices):
                parts: list[np.ndarray] = [np.zeros(SAMPLE_RATE, np.float32)]
                for i, phrase in enumerate(case.said):
                    if i:
                        parts.append(np.zeros(SAMPLE_RATE, np.float32))
                    parts.append(speak(voice, phrase))
                parts.append(np.zeros(2 * SAMPLE_RATE, np.float32))
                stream = np.concatenate(parts)
                if condition != "clean":
                    start = int(rng.integers(0, len(noise) - len(stream)))
                    stream = mix(stream, np.asarray(noise[start : start + len(stream)], np.float32) / 32768.0, args.snr)
                before = len(heard)
                t0 = time.perf_counter()
                feed(stream)
                got = [normalize(q.text) for q in heard[before:]]
                ok = got == ([case.expect] if case.expect is not None else [])
                outcomes.append(ok)
                total += 1
                failures += not ok
                if not ok:
                    print(f"   MISS {TEST_VOICES[v]}: heard {got!r} ({time.perf_counter() - t0:.1f} s)")
            label = f"heard as {case.expect!r}" if case.expect else "ignored"
            print(f"{sum(outcomes)}/{len(outcomes)}  [{case.mode}] {' / '.join(case.said)} -> {label}")

    pipe.set_mode("none")
    seconds = int(args.noise_minutes * 60)
    start = int(rng.integers(0, len(noise) - seconds * SAMPLE_RATE))
    before, starts = len(heard), pipe.open_starts
    feed(np.asarray(noise[start : start + seconds * SAMPLE_RATE], np.float32) / 32768.0)
    print(
        f"\ncar noise alone, {seconds} s, no wake word: {pipe.open_starts - starts} speech starts, "
        f"{len(heard) - before} answered {[q.text for q in heard[before:]]}"
    )
    print(f"\n{total - failures}/{total} cases as expected")
    return 0 if failures == 0 and len(heard) == before else 1


if __name__ == "__main__":
    raise SystemExit(main())
