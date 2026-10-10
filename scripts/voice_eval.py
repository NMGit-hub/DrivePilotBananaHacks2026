"""Mic-free voice benchmark (BUILD_PLAN M4): synthetic spoken questions through the real voice pipeline.

    .venv/Scripts/python scripts/voice_eval.py                 # all questions, clean + car noise, false wakes
    .venv/Scripts/python scripts/voice_eval.py --limit 40 --no-false-wakes
    .venv/Scripts/python scripts/voice_eval.py --audio-out     # also time answers/alerts on the real speaker

Each question from the router tables (tests/test_router.py) is spoken as "Hey Drive Pilot, <question>" by one of
the six Piper voices held out from wake-word training (scripts/wakeword/common.py), placed in a stream with a
second of lead-in and two of tail, optionally mixed with held-out car noise (DEMAND TCAR, a car never used in
training) at 10 dB SNR, and fed in 80 ms chunks through `VoicePipeline` (wake word → Silero endpoint →
faster-whisper on the GPU). Reported against the M4 targets:
- wake rate >= 95%, and 0 false wakes on 10 minutes of other speech (LibriSpeech test-clean) + car noise;
- the transcript routes like the original text >= 95% (word error rate too);
- end of question → first audio of a local answer < 1.0 s (median and p90). Audio is fed faster than real time,
  so this is computed from measured parts: endpoint delay (stream time) + the part of Whisper's time not hidden
  by speculation + answering + Piper synthesis of the first sentence + the speaker's output latency;
- with --audio-out: alert (chime + pre-rendered clip) say() → first sound at the speaker < 0.3 s.
Writes runs/<stamp>_voice/voice_eval.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.alerts import LIGHT_GREEN, YOUR_TURN  # noqa: E402
from answers.assistant import Assistant  # noqa: E402
from answers.speech import TextSpeaker  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from state.stop_queue import EGO_KEY, Approach, QueueEntry  # noqa: E402
from state.world import WorldSnapshot, empty_snapshot  # noqa: E402
from tests.test_router import ADVERSARIAL, TABLE  # noqa: E402
from voice.audio import CHUNK, SAMPLE_RATE, resample, to_int16  # noqa: E402
from voice.pipeline import HeardQuestion, VoicePipeline  # noqa: E402
from voice.router import RouteKind, normalize, route  # noqa: E402
from voice.stt import Transcriber  # noqa: E402
from voice.tts import Tts  # noqa: E402
from voice.wakeword import WakeWordDetector  # noqa: E402

TEST_VOICES = [
    "en_GB-alan-medium",
    "en_US-joe-medium",
    "en_US-kristin-medium",
    "en_GB-northern_english_male-medium",
    "en_US-hfc_female-medium",
    "en_US-reza_ibrahim-medium",
]
VOICES_DIR = REPO_ROOT / "data" / "wakeword" / "piper_voices"
CAR_NOISE = REPO_ROOT / "data" / "wakeword" / "noise" / "test_car.npy"
LIBRISPEECH = REPO_ROOT / "data" / "wakeword" / "librispeech" / "LibriSpeech" / "test-clean"
LOCAL_KINDS = {RouteKind.RIGHT_OF_WAY, RouteKind.WATCH, RouteKind.CONTROL, RouteKind.GAP_CHECK}
# Is the wake phrase intelligible in a synthetic clip? Some Piper voices mispronounce it ("a drive highlight",
# "they drive pilot"), like the voice the wake-word training excluded; recall is also reported on clips where
# Whisper hears the phrase (the wake-word evaluation's "Whisper-verified" set).
WAKE_HEARD = re.compile(r"\b(?:hey|hi)\W+drive\W*pilot", re.IGNORECASE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance / reference length (on router-normalized text)."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise so that speech RMS / noise RMS = snr_db."""
    rms_speech = float(np.sqrt(np.mean(speech[np.abs(speech) > 1e-3] ** 2))) if np.any(np.abs(speech) > 1e-3) else 0.0
    rms_noise = float(np.sqrt(np.mean(noise**2))) or 1.0
    return speech + noise * (rms_speech / rms_noise / (10 ** (snr_db / 20)))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def demo_snapshot() -> WorldSnapshot:
    """Stopped at a 4-way stop, one car ahead: realistic local answers to time."""
    base = empty_snapshot(20.0)
    order = (QueueEntry("track:2", Approach.AHEAD, 10.0, label="gray truck"), QueueEntry(EGO_KEY, Approach.US, 12.0))
    queue = base.queue.__class__(
        t=20.0, active=True, order=order, departed=(), ego_position=2, tie_groups=(), deadlock=False, your_turn=False
    )
    return WorldSnapshot(
        **{
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            "ego_stopped": True,
            "ego_stopped_since": 12.0,
            "ego_motion_confident": True,
            "queue": queue,
            "at_stop_sign": True,
            "stop_sign_last_seen": 11.0,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N questions (0 = all).")
    parser.add_argument("--snr", type=float, default=10.0, help="Car-noise SNR (dB) for the noisy condition.")
    parser.add_argument("--no-noise", action="store_true", help="Clean condition only.")
    parser.add_argument("--no-false-wakes", action="store_true", help="Skip the 10-minute false-wake test.")
    parser.add_argument("--false-wake-minutes", type=float, default=10.0)
    parser.add_argument("--audio-out", action="store_true", help="Measure answers/alerts on the real speaker.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    cfg = load_config(REPO_ROOT / "config.yaml")
    rng = np.random.default_rng(args.seed)
    from piper import PiperVoice  # deferred: heavy

    voices = {name: PiperVoice.load(str(VOICES_DIR / f"{name}.onnx")) for name in TEST_VOICES}
    noise = np.load(CAR_NOISE, mmap_mode="r")
    stt = Transcriber(cfg.voice.stt, cfg.root)
    stt.warmup()
    wake = WakeWordDetector(cfg.voice.wakeword, cfg.root)
    tts = Tts(cfg.voice.tts, cfg.root)
    tts.warmup()

    intelligible: dict[tuple[str, str], bool] = {}
    questions = list(dict.fromkeys(text for text, _, _ in [*TABLE, *ADVERSARIAL]))
    random.Random(args.seed).shuffle(questions)
    if args.limit:
        questions = questions[: args.limit]
    conditions = ["clean"] if args.no_noise else ["clean", f"car{args.snr:g}dB"]

    heard: list[HeardQuestion] = []
    pipe = VoicePipeline(cfg.voice, wake, stt, heard.append, answer_in_thread=False)
    assistant = Assistant(cfg, demo_snapshot, TextSpeaker())
    output_latency_s = 0.0
    output = None
    if args.audio_out:
        from voice.audio import AudioOutput

        output = AudioOutput(cfg.voice.output_device, volume=0.08)
        output.start()
        output_latency_s = output.latency_s
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for condition in conditions:
        for i, question in enumerate(questions):
            voice_name = TEST_VOICES[i % len(TEST_VOICES)]
            voice = voices[voice_name]
            spoken = f"Hey Drive Pilot, {question}"
            speech = resample(
                np.concatenate([c.audio_float_array for c in voice.synthesize(spoken)]),
                voice.config.sample_rate,
                SAMPLE_RATE,
            )
            if (voice_name, question) not in intelligible:
                heard_as, _ = stt.transcribe_raw(speech)
                intelligible[(voice_name, question)] = bool(WAKE_HEARD.search(heard_as))
            lead, tail = np.zeros(SAMPLE_RATE, np.float32), np.zeros(2 * SAMPLE_RATE, np.float32)
            stream = np.concatenate((lead, speech, tail))
            if condition != "clean":
                start = int(rng.integers(0, len(noise) - len(stream)))
                stream = mix(stream, np.asarray(noise[start : start + len(stream)], np.float32) / 32768.0, args.snr)
            pcm = to_int16(stream)
            pad = (-len(pcm)) % CHUNK
            pcm = np.concatenate((pcm, np.zeros(pad, np.int16)))
            t0 = pipe.t
            speech_end = t0 + (len(lead) + len(speech)) / SAMPLE_RATE
            before_wakes, before = pipe.wakes, len(heard)
            for k in range(0, len(pcm), CHUNK):
                pipe.feed(pcm[k : k + CHUNK])
            pipe.flush(timeout=120.0)
            while pipe.listening:  # still waiting for a question: let it end here, not in the next clip
                pipe.feed(np.zeros(CHUNK, np.int16))
                pipe.flush(timeout=120.0)
            woke = pipe.wakes > before_wakes
            got = heard[before] if len(heard) > before else None
            expected = route(question)
            row: dict[str, Any] = {
                "condition": condition,
                "voice": voice_name,
                "question": question,
                "expected": f"{expected.kind.value}/{expected.intent}",
                "woke": woke,
                "wake_intelligible": intelligible[(voice_name, question)],
            }
            if got is not None:
                actual = route(got.text)
                endpoint_s = got.end_t - speech_end
                if got.speculative and got.speculated_t is not None:
                    hidden = got.end_t - got.speculated_t  # silence that passed while Whisper was already working
                    stt_wait_s = max(0.0, got.stt_ms / 1000 - hidden)
                else:
                    stt_wait_s = got.stt_ms / 1000
                row.update(
                    {
                        "transcript": got.text,
                        "routed": f"{actual.kind.value}/{actual.intent}",
                        "route_ok": (actual.kind, actual.intent) == (expected.kind, expected.intent),
                        "wer": round(word_error_rate(question, got.text), 3),
                        "endpoint_s": round(endpoint_s, 3),
                        "stt_ms": round(got.stt_ms, 1),
                        "speculative": got.speculative,
                        "transcript_latency_s": round(endpoint_s + stt_wait_s, 3),
                    }
                )
                if actual.kind in LOCAL_KINDS:
                    t_answer = time.perf_counter()
                    reply = assistant.ask(got.text)
                    answer_s = time.perf_counter() - t_answer
                    if reply.text:
                        first_sentence = reply.text.split(". ")[0]
                        t_tts = time.perf_counter()
                        audio = tts.synthesize(first_sentence)
                        tts_s = time.perf_counter() - t_tts
                        row["first_audio_s"] = round(endpoint_s + stt_wait_s + answer_s + tts_s + output_latency_s, 3)
                        row["tts_ms"] = round(tts_s * 1000, 1)
                        if output is not None and i < 10:
                            t_play = time.perf_counter()
                            sound_at = output.play(audio, tts.rate)
                            if sound_at is not None:
                                row["device_start_ms"] = round((sound_at - t_play) * 1000, 1)
            results.append(row)
            print(
                f"[{condition}] {i + 1}/{len(questions)} {'W' if woke else '-'} {question!r} -> "
                f"{row.get('transcript')!r} {'OK' if row.get('route_ok') else 'XX'}",
                flush=True,
            )

    summary: dict[str, Any] = {
        "questions": len(questions),
        "conditions": conditions,
        "whisper": cfg.voice.stt.model,
        "device": stt.device,
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        heard_rows = [r for r in rows if "transcript" in r]
        latencies = [r["first_audio_s"] for r in heard_rows if "first_audio_s" in r]
        summary[condition] = {
            "wake_rate": round(sum(r["woke"] for r in rows) / len(rows), 4),
            "wake_rate_intelligible": round(
                sum(r["woke"] for r in rows if r["wake_intelligible"])
                / max(sum(r["wake_intelligible"] for r in rows), 1),
                4,
            ),
            "clips_with_intelligible_wake_phrase": sum(r["wake_intelligible"] for r in rows),
            "wake_rate_by_voice": {
                v: round(
                    sum(r["woke"] for r in rows if r["voice"] == v) / max(sum(r["voice"] == v for r in rows), 1), 3
                )
                for v in TEST_VOICES
            },
            "route_accuracy_of_all": round(sum(bool(r.get("route_ok")) for r in rows) / len(rows), 4),
            "route_accuracy_when_heard": round(sum(r["route_ok"] for r in heard_rows) / max(len(heard_rows), 1), 4),
            "mean_wer": round(statistics.mean([r["wer"] for r in heard_rows]), 4) if heard_rows else None,
            "stt_ms_median": round(statistics.median([r["stt_ms"] for r in heard_rows]), 1) if heard_rows else None,
            "speculative_share": round(sum(r["speculative"] for r in heard_rows) / max(len(heard_rows), 1), 3),
            "endpoint_s_median": round(statistics.median([r["endpoint_s"] for r in heard_rows]), 3)
            if heard_rows
            else None,
            "local_first_audio_s_median": round(statistics.median(latencies), 3) if latencies else None,
            "local_first_audio_s_p90": round(percentile(latencies, 90), 3) if latencies else None,
        }

    if not args.no_false_wakes:
        files = sorted(LIBRISPEECH.rglob("*.flac"))
        random.Random(args.seed).shuffle(files)
        target = int(args.false_wake_minutes * 60 * SAMPLE_RATE)
        parts, total = [], 0
        for path in files:
            audio, rate = sf.read(path, dtype="float32")
            parts.append(resample(audio, rate, SAMPLE_RATE))
            total += len(parts[-1])
            if total >= target:
                break
        speech = np.concatenate(parts)[:target]
        start = int(rng.integers(0, len(noise) - len(speech)))
        stream = mix(speech, np.asarray(noise[start : start + len(speech)], np.float32) / 32768.0, 10.0)
        pcm = to_int16(stream)
        before = pipe.wakes
        for k in range(0, len(pcm) - CHUNK + 1, CHUNK):
            pipe.feed(pcm[k : k + CHUNK])
        pipe.flush(timeout=120.0)
        summary["false_wakes"] = {"minutes": round(len(pcm) / SAMPLE_RATE / 60, 2), "wakes": pipe.wakes - before}

    if output is not None:
        from answers.speech import Priority, Utterance
        from voice.speaker import AudioSpeaker
        from voice.tts import ClipCache

        clips = ClipCache(tts, cfg.resolve_path(cfg.voice.tts.clips_dir))
        clips.prepare([YOUR_TURN, LIGHT_GREEN])
        speaker = AudioSpeaker(tts, output, clips).start()
        for _ in range(5):
            speaker.say(Utterance(YOUR_TURN, "alert", Priority.REQUESTED, clip="your_turn"))
            speaker.wait_idle()
        alerts = [r.latency_ms for r in speaker.records if r.kind == "alert" and r.latency_ms is not None]
        summary["alert_latency_ms"] = {"median": round(statistics.median(alerts), 1), "max": round(max(alerts), 1)}
        speaker.close()
        output.close()

    checks = {}
    for condition in conditions:
        s = summary[condition]
        checks[f"{condition}: wake >= 95% (clips where the phrase is intelligible)"] = (
            s["wake_rate_intelligible"] >= 0.95
        )
        checks[f"{condition}: routed right >= 95%"] = s["route_accuracy_of_all"] >= 0.95
        if s["local_first_audio_s_median"] is not None:
            checks[f"{condition}: local answer audio < 1.0 s (median, p90)"] = (
                s["local_first_audio_s_median"] < 1.0 and s["local_first_audio_s_p90"] < 1.0
            )
    if "false_wakes" in summary:
        checks["0 false wakes on 10 min of speech + car noise"] = summary["false_wakes"]["wakes"] == 0
    if "alert_latency_ms" in summary:
        checks["alert sound < 0.3 s"] = summary["alert_latency_ms"]["max"] < 300
    summary["checks"] = checks

    run_dir = cfg.resolve_path(cfg.logging.runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_voice"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "voice_eval.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {run_dir / 'voice_eval.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
