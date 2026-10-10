"""Full-pipeline benchmark: the whole app on a dev clip with spoken questions, measured against PLAN.md.

    .venv/Scripts/python scripts/bench_app.py              # default clip and questions, real speaker (quiet)
    .venv/Scripts/python scripts/bench_app.py --silent     # no sound (a stand-in output with the same timing)
    .venv/Scripts/python scripts/bench_app.py --ui         # also serve the page (watch it at localhost:8000)

The app runs as `python -m app` does: real-time replay, YOLO26s + ego motion on the GPU, world state, alerts,
Whisper, Piper, the vision model. Only the microphone is replaced: a feeder plays synthesized "Hey DrivePilot,
<question>" audio (a held-out Piper voice, over car noise at 10 dB SNR) into the voice pipeline in real time,
at set video times, the way a microphone delivers 80 ms chunks. Each question's latency is measured from the
true end of its speech to the first sound of the answer at the speaker. Targets (BUILD_PLAN M7):
local answers < 1.0 s and model answers < 2.5 s (median and p90); alerts < 0.3 s from the frame they were
detected in to the sound; perception keeps up (>= 25.2 fps processed, 84% of the 30 fps video).
The report goes to the run folder as bench_app.json; the exit code is 0 only if every target is met.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app import (  # noqa: E402
    FIXED_ALERT_TEXTS,
    DrivePilotApp,
    PageActions,
    apply_cli,
    build_app,
    build_parser,
    make_voice_pipeline,
)
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from core.runlog import RunRecorder  # noqa: E402
from voice.audio import CHUNK, SAMPLE_RATE, resample, to_int16  # noqa: E402
from voice.router import route  # noqa: E402

VOICES_DIR = REPO_ROOT / "data" / "wakeword" / "piper_voices"
CAR_NOISE = REPO_ROOT / "data" / "wakeword" / "noise" / "test_car.npy"
DEFAULT_CLIP = "data/videos/dev/red_light_turns_green_sf__b97b9ee8-612c36ef.mov"
# (video time, question): moving until ~22 s, stopped at a red light, green at 36.4 s. Spaced so each answer
# starts before the next question (a new wake word stops the speech): ~7 s after a model question, ~5 s otherwise.
DEFAULT_QUESTIONS = [
    (2.0, "what's the speed limit here?"),
    (9.0, "what does that sign say?"),
    (16.0, "what's my next turn?"),
    (21.5, "who got here first?"),
    (26.5, "tell me when it's green"),
    (31.0, "can I go?"),
]
TARGETS = {"local": 1.0, "model": 2.5}
ALERT_TARGET_MS = 300.0
FPS_TARGET = 25.2


@dataclass(slots=True)
class Spoken:
    """One question as fed to the pipeline."""

    t: float  # video time to start saying it
    text: str
    audio: np.ndarray  # float32, 16 kHz
    speech_end_at: float | None = None  # perf_counter of the last speech sample (set while feeding)
    started_at: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SilentOutput:
    """Stands in for `AudioOutput`: plays nothing, but takes as long, and can be interrupted."""

    rate = 48000
    latency_s = 0.0

    def __init__(self) -> None:
        self._stop = threading.Event()

    def start(self) -> None:
        pass

    def play(self, audio: np.ndarray, rate: int, timeout: float | None = None) -> float | None:
        self._stop.clear()
        started = time.perf_counter()
        self._stop.wait(len(audio) / rate)
        return started

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()


class Feeder(threading.Thread):
    """Feeds 80 ms chunks in real time: car noise, and each question when the video reaches its time."""

    def __init__(self, app: DrivePilotApp, pipeline: Any, questions: list[Spoken], noise: np.ndarray, snr: float):
        super().__init__(name="feeder", daemon=True)
        self.app, self.pipeline, self.questions = app, pipeline, list(questions)
        self.noise = noise
        rms = statistics.mean(float(np.sqrt(np.mean(q.audio[np.abs(q.audio) > 1e-3] ** 2))) for q in questions)
        self.noise_gain = rms / (float(np.sqrt(np.mean(noise[: SAMPLE_RATE * 60] ** 2))) or 1.0) / 10 ** (snr / 20)
        self.stop_event = threading.Event()

    def run(self) -> None:
        while self.app.worker is None or self.app.worker.processed == 0:  # wait for the video to start
            if self.stop_event.wait(0.05):
                return
        pending = list(self.questions)
        current: Spoken | None = None
        position = 0
        noise_at = 0
        start = time.perf_counter()
        k = 0
        while not self.stop_event.is_set():
            due = start + (k + 1) * CHUNK / SAMPLE_RATE  # a microphone hands over a chunk once it is complete
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            k += 1
            if current is None and pending and self.app.media_time() >= pending[0].t:
                current, position = pending.pop(0), 0
                current.started_at = time.perf_counter()
            chunk = np.zeros(CHUNK, np.float32)
            if current is not None:
                piece = current.audio[position : position + CHUNK]
                chunk[: len(piece)] += piece
                position += CHUNK
                if position >= len(current.audio):
                    # The last speech sample was `position - len(audio)` samples before the end of this chunk.
                    late = (position - len(current.audio)) / SAMPLE_RATE
                    current.speech_end_at = due - late
                    current = None
            if noise_at + CHUNK > len(self.noise):
                noise_at = 0
            chunk += np.asarray(self.noise[noise_at : noise_at + CHUNK], np.float32) / 32768.0 * self.noise_gain
            noise_at += CHUNK
            self.pipeline.feed(to_int16(chunk))


def synthesize(voice_name: str, questions: list[tuple[float, str]]) -> list[Spoken]:
    from piper import PiperVoice

    voice = PiperVoice.load(str(VOICES_DIR / f"{voice_name}.onnx"))
    out = []
    for t, text in questions:
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(f"Hey Drive Pilot, {text}")])
        out.append(Spoken(t, text, resample(audio, voice.config.sample_rate, SAMPLE_RATE).astype(np.float32)))
    return out


def p90(values: list[float]) -> float | None:
    return float(np.percentile(values, 90)) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, default=REPO_ROOT / DEFAULT_CLIP)
    parser.add_argument("--voice", default="en_US-joe-medium", help="A held-out Piper voice (not in training).")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--silent", action="store_true", help="Don't play sound (same timing).")
    parser.add_argument("--volume", type=float, default=0.08)
    parser.add_argument("--ui", action="store_true", help="Serve the page too.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml", overrides=[*args.overrides, f"voice.output_volume={args.volume}"])
    app_args = build_parser().parse_args(["--video", str(args.clip), "--no-voice", "--exit-at-end"])
    app_args.no_ui = not args.ui
    cfg = apply_cli(cfg, app_args)
    questions = synthesize(args.voice, DEFAULT_QUESTIONS)
    noise = np.load(CAR_NOISE, mmap_mode="r")
    with RunRecorder(cfg.resolve_path(cfg.logging.runs_dir), "bench_app", cfg.redacted_dump()) as recorder:
        setup_logging("INFO", log_file=recorder.log_file, secrets=cfg.secrets.values())
        speaker = None
        if args.silent:
            from voice.speaker import AudioSpeaker
            from voice.tts import ClipCache, Tts

            tts = Tts(cfg.voice.tts, cfg.root)
            tts.warmup()
            clips = ClipCache(tts, cfg.resolve_path(cfg.voice.tts.clips_dir))
            clips.prepare(list(FIXED_ALERT_TEXTS))
            speaker = AudioSpeaker(tts, SilentOutput(), clips).start()  # type: ignore[arg-type]
        app = build_app(cfg, app_args, recorder, speaker=speaker)
        pipeline = make_voice_pipeline(app, cfg)
        app.attach_voice(pipeline, None)
        app.status["voice"] = "bench feeder"
        server = None
        if app.hub is not None:
            from ui.server import UiServer, create_server

            server = UiServer(create_server(app.hub, PageActions(app)), cfg.ui.host, cfg.ui.port)
            server.start()
            print(f"page: {server.url}")
        time.sleep(3.0)  # let the vision model's startup check finish (it opens the connection)
        feeder = Feeder(app, pipeline, questions, noise, args.snr)
        feeder.start()
        code = app.run()
        feeder.stop_event.set()
        feeder.join(2.0)
        if server is not None:
            server.stop()
        report = build_report(app, questions, args)
        (recorder.dir / "bench_app.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_report(report)
        print(f"\nreport: {recorder.dir / 'bench_app.json'}")
    return 0 if code == 0 and all(c["ok"] for c in report["checks"]) else 1


def build_report(app: DrivePilotApp, questions: list[Spoken], args: argparse.Namespace) -> dict[str, Any]:
    summary = app.summary()
    voiced = sorted((q for q in app.questions.values() if q.via == "voice"), key=lambda q: q.qid)
    rows = []
    unheard = list(questions)
    by_source: dict[str, list[float]] = {}
    for q in voiced:
        # Match each heard question to the spoken one that ended just before it was answered.
        ended = q.ended_at or 0.0
        match = next((s for s in unheard if s.speech_end_at is not None and s.speech_end_at <= ended + 2.0), None)
        if match is None:
            continue
        unheard.remove(match)
        expected = route(match.text)
        first_sound = (q.ended_at or 0.0) + q.first_audio_s if q.first_audio_s is not None else None
        latency = first_sound - match.speech_end_at if first_sound is not None and match.speech_end_at else None
        routed_ok = q.route == f"{expected.kind.value}/{expected.intent}"
        kind = "local" if q.source in ("local", "nav", "control") else q.source
        if latency is not None:
            by_source.setdefault(kind, []).append(latency)
        rows.append(
            {
                "t": round(match.t, 1),
                "said": match.text,
                "heard": q.text,
                "route": q.route,
                "routed_ok": routed_ok,
                "source": q.source,
                "answer": q.answer,
                "first_audio_s": None if latency is None else round(latency, 3),
                **{k: q.voice.get(k) for k in ("endpoint_s", "stt_ms", "speculative")},
            }
        )
    alerts = [m.seconds * 1000.0 for m in app.latency.alerts if not m.key.startswith("nav_")]
    perception = summary["perception"]
    checks = []
    for kind, target in TARGETS.items():
        values = by_source.get(kind, [])
        median = statistics.median(values) if values else None
        top = p90(values)
        checks.append(
            {
                "name": f"{kind} answers: question end -> first sound (median, p90)",
                "value": None if median is None else f"{median:.2f} s, {top:.2f} s",
                "target": f"< {target} s",
                "ok": median is not None and top is not None and median < target and top < target,
            }
        )
    checks.append(
        {
            "name": "alerts: detection frame -> sound (median, max)",
            "value": f"{statistics.median(alerts):.0f} ms, {max(alerts):.0f} ms" if alerts else None,
            "target": f"< {ALERT_TARGET_MS:.0f} ms",
            "ok": bool(alerts) and max(alerts) < ALERT_TARGET_MS,
        }
    )
    checks.append(
        {
            "name": "perception keeps up",
            "value": f"{perception.get('fps')} fps",
            "target": f">= {FPS_TARGET} fps",
            "ok": (perception.get("fps") or 0) >= FPS_TARGET,
        }
    )
    checks.append(
        {
            "name": "every question heard and routed right",
            "value": f"{sum(1 for r in rows if r['routed_ok'])}/{len(questions)}",
            "target": f"{len(questions)}/{len(questions)}",
            "ok": len(rows) == len(questions) and all(bool(r["routed_ok"]) for r in rows),
        }
    )
    return {
        "clip": str(args.clip),
        "voice": args.voice,
        "snr_db": args.snr,
        "silent": args.silent,
        "questions": rows,
        "unheard": [s.text for s in unheard],
        "alerts_ms": [round(a, 1) for a in alerts],
        "perception": perception,
        "voice_counters": summary["voice"],
        "checks": checks,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n{'t':>5}  {'latency':>8}  {'source':6}  said -> heard -> answer")
    for row in report["questions"]:
        latency = "-" if row["first_audio_s"] is None else f"{row['first_audio_s']:.2f} s"
        flag = "" if row["routed_ok"] else "  [MISROUTED]"
        print(f"{row['t']:5.1f}  {latency:>8}  {row['source']:6}  {row['said']!r} -> {row['heard']!r}{flag}")
        print(f"{'':23}{row['answer']}")
    if report["unheard"]:
        print("not heard:", report["unheard"])
    unplayed = [row["said"] for row in report["questions"] if row["first_audio_s"] is None]
    if unplayed:
        print("answered but never played (stopped by the next question or an alert):", unplayed)
    print(f"alerts (ms): {report['alerts_ms']}")
    print("\ntargets:")
    for check in report["checks"]:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}: {check['value']} (target {check['target']})")


if __name__ == "__main__":
    raise SystemExit(main())
