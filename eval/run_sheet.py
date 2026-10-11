"""Test-sheet runner (PLAN.md "Test sheet"): ask every row's question at its moment in its clip, have an AI judge
mark the answer against the expected one, and report accuracy per category and answer times.

    .venv/Scripts/python eval/run_sheet.py                       # eval/test_sheet.csv, typed questions
    .venv/Scripts/python eval/run_sheet.py --voice               # questions spoken (held-out Piper voice) via Whisper
    .venv/Scripts/python eval/run_sheet.py --rows 1-5,9 --type lights
    .venv/Scripts/python eval/run_sheet.py --previous runs/<time>_sheet/results.csv   # keep your manual verdicts

The sheet is a CSV in PLAN.md's format: clip (a file name, or part of one, under data/videos/), time_s, question,
expected, type (the category), notes. An expected answer starting with "ALERT@<seconds>:" also requires the
alert to be spoken within ±1.5 s of that video time ("tell me when it's green"; unrequested safety warnings such
as a pedestrian in our path don't count as it). The clips are BDD100K dashcam videos, ~60° across: the geometry
of V2 features (parking, pedestrians) uses that field of view (`--hfov`).

Each clip is processed frame by frame from its start (every frame, so results don't depend on machine speed), so
the tracker has its full history at each row's time. Questions go through the same `Assistant` as the app: local
answers, the vision model with the latest frames, voice controls. Navigation needs a GPX track, which the dev
clips don't have, so navigation is covered by tests/test_nav*.py until our own drives are recorded.

Answer time: "answer" is routing + answering until the first sentence is ready, plus Piper synthesizing it.
With --voice, the spoken question also goes through the wake word, end-of-question detection and Whisper, and
"first word" = end of speech → end detected → transcript → first sentence → synthesized (measured offline; the
real-time measurement with everything running at once is scripts/bench_app.py).

Results go to runs/<time>_sheet/: results.csv (one row per question, with an empty `manual` column for your
verdict: right / partly / wrong; it wins over the judge's), report.json and report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.assistant import Assistant  # noqa: E402
from answers.scene import SceneHandler  # noqa: E402
from answers.speech import Priority, TextSpeaker  # noqa: E402
from answers.vision import VisionError, VisionModel, VisionRequest, create_vision_model  # noqa: E402
from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from core.runlog import RunRecorder  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.lanes import LaneMasks  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402
from state.lanes import LaneTracker  # noqa: E402
from state.memory import SceneMemory  # noqa: E402
from state.pedestrians import DrivableArea  # noqa: E402
from state.world import WorldState  # noqa: E402

log = get_logger("eval.sheet")

DEFAULT_SHEET = REPO_ROOT / "eval" / "test_sheet.csv"
VIDEOS = REPO_ROOT / "data" / "videos"
ALERT = re.compile(r"^\s*ALERT@(?P<t>\d+(?:\.\d+)?)\s*:\s*(?P<rest>.*)$", re.DOTALL)
ALERT_TOLERANCE_S = 1.5
JUDGE_ATTEMPTS = 2
VERDICTS = ("right", "partly", "wrong")
JUDGE_SYSTEM = (
    "You grade a driving co-pilot's spoken answers against the expected answer, which a person wrote after "
    "looking at the video. Grade:\n"
    '- "right": the answer gives the expected facts (wording may differ) and says nothing false or unsafe;\n'
    '- "partly": incomplete, or hedges where the expected answer is definite, but nothing false or unsafe;\n'
    '- "wrong": misses or contradicts the expected facts, invents details, or tells the driver it is safe to go, '
    "turn, merge or pass when the expected answer does not.\n"
    "Extra correct detail or a reminder to check before going is fine. If the expected answer says what the "
    "answer must not do (e.g. never say it's safe), grade on that."
)
JUDGE_FORMAT = 'Reply as {"verdict": "right" | "partly" | "wrong", "reason": "<one short sentence>"}.'


@dataclass(slots=True)
class Row:
    """One question of the sheet and what happened to it."""

    index: int
    clip: str
    time_s: float
    question: str
    expected: str
    type: str
    notes: str = ""
    alert_at: float | None = None  # expected alert time (ALERT@ rows)
    heard: str | None = None  # --voice: what Whisper heard
    answer: str | None = None
    source: str = ""
    route: str = ""
    alert_said_at: float | None = None
    answer_ms: float | None = None  # first sentence ready
    tts_ms: float | None = None  # synthesizing it
    endpoint_s: float | None = None  # --voice: end of speech → end detected
    stt_wait_s: float | None = None  # --voice: end detected → transcript
    answer_s: float | None = None  # first sentence ready + synthesized
    first_word_s: float | None = None  # --voice: end of speech → first sentence synthesized
    judge: str = ""
    judge_reason: str = ""
    manual: str = ""
    tag: str = ""  # V2-M4: what the answer's hidden tag pointed at ("track 12", "image at 8.0 s")
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        """Identifies the row across runs (for carrying manual verdicts over)."""
        return self.clip, f"{self.time_s:g}", self.question

    @property
    def verdict(self) -> str:
        """The manual verdict if given, else the judge's."""
        return self.manual if self.manual in VERDICTS else self.judge


def load_sheet(path: Path) -> list[Row]:
    """Rows of a sheet CSV (1-based indexes)."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, record in enumerate(csv.DictReader(f), start=1):
            expected = (record.get("expected") or "").strip()
            match = ALERT.match(expected)
            rows.append(
                Row(
                    index=i,
                    clip=record["clip"].strip(),
                    time_s=float(record["time_s"]),
                    question=record["question"].strip(),
                    expected=match.group("rest").strip() if match else expected,
                    type=(record.get("type") or "other").strip(),
                    notes=(record.get("notes") or "").strip(),
                    alert_at=float(match.group("t")) if match else None,
                )
            )
    return rows


def parse_selection(text: str) -> set[int]:
    """ "1-5,9" -> {1, 2, 3, 4, 5, 9}."""
    picked: set[int] = set()
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        picked |= set(range(int(a), int(b or a) + 1))
    return picked


def find_clip(name: str) -> Path:
    """The video under data/videos/ whose file name contains `name`."""
    matches = [p for p in VIDEOS.rglob("*") if p.is_file() and name.lower() in p.name.lower()]
    if len(matches) != 1:
        raise SystemExit(f"clip {name!r}: {len(matches)} matching files under {VIDEOS} (need exactly one)")
    return matches[0]


# ------------------------------------------------------------------------------------------------- voice


class SpokenQuestions:
    """--voice: speak each question with a held-out Piper voice and put it through the real voice pipeline."""

    def __init__(self, cfg: AppConfig, voice_name: str) -> None:
        from piper import PiperVoice

        from voice.pipeline import HeardQuestion, VoicePipeline
        from voice.stt import Transcriber
        from voice.wakeword import WakeWordDetector

        self._voice = PiperVoice.load(str(REPO_ROOT / "data" / "wakeword" / "piper_voices" / f"{voice_name}.onnx"))
        stt = Transcriber(cfg.voice.stt, cfg.root)
        stt.warmup()
        self.heard: list[HeardQuestion] = []
        self.pipeline = VoicePipeline(
            cfg.voice, WakeWordDetector(cfg.voice.wakeword, cfg.root), stt, self.heard.append, answer_in_thread=False
        )

    def ask(self, question: str) -> tuple[str | None, float | None, float | None]:
        """(what was heard, end of speech → end detected, end detected → transcript); None if nothing was heard."""
        import numpy as np

        from voice.audio import CHUNK, SAMPLE_RATE, resample, to_int16

        audio = np.concatenate([c.audio_float_array for c in self._voice.synthesize(f"Hey Drive Pilot, {question}")])
        speech = resample(audio, self._voice.config.sample_rate, SAMPLE_RATE)
        stream = np.concatenate((np.zeros(SAMPLE_RATE // 2, np.float32), speech, np.zeros(2 * SAMPLE_RATE, np.float32)))
        pcm = to_int16(stream)
        pcm = np.concatenate((pcm, np.zeros((-len(pcm)) % CHUNK, np.int16)))
        before, start = len(self.heard), self.pipeline.t
        speech_end = start + (SAMPLE_RATE // 2 + len(speech)) / SAMPLE_RATE
        for k in range(0, len(pcm), CHUNK):
            self.pipeline.feed(pcm[k : k + CHUNK])
        self.pipeline.flush(timeout=120.0)
        while self.pipeline.listening:  # still waiting for a question: let it end here, not in the next row
            self.pipeline.feed(np.zeros(CHUNK, np.int16))
            self.pipeline.flush(timeout=120.0)
        if len(self.heard) == before:
            return None, None, None
        got = self.heard[before]
        return got.text, got.end_t - speech_end, got.transcript_at - got.end_detected_at


# ------------------------------------------------------------------------------------------------- run


def run_clip(
    cfg: AppConfig,
    clip: Path,
    rows: list[Row],
    detector: Detector,
    vision: VisionModel | None,
    tts: Any,
    voice: SpokenQuestions | None,
    out_dir: Path | None = None,
    lane_model: Callable[[Frame], LaneMasks] | None = None,
) -> None:
    """Play the clip frame by frame and answer its rows at their times (`out_dir`: where to save the marked
    image of answers that pointed at an object). `lane_model`: YOLOPv2, run as the app's lanes thread does (at
    lanes.fps, here in lock-step): lane detection's context for the vision model, the drivable area for
    pedestrian warnings."""
    speaker = TextSpeaker()
    world_holder: list[WorldState] = []
    memory = SceneMemory(cfg.memory)  # as in the app: "what was that sign?" and numbered objects
    latest_tracks: list[Any] = [()]
    lanes = LaneTracker(cfg.lanes, cfg.effective_hfov_deg)
    drivable: list[DrivableArea | None] = [None]
    lane_t = [-1e9]
    tags: list[str] = []
    scene = (
        SceneHandler(
            cfg,
            vision,
            lambda n, s: world_holder[0].recent_frames(n, s),
            lambda: lanes.context() if lanes.snapshot().valid else [],
            tracks=lambda: latest_tracks[0],
            memory=memory,
            on_highlight=lambda track_id: tags.append(f"track {track_id}"),
            on_image=lambda frame: tags.append(f"image at {frame.t:.1f} s"),
        )
        if vision
        else None
    )
    assistant = Assistant(cfg, lambda: world_holder[0].snapshot(), speaker, model=scene)
    assistant.on_highlight = lambda track_id: tags.append(f"track {track_id}")
    world = WorldState(cfg, on_event=assistant.on_world_event, drivable_source=lambda: drivable[0])
    world_holder.append(world)
    pending = sorted(rows, key=lambda r: r.time_s)

    def on_result(result: PerceptionResult) -> None:
        frame = result.frame
        if lane_model is not None and frame.t - lane_t[0] >= 1.0 / cfg.lanes.fps - 1e-6:
            lane_t[0] = frame.t
            masks = lane_model(frame)
            lanes.update(frame.t, masks, None)
            drivable[0] = DrivableArea(masks.t, masks.drivable)
        world.on_perception(result)
        latest_tracks[0] = tuple(result.tracks)
        memory.add(result.frame, result.tracks)
        while pending and result.frame.t >= pending[0].time_s:
            ask(pending.pop(0))

    def ask(row: Row) -> None:
        text = row.question
        if voice is not None:
            heard, row.endpoint_s, row.stt_wait_s = voice.ask(row.question)
            row.heard = heard
            if heard is None:
                row.answer, row.source = None, "not heard"
                return
            text = heard
        said_before = len(speaker.said)
        tags.clear()
        if scene is not None:
            scene.last_plan = None
        reply = assistant.ask(text)
        row.answer, row.source = reply.text, reply.source
        plan = scene.last_plan if scene is not None else None
        if plan is not None:  # which number the model gave, to check it on the marked image
            numbers = {m.track_id: m.number for m in plan.marks.values()}
            tags[:] = [
                f"{tag} (#{numbers[int(tag.split()[1])]})"
                if tag.startswith("track ") and int(tag.split()[1]) in numbers
                else tag
                for tag in tags
            ]
        row.tag = "; ".join(tags)
        if tags and plan is not None and plan.marked is not None and out_dir is not None:
            cv2.imwrite(str(out_dir / f"row{row.index:02d}_marked.jpg"), plan.marked)  # (to check the tag)
        row.route = f"{reply.route.kind.value}/{reply.route.intent}"
        row.answer_ms = None if reply.first_ms is None else round(reply.first_ms, 1)
        first = next((u.text for u in speaker.said[said_before:] if u.kind == "answer"), None)
        if first is not None and tts is not None:
            start = time.perf_counter()
            tts.synthesize(first)
            row.tts_ms = round((time.perf_counter() - start) * 1000.0, 1)
        if row.answer_ms is not None:
            row.answer_s = round(row.answer_ms / 1000.0 + (row.tts_ms or 0.0) / 1000.0, 3)
            if voice is not None and row.endpoint_s is not None and row.stt_wait_s is not None:
                row.first_word_s = round(row.answer_s + row.endpoint_s + max(row.stt_wait_s, 0.0), 3)
        log.info("asked", clip=clip.name, t=row.time_s, question=text, answer=row.answer, source=row.source)

    source = FileSource(clip, realtime=False)
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(cfg.ego_motion))
    worker.add_listener(on_result)
    source.start()
    worker.start()
    worker.join()
    source.stop()
    if worker.error is not None:
        raise RuntimeError(f"perception failed on {clip.name}") from worker.error
    for row in pending:  # past the end of the clip
        row.answer, row.source = None, "past the end of the clip"
    for row in rows:
        if row.alert_at is not None:  # the alert a watch promised (not an unrequested safety warning)
            said = [
                u.t for u in speaker.said if u.kind == "alert" and u.priority < Priority.SAFETY and u.t >= row.time_s
            ]
            row.alert_said_at = round(said[0], 2) if said else None


def judge(model: VisionModel | None, row: Row, temperature: float | None = 0.0) -> None:
    """Have the AI judge mark the answer."""
    if row.answer is None:
        row.judge, row.judge_reason = "wrong", f"no answer ({row.source})"
        return
    alert = ""
    if row.alert_at is not None:
        if row.alert_said_at is None:
            row.judge, row.judge_reason = "wrong", "the alert was never spoken"
            return
        if abs(row.alert_said_at - row.alert_at) > ALERT_TOLERANCE_S:
            row.judge, row.judge_reason = "wrong", f"alert at {row.alert_said_at} s, expected {row.alert_at} s"
            return
        alert = f"\n(The promised alert was spoken at the right moment: {row.alert_said_at} s.)"
    if model is None:
        row.judge, row.judge_reason = "", "no judge (vision model not available)"
        return
    request = VisionRequest(
        system=JUDGE_SYSTEM,
        text=(
            f"Driver's question: {row.question}\nExpected answer: {row.expected}\n"
            f"Co-pilot's answer: {row.answer}{alert}\n\n{JUDGE_FORMAT}"
        ),
        max_tokens=300,
        temperature=temperature,
    )
    request = VisionRequest(
        system=request.system + "\nReply with a single JSON object only, no prose and no code fences.",
        text=request.text,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    error = ""
    for _ in range(JUDGE_ATTEMPTS):
        try:
            verdict, reason = parse_verdict(model.complete(request, 8.0, 30.0))
        except VisionError as exc:
            error = str(exc)
            continue
        if verdict:
            row.judge, row.judge_reason = verdict, reason
            return
        error = "no verdict in the reply"
    row.judge, row.judge_reason = "", f"judge failed: {error}"


def parse_verdict(text: str) -> tuple[str, str]:
    """(verdict, reason) from the judge's reply; tolerates broken JSON (e.g. unescaped quotes in the reason)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is not None:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            verdict = str(data.get("verdict", "")).lower()
            return (verdict if verdict in VERDICTS else ""), str(data.get("reason", ""))[:300]
    verdict_match = re.search(r'"verdict"\s*:\s*"(right|partly|wrong)"', text, re.IGNORECASE)
    reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*}?\s*$', text, re.DOTALL)
    verdict = verdict_match.group(1).lower() if verdict_match else ""
    return verdict, (reason_match.group(1) if reason_match else "")[:300]


# ---------------------------------------------------------------------------------------------- report


def summarize(rows: list[Row]) -> dict[str, Any]:
    """Accuracy and answer times per category and overall."""

    def block(group: list[Row]) -> dict[str, Any]:
        counts = {v: sum(1 for r in group if r.verdict == v) for v in VERDICTS}
        judged = sum(counts.values())
        times = [r.first_word_s for r in group if r.first_word_s is not None]
        answer = [r.answer_s for r in group if r.answer_s is not None]
        return {
            "n": len(group),
            **counts,
            "unjudged": len(group) - judged,
            "accuracy": round(counts["right"] / judged, 3) if judged else None,
            "accuracy_with_partly": round((counts["right"] + 0.5 * counts["partly"]) / judged, 3) if judged else None,
            "median_first_word_s": round(statistics.median(times), 3) if times else None,
            "median_answer_s": round(statistics.median(answer), 3) if answer else None,
        }

    types = sorted({r.type for r in rows})
    return {"overall": block(rows), "by_type": {t: block([r for r in rows if r.type == t]) for t in types}}


def write_outputs(folder: Path, rows: list[Row], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    columns = [
        "index", "clip", "time_s", "type", "question", "heard", "expected", "answer", "source", "route",
        "alert_at", "alert_said_at", "answer_ms", "tts_ms", "answer_s", "endpoint_s", "stt_wait_s", "first_word_s",
        "judge", "judge_reason", "manual", "verdict", "tag", "notes",
    ]  # fmt: skip
    with (folder / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**asdict(row), "verdict": row.verdict})
    (folder / "report.json").write_text(
        json.dumps({**meta, "summary": summary, "rows": [{**asdict(r), "verdict": r.verdict} for r in rows]}, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# Test sheet: {meta['sheet']}",
        "",
        f"{meta['mode']} questions, judge: {meta['judge']}, {meta['when']}",
        "",
        "| type | n | right | partly | wrong | accuracy | median answer (s) | median first word (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, b in [*summary["by_type"].items(), ("**all**", summary["overall"])]:
        lines.append(
            f"| {name} | {b['n']} | {b['right']} | {b['partly']} | {b['wrong']} | {b['accuracy']} | "
            f"{b['median_answer_s']} | {b['median_first_word_s']} |"
        )
    lines += ["", "| # | clip | t | question | answer | verdict | why |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        answer = (r.answer or f"({r.source})").replace("|", "/")
        why = r.judge_reason.replace("|", "/")
        lines.append(f"| {r.index} | {r.clip} | {r.time_s:g} | {r.question} | {answer} | {r.verdict} | {why} |")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--rows", default=None, help="Only these rows, e.g. 1-5,9 (1 = first data row).")
    parser.add_argument("--type", default=None, help="Only rows of this type.")
    parser.add_argument("--voice", action="store_true", help="Speak the questions (held-out Piper voice) via Whisper.")
    parser.add_argument("--voice-name", default="en_US-joe-medium")
    parser.add_argument("--previous", type=Path, default=None, help="An earlier results.csv: keep its manual verdicts.")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-lanes", action="store_true", help="Without YOLOPv2 lane detection (V2-M5).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--hfov", type=float, default=60.0, help="Field of view of the clips (BDD100K: ~60°).")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml", overrides=[f"video.file.hfov_deg={args.hfov}", *args.overrides])
    rows = load_sheet(args.sheet)
    if args.rows:
        picked = parse_selection(args.rows)
        rows = [r for r in rows if r.index in picked]
    if args.type:
        rows = [r for r in rows if r.type == args.type]
    if not rows:
        print("no rows selected")
        return 2
    if args.previous is not None:
        with args.previous.open(encoding="utf-8", newline="") as f:
            manual = {
                (p["clip"], f"{float(p['time_s']):g}", p["question"]): p.get("manual", "") for p in csv.DictReader(f)
            }
        for row in rows:
            row.manual = (manual.get(row.key) or "").strip().lower()

    with RunRecorder(cfg.resolve_path(cfg.logging.runs_dir), "sheet", cfg.redacted_dump()) as recorder:
        setup_logging("INFO", log_file=recorder.log_file, secrets=cfg.secrets.values())
        try:
            vision: VisionModel | None = create_vision_model(cfg)
        except VisionError as exc:
            print(f"vision model not available ({exc}): model questions will say so, and nothing is judged")
            vision = None
        try:
            from voice.tts import Tts

            tts: Any = Tts(cfg.voice.tts, cfg.root)
            tts.warmup()
        except Exception as exc:  # noqa: BLE001 - timing only; the sheet still runs without Piper
            print(f"Piper not available ({exc}): answer times exclude speech synthesis")
            tts = None
        voice = SpokenQuestions(cfg, args.voice_name) if args.voice else None
        detector = Detector(cfg.perception, cfg.root)
        lane_model = None
        if cfg.lanes.enabled and cfg.resolve_path(cfg.lanes.weights).is_file() and not args.no_lanes:
            from perception.lanes import LaneModel

            lane_model = LaneModel(cfg.lanes, cfg.root)
        clips: dict[str, list[Row]] = {}
        for row in rows:
            clips.setdefault(row.clip, []).append(row)
        for name, clip_rows in clips.items():
            path = find_clip(name)
            print(f"{path.name}: {len(clip_rows)} question(s)", flush=True)
            run_clip(cfg, path, clip_rows, detector, vision, tts, voice, out_dir=recorder.dir, lane_model=lane_model)
        judge_model = None if args.no_judge else vision
        for row in rows:
            judge(judge_model, row, cfg.vision.temperature)
            print(f"  {row.index:3}. [{row.verdict or '?':6}] {row.question} -> {row.answer}", flush=True)
        summary = summarize(rows)
        meta = {
            "sheet": str(args.sheet),
            "mode": "spoken" if args.voice else "typed",
            "judge": "none" if judge_model is None else judge_model.name,
            "when": time.strftime("%Y-%m-%d %H:%M"),
        }
        write_outputs(recorder.dir, rows, summary, meta)
        print(json.dumps(summary, indent=2))
        print(f"\nresults: {recorder.dir / 'results.csv'}\nreport: {recorder.dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
