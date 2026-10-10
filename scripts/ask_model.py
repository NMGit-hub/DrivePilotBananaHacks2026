"""Ask the vision model questions at given moments of dev clips (live API calls) and time the answers.

    .venv/Scripts/python scripts/ask_model.py --ask "data/videos/dev/x.mov@14.4=what did that sign say?"
    .venv/Scripts/python scripts/ask_model.py --suite eval/model_questions.yaml [--provider claude]

For each question the clip is replayed (lock-step perception + world state) from a few seconds before the
moment, so the tracker context is real, then the question goes through `SceneHandler` exactly as in the app.
Reports the answer, time to the first sentence (what the driver waits for before speech can start) and the
total time; with a suite file, also whether the answer contains the expected words (a quick check; the M8
test sheet has the full judge). Uses the API key from .env; each question is one API call.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.scene import SceneHandler  # noqa: E402
from answers.vision import create_vision_model  # noqa: E402
from capture.file_source import FileSource  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402
from state.world import WorldState  # noqa: E402
from voice.router import RouteKind, route  # noqa: E402

WARMUP_S = 5.0


@dataclass
class Question:
    """One question at one moment of a clip."""

    clip: Path
    t: float
    text: str
    expect_any: tuple[str, ...] = ()  # the answer should contain one of these (case-insensitive)


def parse_ask(item: str) -> Question:
    """ "path@14.4=question" """
    where, _, text = item.partition("=")
    clip, _, t = where.rpartition("@")
    return Question(Path(clip), float(t), text.strip())


def load_suite(path: Path) -> list[Question]:
    """A YAML list of {clip, t, question, expect_any}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Question(REPO_ROOT / row["clip"], float(row["t"]), row["question"], tuple(row.get("expect_any", ())))
        for row in data["questions"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ask", action="append", default=[], type=parse_ask, help='"<clip>@<seconds>=<question>"')
    parser.add_argument("--suite", type=Path, help="YAML file of questions with expected words.")
    parser.add_argument("--provider", help="Override vision_model (claude, gemini, ollama, mock).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    args = parser.parse_args(argv)

    overrides = list(args.set) + ([f"vision_model={args.provider}"] if args.provider else [])
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=overrides)
    setup_logging("WARNING", secrets=cfg.secrets.values())
    questions = list(args.ask) + (load_suite(args.suite) if args.suite else [])
    model = create_vision_model(cfg)
    detector = Detector(cfg.perception, cfg.root)
    firsts, totals, checks = [], [], []

    for q in questions:
        world = WorldState(cfg)
        handler = SceneHandler(cfg, model, world.recent_frames)
        done: list[tuple[list[str], float, float]] = []

        def listener(
            result: PerceptionResult,
            q: Question = q,
            world: WorldState = world,
            handler: SceneHandler = handler,
            done: list[tuple[list[str], float, float]] = done,
        ) -> None:
            world.on_perception(result)
            if done or result.frame.t < q.t:
                return
            r = route(q.text)
            snap = world.snapshot()
            assert snap is not None
            start = time.perf_counter()
            stream = handler.answer(r, snap)
            said = [next(stream)]
            first = time.perf_counter() - start
            said += list(stream)
            done.append((said, first, time.perf_counter() - start))
            if r.kind not in (RouteKind.MODEL, RouteKind.GAP_CHECK):
                print(f"  (note: the router sends this to {r.kind.value}, asked the model directly here)")

        source = FileSource(q.clip, realtime=False, start_s=max(0.0, q.t - WARMUP_S))
        worker = PerceptionWorker(source, detector, EgoMotionEstimator(cfg.ego_motion))
        worker.add_listener(listener)
        source.start()
        worker.start()
        while worker.running and not done:
            time.sleep(0.05)
        worker.stop()
        source.stop()
        if not done:
            print(f"{q.clip.name}@{q.t}: the clip ended before {q.t} s")
            continue
        said, first, total = done[0]
        firsts.append(first)
        totals.append(total)
        answer = " ".join(said)
        verdict = ""
        if q.expect_any:
            ok = any(re.search(rf"\b{re.escape(str(word).lower())}\b", answer.lower()) for word in q.expect_any)
            checks.append(ok)
            verdict = "  [OK]" if ok else f"  [MISSING one of {list(q.expect_any)}]"
        print(f"{q.clip.stem.split('__')[-1][:8]} @{q.t:5.1f}s  Q: {q.text}")
        print(f"    A ({first * 1000:.0f} ms to first sentence, {total * 1000:.0f} ms total): {answer}{verdict}")

    if firsts:
        print(
            f"\n{len(firsts)} questions, provider {model.name}: first sentence median "
            f"{statistics.median(firsts) * 1000:.0f} ms, max {max(firsts) * 1000:.0f} ms; total median "
            f"{statistics.median(totals) * 1000:.0f} ms"
            + (f"; expected words present in {sum(checks)}/{len(checks)}" if checks else "")
        )
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
