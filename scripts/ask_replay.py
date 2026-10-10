"""Replay a clip through the full perception + world-state + assistant path and ask typed questions at set times.

    .venv/Scripts/python scripts/ask_replay.py --clip data/videos/dev/<name>.mov \
        --ask "22.5=who got here first?" --ask "23=can I go?" --ask "30=tell me when it's green"

Every frame is processed (lock-step), questions are asked when the video reaches their time, and the answers
plus any alerts are printed with their video times. The vision model and navigation are not connected here
(M5 / M6), so only local answers are meaningful.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.assistant import Assistant  # noqa: E402
from answers.speech import TextSpeaker, Utterance  # noqa: E402
from capture.file_source import FileSource  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from perception.pipeline import PerceptionResult, PerceptionWorker  # noqa: E402
from state.world import WorldEvent, WorldState  # noqa: E402


class PrintingSpeaker(TextSpeaker):
    """Prints alerts as they happen (answers are printed with their question)."""

    def say(self, utterance: Utterance) -> None:
        """Record and print alerts."""
        super().say(utterance)
        if utterance.kind == "alert":
            print(f"{utterance.t:6.2f}  ALERT [{utterance.priority.name}] {utterance.text}")


def parse_ask(item: str) -> tuple[float, str]:
    """ "22.5=who got here first?" -> (22.5, "who got here first?")."""
    t, _, question = item.partition("=")
    return float(t), question.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--ask", action="append", default=[], type=parse_ask, help='"<seconds>=<question>"')
    parser.add_argument("--events", action="store_true", help="Also print world events.")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    setup_logging("WARNING")
    pending = sorted(args.ask)
    speaker = PrintingSpeaker()
    world_holder: list[WorldState] = []
    assistant = Assistant(cfg, lambda: world_holder[0].snapshot(), speaker)

    def on_event(event: WorldEvent) -> None:
        if args.events:
            detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
            print(f"{event.t:6.2f}  event {event.kind} {detail}")
        assistant.on_world_event(event)

    world = WorldState(cfg, on_event=on_event)
    world_holder.append(world)

    def listener(result: PerceptionResult) -> None:
        world.on_perception(result)
        while pending and result.frame.t >= pending[0][0]:
            _, question = pending.pop(0)
            reply = assistant.ask(question)
            print(f"{result.frame.t:6.2f}  Q: {question}")
            print(f"        A ({reply.source}, {reply.elapsed_ms:.2f} ms): {reply.text}")

    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(args.clip, realtime=False)
    worker = PerceptionWorker(source, detector, EgoMotionEstimator(cfg.ego_motion))
    worker.add_listener(listener)
    source.start()
    worker.start()
    while worker.running:
        time.sleep(0.1)
    worker.stop()
    source.stop()
    if worker.error is not None:
        raise RuntimeError("perception failed") from worker.error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
