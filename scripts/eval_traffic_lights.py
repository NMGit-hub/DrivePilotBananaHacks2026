"""Score the traffic-light reader against eval/light_labels.yaml.

    .venv/Scripts/python scripts/eval_traffic_lights.py [--sheet OUT.png]

Every 0.25 s inside a labeled interval the smoothed light state is compared with the label:
right (same color), abstain (UNKNOWN: the answer will say it can't see the light), or wrong (a different
color; red/green confusions are the dangerous kind). Also reports when "turned green" fired versus the label.
Detections are computed once per clip on the GPU and cached in data/cache/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from perception.detector import Track  # noqa: E402
from perception.ego_motion import EgoMotionEstimator  # noqa: E402
from state.traffic_light import TrafficLightTracker, classify_light  # noqa: E402

LABELS = REPO_ROOT / "eval" / "light_labels.yaml"
CACHE = REPO_ROOT / "data" / "cache"
STEP_S = 0.25


def cached_tracks(cfg: AppConfig, path: Path) -> dict[int, list[Track]]:
    """All tracks (every class) per frame index, cached as JSON."""
    cache = CACHE / f"tracks_{path.stem.split('__')[-1]}_{cfg.perception.imgsz}.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {int(i): [Track(t["id"], t["cls"], t["conf"], tuple(t["box"])) for t in ts] for i, ts in raw.items()}
    from perception.detector import Detector

    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(path, realtime=False)
    source.start()
    out: dict[int, list[Track]] = {}
    while (frame := source.read(timeout=10)) is not None:
        out[frame.index] = detector(frame).tracks
    source.stop()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({i: [t.to_record() | {"conf": t.conf} for t in ts] for i, ts in out.items()}), encoding="utf-8"
    )
    return out


def label_at(t: float, states: list[list[Any]], duration: float) -> str | None:
    """Labeled color at time t, or None if t is outside every interval."""
    for start, end, color in states:
        if start <= t <= (end if end is not None else duration + 1):
            return str(color)
    return None


def main(argv: list[str] | None = None) -> int:
    """Evaluate every labeled clip and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, help="also write a sheet of light crops with their readings")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    cfg = load_config(overrides=args.overrides)
    setup_logging("WARNING")
    tl_cfg = cfg.traffic_light
    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))["clips"]
    totals = {"right": 0, "abstain": 0, "wrong": 0, "red_green_swaps": 0}
    tiles: list[np.ndarray] = []
    for clip in labels:
        path = next((cfg.root / "data" / "videos" / "dev").glob(f"*__{clip['bdd_id']}.mov"))
        tracks = cached_tracks(cfg, path)
        tracker = TrafficLightTracker(tl_cfg)
        ego = EgoMotionEstimator(cfg.ego_motion)  # the real stopped/moving signal drives light anchoring
        source = FileSource(path, realtime=False)
        source.start()
        counts = {"right": 0, "abstain": 0, "wrong": 0, "red_green_swaps": 0}
        green_at: float | None = None
        next_sample, next_tile = 0.0, 0.0
        duration = source.duration
        while (frame := source.read(timeout=10)) is not None:
            frame_tracks = tracks.get(frame.index, [])
            stopped = ego.update(frame, [t.box for t in frame_tracks]).stopped
            reading, turned_green = tracker.update(frame.t, frame.image, frame_tracks, ego_stopped=stopped)
            if turned_green and green_at is None:
                green_at = frame.t
            if frame.t >= next_sample:
                next_sample += STEP_S
                expected = label_at(frame.t, clip["states"], duration)
                if expected is not None:
                    got = reading.color.value
                    key = "right" if got == expected else ("abstain" if got == "unknown" else "wrong")
                    counts[key] += 1
                    if {got, expected} == {"red", "green"}:
                        counts["red_green_swaps"] += 1
            if args.sheet is not None and frame.t >= next_tile and label_at(frame.t, clip["states"], duration):
                next_tile = frame.t + 3.0
                for light in tracker.relevant(frame_tracks, frame.width, frame.height)[:2]:
                    reading_one = classify_light(frame.image, light.box, tl_cfg)
                    x1, y1, x2, y2 = (int(v) for v in light.box)
                    crop = cv2.resize(
                        frame.image[max(0, y1 - 3) : y2 + 3, max(0, x1 - 3) : x2 + 3],
                        (60, 120),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    tile = np.full((140, 60, 3), 255, np.uint8)
                    tile[:120] = crop
                    cv2.putText(
                        tile,
                        f"{reading_one.color.value[:3]} {frame.t:.0f}",
                        (1, 133),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (0, 0, 0),
                        1,
                    )
                    tiles.append(tile)
        source.stop()
        n = sum(counts[k] for k in ("right", "abstain", "wrong"))
        line = (
            f"{clip['bdd_id']}: right {counts['right']}/{n}, abstain {counts['abstain']}, wrong {counts['wrong']}"
            f" (red/green swaps {counts['red_green_swaps']})"
        )
        if "turns_green" in clip:
            line += (
                f"; turned green at {green_at:.2f} s (label {clip['turns_green']} s)"
                if green_at is not None
                else f"; turned green NOT detected (label {clip['turns_green']} s)"
            )
        elif green_at is not None:
            line += f"; FALSE turned-green at {green_at:.2f} s"
        print(line)
        for k in totals:
            totals[k] += counts[k]
    n = totals["right"] + totals["abstain"] + totals["wrong"]
    print(
        f"\nall: right {totals['right']}/{n} ({100 * totals['right'] / max(n, 1):.0f}%), abstain {totals['abstain']}, "
        f"wrong {totals['wrong']}, red/green swaps {totals['red_green_swaps']}"
    )
    if args.sheet is not None and tiles:
        rows = [
            np.hstack(tiles[i : i + 20] + [np.full_like(tiles[0], 255)] * (20 - len(tiles[i : i + 20])))
            for i in range(0, len(tiles), 20)
        ]
        cv2.imwrite(str(args.sheet), np.vstack(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
