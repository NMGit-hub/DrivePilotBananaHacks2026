"""Score "are we stopped?" detection against the hand-labeled stops in eval/ego_stop_labels.yaml.

    .venv/Scripts/python scripts/eval_ego_motion.py            # score the configured metric/thresholds
    .venv/Scripts/python scripts/eval_ego_motion.py --sweep    # grid-search thresholds for every metric

Detector boxes (needed to mask moving objects) are computed once per clip on the GPU and cached in
data/cache/. The motion measures are recomputed on the CPU each run, then replayed through the same
`StopStateMachine` the live pipeline uses. Pass criterion (BUILD_PLAN.md M1): on every firm (non-soft)
label, the detected stop start and end are within ±0.5 s, and there are no false stops.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capture.file_source import FileSource  # noqa: E402
from core.config import AppConfig, EgoMotionConfig, load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from perception.ego_motion import EgoMotionEstimator, StopStateMachine  # noqa: E402

LABELS = REPO_ROOT / "eval" / "ego_stop_labels.yaml"
CACHE = REPO_ROOT / "data" / "cache"
TOLERANCE_S = 0.5
METRICS = ("zoom", "horizontal", "magnitude", "combined")  # recorded for analysis


@dataclass
class ClipMeasures:
    """Per-frame motion measures for one clip: `values` = long baseline (short as fallback), `short` = consecutive."""

    bdd_id: str
    t: np.ndarray
    values: dict[str, list[float | None]]
    duration: float
    short: dict[str, list[float | None]] | None = None


def clip_path(cfg: AppConfig, bdd_id: str) -> Path:
    """Locate a dev clip by its BDD id."""
    matches = sorted((cfg.root / "data" / "videos" / "dev").glob(f"*__{bdd_id}.mov"))
    if not matches:
        raise SystemExit(f"clip {bdd_id} missing: run scripts/fetch_dev_clips.py")
    return matches[0]


def detector_boxes(cfg: AppConfig, path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Boxes per frame index from YOLO26s + ByteTrack (cached)."""
    cache = CACHE / f"boxes_{path.stem.split('__')[-1]}_{cfg.perception.imgsz}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        return {int(i): [tuple(b) for b in data[f"f{int(i)}"]] for i in data["indices"]}
    from perception.detector import Detector

    detector = Detector(cfg.perception, cfg.root)
    source = FileSource(path, realtime=False)
    source.start()
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    while (frame := source.read(timeout=10)) is not None:
        boxes[frame.index] = [t.box for t in detector(frame).tracks]
    source.stop()
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {f"f{i}": np.asarray(b, dtype=np.float32).reshape(-1, 4) for i, b in boxes.items()}
    np.savez_compressed(cache, indices=np.asarray(list(boxes)), **arrays)  # type: ignore[arg-type]
    return boxes


def measure_clip(cfg: AppConfig, ego_cfg: EgoMotionConfig, bdd_id: str) -> ClipMeasures:
    """Run the estimator's measurement over the whole clip."""
    path = clip_path(cfg, bdd_id)
    boxes = detector_boxes(cfg, path)
    estimator = EgoMotionEstimator(ego_cfg)
    source = FileSource(path, realtime=False)
    source.start()
    ts: list[float] = []
    values: dict[str, list[float | None]] = {m: [] for m in METRICS}
    shorts: dict[str, list[float | None]] = {m: [] for m in METRICS}
    while (frame := source.read(timeout=10)) is not None:
        short, long, _ = estimator.measure(frame, boxes.get(frame.index, []))
        ts.append(frame.t)
        for m in METRICS:
            # same rule as EgoMotionEstimator.update: long baseline, short as the fallback
            value = long.get(m, ego_cfg.zoom_weight)
            values[m].append(value if value is not None else short.get(m, ego_cfg.zoom_weight))
            shorts[m].append(short.get(m, ego_cfg.zoom_weight))
    source.stop()
    return ClipMeasures(bdd_id, np.asarray(ts), values, duration=ts[-1], short=shorts)


def detect_stops(
    measures: ClipMeasures, metric: str, ego_cfg: EgoMotionConfig, below: float, above: float
) -> list[tuple[float, float | None]]:
    """Replay the measures through the stop state machine; return [start, end) intervals (end None = still)."""
    machine = StopStateMachine(below, above, ego_cfg.stopped_after_s, ego_cfg.smoothing_s)
    intervals: list[tuple[float, float | None]] = []
    current: float | None = None
    for t, value in zip(measures.t, measures.values[metric], strict=True):
        machine.update(float(t), value)
        if machine.stopped and current is None:
            current = machine.stopped_since
        elif not machine.stopped and current is not None:
            intervals.append((current, float(t)))
            current = None
    if current is not None:
        intervals.append((current, None))
    return intervals


def score(label: dict[str, Any], detected: list[tuple[float, float | None]], duration: float) -> dict[str, Any]:
    """Compare detected intervals with one clip's labels."""

    def overlaps(a: tuple[float | None, float | None], b: tuple[float | None, float | None]) -> bool:
        a_start, b_start = a[0] if a[0] is not None else -1.0, b[0] if b[0] is not None else -1.0
        a_end = a[1] if a[1] is not None else duration + 1
        b_end = b[1] if b[1] is not None else duration + 1
        return a_start < b_end and b_start < a_end

    labeled = [tuple(s) for s in label.get("stops", [])]
    soft = [tuple(s) for s in label.get("soft_stops", [])]
    errors, missed = [], []
    for lab in labeled:
        hits = [d for d in detected if overlaps(d, lab)]
        if not hits:
            missed.append(lab)
            continue
        # A stop that is already in progress when the clip begins has no observable start: not scored.
        start_err = None if lab[0] is None else hits[0][0] - lab[0]
        end_err: float | None
        if lab[1] is None:
            end_err = 0.0 if hits[-1][1] is None else float("inf")
        elif hits[-1][1] is None:
            # Still "stopped" when the clip ends. If the car started moving less than the tolerance before the
            # end of the clip, the detector never had a fair chance: the end is not scorable (not a pass, not a
            # failure). Otherwise the stop end was missed.
            end_err = None if duration - lab[1] < TOLERANCE_S else float("inf")
        else:
            end_err = hits[-1][1] - lab[1]
        errors.append({"label": lab, "start_err": start_err, "end_err": end_err, "pieces": len(hits)})
    moving = [tuple(r) for r in label.get("moving", [])]
    false = [d for d in detected if not any(overlaps(d, lab) for lab in labeled + soft)]
    false += [d for d in detected if any(overlaps(d, r) for r in moving) and d not in false]
    soft_hits = [s for s in soft if any(overlaps(d, s) for d in detected)]
    return {"errors": errors, "missed": missed, "false": false, "soft_hits": soft_hits, "soft_total": len(soft)}


def passes(result: dict[str, Any]) -> bool:
    """Every labeled stop found once, boundaries within tolerance, no false stops."""
    return (
        not result["missed"]
        and not result["false"]
        and all(
            (e["start_err"] is None or abs(e["start_err"]) <= TOLERANCE_S)
            and (e["end_err"] is None or abs(e["end_err"]) <= TOLERANCE_S)
            and e["pieces"] == 1
            for e in result["errors"]
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Measure every labeled clip, then score (or sweep) the stop detector."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="grid-search thresholds per metric")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # "±" on a cp1252 Windows console
    cfg = load_config(overrides=args.overrides)
    setup_logging("WARNING")
    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))["clips"]
    measured = {c["bdd_id"]: measure_clip(cfg, cfg.ego_motion, c["bdd_id"]) for c in labels}

    def evaluate(
        below: float, above: float, after: float, smoothing: float, clips: list[dict[str, Any]]
    ) -> tuple[int, float, list[tuple[dict[str, Any], dict[str, Any]]]]:
        ego = cfg.ego_motion.model_copy(update={"stopped_after_s": after, "smoothing_s": smoothing})
        rows, ok, worst = [], 0, 0.0
        for label in clips:
            m = measured[label["bdd_id"]]
            result = score(label, detect_stops(m, cfg.ego_motion.metric, ego, below, above), m.duration)
            rows.append((label, result))
            ok += passes(result)
            for e in result["errors"]:
                worst = max(worst, abs(e["start_err"] or 0.0), abs(e["end_err"] or 0.0))
        return ok, worst, rows

    e = cfg.ego_motion
    if args.sweep:
        tune = [lab for lab in labels if not lab.get("holdout")]  # never tune on held-out clips
        print(f"metric={e.metric} baseline_s={e.baseline_s}; sweeping on {len(tune)} non-held-out clips")
        values = np.asarray([v for lab in tune for v in measured[lab["bdd_id"]].values[e.metric] if v is not None])
        grid = np.unique(np.round(np.percentile(values, np.linspace(5, 50, 19)), 4))
        results = []
        for below, ratio, after, smoothing in itertools.product(grid, (2.0, 3.0), (0.3, 0.5), (0.0, 0.2, 0.3)):
            ok, worst, _ = evaluate(float(below), float(below * ratio), after, smoothing, tune)
            results.append((ok, -worst, float(below), float(below * ratio), after, smoothing))
        results.sort(reverse=True)
        print(f"{'clips ok':>8s} {'worst':>6s} {'below':>8s} {'above':>8s} {'after':>6s} {'smooth':>6s}")
        for ok, neg_worst, below, above, after, smoothing in results[:8]:
            print(f"{ok:5d}/{len(tune):<2d} {-neg_worst:6.2f} {below:8.4g} {above:8.4g} {after:6.2f} {smoothing:6.2f}")
        return 0

    ok, worst, rows = evaluate(e.stopped_below, e.moving_above, e.stopped_after_s, e.smoothing_s, labels)
    print(
        f"metric={e.metric} baseline_s={e.baseline_s} stopped_below={e.stopped_below} "
        f"moving_above={e.moving_above} stopped_after_s={e.stopped_after_s} smoothing_s={e.smoothing_s}\n"
    )
    for label, result in rows:
        print(
            f"[{'PASS' if passes(result) else 'FAIL'}] {label['bdd_id']}{' (held out)' if label.get('holdout') else ''}"
        )
        for err in result["errors"]:
            split = "" if err["pieces"] == 1 else f" (split into {err['pieces']} pieces)"
            end = "n/a (clip ends too soon after)" if err["end_err"] is None else f"{err['end_err']:+.2f} s"
            start = "n/a (stopped at clip start)" if err["start_err"] is None else f"{err['start_err']:+.2f} s"
            print(f"    stop {err['label']}: start {start}, end {end}{split}")
        for lab in result["missed"]:
            print(f"    MISSED stop {lab}")
        for d in result["false"]:
            print(f"    FALSE stop {d[0]:.2f}-{'end' if d[1] is None else f'{d[1]:.2f}'}")
        if result["soft_total"]:
            print(f"    soft near-stops detected: {len(result['soft_hits'])}/{result['soft_total']} (not scored)")
    held = [r for lab, r in rows if lab.get("holdout")]
    print(
        f"\nclips passing: {ok}/{len(labels)} (tolerance ±{TOLERANCE_S} s, worst boundary error {worst:.2f} s)"
        + (f"; held out: {sum(passes(r) for r in held)}/{len(held)}" if held else "")
    )
    return 0 if ok == len(labels) else 1


if __name__ == "__main__":
    sys.exit(main())
