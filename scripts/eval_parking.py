"""Score the parking tracker on the hand-labeled BDD100K parking-lot clips (V2-M1; eval/parking_eval.py).

    .venv/Scripts/python scripts/cache_perception.py --labels eval/parking_labels.yaml --videos data/videos/parking_lot
    .venv/Scripts/python scripts/eval_parking.py                      # summary + per-clip table
    .venv/Scripts/python scripts/eval_parking.py --events             # + every event with what it matched
    .venv/Scripts/python scripts/eval_parking.py --set parking.min_visible_depth_m=0.8
    .venv/Scripts/python scripts/eval_parking.py --sweep parking.gap_ratio=1.5,1.7,2.0

Replays cached perception (tracks + ego motion of every frame) through `ParkingTracker` as `WorldState` drives
it. BDD100K dashcams see ~60° across (`--hfov`); the tracker measures its own scale from the stall pitch, so the
exact value matters little. Writes runs/<time>_parking/report.json (+ events.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from eval.parking_eval import (  # noqa: E402
    ClipLabels,
    ClipScore,
    EventMatch,
    Summary,
    load_labels,
    replay,
    score_clip,
    summarize,
)
from eval.perception_cache import load_cache  # noqa: E402
from nav.gps import GpxReplay  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache" / "perception"
GPS_DIR = REPO_ROOT / "data" / "gps"
LABELS = REPO_ROOT / "eval" / "parking_labels.yaml"


def _score_one(args: tuple[ClipLabels, list[str], float, bool]) -> ClipScore:
    clip, overrides, hfov, use_gps = args
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=overrides)
    frames = load_cache(CACHE_DIR / f"{clip.id}.pkl", cfg)
    gpx = GPS_DIR / f"{clip.id}.gpx"
    speed = None
    if use_gps and gpx.exists():
        replay_gps = GpxReplay(gpx)

        def speed(t: float) -> float | None:
            fix = replay_gps.fix_at(t)
            return None if fix is None else fix.speed_mps

    events = replay(frames, cfg.parking, hfov, speed)
    duration = frames[-1].t if frames else 0.0
    return score_clip(clip, events, duration, parked_delay_s=cfg.parking.parked_after_s)


def run(
    clips: list[ClipLabels], overrides: list[str], hfov: float, use_gps: bool, pool: ProcessPoolExecutor
) -> list[ClipScore]:
    """Score every clip (in parallel)."""
    return list(pool.map(_score_one, [(c, overrides, hfov, use_gps) for c in clips]))


def describe_match(m: EventMatch) -> str:
    """One event line."""
    e, sp = m.event, m.event.space
    assert sp is not None
    what = (
        f"HIT {sp.side} t_vis={m.spot.t_visible} ({m.spot.certainty}, {m.spot.restriction}, {m.spot.stalls} st)"
        if m.spot is not None
        else ("FALSE look-alike: " + m.look_alike.what[:60] if m.look_alike is not None else "FALSE")
    )
    return (
        f"  {e.t:6.2f} {e.kind:<9} {sp.side:<5} z={sp.z:5.1f} stalls={sp.stalls} before={sp.cars_before} "
        f"cars={sp.near.track_id}->{sp.far.track_id} depth={sp.visible_depth:4.1f} pitch={sp.pitch:3.1f}  {what}"
    )


def summary_line(s: Summary) -> str:
    """The headline numbers."""
    delays = sorted(s.delays)
    median = delays[len(delays) // 2] if delays else float("nan")
    return (
        f"recall {s.found}/{s.targets} = {s.recall:.0%} (with a neighbour before: {s.found_with_neighbour}/"
        f"{s.targets_with_neighbour}; candidates {s.candidate_found}) | false spots {s.false_spots} "
        f"({s.look_alike_hits} look-alikes) = {s.false_per_10min:.1f}/10 min of {s.lot_minutes:.1f} | "
        f"delay median {median:+.1f} s | lot {s.lot_clips_offered}/{s.lot_clips} clips, {s.lot_false} false | "
        f"parked {s.parked_right}/{s.parked_labels}, {s.parked_false} false | pull-outs {s.pulls_found}/"
        f"{s.pull_labels}, {s.pull_false} false"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--hfov", type=float, default=60.0, help="Camera field of view of the clips (degrees).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    parser.add_argument("--sweep", action="append", default=[], help="key=v1,v2,...: one summary line per value.")
    parser.add_argument("--clip", action="append", default=[], help="Only these clip ids (prefix).")
    parser.add_argument("--events", action="store_true", help="Print every spot/candidate event.")
    parser.add_argument("--no-gps", action="store_true", help="Our speed from the parked cars only (no GPX).")
    parser.add_argument("--out", type=Path, default=None, help="Report folder (default runs/<time>_parking).")
    args = parser.parse_args(argv)

    clips = [c for c in load_labels(args.labels) if c.usable]
    if args.clip:
        clips = [c for c in clips if any(c.id.startswith(p) for p in args.clip)]
    missing = [c.id for c in clips if not (CACHE_DIR / f"{c.id}.pkl").exists()]
    if missing:
        print(f"no perception cache for {len(missing)} clips (run scripts/cache_perception.py): {missing[:3]}")
        clips = [c for c in clips if c.id not in missing]
    cfg: AppConfig = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    del cfg  # (validates the overrides early)

    with ProcessPoolExecutor(max_workers=8) as pool:
        if args.sweep:
            for spec in args.sweep:
                key, values = spec.split("=", 1)
                for value in values.split(","):
                    scores = run(clips, [*args.set, f"{key}={value}"], args.hfov, not args.no_gps, pool)
                    print(f"{key}={value:<6} {summary_line(summarize(scores))}", flush=True)
            return 0
        started = time.perf_counter()
        scores = run(clips, args.set, args.hfov, not args.no_gps, pool)
        elapsed = time.perf_counter() - started

    lines: list[str] = []
    for s in scores:
        c = s.clip
        targets = s.targets
        found = sum(i in s.hits for i in targets)
        lots = " ".join(f"{t:.1f}{'' if ok else '!'}" for t, ok in s.lot_events) or "-"
        parked = " ".join(f"{t:.1f}{'' if ok else '!'}" for t, ok in s.parked_events) or "-"
        pulls = " ".join(f"{t:.1f}{side[0]}{'' if ok else '!'}" for t, side, ok in s.pull_events) or "-"
        lines.append(
            f"{c.id}  {c.scene:<10} spots {found}/{len(targets)}  false {len(s.false_spots)}  events "
            f"{len(s.spots)}  lot {lots} (labels {list(c.in_lot)})  parked {parked} (label {c.parked_at})  "
            f"pull {pulls} ({len(c.pull_outs)})"
        )
        if args.events:
            for i in targets:
                spot = c.spots[i]
                got = f"found at {s.hits[i]:.2f}" if i in s.hits else "MISSED"
                lines.append(
                    f"    target {spot.side} {spot.t_visible}-{spot.t_pass} {spot.stalls} st before="
                    f"{spot.neighbor_before}: {got}"
                )
            for m in sorted([*s.spots, *s.candidates], key=lambda m: (m.event.t, m.event.kind)):
                lines.append(describe_match(m))
    summary = summarize(scores)
    lines.append("")
    lines.append(summary_line(summary))
    gps = "no GPS" if args.no_gps else "GPS speed"
    lines.append(f"({len(scores)} clips replayed in {elapsed:.1f} s; hfov {args.hfov}; {gps}; overrides {args.set})")
    print("\n".join(lines))

    out = args.out or REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_parking"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.txt").write_text("\n".join(lines), encoding="utf-8")
    report = {"overrides": args.set, "hfov": args.hfov, "gps": not args.no_gps, "summary": summary.as_dict()}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
