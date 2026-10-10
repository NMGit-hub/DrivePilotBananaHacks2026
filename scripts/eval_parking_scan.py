"""Measure the periodic model scan for open parking spots on the hand-labeled BDD100K clips (V2-M1).

    .venv/Scripts/python scripts/eval_parking_scan.py --prompt v1 --dry-run          # calls needed, no network
    .venv/Scripts/python scripts/eval_parking_scan.py --prompt v1                    # scan every 2 s, score
    .venv/Scripts/python scripts/eval_parking_scan.py --prompt v1 --every 1.0        # every 1 s (reuses the 2 s calls)
    .venv/Scripts/python scripts/eval_parking_scan.py --prompt v1 --offline --grid   # re-score cached replies only
    .venv/Scripts/python scripts/eval_parking_scan.py --prompt v1 --offline --events --save-frames

For each usable clip of eval/parking_labels.yaml, frames are decoded at t = 0, every, 2*every, ... (sequential decode,
the frame whose timestamp is nearest), sent to the vision model (config.yaml `vision_model`, Claude Haiku by
default) with the prompt version's question, and every raw reply is cached under
data/cache/parking_scan/<prompt version>[@<model>]/<clip>/<t>.json (with the call's latency), so re-scoring and decision
changes cost nothing. The replies go through `SpotAnnouncer` (the app's decision logic) and each announcement is
tied to the labels with `eval.parking_eval.match_report`, as the geometry tracker's spot events are. Writes
runs/<time>_parking_scan/report.json (+ events.txt; frames/ with --save-frames).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answers.vision import VisionError, VisionModel, create_vision_model  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import get_logger  # noqa: E402
from eval.parking_eval import ClipLabels, load_labels  # noqa: E402
from eval.parking_scan import (  # noqa: E402
    PROMPTS,
    AnnouncementMatch,
    ScanClipScore,
    ScanDecision,
    ScanPrompt,
    ScanResult,
    ScanSummary,
    median,
    parse_scan,
    score_scan,
    summarize_scan,
)

log = get_logger("eval_parking_scan")

LABELS = REPO_ROOT / "eval" / "parking_labels.yaml"
VIDEOS = REPO_ROOT / "data" / "videos" / "parking_lot"
CACHE = REPO_ROOT / "data" / "cache" / "parking_scan"
PRICES = {  # $ per million input / output tokens (claude-api skill, cached 2026-09-25)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}


# ------------------------------------------------------------------------------------------------ frames


def clip_duration(video: Path) -> float:
    """Timestamp of the last frame (s), by a quick sequential pass (BDD .mov files have variable frame rates);
    remembered in data/cache/parking_scan/durations.json."""
    store = CACHE / "durations.json"
    known: dict[str, float] = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    if video.stem in known:
        return known[video.stem]
    cap = cv2.VideoCapture(str(video))
    last = 0.0
    try:
        while cap.grab():
            last = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    finally:
        cap.release()
    known[video.stem] = round(last, 3)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
    return known[video.stem]


def read_frames(video: Path, times: list[float]) -> dict[float, np.ndarray]:
    """The frame nearest to each wanted time (one sequential decode; grab without decoding the rest)."""
    want = sorted(set(times))
    out: dict[float, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    try:
        half = 0.5 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        i = 0
        while i < len(want) and cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t < want[i] - half:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            while i < len(want) and want[i] <= t + half:
                out[want[i]] = img
                i += 1
    finally:
        cap.release()
    return out


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    """(width, height) of a JPEG image."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return (0, 0) if image is None else (image.shape[1], image.shape[0])


def scan_times(duration: float, every: float) -> list[float]:
    """0, every, 2*every, ... up to the clip's end (rounded so 2 s times are a subset of 1 s ones)."""
    n = int(duration / every + 1e-6)
    return [round(k * every, 2) for k in range(n + 1) if k * every <= duration]


# ------------------------------------------------------------------------------------------------- cache


def cache_path(version: str, clip: str, t: float) -> Path:
    """Where one scan's reply is cached."""
    return CACHE / version / clip / f"{t:06.2f}.json"


def load_cached(version: str, clip: str, t: float) -> dict[str, Any] | None:
    """The cached record of a scan, or None."""
    path = cache_path(version, clip, t)
    if not path.exists():
        return None
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def save_cached(version: str, clip: str, t: float, record: dict[str, Any]) -> None:
    """Write a scan's record (atomically)."""
    path = cache_path(version, clip, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------- calls


@dataclasses.dataclass
class CallStats:
    """Counters for this run's model calls."""

    made: int = 0
    failed: int = 0
    input_tokens: float = 0.0  # rough estimates (images w*h/750, text ~4 chars per token)
    output_tokens: float = 0.0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def estimate_tokens(prompt: ScanPrompt, image_sizes: list[tuple[int, int]], reply: dict[str, Any]) -> tuple[int, int]:
    """Rough input / output token counts of one call (for the cost estimate)."""
    image = sum(w * h / 750 for w, h in image_sizes)
    text = (len(prompt.system) + len(prompt.text) + 120) / 4
    return round(image + text), round(len(json.dumps(reply)) / 3.5)


def scan_clip(
    clip: ClipLabels,
    prompt: ScanPrompt,
    times: list[float],
    model: VisionModel,
    cfg: AppConfig,
    pool: ThreadPoolExecutor,
    stats: CallStats,
    first_token_s: float,
    total_s: float,
) -> None:
    """Call the model for every uncached scan time of the clip and cache the replies."""
    video = VIDEOS / f"{clip.id}.mov"
    missing = [t for t in times if load_cached(prompt.version, clip.id, t) is None]
    if not missing:
        return
    earlier = {t: max(0.0, round(t - prompt.earlier_s, 3)) for t in missing} if prompt.earlier_s else {}
    frames = read_frames(video, missing + list(earlier.values()))

    def call(t: float) -> None:
        now = frames.get(t)
        if now is None:
            log.warning("no frame", clip=clip.id, t=t)
            return
        before = frames.get(earlier[t]) if t in earlier else None
        request = prompt.request(now, before, cfg.vision)
        started = time.perf_counter()
        try:
            reply = model.complete_json(request, first_token_s, total_s)
        except VisionError as exc:
            with stats.lock:
                stats.failed += 1
            log.warning("scan failed", clip=clip.id, t=t, error=str(exc))
            return
        latency = time.perf_counter() - started
        tokens_in, tokens_out = estimate_tokens(prompt, [jpeg_size(im.jpeg) for im in request.images], reply)
        with stats.lock:
            stats.made += 1
            stats.input_tokens += tokens_in
            stats.output_tokens += tokens_out
        save_cached(
            prompt.version,
            clip.id,
            t,
            {
                "version": prompt.version,
                "clip": clip.id,
                "t": t,
                "earlier_t": earlier.get(t),
                "model": getattr(model, "model", model.name),
                "latency_s": round(latency, 3),
                "tokens_est": [tokens_in, tokens_out],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reply": reply,
            },
        )

    for future in as_completed([pool.submit(call, t) for t in missing]):
        future.result()


# ----------------------------------------------------------------------------------------------- scoring


def load_results(
    clip: ClipLabels, prompt: ScanPrompt, times: list[float], until: float | None = None
) -> tuple[list[ScanResult], list[float]]:
    """The cached scans of a clip as ScanResults (missing ones skipped; none after `until`), and their latencies."""
    results, latencies = [], []
    for t in times:
        if until is not None and t > until:
            break
        record = load_cached(prompt.version, clip.id, t)
        if record is None:
            continue
        latency = record.get("latency_s")
        results.append(parse_scan(record["reply"], t, prompt.kind, latency))
        if latency is not None:
            latencies.append(float(latency))
    return results, latencies


def describe_match(m: AnnouncementMatch) -> str:
    """One announcement line."""
    a, r = m.announcement, m.announcement.report
    if m.spot is not None:
        spot = m.spot
        what = f"HIT {spot.side} t_vis={spot.t_visible} ({spot.certainty}, {spot.restriction}, {spot.stalls} st)"
        what += " repeat" if m.repeat else ""
    elif m.look_alike is not None:
        what = "FALSE look-alike: " + m.look_alike.what[:60]
    else:
        what = "FALSE"
    return (
        f"  {a.t:6.2f} (first {a.first_t:5.2f}, spoken {a.spoken_t:6.2f}) {r.side:<5} ahead={r.cars_ahead} "
        f"before={r.before} stalls={r.stalls} {r.restriction} {r.confidence}  {what}"
    )


def summary_line(s: ScanSummary) -> str:
    """The headline numbers."""
    return (
        f"recall {s.found}/{s.targets} = {s.recall:.0%} (with a neighbour before {s.found_with_neighbour}/"
        f"{s.targets_with_neighbour}, without {s.found - s.found_with_neighbour}/"
        f"{s.targets - s.targets_with_neighbour}) | false {s.false_alarms} ({s.look_alike_hits} look-alikes) = "
        f"{s.false_per_10min:.1f}/10 min of {s.lot_minutes:.1f} | announcements {s.announcements} "
        f"(repeats {s.repeats}, restricted spots {s.restricted}) | delay median {median(s.delays):+.1f} s "
        f"(spoken {median(s.spoken_delays):+.1f} s, late {s.late})"
    )


def grid() -> list[ScanDecision]:
    """Decision settings compared by --grid."""
    out = []
    for conf in ("medium", "high"):
        for confirm in (1, 2):
            for closed in (False, True):
                out.append(ScanDecision(min_confidence=conf, confirm_scans=confirm, require_closed=closed))
    out.append(ScanDecision(min_confidence="high", require_closed=True, require_neighbour=True))
    out.append(ScanDecision(min_confidence="high", require_closed=True, require_lot=True))
    out.append(ScanDecision(min_confidence="high", require_closed=True, use_before=False))
    out.append(ScanDecision(min_confidence="high", require_closed=True, track_gap_s=2.5))
    out.append(ScanDecision(min_confidence="low"))
    return out


def save_frames(scores: list[ScanClipScore], out: Path, limit_hits: int) -> int:
    """Write the frame of every false alarm and of the first `limit_hits` hits (with the report stamped on it)."""
    folder = out / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in scores:
        chosen = [m for m in s.matches if m.false_alarm] + [m for m in s.matches if not m.false_alarm][:limit_hits]
        if not chosen:
            continue
        frames = read_frames(VIDEOS / f"{s.clip.id}.mov", [m.announcement.t for m in chosen])
        for m in chosen:
            image = frames.get(m.announcement.t)
            if image is None:
                continue
            image = image.copy()
            tag = "FALSE" if m.false_alarm else "HIT"
            lines = [f"{s.clip.id[:8]} t={m.announcement.t:.2f} {tag}", m.announcement.text[:90]]
            for k, line in enumerate(lines):
                y = 36 + 36 * k
                cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            name = f"{tag.lower()}_{s.clip.id[:8]}_{m.announcement.t:06.2f}_{m.announcement.report.side}.jpg"
            cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
    return n


# -------------------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="v1", choices=sorted(PROMPTS), help="Prompt version (its own cache).")
    parser.add_argument("--every", type=float, default=2.0, help="Seconds between scans.")
    parser.add_argument("--model", default=None, help="Another Claude model (cached under <prompt>@<model>).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value (e.g. vision.*).")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--clip", action="append", default=[], help="Only these clip ids (prefix).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel model calls (<= 4).")
    parser.add_argument("--max-calls", type=int, default=700, help="Refuse to make more model calls than this.")
    parser.add_argument("--dry-run", action="store_true", help="Count the calls needed; no network.")
    parser.add_argument("--offline", action="store_true", help="Score cached replies only (no calls).")
    parser.add_argument("--first-token-s", type=float, default=None, help="Default: vision.first_token_timeout_s.")
    parser.add_argument("--total-s", type=float, default=None, help="Default: vision.total_timeout_s.")
    parser.add_argument("--min-confidence", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--confirm", type=int, default=1, help="Agreeing scans before announcing.")
    parser.add_argument("--gap", type=float, default=4.5, help="Track gap (s): same spot if seen again within it.")
    parser.add_argument("--require-lot", action="store_true", help="Ignore scans the model says aren't in a lot.")
    parser.add_argument("--no-before", action="store_true", help="Don't use the car-before colour to dedupe.")
    parser.add_argument("--closed", action="store_true", help="Only gaps with a parked vehicle listed after them.")
    parser.add_argument("--neighbour", action="store_true", help="Only spots with a parked vehicle right before them.")
    parser.add_argument(
        "--stop-at-parked",
        action="store_true",
        help="End the watch once parked (labeled parked_at + parking.parked_after_s), as the app does.",
    )
    parser.add_argument("--grid", action="store_true", help="Also print a summary line per decision setting.")
    parser.add_argument("--events", action="store_true", help="Print every announcement with what it matched.")
    parser.add_argument("--save-frames", action="store_true", help="Save false-alarm and hit frames.")
    parser.add_argument("--hits-per-clip", type=int, default=2, help="Hit frames saved per clip (--save-frames).")
    parser.add_argument("--out", type=Path, default=None, help="Report folder (default runs/<time>_parking_scan).")
    args = parser.parse_args(argv)

    prompt = PROMPTS[args.prompt]
    overrides = list(args.set)
    if args.model:  # another Claude model: its own cache folder
        overrides.insert(0, f"vision.claude.model={args.model}")
        prompt = dataclasses.replace(prompt, version=f"{prompt.version}@{args.model}")
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=overrides)
    decision = ScanDecision(
        min_confidence=args.min_confidence,
        confirm_scans=args.confirm,
        track_gap_s=args.gap,
        require_lot=args.require_lot,
        use_before=not args.no_before,
        require_closed=args.closed,
        require_neighbour=args.neighbour,
    )
    clips = [c for c in load_labels(args.labels) if c.usable]
    if args.clip:
        clips = [c for c in clips if any(c.id.startswith(p) for p in args.clip)]

    durations = {c.id: clip_duration(VIDEOS / f"{c.id}.mov") for c in clips}
    times = {c.id: scan_times(durations[c.id], args.every) for c in clips}
    needed = sum(1 for c in clips for t in times[c.id] if load_cached(prompt.version, c.id, t) is None)
    total = sum(len(v) for v in times.values())
    print(f"prompt {prompt.version}: {len(clips)} clips, {total} scans every {args.every} s, {needed} not cached")
    if args.dry_run:
        return 0

    stats = CallStats()
    started = time.perf_counter()
    if needed and not args.offline:
        if needed > args.max_calls:
            print(f"refusing: {needed} calls > --max-calls {args.max_calls}")
            return 2
        model = create_vision_model(cfg)
        first_s = args.first_token_s or cfg.vision.first_token_timeout_s
        total_s = args.total_s or cfg.vision.total_timeout_s
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4)), thread_name_prefix="scan") as pool:
            for k, c in enumerate(clips, 1):
                scan_clip(c, prompt, times[c.id], model, cfg, pool, stats, first_s, total_s)
                print(f"  [{k}/{len(clips)}] {c.id}: calls so far {stats.made} (failed {stats.failed})", flush=True)
        model.close()
    elapsed = time.perf_counter() - started

    scores: list[ScanClipScore] = []
    latencies: list[float] = []
    tokens_in = tokens_out = 0.0
    until = {
        c.id: c.parked_at + cfg.parking.parked_after_s if args.stop_at_parked and c.parked_at is not None else None
        for c in clips
    }
    for c in clips:
        results, lat = load_results(c, prompt, times[c.id], until[c.id])
        latencies += lat
        for t in times[c.id]:
            record = load_cached(prompt.version, c.id, t)
            if record is not None and record.get("tokens_est"):
                tokens_in += record["tokens_est"][0]
                tokens_out += record["tokens_est"][1]
        scores.append(score_scan(c, durations[c.id], results, decision))

    lines: list[str] = []
    for s in scores:
        c = s.clip
        found = sum(i in s.hits for i in s.targets)
        lines.append(
            f"{c.id}  {c.scene:<10} scans {s.scans:>3}  targets {found}/{len(s.targets)}  announcements "
            f"{len(s.matches):>2}  false {len(s.false_alarms)}  lot {c.lot_s:4.1f} s"
        )
        if args.events:
            for i in s.targets:
                spot = c.spots[i]
                a = s.hits.get(i)
                got = f"found at {a.t:.2f}" if a is not None else "MISSED"
                lines.append(
                    f"    target {spot.side} {spot.t_visible}-{spot.t_pass} {spot.stalls} st before="
                    f"{spot.neighbor_before}: {got}"
                )
            lines += [describe_match(m) for m in s.matches]
    summary = summarize_scan(scores)
    lat = sorted(latencies)
    p90 = float(np.percentile(lat, 90)) if lat else float("nan")
    price_in, price_out = PRICES.get(cfg.vision.claude.model, (float("nan"), float("nan")))
    cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
    lines.append("")
    lines.append(f"decision: {decision.label()}")
    lines.append(summary_line(summary))
    lines.append(
        f"latency median {median(lat):.2f} s, p90 {p90:.2f} s over {len(lat)} cached calls | this run: "
        f"{stats.made} calls, {stats.failed} failed, {elapsed:.0f} s | est. cost of all {len(lat)} cached calls "
        f"${cost:.2f} (~{tokens_in / max(len(lat), 1):.0f} in / {tokens_out / max(len(lat), 1):.0f} out tokens each)"
    )
    grid_rows: list[dict[str, Any]] = []
    if args.grid:
        lines.append("")
        for d in grid():
            g = summarize_scan(
                [
                    score_scan(
                        s.clip, s.duration, load_results(s.clip, prompt, times[s.clip.id], until[s.clip.id])[0], d
                    )
                    for s in scores
                ]
            )
            lines.append(f"{d.label():<52} {summary_line(g)}")
            grid_rows.append({"decision": dataclasses.asdict(d), "summary": g.as_dict()})
    print("\n".join(lines))

    out = args.out or REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_parking_scan"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.txt").write_text("\n".join(lines), encoding="utf-8")
    report = {
        "prompt": prompt.version,
        "every_s": args.every,
        "model": cfg.vision.claude.model if cfg.vision_model == "claude" else cfg.vision_model,
        "decision": dataclasses.asdict(decision),
        "stop_at_parked": args.stop_at_parked,
        "summary": summary.as_dict(),
        "latency": {"median_s": median(lat), "p90_s": p90, "calls": len(lat)},
        "this_run": {"calls": stats.made, "failed": stats.failed, "seconds": round(elapsed, 1)},
        "cost_estimate_usd": round(cost, 3),
        "clips": [
            {
                "id": s.clip.id,
                "scans": s.scans,
                "targets": len(s.targets),
                "found": sum(i in s.hits for i in s.targets),
                "announcements": len(s.matches),
                "false": len(s.false_alarms),
                "events": [describe_match(m).strip() for m in s.matches],
            }
            for s in scores
        ],
        "grid": grid_rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.save_frames:
        n = save_frames(scores, out, args.hits_per_clip)
        print(f"saved {n} frames to {out / 'frames'}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
