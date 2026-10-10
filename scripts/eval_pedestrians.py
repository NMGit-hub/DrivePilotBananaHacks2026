"""Pedestrian and cyclist warnings on the dev clips (V2-M3; state/pedestrians.py, eval/pedestrian_eval.py).

    .venv/Scripts/python scripts/cache_perception.py ...          # once: perception of every dev clip on disk
    .venv/Scripts/python scripts/eval_pedestrians.py              # every alert + every in-path-but-silent episode
    .venv/Scripts/python scripts/eval_pedestrians.py --frames     # + JPEGs of each alert and episode (zone drawn)
    .venv/Scripts/python scripts/eval_pedestrians.py --clip red_light_construction --set pedestrians.headway_s=2.5

Replays cached perception through the real WorldState and AlertEngine, with each clip's GPS speed (data/gps/<bdd
id>.gpx). BDD100K dashcams see ~60° across (`--hfov`). Prints, per clip: every spoken warning (time, frame, text,
track, where it was), every episode where someone was in (or entering) the path zone without a warning and why
(we were stopped, GPS said we were barely moving, still confirming, already warned), and the per-frame cost of the
pedestrian update. Writes runs/<time>_pedestrians/ (report.txt, frames/).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from eval.pedestrian_eval import Replay, episodes, replay  # noqa: E402
from eval.perception_cache import CachedFrame, load_cache, read_frame  # noqa: E402
from nav.gps import GpxReplay  # noqa: E402
from state.parking import Camera  # noqa: E402
from state.pedestrians import PedestrianSnapshot  # noqa: E402

COLORS = {"in_path": (0, 0, 255), "entering": (0, 128, 255), "other": (0, 215, 255)}


def draw(
    image: np.ndarray, snap: PedestrianSnapshot, horizon: float, cfg: AppConfig, frame: CachedFrame, note: str
) -> np.ndarray:
    """The frame with the path zone, the horizon row and every pedestrian/cyclist box."""
    out = image.copy()
    h, w = out.shape[:2]
    cam = Camera(snap.focal_px, snap.aim_x, cfg.parking.camera_height_m)  # centred on where we're heading
    cv2.line(out, (0, round(horizon)), (w, round(horizon)), (0, 255, 255), 1)
    cv2.line(out, (round(snap.aim_x), round(horizon) - 12), (round(snap.aim_x), round(horizon) + 12), (255, 0, 255), 2)
    zone = snap.zone
    if zone is not None and snap.focal_px > 0:
        z_near = max(1.0, cam.f * cam.height_m / max(h - horizon, 1.0))
        for closing, color in ((None, (0, 220, 0)), (0.0, (0, 120, 0))):  # at our speed / for things we don't close on
            length = zone.length(closing)
            zs = np.linspace(z_near, max(length, z_near + 0.1), 24)
            left = [cam.project(zone.center(z) - zone.half_width_at(z), z, 0.0, horizon) for z in zs]
            right = [cam.project(zone.center(z) + zone.half_width_at(z), z, 0.0, horizon) for z in zs]
            poly = np.array([*left, *reversed(right)], np.int32)
            cv2.polylines(out, [poly], True, color, 2)
    for p in snap.people:
        state = "in_path" if p.in_path else ("entering" if p.entering else "other")
        x1, y1, x2, y2 = (round(v) for v in p.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[state], 2)
        vx = "" if p.vx is None else f" vx{p.vx:+.1f}"
        label = f"#{p.track_id} {p.kind[0]} z{p.z:.0f} x{p.x:+.1f}{vx}{' W' if p.alerted else ''}"
        cv2.putText(out, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[state], 2)
    still = "moving" if snap.moving else f"still ({snap.still_reason})"
    speed = "-" if zone is None else f"{zone.speed:.1f}"
    header = f"t={frame.t:.2f} #{frame.index} {still} v={speed} turn={snap.turning} {note}"
    cv2.putText(out, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def gps_speed(gps: GpxReplay) -> Callable[[float], float | None]:
    """The clip's GPS speed (m/s) at video time t, None without a fix (WorldState's `speed_source`)."""

    def speed(t: float) -> float | None:
        fix = gps.fix_at(t)
        return None if fix is None else fix.speed_mps

    return speed


def strip(
    clip: Path, frames: list[CachedFrame], run: Replay, cfg: AppConfig, indices: list[int], note: str
) -> np.ndarray:
    """Half-size frames side by side."""
    tiles = []
    for k in indices:
        k = min(max(k, 0), len(frames) - 1)
        image = read_frame(clip, frames[k].index)
        if image is None:
            continue
        tile = draw(image, run.snapshots[k], run.horizons[k], cfg, frames[k], note)
        tiles.append(cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2)))
    return np.hstack(tiles) if tiles else np.zeros((10, 10, 3), np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data", help="Folder with videos/, cache/, gps/.")
    parser.add_argument("--videos", default="videos/dev", help="Clips folder inside --data.")
    parser.add_argument("--hfov", type=float, default=60.0, help="Camera field of view of the clips (degrees).")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    parser.add_argument("--clip", action="append", default=[], help="Only clips whose name contains this.")
    parser.add_argument("--frames", action="store_true", help="Save JPEGs of every alert and silent episode.")
    parser.add_argument("--no-gps", action="store_true", help="Replay without the clips' GPS speed.")
    parser.add_argument("--min-episode", type=int, default=3, help="Shorter silent episodes are not drawn.")
    parser.add_argument("--out", type=Path, default=None, help="Report folder (default runs/<time>_pedestrians).")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml", overrides=[f"video.file.hfov_deg={args.hfov}", *args.set])
    clips = sorted((args.data / args.videos).glob("*.mov"))
    if args.clip:
        clips = [c for c in clips if any(p in c.stem for p in args.clip)]
    out = args.out or REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_pedestrians"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    costs: list[float] = []
    total_alerts = 0

    def emit(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    for clip in clips:
        cache = args.data / "cache" / "perception" / f"{clip.stem}.pkl"
        if not cache.exists():
            emit(f"== {clip.stem}: no perception cache (scripts/cache_perception.py)")
            continue
        frames = load_cache(cache, cfg)
        bdd = clip.stem.split("__")[-1]
        gpx = args.data / "gps" / f"{bdd}.gpx"
        speed = gps_speed(GpxReplay(gpx)) if not args.no_gps and gpx.exists() else None
        run = replay(frames, cfg, speed)
        costs += run.cost_ms
        name = clip.stem.split("__")[0]
        cost = sorted(run.cost_ms)
        emit(
            f"\n== {name} ({len(frames)} frames, gps={'no' if speed is None else 'yes'}): {len(run.alerts)} alerts | "
            f"pedestrian update {statistics.mean(cost):.3f} ms mean, {cost[int(0.99 * (len(cost) - 1))]:.3f} p99, "
            f"{cost[-1]:.3f} max; whole world update {statistics.mean(run.world_ms):.2f} ms mean"
        )
        total_alerts += len(run.alerts)
        spoken = {(a.index, a.event.person.track_id) for a in run.alerts}
        for index, e in run.events:
            p = e.person
            said = next((a.text for a in run.alerts if (a.index, p.track_id) == (index, p.track_id)), None)
            vx = "-" if p.vx is None else f"{p.vx:+.2f}"
            what = "ALERT" if (index, p.track_id) in spoken else "event"
            emit(
                f"  {what} t={e.t:6.2f} #{index:<4} track {p.track_id:<5}"
                f" {p.kind:<10} {e.how:<8} {e.side:<5} x={p.x:+5.1f} z={p.z:5.1f} vx={vx} "
                f"-> {said or '(suppressed by the alert engine)'}"
            )
            if args.frames:
                k = next(i for i, f in enumerate(frames) if f.index == index)
                sheet = strip(clip, frames, run, cfg, [k - 15, k, k + 15], f"ALERT #{p.track_id}")
                cv2.imwrite(str(out / "frames" / f"{name}_alert_{e.t:06.2f}_{p.track_id}.jpg"), sheet)
        for ep in episodes(run, frames, cfg.pedestrians.confirm_frames):
            n = ep.i1 - ep.i0 + 1
            reasons = ", ".join(f"{r} {c}" for r, c in sorted(ep.reasons.items(), key=lambda kv: -kv[1]))
            emit(
                f"  silent t={ep.t0:6.2f}-{ep.t1:6.2f} track {ep.track_id:<5} {ep.kind:<10} {ep.how:<8} "
                f"min z={ep.min_z:5.1f} frames={n}: {reasons}"
            )
            if args.frames and n >= args.min_episode:
                k0 = next(i for i, f in enumerate(frames) if f.index == ep.i0)
                k1 = next(i for i, f in enumerate(frames) if f.index == ep.i1)
                sheet = strip(clip, frames, run, cfg, [k0, (k0 + k1) // 2, k1], f"silent #{ep.track_id}")
                cv2.imwrite(str(out / "frames" / f"{name}_silent_{ep.t0:06.2f}_{ep.track_id}.jpg"), sheet)
    if costs:
        ordered = sorted(costs)
        emit(
            f"\n{total_alerts} alerts on {len(clips)} clips; pedestrian update over {len(costs)} frames: "
            f"{statistics.mean(costs):.3f} ms mean, {ordered[int(0.99 * (len(ordered) - 1))]:.3f} ms p99, "
            f"{ordered[-1]:.3f} ms max (overrides {args.set})"
        )
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
