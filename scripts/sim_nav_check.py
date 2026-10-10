"""Drive saved routes (simulated GPS) through the real navigation handler and check every alert (V2-M0).

    .venv/Scripts/python scripts/sim_nav_check.py                      # every route in data/routes, seed 7
    .venv/Scripts/python scripts/sim_nav_check.py --seeds 1-20 --quiet  # robustness over GPS noise
    .venv/Scripts/python scripts/sim_nav_check.py --routes data/routes/redmond_town_center.json --missed-turn 3

Each route is driven twice: along the route (stopping before its first two turns), and with a missed turn (by
default the first left/right turn at least 150 m from the start and more than 500 m from the destination, where
leaving the route would count as turning in at it; a route without one, like the 500 m Carmel roundabout route,
gets only the first drive). The route is served from the saved Mapbox
response, so nothing calls Mapbox; the navigation settings (lane reminder distances, off-route rule, arrival) come
from config.yaml. `eval/nav_check.py` says what is checked. Prints each drive's alerts with the distance to their
maneuver (true and as the handler measured it) and the checks, then a table. Exits 1 if any check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import NavConfig, load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from eval.nav_check import NavCheck, check_drive, report_lines  # noqa: E402
from nav.mapbox import Place  # noqa: E402
from nav.progress import NEAR_END_ALONG_M  # noqa: E402
from nav.route import Route, parse_directions  # noqa: E402
from nav.sim_drive import SimParams, simulate, summary_lines  # noqa: E402

TURNS = {"left", "right", "sharp left", "sharp right"}
EXIT_RING = {"exit roundabout", "exit rotary"}
STOP_KINDS = ("turn", "end of road")  # stop signs and lights; roundabouts are yields
AUTO_STOPS = 2
MIN_MISSED_START_M = 150.0  # miss a turn only once we're under way


@dataclass(frozen=True)
class Saved:
    """A saved route file."""

    name: str
    place: Place
    response: dict[str, Any]
    route: Route


def load_saved(path: Path) -> Saved:
    """Read `data/routes/<name>.json` ({"place": ..., "directions": raw Mapbox response})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["place"]
    place = Place(raw["name"], raw.get("address", raw["name"]), (float(raw["location"][0]), float(raw["location"][1])))
    return Saved(path.stem, place, data["directions"], parse_directions(data["directions"], place.name))


def seeds_arg(text: str) -> list[int]:
    """ "7", "1,2,5" or "1-20"."""
    out: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out += list(range(int(low), int(high or low) + 1))
    return out


def stop_steps(route: Route) -> list[int]:
    """The first few plain turns: where to stand still on the drive along the route."""
    return [s.index for s in route.steps[1:-1] if s.maneuver in STOP_KINDS and s.modifier in TURNS][:AUTO_STOPS]


def missed_candidates(route: Route) -> list[int]:
    """Turns to miss, best first: plain turns, then ends of roads and turning ramps; once under way, and far enough
    from the destination that leaving the route there can't count as turning in at it (nav/progress.py)."""
    turns = [
        s
        for s in route.steps[1:-1]
        if s.modifier in TURNS
        and not s.is_roundabout
        and s.maneuver not in EXIT_RING
        and s.start_m >= MIN_MISSED_START_M
        and route.line.length_m - s.start_m > NEAR_END_ALONG_M
    ]
    return [s.index for s in sorted(turns, key=lambda s: (s.maneuver != "turn", s.index))]


def run_route(saved: Saved, cfg: NavConfig, params: SimParams, seed: int, missed_turn: int | None) -> list[NavCheck]:
    """The drive along the route and the missed-turn drive."""
    drive = simulate(saved.route, params, stop_steps(saved.route), None, seed)
    results = [check_drive(drive, saved.response, saved.place, cfg)]
    candidates = [missed_turn] if missed_turn is not None else missed_candidates(saved.route)
    for index in candidates:
        try:
            drive = simulate(saved.route, params, (), index, seed)
        except ValueError as e:
            print(f"  {saved.name}: can't miss step {index}: {e}")
            continue
        results.append(check_drive(drive, saved.response, saved.place, cfg))
        break
    else:
        print(
            f"  {saved.name}: no turn to miss (each is in the first {MIN_MISSED_START_M:g} m, a roundabout, or within "
            f"{NEAR_END_ALONG_M:g} m of the destination, where leaving the route counts as turning in)"
        )
    return results


def describe(result: NavCheck) -> str:
    """One table row's measurements."""

    def measured(kind: str) -> list[tuple[float, float]]:  # (true distance, due distance) of checked alerts
        return [
            (r.true_m, r.target_m)
            for r in result.alerts
            if r.kind == kind and r.ok is not None and r.true_m is not None and r.target_m is not None
        ]

    cells = []
    lanes = measured("nav_lanes")
    if lanes:
        errors = [(true - due) / due for true, due in lanes]
        cells.append(f"{len(lanes)} lane reminders at {min(errors):+.1%}..{max(errors):+.1%} of their distance")
    rounds = measured("nav_roundabout")
    if rounds:
        cells.append("roundabout said " + ", ".join(f"{true:.0f} m (due {due:.0f})" for true, due in rounds))
    exits = measured("nav_not_this_exit")
    if exits:
        cells.append("'not this one' " + ", ".join(f"{-true:+.0f}" for true, _ in exits) + " m past the exits")
    arrived = measured("nav_arrived")
    if arrived:
        cells.append(f"arrival {arrived[0][0]:.0f} m from the end")
    if result.drive.missed is not None and result.left_route_t is not None and result.reroute_t is not None:
        cells.append(f"reroute asked {result.reroute_t - result.left_route_t:.1f} s after leaving the route")
    return "; ".join(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", type=Path, nargs="*", default=[REPO_ROOT / "data" / "routes"])
    parser.add_argument("--seeds", type=seeds_arg, default=[7], help="7, 1,2,5 or 1-20")
    parser.add_argument("--noise-m", type=float, default=SimParams().noise_m, help="GPS error std per axis (m).")
    parser.add_argument("--missed-turn", type=int, default=None, help="Step to miss (default: chosen per route).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--quiet", action="store_true", help="Only the table.")
    args = parser.parse_args(argv)

    setup_logging("ERROR")  # the handler logs every alert and reroute; the report says it better
    cfg = load_config(args.config, env_file=None).nav
    files = sorted(f for p in args.routes for f in (p.glob("*.json") if p.is_dir() else [p]))
    if not files:
        print(f"no saved routes in {', '.join(map(str, args.routes))} (scripts/fetch_route.py saves them)")
        return 2
    params = SimParams(noise_m=args.noise_m)
    print(
        f"nav settings: lane reminders at {', '.join(f'{m:g}' for m in cfg.lane_reminders_m)} m; off route > "
        f"{cfg.off_route_m:g} m for {cfg.off_route_s:g} s; arrival within {cfg.arrival_m:g} m on the route or "
        f"{cfg.arrival_radius_m:g} m away; GPS noise {params.noise_m:g} m per axis"
    )
    rows: list[tuple[str, str, int, NavCheck]] = []
    for path in files:
        saved = load_saved(path)
        for seed in args.seeds:
            for result in run_route(saved, cfg, params, seed, args.missed_turn):
                drive = result.drive
                label = "along the route" if drive.missed is None else f"missed turn at step {drive.missed.index}"
                rows.append((saved.name, label, seed, result))
                if args.quiet:
                    continue
                print(f"\n=== {saved.name}, {label}, seed {seed}: {'PASS' if result.ok else 'FAIL'}")
                print("\n".join(summary_lines(drive)))
                print("\n".join(report_lines(result)))

    print(f"\n{'route':<22} {'drive':<24} {'seed':>4}  {'checks':>7}  measurements")
    for name, label, seed, result in rows:
        passed = sum(c.ok for c in result.checks)
        print(f"{name:<22} {label:<24} {seed:>4}  {passed:>3}/{len(result.checks):<3}  {describe(result)}")
        for check in result.checks:
            if not check.ok:
                print(f"{'':<22} FAIL {check.name}: {check.detail}")
    failed = sum(not r.ok for *_, r in rows)
    print(f"\n{len(rows) - failed}/{len(rows)} drives passed every check")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
