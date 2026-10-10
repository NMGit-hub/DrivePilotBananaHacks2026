"""Simulate a drive along a saved Mapbox route and write it as a GPX track the app replays (HANDOFF.md 5B).

    .venv/Scripts/python scripts/route_to_gpx.py --route data/routes/redmond_town_center.json
    .venv/Scripts/python scripts/route_to_gpx.py --route data/routes/redmond_town_center.json
        --out data/gps/sim_redmond_missed.gpx --stop-before 1,2 --stop-s 20 --missed-turn 3 --seed 7 --noise-m 4
    .venv/Scripts/python -m app --video <any clip> --gpx data/gps/sim_redmond_town_center.gpx
        --set video.file.loop=true
    (indented lines continue the command above; in the app, say or type "I'm going to <the place>" while the
    simulated car still waits at the start)

Realistic speeds from the route's step durations, slowdowns before maneuvers, optional stops and a missed turn,
one fix a second with a few metres of slowly wandering GPS error (see `nav/sim_drive.py`). Prints the drive:
distance, time, speeds and when each maneuver is reached. `scripts/fetch_route.py` saves routes; the GPX goes to
`data/gps/` (gitignored), with a `video_start` waypoint at its first fix (video t=0).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nav.route import parse_directions  # noqa: E402
from nav.sim_drive import DEFAULT_START_TIME, SimParams, simulate, summary_lines  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "gps"


def step_list(text: str) -> list[int]:
    """ "3,5" -> [3, 5]."""
    return [int(v) for v in text.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    defaults = SimParams()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", type=Path, required=True, help="Saved route (data/routes/<name>.json).")
    parser.add_argument("--out", type=Path, default=None, help="GPX file (default: data/gps/sim_<route>.gpx).")
    parser.add_argument("--stop-before", type=step_list, default=[], help="Step numbers to stop before: 3,5")
    parser.add_argument("--stop-s", type=float, default=defaults.stop_s, help="Seconds standing at each stop.")
    parser.add_argument("--missed-turn", type=int, default=None, help="Step number to drive straight past.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-m", type=float, default=defaults.noise_m, help="GPS error std per axis (m).")
    parser.add_argument("--start-wait-s", type=float, default=defaults.start_wait_s, help="Standing at the start.")
    parser.add_argument(
        "--start-time", type=datetime.fromisoformat, default=DEFAULT_START_TIME, help="GPX time of the first fix."
    )
    args = parser.parse_args(argv)

    data = json.loads(args.route.read_text(encoding="utf-8"))
    place = data.get("place") or {}
    route = parse_directions(data.get("directions", data), str(place.get("name", args.route.stem)))
    params = replace(defaults, stop_s=args.stop_s, noise_m=args.noise_m, start_wait_s=args.start_wait_s)
    try:
        drive = simulate(route, params, args.stop_before, args.missed_turn, args.seed)
    except ValueError as e:
        print(f"can't simulate that drive: {e}", file=sys.stderr)
        return 2
    out = args.out or DEFAULT_OUT / f"sim_{args.route.stem}.gpx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(drive.to_gpx(args.start_time), encoding="utf-8")
    print("\n".join(summary_lines(drive)))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
