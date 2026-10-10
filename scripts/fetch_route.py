"""Fetch a real Mapbox route and save it for offline replays and tests (data/routes/, gitignored).

    .venv/Scripts/python scripts/fetch_route.py --from 47.6158,-122.2037 --to "Redmond Town Center"
    .venv/Scripts/python scripts/fetch_route.py --from 47.5440,-122.0170 --to "Swedish Issaquah" --name issaquah

Prints the steps (maneuver, distance along the route, lanes, roundabout exits) so routes can be checked by eye.
Saved responses stay out of git (Mapbox terms limit storing results); tests use hand-made fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from nav.mapbox import MapboxClient  # noqa: E402
from nav.route import parse_directions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="origin", required=True, help="lat,lon")
    parser.add_argument("--to", required=True, help="Place name to geocode.")
    parser.add_argument("--heading", type=float, default=None)
    parser.add_argument("--name", default=None, help="File name (default: from the place).")
    args = parser.parse_args(argv)

    cfg = load_config(REPO_ROOT / "config.yaml")
    setup_logging("WARNING", secrets=cfg.secrets.values())
    if cfg.secrets.mapbox_access_token is None:
        print("MAPBOX_ACCESS_TOKEN is not set (.env)")
        return 2
    lat, lon = (float(v) for v in args.origin.split(","))
    client = MapboxClient(cfg.secrets.mapbox_access_token.get_secret_value())
    place = client.geocode(args.to, near=(lat, lon))
    response = client.directions((lat, lon), place.location, args.heading)
    route = parse_directions(response, place.name)
    name = args.name or place.name.lower().replace(" ", "_")
    out = REPO_ROOT / "data" / "routes" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "place": {"name": place.name, "address": place.address, "location": list(place.location)},
        "directions": response,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"{place.name} ({place.address}): {route.distance_m / 1609.34:.1f} mi, {route.duration_s / 60:.0f} min, "
        f"{len(route.steps)} steps -> {out}"
    )
    for step in route.steps:
        lanes = "".join(("V" if lane.valid else "x") for lane in step.lanes)
        extra = f" lanes[{lanes}]" if lanes else ""
        if step.is_roundabout:
            extra += f" exit {step.roundabout_exit}, exits at {[round(m) for m in step.roundabout_exits_m]}"
        if step.exit_ref:
            extra += f" exit-ref {step.exit_ref}"
        print(f"  {step.start_m:7.0f} m  {step.maneuver:16} {step.modifier or ''!s:12} {step.instruction}{extra}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
