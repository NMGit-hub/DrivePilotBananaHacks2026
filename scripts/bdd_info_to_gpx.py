"""Convert BDD100K clips' GPS ("info" files) to GPX tracks the app replays with the clip's video.

    .venv/Scripts/python scripts/bdd_info_to_gpx.py --dev                    # every clip in eval/dev_clips.yaml
    .venv/Scripts/python scripts/bdd_info_to_gpx.py b97b9ee8-612c36ef c4130097-887da34f
    .venv/Scripts/python -m app --video data/videos/dev/<name>__<id>.mov --gpx data/gps/<id>.gpx

Reads `data/bdd100k/bdd100k_info.zip` without unpacking it (HANDOFF.md section 5A) and writes
`data/gps/<clip id>.gpx` (gitignored). The GPX's `video_start` waypoint lines it up with the video; see
`nav/bdd_info.py` for what is cleaned. Exits with status 1 if any clip failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nav.bdd_info import BddInfoError, parse_info, read_info, to_gpx  # noqa: E402

DEFAULT_ZIP = REPO_ROOT / "data" / "bdd100k" / "bdd100k_info.zip"
DEFAULT_OUT = REPO_ROOT / "data" / "gps"


def dev_clip_ids() -> list[str]:
    """The BDD100K ids of the dev clips."""
    clips = yaml.safe_load((REPO_ROOT / "eval" / "dev_clips.yaml").read_text(encoding="utf-8"))["clips"]
    return [str(c["bdd_id"]) for c in clips]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="*", help="BDD100K clip ids (xxxxxxxx-xxxxxxxx).")
    parser.add_argument("--dev", action="store_true", help="Also every clip in eval/dev_clips.yaml.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="bdd100k_info.zip")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder for the GPX files.")
    parser.add_argument("--max-accuracy-m", type=float, default=100.0, help="Drop fixes less accurate than this.")
    parser.add_argument("--gap-s", type=float, default=5.0, help="Split the track where fixes are further apart.")
    args = parser.parse_args(argv)
    ids = list(dict.fromkeys([*args.clips, *(dev_clip_ids() if args.dev else [])]))
    if not ids:
        parser.error("give clip ids or --dev")
    if not args.zip.is_file():
        print(f"not found: {args.zip} (HANDOFF.md section 5A)", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for clip_id in ids:
        try:
            track = parse_info(read_info(args.zip, clip_id), clip_id, args.max_accuracy_m, args.gap_s)
        except BddInfoError as e:
            failed += 1
            print(f"{clip_id}  FAILED: {e}")
            continue
        path = args.out / f"{clip_id}.gpx"
        path.write_text(to_gpx(track), encoding="utf-8")
        speeds = [f.speed_mps for s in track.segments for f in s if f.speed_mps is not None]
        dropped = ", ".join(f"{n} {why}" for why, n in track.dropped.items()) or "none"
        print(
            f"{clip_id}  {track.fixes} fixes in {len(track.segments)} segment(s); first fix "
            f"{track.first_fix_after_start_s:+.2f} s from the video start; "
            f"max speed {max(speeds, default=0.0) * 2.23694:.0f} mph; dropped: {dropped}  -> {path}"
        )
    print(f"{len(ids) - failed}/{len(ids)} converted")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
