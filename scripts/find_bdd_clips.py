"""Find BDD100K clips by scene ("parking lot", "highway", ...), check their GPS, and fetch some with their GPX.

    .venv/Scripts/python scripts/find_bdd_clips.py --scene "parking lot"                 # list candidates
    .venv/Scripts/python scripts/find_bdd_clips.py --scene "parking lot" --fetch 30      # download 30 + GPX
    .venv/Scripts/python scripts/find_bdd_clips.py --scene highway --min-fast-s 20 --fetch 10
    .venv/Scripts/python scripts/find_bdd_clips.py --scene "parking lot" --min-crawl-s 15 --max-mph 15 \
        --out parking_crawl --fetch 30      # cars crawling through a lot the whole clip (looking for a spot)

Scene labels: BDD100K tags the frame at 10 s of every clip with a scene; a public copy of those tags
(`dronefreak/BDD100K-Scenario-Classification` on Hugging Face, file names = clip ids, no login) is downloaded
once (~11 small metadata files) and cached in data/bdd100k/scene_labels.json. The GPS ("info" files,
HANDOFF.md 5A) says how the car moves: `--min-slow-s` keeps clips that move slowly (1-16 mph: not parked for the
whole clip); `--min-crawl-s` + `--max-mph` keep clips that crawl at parking-search speed (1-8 mph) and never go
faster than that cap; `--min-fast-s` keeps clips that drive at speed. Clips already in a `--skip` folder are left
out. Videos come by
range request from the archive mirror (scripts/fetch_dev_clips.py) into data/videos/<scene>/<id>.mov, GPX
into data/gps/<id>.gpx (both gitignored). Candidates are picked in a seeded random order.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_dev_clips import download_clip, load_index  # noqa: E402

from nav.bdd_info import BddInfoError, parse_info, read_info, to_gpx  # noqa: E402

SCENE_BASE = "https://huggingface.co/datasets/dronefreak/BDD100K-Scenario-Classification/resolve/main/data/images/"
SCENE_SHARDS = {"valid": 2, "train": 9}
SCENE_CACHE = REPO_ROOT / "data" / "bdd100k" / "scene_labels.json"
INFO_ZIP = REPO_ROOT / "data" / "bdd100k" / "bdd100k_info.zip"
GPS_DIR = REPO_ROOT / "data" / "gps"
SLOW_MPS = (0.5, 7.0)  # moving, but at parking-lot speed (1-16 mph)
CRAWL_MPS = (0.5, 3.6)  # looking for a spot (1-8 mph)
FAST_MPS = 15.0  # ~34 mph


def scene_labels() -> dict[str, str]:
    """Clip id → scene label for every labeled BDD100K clip (train + val), cached after the first download."""
    if SCENE_CACHE.exists():
        return json.loads(SCENE_CACHE.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for split, shards in SCENE_SHARDS.items():
        for k in range(shards):
            url = f"{SCENE_BASE}{split}/shard_{k:03d}/metadata.jsonl"
            with urllib.request.urlopen(url, timeout=60) as response:
                for line in response.read().decode("utf-8").splitlines():
                    row = json.loads(line)
                    labels[Path(row["file_name"]).stem] = row["label"]
    SCENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCENE_CACHE.write_text(json.dumps(labels), encoding="utf-8")
    return labels


@dataclass(frozen=True, slots=True)
class Motion:
    """How the car moves in a clip, from its GPS."""

    slow_s: float  # seconds at parking-lot speed
    fast_s: float  # seconds above FAST_MPS
    max_mph: float
    crawl_s: float = 0.0  # seconds at parking-search speed


def motion(clip_id: str) -> Motion | None:
    """Seconds slow / fast and the top speed, or None if the clip's GPS is unusable."""
    try:
        track = parse_info(read_info(INFO_ZIP, clip_id), clip_id)
    except BddInfoError:
        return None
    fixes = [f for s in track.segments for f in s if track.video_start <= f.time <= track.video_end]
    speeds = [f.speed_mps for f in fixes if f.speed_mps is not None]
    if not speeds:
        return None
    return Motion(
        float(sum(SLOW_MPS[0] <= v <= SLOW_MPS[1] for v in speeds)),
        float(sum(v >= FAST_MPS for v in speeds)),
        max(speeds) * 2.23694,
        float(sum(CRAWL_MPS[0] <= v <= CRAWL_MPS[1] for v in speeds)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help='e.g. "parking lot", "highway", "residential"')
    parser.add_argument("--min-slow-s", type=float, default=0.0, help="At least this long at 1-16 mph.")
    parser.add_argument("--min-fast-s", type=float, default=0.0, help="At least this long above ~34 mph.")
    parser.add_argument("--min-crawl-s", type=float, default=0.0, help="At least this long at 1-8 mph.")
    parser.add_argument("--max-mph", type=float, default=None, help="Never faster than this (GPS).")
    parser.add_argument("--out", default=None, help="Folder name under data/videos (default: the scene).")
    parser.add_argument("--skip", action="append", default=[], help="Leave out clips already in this folder.")
    parser.add_argument("--fetch", type=int, default=0, help="Download this many candidates (+ their GPX).")
    parser.add_argument("--limit", type=int, default=400, help="Check the GPS of at most this many clips.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    labels = scene_labels()
    ids = sorted(i for i, scene in labels.items() if scene == args.scene)
    print(f"{len(ids)} clips labeled {args.scene!r}")
    if not INFO_ZIP.is_file():
        print(f"not found: {INFO_ZIP} (HANDOFF.md 5A)", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(ids)
    index = load_index()
    have = {p.stem for folder in args.skip for p in (REPO_ROOT / "data" / "videos" / folder).glob("*.mov")}
    candidates: list[tuple[str, Motion]] = []
    for clip_id in ids[: args.limit]:
        if clip_id not in index or clip_id in have:
            continue
        m = motion(clip_id)
        if m is None or m.slow_s < args.min_slow_s or m.fast_s < args.min_fast_s or m.crawl_s < args.min_crawl_s:
            continue
        if args.max_mph is not None and m.max_mph > args.max_mph:
            continue
        candidates.append((clip_id, m))
    print(f"{len(candidates)} with usable GPS that pass the motion filter (of {min(len(ids), args.limit)} checked)")
    out_dir = REPO_ROOT / "data" / "videos" / (args.out or args.scene.replace(" ", "_"))
    for clip_id, m in candidates[: max(args.fetch, 20)]:
        split = index[clip_id]["split"]
        print(
            f"  {clip_id} ({split})  slow {m.slow_s:4.0f} s  crawl {m.crawl_s:4.0f} s  fast {m.fast_s:4.0f} s  "
            f"max {m.max_mph:3.0f} mph"
        )
    for clip_id, _ in candidates[: args.fetch]:
        dest = out_dir / f"{clip_id}.mov"
        if not dest.exists():
            print(f"downloading {dest.name} ({int(index[clip_id]['size']) / 1e6:.1f} MB)")
            download_clip(index[clip_id], dest)
        GPS_DIR.mkdir(parents=True, exist_ok=True)
        track = parse_info(read_info(INFO_ZIP, clip_id), clip_id)
        (GPS_DIR / f"{clip_id}.gpx").write_text(to_gpx(track), encoding="utf-8")
    if args.fetch:
        print(f"fetched {min(args.fetch, len(candidates))} into {out_dir} (GPX in {GPS_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
