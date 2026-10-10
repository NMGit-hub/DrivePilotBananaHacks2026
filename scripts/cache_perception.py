"""Cache perception results (tracks + ego motion, every frame) for clips, for fast replays of downstream logic.

    .venv/Scripts/python scripts/cache_perception.py --labels eval/parking_labels.yaml --videos data/videos/parking_lot
    .venv/Scripts/python scripts/cache_perception.py --clip data/videos/dev/<name>.mov

Writes data/cache/perception/<clip stem>.pkl (gitignored); clips already cached with the current perception
settings are skipped unless --force. Labels files: every clip with `usable: true` (or all with --all).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from eval.perception_cache import build_cache, load_cache  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache" / "perception"
log = get_logger("cache_perception")


def cache_path(clip: Path) -> Path:
    """Where a clip's perception cache lives."""
    return CACHE_DIR / f"{clip.stem}.pkl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, action="append", default=[], help="A video (repeatable).")
    parser.add_argument("--labels", type=Path, help="A labels YAML with a `clips:` list of ids.")
    parser.add_argument("--videos", type=Path, help="Folder of <id>.mov for --labels.")
    parser.add_argument("--all", action="store_true", help="With --labels: also clips marked unusable.")
    parser.add_argument("--force", action="store_true", help="Rebuild caches that exist.")
    parser.add_argument("--set", action="append", default=[], help="Config override key=value.")
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    cfg = load_config(REPO_ROOT / "config.yaml", overrides=args.set)
    clips = list(args.clip)
    if args.labels is not None:
        if args.videos is None:
            parser.error("--labels needs --videos")
        data = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
        clips += [args.videos / f"{c['id']}.mov" for c in data["clips"] if args.all or c.get("usable", True)]
    if not clips:
        parser.error("no clips")
    for clip in clips:
        out = cache_path(clip)
        if out.exists() and not args.force:
            try:
                load_cache(out, cfg)
                print(f"cached   {clip.name}")
                continue
            except ValueError as exc:
                print(f"stale    {clip.name}: {exc}")
        started = time.perf_counter()
        n = build_cache(cfg, clip, out)
        elapsed = time.perf_counter() - started
        print(f"built    {clip.name}: {n} frames in {elapsed:.0f} s ({n / max(elapsed, 1e-6):.0f} fps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
