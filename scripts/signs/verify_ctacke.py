"""Which ctacke speed-limit crops are actually readable? Asks Claude to read each one (same grids as the labeler).

    .venv/Scripts/python scripts/signs/verify_ctacke.py --env-file <path to .env>

ctacke labels every frame of a sign sequence with the sign's value, including dark, blurred and distant frames
where no one can read the number. Training the value classifier on those teaches it to guess confidently, so each
crop is re-read: the value counts only when the model reads the same number; otherwise the crop is "unreadable"
and trains the classifier's reject class instead. Output: data/signs/ctacke_readable.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.signs import label_bdd_signs as lab  # noqa: E402
from scripts.signs.prepare_data import ctacke_samples  # noqa: E402

OUT = REPO_ROOT / "data" / "signs" / "ctacke_readable.jsonl"


def main(argv: list[str] | None = None) -> int:
    """Re-read every ctacke crop; write key, labeled value, read value."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    args = p.parse_args(argv)
    crops: list[lab.Crop] = []
    labeled: dict[str, int | None] = {}
    for split in ("train", "val"):
        for s in ctacke_samples(split):
            for k, b in enumerate(s.boxes):
                key = f"ctacke/{split}/{s.image.parent.name}/{s.image.stem}/{k}"
                crops.append(lab.Crop(key, s.image, b.xyxy))
                labeled[key] = b.value
    tmp = OUT.with_suffix(".raw.jsonl")
    done = set()
    if tmp.exists():
        done = {json.loads(line)["key"] for line in tmp.read_text(encoding="utf-8").splitlines() if line}
    todo = [c for c in crops if c.key not in done]
    per = lab.COLS * lab.ROWS
    grids = [todo[i : i + per] for i in range(0, len(todo), per)]
    print(f"{len(crops)} ctacke crops, {len(todo)} to read in {len(grids)} grids")
    labeler = lab.Labeler(args.model, lab.read_key(args.env_file), tmp, args.max_usd)
    # The labeler writes paths relative to the BDD folder; ctacke images live elsewhere, so patch the root.
    lab.BDD_ROOT = REPO_ROOT
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(labeler.label, grids))
    rows = [json.loads(line) for line in tmp.read_text(encoding="utf-8").splitlines() if line]
    readable = 0
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            ok = r["kind"] in ("speed_limit", "school_speed_limit") and r["value"] == labeled[r["key"]]
            readable += ok
            f.write(
                json.dumps({"key": r["key"], "labeled": labeled[r["key"]], "read": r["value"], "readable": ok}) + "\n"
            )
    print(f"{readable}/{len(rows)} readable (read value == label), ${labeler.usd:.2f} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
