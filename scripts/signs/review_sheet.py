"""Contact sheets of labeled sign crops, for checking labels by eye.

    .venv/Scripts/python scripts/signs/review_sheet.py data/signs/bdd_labels/valid.jsonl --kind speed_limit
    .venv/Scripts/python scripts/signs/review_sheet.py <labels.jsonl> --kind not_a_sign --sample 96 --out sheet.png

Each tile is the box with 40% context at a fixed size, captioned with its row number in the (filtered) list,
the label's value and the box height in pixels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
BDD_ROOT = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
TILE = 112


def tile(rec: dict, n: int, cache: dict[str, Image.Image]) -> Image.Image:
    """One captioned crop tile."""
    path = str(BDD_ROOT / rec["image"])
    if path not in cache:
        cache[path] = Image.open(path).convert("RGB")
    img = cache[path]
    x1, y1, x2, y2 = rec["box"]
    pad = 0.4 * max(x2 - x1, y2 - y1)
    crop = img.crop((int(x1 - pad), int(y1 - pad), int(x2 + pad + 1), int(y2 + pad + 1)))
    s = TILE / max(crop.width, crop.height)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.Resampling.BICUBIC)
    out = Image.new("RGB", (TILE, TILE + 14), "white")
    out.paste(crop, ((TILE - crop.width) // 2, 14 + (TILE - crop.height) // 2))
    ImageDraw.Draw(out).text(
        (2, 0),
        f"{n} {rec['kind'][:9]} {rec.get('value') or ''} h{int(y2 - y1)}",
        fill="black",
        font=ImageFont.load_default(size=11),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Write a contact sheet PNG of the selected labels."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", type=Path)
    p.add_argument("--kind", default=None, help="only this kind (default: all)")
    p.add_argument("--sample", type=int, default=0, help="random sample of this many (0 = all, in order)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=120)
    p.add_argument("--cols", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    recs = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
    if args.kind:
        recs = [r for r in recs if r["kind"] == args.kind]
    if args.sample:
        recs = random.Random(args.seed).sample(recs, min(args.sample, len(recs)))
    chosen = list(enumerate(recs))[args.start : args.start + args.count]
    if not chosen:
        print("nothing to show")
        return 1
    rows = (len(chosen) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * (TILE + 2), rows * (TILE + 16)), (200, 200, 200))
    cache: dict[str, Image.Image] = {}
    for k, (n, rec) in enumerate(chosen):
        sheet.paste(tile(rec, n, cache), ((k % args.cols) * (TILE + 2), (k // args.cols) * (TILE + 16)))
    out = args.out or REPO_ROOT / "data" / "signs" / f"sheet_{args.kind or 'all'}_{args.start}.png"
    sheet.save(out)
    print(f"{len(recs)} matching labels; wrote {len(chosen)} tiles to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
