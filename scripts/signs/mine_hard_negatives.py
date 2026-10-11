"""Hard negatives for the value classifier: BDD train signs that are NOT speed limits but read as a number.

    .venv/Scripts/python scripts/signs/mine_hard_negatives.py --weights models/signs/speed_value_cls.pt

Every BDD100K train-split sign box labeled "other" or "not_a_sign" (route shields, parking and text signs...) is
cropped like a detection and classified; the ones the classifier reads as a speed value (top-1 not "none", p >=
--min-p) are written to data/signs/hardneg_cls.jsonl with a contact sheet (data/signs/hardneg_cls.png) for a
check by eye (pseudo-labels can be wrong: a real speed limit labeled "other" must not become a negative; list
its row number in --exclude). prepare_data.py cls then adds them to the reject class with jittered copies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.signs import crop_square, parse_value_class  # noqa: E402
from scripts.signs.prepare_data import BDD_IMAGES, dev_clip_ids, read_labels  # noqa: E402

OUT = REPO_ROOT / "data" / "signs" / "hardneg_cls.jsonl"
SHEET = REPO_ROOT / "data" / "signs" / "hardneg_cls.png"


def main(argv: list[str] | None = None) -> int:
    """Classify every non-speed BDD train sign crop; keep those read as a number."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "signs" / "speed_value_cls.pt")
    p.add_argument("--min-p", type=float, default=0.3)
    p.add_argument("--min-h", type=float, default=14.0)
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="rows (from the sheet) that are real limits")
    args = p.parse_args(argv)
    model = YOLO(str(args.weights), task="classify")
    banned = dev_clip_ids()
    recs = [
        r
        for r in read_labels("train")
        if r["kind"] in ("other", "not_a_sign") and r["box"][3] - r["box"][1] >= args.min_h
    ]
    recs.sort(key=lambda r: r["image"])
    found = []
    cache: tuple[str, np.ndarray | None] = ("", None)
    batch: list[tuple[dict, np.ndarray]] = []

    def flush() -> None:
        if not batch:
            return
        results: Any = model.predict([c for _, c in batch], imgsz=96, device=0, verbose=False)
        for (rec, crop), res in zip(batch, results, strict=True):
            probs = res.probs.data.float().cpu().numpy()
            top = int(np.argmax(probs))
            value = parse_value_class(model.names[top])
            if value is not None and probs[top] >= args.min_p:
                found.append({**rec, "read": value, "p": round(float(probs[top]), 3), "crop": crop})
        batch.clear()

    for r in recs:
        if Path(r["image"]).stem in banned:
            continue
        if cache[0] != r["image"]:
            cache = (r["image"], cv2.imread(str(BDD_IMAGES / r["image"])))
        img = cache[1]
        if img is None:
            continue
        crop = crop_square(img, tuple(r["box"]), 96)
        if crop is not None:
            batch.append((r, crop))
        if len(batch) >= 64:
            flush()
    flush()
    found.sort(key=lambda f: -f["p"])
    excluded = set(args.exclude)
    with OUT.open("w", encoding="utf-8") as f:
        for n, rec in enumerate(found):
            if n not in excluded:
                f.write(json.dumps({k: v for k, v in rec.items() if k != "crop"}) + "\n")
    cols = 16
    tiles = []
    for n, rec in enumerate(found):
        t = cv2.resize(rec["crop"], (96, 96))
        cv2.putText(t, f"{n}:{rec['read']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((96, 96, 3), np.uint8))
    if tiles:
        rows = [np.concatenate(tiles[i : i + cols], axis=1) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(SHEET), np.concatenate(rows, axis=0))
    print(f"{len(recs)} non-speed crops checked, {len(found)} read as a value (p >= {args.min_p}) -> {OUT}, {SHEET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
