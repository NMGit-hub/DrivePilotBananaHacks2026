"""Hard negatives for the detector: BDD train frames without any of our signs where the detector still fires.

    .venv/Scripts/python scripts/signs/mine_det_negatives.py --weights runs/signs/det/weights/best.pt

Runs the detector (PyTorch, GPU) on every pseudo-labeled BDD100K train frame that holds none of our 13 kinds,
outside the held-out val fraction (the tune set); frames with a detection above --conf that doesn't sit on an "unclear"
(masked) sign are written to data/signs/hardneg_det_raw.txt. Most of those hits turned out to be real signs BDD
never boxed, so they are NOT used as they are: dump_det_hits.py makes contact sheets of every hit and
curate_det_negatives.py writes the frames checked by eye to data/signs/hardneg_det.txt, which prepare_data.py det
adds as negative frames (route shields, advisory plates and look-alikes learned as "not ours").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.signs.prepare_data import bdd_samples, in_val  # noqa: E402

OUT = REPO_ROOT / "data" / "signs" / "hardneg_det_raw.txt"  # unchecked: curate_det_negatives.py reads it


def main(argv: list[str] | None = None) -> int:
    """Find negative frames with confident false detections."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args(argv)
    model = YOLO(str(args.weights), task="detect")
    _pos, all_neg = bdd_samples("train")
    neg = [s for s in all_neg if not in_val(s.image)]  # the val fraction stays untouched (tune set)
    hits: list[tuple[str, float, str]] = []
    for n, s in enumerate(neg):
        res: Any = model.predict(str(s.image), imgsz=1280, conf=args.conf, device=0, half=True, verbose=False)
        boxes = res[0].boxes
        masked = [b.xyxy for b in s.boxes if b.kind is None]
        best = None
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True):
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            if any(m[0] <= cx <= m[2] and m[1] <= cy <= m[3] for m in masked):
                continue  # on a sign of unknown kind: may be right
            if best is None or conf > best[0]:
                best = (conf, model.names[int(cls)])
        if best is not None:
            hits.append((s.image.relative_to(REPO_ROOT).as_posix(), round(best[0], 3), best[1]))
        if n % 500 == 0:
            print(f"{n}/{len(neg)} frames, {len(hits)} with false detections", flush=True)
    OUT.write_text("\n".join(f"{img}\t{conf}\t{kind}" for img, conf, kind in hits) + "\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for _, _, k in hits:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(hits)} of {len(neg)} negative frames have a detection >= {args.conf}: {kinds} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
