"""Signs BDD's annotators never boxed, found by the trained detector, so training masks them instead of teaching
them as background.

    .venv/Scripts/python scripts/signs/mine_unboxed.py --weights runs/signs/det/weights/best.pt

BDD100K boxes many but not all traffic signs: checking the mined hard negatives by eye (curate_det_negatives.py)
showed most confident detections on sign-free frames were real, unboxed one-way, school-zone, stop and speed-limit
signs. In training those are background. This runs the detector (PyTorch, GPU) on every BDD train frame outside
the val fraction and writes each detection >= --conf that touches no human sign box to
data/signs/unboxed_det.jsonl (with kind and confidence); prepare_data.py det paints them over (like signs of unknown
kind) in training frames and synthetic backgrounds, except speed-kind detections: about a third of these hits are
look-alikes, and a route shield or advisory plate must stay a negative. The curated hard-negative frames are
skipped: their detections were checked and must stay background. Evaluation labels never use this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.signs.prepare_data import (  # noqa: E402
    BDD_IMAGES,
    HARDNEG_DET,
    UNBOXED,
    bdd_samples,
    bdd_sign_boxes,
    in_val,
    rel_path,
    touches_any,
)


def main(argv: list[str] | None = None) -> int:
    """Write the unboxed detections of the training frames."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args(argv)
    hard = set()
    if HARDNEG_DET.exists():
        hard = {line.split("\t")[0] for line in HARDNEG_DET.read_text(encoding="utf-8").splitlines() if line}
    pos, neg = bdd_samples("train")
    frames = [s for s in pos + neg if not in_val(s.image) and rel_path(s.image) not in hard]
    meta = bdd_sign_boxes("train")
    model = YOLO(str(args.weights), task="detect")
    out: list[dict[str, Any]] = []
    n_boxes = 0
    for n, s in enumerate(frames):
        human = list(meta.get(s.image.relative_to(BDD_IMAGES).as_posix(), {}).values())
        preds: Any = model.predict(str(s.image), imgsz=1280, conf=args.conf, device=0, verbose=False)
        res = preds[0]
        hits = [
            ([round(v, 1) for v in xyxy], model.names[int(c)], round(float(p), 3))
            for xyxy, c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=True)
            if not touches_any(tuple(xyxy), human)
        ]
        if hits:
            out.append(
                {
                    "image": rel_path(s.image),
                    "boxes": [h[0] for h in hits],
                    "kinds": [h[1] for h in hits],
                    "confs": [h[2] for h in hits],
                }
            )
            n_boxes += len(hits)
        if n % 1000 == 0:
            print(f"{n}/{len(frames)} frames, {n_boxes} unboxed detections", flush=True)
    UNBOXED.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    print(f"{n_boxes} unboxed detections >= {args.conf} in {len(out)} of {len(frames)} training frames -> {UNBOXED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
