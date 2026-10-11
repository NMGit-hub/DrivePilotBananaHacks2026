"""Every detection in the mined hard-negative frames, as review_sheet.py records, for checking them by eye.

    .venv/Scripts/python scripts/signs/dump_det_hits.py --weights runs/signs/det/weights/best.pt
    .venv/Scripts/python scripts/signs/review_sheet.py data/signs/hardneg_det_dets.jsonl --kind speed_limit --count 132

Re-runs the detector (PyTorch, GPU, conf >= 0.3 like mine_det_negatives.py) on the frames listed in
data/signs/hardneg_det_raw.txt and writes each detection that doesn't sit on a masked ("unclear") sign to
data/signs/hardneg_det_dets.jsonl (sorted by kind, then confidence), and the non-speed kinds again to
data/signs/hardneg_det_rest.jsonl: the tile numbers of those sheets are what curate_det_negatives.py refers to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.sign_classes import VALUE_KINDS  # noqa: E402
from scripts.signs.prepare_data import BDD_IMAGES, DATA, bdd_samples, rel_path  # noqa: E402

RAW = DATA / "hardneg_det_raw.txt"
DETS = DATA / "hardneg_det_dets.jsonl"
REST = DATA / "hardneg_det_rest.jsonl"


def main(argv: list[str] | None = None) -> int:
    """Write the detection records of the mined frames."""
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args(argv)
    mined = {line.split("\t")[0] for line in RAW.read_text(encoding="utf-8").splitlines() if line}
    _pos, neg = bdd_samples("train")
    model = YOLO(str(args.weights), task="detect")
    out: list[dict[str, Any]] = []
    for s in neg:
        if rel_path(s.image) not in mined:
            continue
        preds: Any = model.predict(str(s.image), imgsz=1280, conf=args.conf, device=0, verbose=False)
        res = preds[0]
        masked = [b.xyxy for b in s.boxes if b.kind is None]
        for xyxy, conf, cls in zip(
            res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist(), strict=True
        ):
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            if any(m[0] <= cx <= m[2] and m[1] <= cy <= m[3] for m in masked):
                continue
            out.append(
                {
                    "image": s.image.relative_to(BDD_IMAGES).as_posix(),
                    "box": [round(v, 1) for v in xyxy],
                    "kind": model.names[int(cls)],
                    "value": round(conf, 2),  # review_sheet captions show this field: here the confidence
                }
            )
    out.sort(key=lambda r: (r["kind"], -r["value"]))
    DETS.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    rest = [r for r in out if r["kind"] not in VALUE_KINDS]
    REST.write_text("\n".join(json.dumps(r) for r in rest) + "\n", encoding="utf-8")
    print(f"{len(out)} detections in {len(mined)} frames -> {DETS} ({len(rest)} non-speed -> {REST})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
