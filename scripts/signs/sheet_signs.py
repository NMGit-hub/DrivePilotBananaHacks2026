"""What the sign detector sees for the test sheet's sign questions (eval/test_sheet.csv rows of type "signs").

    .venv/Scripts/python scripts/signs/sheet_signs.py --videos <folder with the dev clips>

For each sign row, the frames of the 2 s before the question (every 0.25 s) go through SignDetector at the app's
thresholds; the kinds (and speed values) seen are summarized next to the sheet's expected answer. This is the
"after" for the Best Fine-tune write-up: the vision model alone scored 2/6 on these rows (PROGRESS.md). Feeding
these readings into the vision prompt is the integrating session's job.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import Frame  # noqa: E402
from perception.signs import SignDetector  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the detector on each sign row's recent frames and print what it saw."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, default=REPO_ROOT / "data" / "videos" / "dev")
    p.add_argument("--window", type=float, default=2.0, help="seconds before the question")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "signs" / "sheet_signs.json")
    args = p.parse_args(argv)
    cfg = load_config()
    det = SignDetector(cfg.signs, cfg.root)
    rows = [r for r in csv.DictReader((REPO_ROOT / "eval" / "test_sheet.csv").open(encoding="utf-8"))]
    results: list[dict[str, Any]] = []
    for r in rows:
        if r["type"] != "signs":
            continue
        clip = next(iter(args.videos.glob(f"{r['clip']}__*.mov")), None)
        if clip is None:
            print(f"missing clip for {r['clip']}")
            continue
        cap = cv2.VideoCapture(str(clip))
        seen: dict[str, dict[str, Any]] = defaultdict(lambda: {"frames": 0, "max_conf": 0.0, "values": set()})
        t = max(0.0, float(r["time_s"]) - args.window)
        n = 0
        while t <= float(r["time_s"]) + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if ok:
                n += 1
                for d in det.detect(Frame(img, t=t, index=n, captured_at=0.0)):
                    s = seen[d.kind]
                    s["frames"] += 1
                    s["max_conf"] = max(s["max_conf"], round(d.conf, 2))
                    if d.value is not None:
                        s["values"].add(d.value)
            t += 0.25
        cap.release()
        summary = {k: {**v, "values": sorted(v["values"])} for k, v in seen.items()}
        results.append(
            {
                "clip": r["clip"],
                "time_s": r["time_s"],
                "question": r["question"],
                "expected": r["expected"],
                "frames": n,
                "detector": summary,
            }
        )
        print(f"\n{r['clip']} @ {r['time_s']}s: {r['question']}\n  expected: {r['expected'][:110]}")
        print(f"  detector ({n} frames): {json.dumps(summary)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
