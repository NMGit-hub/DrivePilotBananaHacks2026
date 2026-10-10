"""Write the hand-checked BDD100K held-out sign set: eval/sign_labels_bdd.json (tracked).

    .venv/Scripts/python scripts/signs/make_heldout.py

Source: the first 2,623 frames of the BDD100K "valid" split mirror (shard_000) whose human "traffic sign" boxes
were classified by Claude Opus 5.5 (scripts/signs/label_bdd_signs.py), then checked by eye:
- every speed-limit / school-speed-limit box (92) was inspected on the full-resolution frame crop: all kinds and
  all read values were right; 12 are not clearly readable (REVIEW below), so they don't count toward the
  "clearly readable" recall (a detection there may still not report a wrong value);
- recall of the labeler for speed limits: the 264 largest of 774 upright, low-saturation signs it called
  "other"/"unclear" were inspected: no readable speed limit was missed;
- other kinds keep Opus's labels (spot-checked on contact sheets, not exhaustively).
Boxes labeled "unclear", sign boxes too small to label (< 12 px tall) and REVIEW["ignore"] are ignore regions: a
detection there is neither a hit nor a false alarm (but a wrong speed value there still counts as wrong).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.sign_classes import KIND_INDEX  # noqa: E402
from scripts.signs.prepare_data import bdd_sign_boxes, dev_clip_ids, read_labels  # noqa: E402

OUT = REPO_ROOT / "eval" / "sign_labels_bdd.json"

# Hand review of the 92 speed-limit boxes (keys: valid/<image>/<box index>), done on full-resolution crops.
REVIEW = {
    # Readable to a person only with doubt (dark, washed out, blurred, cut by the frame) or no number visible.
    "not_clearly_readable": [
        "valid/0605d7f1-4e52a6f0/13",  # "40" in reddish night light, 26 px
        "valid/083cbb05-fd4bd01d/5",  # blurred 25, 23 px
        "valid/0bbd0cc8-efe17cb6/18",  # dark, number not visible
        "valid/0c5cdb09-421eaaa8/15",  # dark, number not visible
        "valid/189ca5ba-7c87db20/13",  # only the SPEED LIMIT header in the box
        "valid/1d78d5cb-3d9b5d09/9",  # top cut off, 21 px
        "valid/1fd0f6e8-8d31211d/12",  # dark and blurred
        "valid/254e87ff-16a80849/15",  # possibly 15, too blurred
        "valid/25f855cd-9954f3c8/17",  # overexposed, 20 px
        "valid/29b74173-8e715c9f/1",  # dark, number not visible
        "valid/2b1ee38e-50467470/4",  # blurred
    ],
    # Kind itself doubtful: neither a hit nor a miss.
    "ignore": [
        "valid/0145b233-ee9c5b63/12",  # "20" on an orange/red panel: not clearly an R2-1
    ],
}


def main() -> int:
    """Build the held-out file from the valid pseudo-labels plus the review above."""
    recs = [r for r in read_labels("valid")]
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_image[r["image"]].append(r)
    signs_meta = bdd_sign_boxes("valid")
    banned = dev_clip_ids()
    unclear_read = set(REVIEW["not_clearly_readable"])
    ignore_keys = set(REVIEW["ignore"])
    images = []
    counts: Counter = Counter()
    for rel in sorted(by_image):
        if Path(rel).stem in banned:
            continue
        signs, ignore, negatives = [], [], []
        labeled = set()
        for r in by_image[rel]:
            labeled.add(r["key"].rsplit("/", 1)[1])
            if r["key"] in ignore_keys or r["kind"] == "unclear":
                ignore.append(r["box"])
            elif r["kind"] in KIND_INDEX:
                readable = r["value"] is not None and r["key"] not in unclear_read
                signs.append(
                    {
                        "key": r["key"],
                        "box": r["box"],
                        "kind": r["kind"],
                        "value": r["value"],
                        "readable": readable if r["kind"] in ("speed_limit", "school_speed_limit") else None,
                    }
                )
                counts[r["kind"]] += 1
                counts["readable_speed"] += int(bool(readable) and r["kind"] == "speed_limit")
            else:
                negatives.append(r["box"])
        for k, xyxy in signs_meta.get(rel, {}).items():
            if k not in labeled:  # too small to have been labeled
                ignore.append([round(v, 1) for v in xyxy])
        images.append({"image": rel, "signs": signs, "ignore": ignore, "negatives": negatives})
    data = {
        "description": __doc__.strip().splitlines()[0],
        "source": "BDD100K valid (dronefreak/BDD100K mirror, shard_000), 1280x720; "
        "labels: Claude Opus 5.5 + hand review",
        "review": REVIEW,
        "counts": dict(counts),
        "images": images,
    }
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{len(images)} images, {dict(counts)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
