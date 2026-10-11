"""Write the hand-checked BDD100K held-out sign sets (tracked).

    .venv/Scripts/python scripts/signs/make_heldout.py                # eval/sign_labels_bdd.json (original)
    .venv/Scripts/python scripts/signs/make_heldout.py --set fresh    # eval/sign_labels_bdd_fresh.json

ORIGINAL: the first 2,623 frames of the BDD100K "valid" split mirror (shard_000) whose human "traffic sign" boxes
were classified by Claude Opus 5.5 (scripts/signs/label_bdd_signs.py), then checked by eye:
- every speed-limit / school-speed-limit box (92) was inspected on the full-resolution frame crop: all kinds and
  all read values were right; 11 are not clearly readable (REVIEW below), so they don't count toward the
  "clearly readable" recall (a value reported there is still judged against the labeled value);
- recall of the labeler for speed limits: the 264 largest of 774 upright, low-saturation signs it called
  "other"/"unclear" were inspected: no readable speed limit was missed;
- the 270 boxes of the smaller kinds (stop, yield, do not enter, no left/right/U turn, no turn on red, wrong way,
  pedestrian crossing, school speed limit) were viewed on contact sheets: all but one looked right (a blue
  square crosswalk sign called pedestrian_crossing, now ignored); one_way (442) and school_zone (158) keep Opus's
  labels: random samples of 96 each were viewed, about 2 doubtful one_way boxes in 96, no wrong school zone.
This set was looked at while developing (interim models were evaluated on it, which led to the look-alike
negatives and the AUDIT below), so it is not a clean test any more: the FRESH set is the headline.

AUDIT (original only): the speed values an interim model reported where the labels had no speed limit were checked
by eye on full-resolution crops. Some were real (school) speed limits that the labeler called "other" or that
BDD's annotators never boxed. Because the audit was triggered by detections it can only add signs a model found,
so each image keeps its audit separately and `scripts/signs/evaluate.py bdd --no-audit` reports without it.

FRESH: the next 2,675 frames of the same split (data/signs/bdd_labels/fresh_valid.jsonl, labeled 2026-10-10 after
development; the one frame shared with the original set is left out), checked by eye BEFORE any model was run on
them (FRESH_REVIEW): all 98 speed-limit boxes on full-resolution crops (no wrong value; 11 not clearly readable:
cut by the frame edge, very dark or washed out; 3 of doubtful kind), and the 528 largest of the 1,235 upright
boxes the labeler called "other"/"not_a_sign" (down to 40 px tall) for missed signs: one missed CITY SPEED LIMIT 25
(relabeled) and 6 doubtful kinds (ignored). Plus SIGN_FREE_FRAMES frames of the split with no human sign box at
all, to measure false alarms on plain driving (BDD doesn't box every sign, so a detection there may still be a
real sign: evaluate.py lists every value reported there).

In both sets, boxes labeled "unclear", sign boxes too small to label (< 12 px tall) and the review's ignore keys
are ignore regions: a detection there is neither a hit nor a false alarm, and a value there is "unverifiable".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.sign_classes import KIND_INDEX  # noqa: E402
from scripts.signs.prepare_data import LABELS, bdd_sign_boxes, dev_clip_ids, read_labels, stable_frac  # noqa: E402

OUT = {
    "original": REPO_ROOT / "eval" / "sign_labels_bdd.json",
    "fresh": REPO_ROOT / "eval" / "sign_labels_bdd_fresh.json",
}
FRESH_LABELS = LABELS / "fresh_valid.jsonl"
SIGN_FREE_FRAMES = 500

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
        "valid/0e7aa9b3-866273e9/4",  # blue square crosswalk sign, not the W11-2 diamond
    ],
}

# key -> (kind, value, readable): labeler said "other", the crop shows one of our kinds.
AUDIT_RELABEL: dict[str, tuple[str, int | None, bool]] = {
    "valid/10903606-c6c54402/2": ("school_speed_limit", 20, True),  # yellow-green SCHOOL panel over 20, night
}
# Labeler said "other" but the kind is doubtful (could be a speed limit): ignore regions.
AUDIT_IGNORE = [
    "valid/047cf4fc-34ad5b08/3",  # red header over "40": not clearly an R2-1
    "valid/2430b667-a2df628e/21",  # blurred "?0" under a lamp
    "valid/28dbc980-86029e53/0",  # yellow-lit sign, "30" under a word that may be SCHOOL
]
# Signs BDD's annotators didn't box (no traffic-sign box there): (image, box from the detection, kind, value, readable).
AUDIT_ADD: list[tuple[str, list[float], str, int | None, bool]] = [
    ("valid/shard_000/12011d41-208b652b.jpg", [752, 285, 768, 315], "school_speed_limit", 20, True),
    ("valid/shard_000/28b3996a-d5e1d98d.jpg", [822, 165, 850, 212], "speed_limit", 25, False),  # dark
]

# Blind hand review of the fresh slice (before any model run), full-resolution crops.
FRESH_REVIEW = {
    "not_clearly_readable": [
        "valid/3fda6604-b95df43c/6",  # 25 washed out by headlights
        "valid/445e5094-8c851e5f/16",  # 30 cut by the top of the frame
        "valid/47e0d7e0-2766582e/21",  # 25 cut by the top of the frame, 14 px
        "valid/53e212ea-860e34d7/0",  # very dark
        "valid/580262f8-6c09e809/0",  # washed out, low contrast
        "valid/5ff046cf-ca98fc5e/13",  # cut by the top of the frame, 12 px
        "valid/632be00e-5371ba08/4",  # 50 very dark
        "valid/656bb701-25082673/3",  # cut by the top of the frame, 15 px
        "valid/69cfdd5d-664b39d7/0",  # blurred, number uncertain (value removed below)
        "valid/6a8c0583-ab5c14fc/2",  # cut by the top of the frame, 15 px
        "valid/6d6fa458-f6bae888/10",  # cut by the top of the frame, 17 px
    ],
    "ignore": [
        "valid/55b35a7e-ec4bc396/19",  # S4-5 "school speed limit ahead" diamond with a 20 panel, not an S5-1
        "valid/69a65575-0775cf8d/4",  # narrow sign with "20" between text lines: kind unclear
        "valid/708a0df2-4035a60f/3",  # box spans a whole pole of signs (speed limit cut at the top)
        "valid/41b93754-45f8218c/18",  # Israeli stop sign (red octagon with a hand)
        "valid/42e09c21-7d43df47/0",  # SCHOOL AHEAD assembly: school zone or not is doubtful
        "valid/48692e9f-5bce10e0/2",  # dark sign with a U-turn-like symbol
        "valid/671b4118-91a48a3c/9",  # box spans a curve arrow and a no-left-turn sign
        "valid/470a3f53-455c8a12/11",  # two red-circle restriction signs, kinds unclear
        "valid/6fef6069-abda4980/21",  # school crossing AHEAD assembly
    ],
}
FRESH_RELABEL: dict[str, tuple[str, int | None, bool]] = {
    "valid/39fba3aa-43b709f1/58": ("speed_limit", 25, True),  # CITY SPEED LIMIT 25 called "other"
    "valid/69cfdd5d-664b39d7/0": ("speed_limit", None, False),  # "55"? too blurred to be sure of the number
}


def build_images(
    recs: list[dict],
    review: dict[str, list[str]],
    relabel: dict[str, tuple[str, int | None, bool]],
    audit: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    """Per-image signs / ignore regions / negatives (and the audit, for the original set) from label records."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_image[r["image"]].append(r)
    signs_meta = bdd_sign_boxes("valid")
    banned = dev_clip_ids()
    unclear_read = set(review["not_clearly_readable"])
    ignore_keys = set(review["ignore"])
    images: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for rel in sorted(by_image):
        if Path(rel).stem in banned:
            continue
        signs, ignore, negatives = [], [], []
        audit_part: dict[str, list] = {"signs": [], "ignore": [], "drop_negatives": []}
        labeled = set()
        for r in by_image[rel]:
            labeled.add(r["key"].rsplit("/", 1)[1])
            kind, value = r["kind"], r["value"]
            forced_readable: bool | None = None
            if r["key"] in relabel:
                kind, value, forced_readable = relabel[r["key"]]
            if audit and r["key"] in AUDIT_RELABEL:
                a_kind, a_value, a_readable = AUDIT_RELABEL[r["key"]]
                audit_part["signs"].append(
                    {"key": r["key"], "box": r["box"], "kind": a_kind, "value": a_value, "readable": a_readable}
                )
                audit_part["drop_negatives"].append(r["box"])
            elif audit and r["key"] in AUDIT_IGNORE:
                audit_part["ignore"].append(r["box"])
                audit_part["drop_negatives"].append(r["box"])
            if r["key"] in ignore_keys or kind == "unclear":
                ignore.append(r["box"])
            elif kind in KIND_INDEX:
                readable = value is not None and r["key"] not in unclear_read
                if forced_readable is not None:
                    readable = forced_readable and value is not None
                signs.append(
                    {
                        "key": r["key"],
                        "box": r["box"],
                        "kind": kind,
                        "value": value,
                        "readable": readable if kind in ("speed_limit", "school_speed_limit") else None,
                    }
                )
                counts[kind] += 1
                counts["readable_speed"] += int(bool(readable) and kind == "speed_limit")
            else:
                negatives.append(r["box"])
        for k, xyxy in signs_meta.get(rel, {}).items():
            if k not in labeled:  # too small to have been labeled
                ignore.append([round(v, 1) for v in xyxy])
        if audit:
            for image, box, a_kind, a_value, a_readable in AUDIT_ADD:
                if image == rel:
                    audit_part["signs"].append(
                        {
                            "key": f"audit/{Path(image).stem}",
                            "box": box,
                            "kind": a_kind,
                            "value": a_value,
                            "readable": a_readable,
                        }
                    )
        entry: dict[str, Any] = {"image": rel, "signs": signs, "ignore": ignore, "negatives": negatives}
        if audit:
            entry["audit"] = audit_part
        images.append(entry)
    return images, counts


def sign_free_frames(exclude: set[str], n: int) -> list[str]:
    """n BDD valid frames (relative paths) with no human traffic-sign box, chosen by a stable hash of the id."""
    banned = dev_clip_ids()
    free = [
        rel
        for rel, boxes in bdd_sign_boxes("valid").items()
        if not boxes and rel not in exclude and Path(rel).stem not in banned
    ]
    return sorted(free, key=lambda rel: stable_frac("free/" + Path(rel).stem))[:n]


def main(argv: list[str] | None = None) -> int:
    """Build one held-out file from the valid pseudo-labels plus the hand review above."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["original", "fresh"], default="original")
    args = p.parse_args(argv)
    original_recs = read_labels("valid")  # data/signs/bdd_labels/valid*.jsonl (not fresh_valid.jsonl)
    if args.set == "original":
        images, counts = build_images(original_recs, REVIEW, {}, audit=True)
        data: dict[str, Any] = {
            "description": __doc__.strip().splitlines()[0],
            "source": "BDD100K valid (dronefreak/BDD100K mirror, shard_000), 1280x720; "
            "labels: Claude Opus 5.5 + hand review",
            "review": REVIEW,
            "audit": {"relabel": AUDIT_RELABEL, "ignore": AUDIT_IGNORE, "add": AUDIT_ADD},
        }
    else:
        taken = {r["image"] for r in original_recs}
        recs = [json.loads(line) for line in FRESH_LABELS.read_text(encoding="utf-8").splitlines() if line]
        recs = [r for r in recs if r["image"] not in taken]
        images, counts = build_images(recs, FRESH_REVIEW, FRESH_RELABEL, audit=False)
        labeled = {im["image"] for im in images}
        free = sign_free_frames(taken | labeled, SIGN_FREE_FRAMES)
        images += [{"image": rel, "signs": [], "ignore": [], "negatives": [], "sign_free": True} for rel in free]
        counts["sign_free_frames"] = len(free)
        data = {
            "description": "Fresh BDD100K held-out slice, hand-checked blind (scripts/signs/make_heldout.py)",
            "source": "BDD100K valid (dronefreak/BDD100K mirror, shard_000 + sign-free frames of shards 000-001), "
            "1280x720; labels: Claude Opus 5.5 + blind hand review",
            "review": FRESH_REVIEW,
            "relabel": FRESH_RELABEL,
        }
    data |= {"counts": dict(counts), "images": images}
    OUT[args.set].write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{len(images)} images, {dict(counts)} -> {OUT[args.set]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
