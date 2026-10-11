"""Keep only the mined detector hard negatives whose false detections were checked by eye as not our signs.

    .venv/Scripts/python scripts/signs/curate_det_negatives.py

mine_det_negatives.py lists BDD train frames that hold none of our labeled kinds but where the detector still
fires (data/signs/hardneg_det_raw.txt). Checking every detection on contact sheets (dump_det_hits.py, then
review_sheet.py, 2026-10-10) showed that most of them are REAL signs that BDD's annotators
never boxed: nearly all one-way, school-zone, stop, yield and do-not-enter hits, and many speed limits (e.g. a
clear SPEED LIMIT 30, two SCHOOL SPEED LIMIT signs). Training those frames as background would teach real signs
as "nothing", so a frame is kept only if every detection in it is clearly not one of our kinds: yellow warning
diamonds and advisory plates (EXIT 40, RAMP 25), route shields (WEST/EAST 25, 101), text regulatory signs (NO
TURNS, STAY IN LANE, STOP HERE ON RED), non-US round limits, no-truck / no-parking signs, logos, windows, lights.
Doubtful tiles (dark, blurred, too small to tell) count as possibly real: their frames are dropped.

Indices below are tile numbers of the sheets: speed = review_sheet --kind speed_limit, school = --kind
school_speed_limit, rest = data/signs/hardneg_det_rest.jsonl (every other kind, in file order).
Reads data/signs/hardneg_det_raw.txt and data/signs/hardneg_det_dets.jsonl, writes data/signs/hardneg_det.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "signs"
DETS = DATA / "hardneg_det_dets.jsonl"
RAW = DATA / "hardneg_det_raw.txt"
OUT = DATA / "hardneg_det.txt"
BDD_PREFIX = "data/signs/raw/bdd100k_det/data/images/"

# speed_limit sheet: possibly real speed limits (or doubtful).
SPEED_EXCLUDE = {
    *(0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33),
    *(34, 35, 36, 39, 42, 43, 46, 49, 51, 53, 58, 59, 60, 61, 62, 63, 68, 69, 72, 75, 76, 78, 83, 84, 86, 87),
    *(91, 93, 94, 98, 101, 103, 104, 106, 110, 111, 112, 113, 114, 116, 120, 121, 123, 125, 127, 131, 132),
}
SCHOOL_EXCLUDE = {0, 1, 2, 7, 11, 21}
# rest sheet, by kind block: do_not_enter / no_left / no_right / no_turn_on_red (0-131) and no_u_turn (132-175)
REST_EXCLUDE = {
    *(0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 31),
    *(32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 52, 53, 58, 59, 63, 64, 65, 67, 68, 69, 70),
    *(71, 72, 75, 76, 77, 78, 79, 80, 81, 83, 85, 92, 97, 100, 102, 105, 109),
    *(134, 135, 136, 138, 139, 140, 142, 144, 145, 146, 148, 149, 150, 151, 156, 157, 158, 159, 160, 168),
    *(172, 173, 174, 175),
    # pedestrian_crossing (420-530): the yellow-green walking-person signs
    *(456, 458, 459, 460, 465, 466, 467, 468, 472, 477, 486, 489, 506, 516),
    # wrong_way (745-761): red rectangles that may be WRONG WAY
    *(745, 747, 748, 751, 753, 754, 756, 757, 758, 759, 760, 761),
}
# Blocks that are mostly real signs: everything excluded except these clear non-signs.
REST_KEEP_ONLY = {
    range(176, 420): {  # one_way
        *(214, 221, 234, 253, 282, 303, 306, 315, 320, 327, 328, 329, 330, 332, 340, 348, 358, 366, 369, 377),
        *(382, 386, 389, 390, 394, 396, 398, 403, 404, 405, 413, 416, 418, 419),
    },
    range(531, 691): {  # school_zone
        *(580, 581, 602, 605, 612, 626, 627, 629, 631, 666, 669, 670, 671, 674, 675, 679, 682, 684, 686),
    },
    range(691, 745): {692, 702, 706, 708, 710, 711, 718, 720, 721, 722, 731, 733, 738, 740},  # stop
    range(762, 802): {785, 788, 798, 801},  # yield
}


def rest_excluded(i: int) -> bool:
    """Whether tile i of the rest sheet is (possibly) a real sign of our kinds."""
    for block, keep in REST_KEEP_ONLY.items():
        if i in block:
            return i not in keep
    return i in REST_EXCLUDE


def main() -> int:
    """Write the curated hard-negative list."""
    recs = [json.loads(line) for line in DETS.read_text(encoding="utf-8").splitlines() if line]
    speed = [r for r in recs if r["kind"] == "speed_limit"]
    school = [r for r in recs if r["kind"] == "school_speed_limit"]
    rest = [r for r in recs if r["kind"] not in ("speed_limit", "school_speed_limit")]
    if len(speed) != 133 or len(school) != 31 or len(rest) != 802:
        raise SystemExit(f"detections changed since the review ({len(speed)}, {len(school)}, {len(rest)}): re-check")
    dropped = {r["image"] for i, r in enumerate(speed) if i in SPEED_EXCLUDE}
    dropped |= {r["image"] for i, r in enumerate(school) if i in SCHOOL_EXCLUDE}
    dropped |= {r["image"] for i, r in enumerate(rest) if rest_excluded(i)}
    raw_lines = [line for line in RAW.read_text(encoding="utf-8").splitlines() if line]
    kept = [line for line in raw_lines if line.split("\t")[0].removeprefix(BDD_PREFIX) not in dropped]
    OUT.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} of {len(raw_lines)} mined frames ({len(dropped)} frames with possibly real signs dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
