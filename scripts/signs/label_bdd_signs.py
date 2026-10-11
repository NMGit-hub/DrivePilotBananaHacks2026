"""Pseudo-label BDD100K "traffic sign" boxes with our US sign taxonomy, using a Claude vision model.

BDD100K's detection labels box every traffic sign by hand but don't say which sign it is. This script crops each
box (with context), tiles the crops into numbered grids and asks Claude which of our kinds each one is
(`perception/sign_classes.py`), with the mph value when readable. The labels are *pseudo-labels*: training data,
never the held-out truth (that is checked by eye on full-resolution frames: scripts/signs/README.md). Boxes
labeled "unclear" are masked out of training images instead of being treated as background.

    .venv/Scripts/python scripts/signs/label_bdd_signs.py --split valid --env-file <path to .env>
    .venv/Scripts/python scripts/signs/label_bdd_signs.py --split train --shards shard_000 --max-usd 20

Resumable: crops already in the output file are skipped. Spend is tracked from the API's usage counts and the run
stops at --max-usd.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perception.sign_classes import KINDS, VALUE_KINDS, parse_speed_value  # noqa: E402

BDD_ROOT = REPO_ROOT / "data" / "signs" / "raw" / "bdd100k_det" / "data" / "images"
OUT_DIR = REPO_ROOT / "data" / "signs" / "bdd_labels"
SIGN_CLASS = 9  # "traffic sign" in the mirror's data.yaml
CELL = 160
BAND = 18
COLS, ROWS = 4, 4
CONTEXT = 0.4  # context around each box, as a fraction of its longer side, per side
# $/token (input, output) of the models we may use.
PRICES = {"claude-haiku-4-5": (1e-6, 5e-6), "claude-sonnet-5-5": (2e-6, 10e-6), "claude-opus-5-5": (4e-6, 20e-6)}
LABELS = (*KINDS, "other", "unclear", "not_a_sign")

PROMPT = f"""Each numbered cell shows one traffic sign from a US dashcam frame (BDD100K). The sign is the object in
the CENTER of the cell; the cell is upscaled, so small signs look blurry. A human drew these boxes around signs,
so nearly every cell does contain a sign. Classify every cell into exactly one kind:

- speed_limit: white rectangular regulatory sign "SPEED LIMIT" + number (MUTCD R2-1). NOT yellow advisory speed
  plates or "EXIT 25", NOT "MINIMUM SPEED", NOT truck or night limits, NOT "SPEED ZONE AHEAD", NOT "YOUR SPEED".
- school_speed_limit: "SCHOOL" + "SPEED LIMIT" + number (often "WHEN FLASHING" / "WHEN CHILDREN ARE PRESENT").
- stop: red octagon STOP.
- yield: red-and-white downward-pointing triangle.
- do_not_enter: red circle with a white horizontal bar.
- wrong_way: red rectangle "WRONG WAY".
- no_turn_on_red: "NO TURN ON RED".
- one_way: black rectangle with a white arrow and "ONE WAY" (vertical or horizontal).
- no_left_turn / no_right_turn / no_u_turn: red circle with a slash over a left / right / U-turn arrow.
- pedestrian_crossing: yellow or fluorescent yellow-green diamond with ONE walking person.
- school_zone: yellow or fluorescent yellow-green PENTAGON with two children.
- other: a sign of any other kind: green street-name and guide signs, lane-use arrows, parking signs, orange
  construction signs, other yellow warning diamonds (curves, merges, signals...), "ALL WAY" plates, route shields.
- unclear: a sign is there but it is too small, dark or blurry to tell its kind with confidence.
- not_a_sign: clearly no sign (a light, a car, a tree), or the blank back of a sign.

Use one of our specific kinds only when you are confident; otherwise "unclear" (if it could be one of them) or
"other" (if it is clearly a different sign). For speed_limit and school_speed_limit, "value" is the mph number
ONLY if you can read it with certainty, else null. A wrong number is much worse than null.

Answer with JSON only, one entry per cell, in cell order:
{{"labels": [{{"i": 1, "kind": "<one of {", ".join(LABELS)}>", "value": null}}, ...]}}"""


@dataclass(frozen=True, slots=True)
class Crop:
    """One BDD sign box. `key` is unique: <split>/<image stem>/<box index within the image's objects>."""

    key: str
    image: Path
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def iter_crops(split: str, min_h: float, min_w: float, shards: list[str] | None) -> Iterator[Crop]:
    """Every downloaded sign box of at least min_w x min_h pixels in the split's metadata files."""
    for meta in sorted((BDD_ROOT / split).glob("shard_*/metadata.jsonl")):
        if shards and meta.parent.name not in shards:
            continue
        for line in meta.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            objs = row["objects"]
            for k, (cat, (x, y, w, h)) in enumerate(zip(objs["categories"], objs["bbox"], strict=True)):
                if cat == SIGN_CLASS and h >= min_h and w >= min_w:
                    image = meta.parent / row["file_name"]
                    if not image.exists():  # not downloaded yet: a later run picks it up
                        continue
                    yield Crop(f"{split}/{image.stem}/{k}", image, (x, y, x + w, y + h))


def crop_cell(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """The box plus CONTEXT per side, letterboxed into a CELL x CELL tile."""
    x1, y1, x2, y2 = box
    pad = CONTEXT * max(x2 - x1, y2 - y1)
    region = img.crop((int(x1 - pad), int(y1 - pad), int(x2 + pad + 1), int(y2 + pad + 1)))
    scale = CELL / max(region.width, region.height)
    region = region.resize(
        (max(1, round(region.width * scale)), max(1, round(region.height * scale))), Image.Resampling.BICUBIC
    )
    tile = Image.new("RGB", (CELL, CELL), (90, 90, 90))
    tile.paste(region, ((CELL - region.width) // 2, (CELL - region.height) // 2))
    return tile


def make_grid(crops: list[Crop], cache: dict[Path, Image.Image]) -> Image.Image:
    """Tile up to COLS x ROWS crops with their 1-based number in a band above each cell."""
    grid = Image.new("RGB", (COLS * CELL, ROWS * (CELL + BAND)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=15)
    for n, c in enumerate(crops):
        if c.image not in cache:
            cache[c.image] = Image.open(c.image).convert("RGB")
        x, y = (n % COLS) * CELL, (n // COLS) * (CELL + BAND)
        draw.text((x + 4, y + 1), str(n + 1), fill="black", font=font)
        grid.paste(crop_cell(cache[c.image], c.box), (x, y + BAND))
    return grid


def parse_reply(text: str, n: int) -> list[tuple[str, int | None]] | None:
    """[(kind, value)] for cells 1..n from the model's JSON, or None if malformed or incomplete."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        entries = json.loads(match.group(0))["labels"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    out: dict[int, tuple[str, int | None]] = {}
    for e in entries:
        try:
            i, kind = int(e["i"]), str(e["kind"])
        except (KeyError, TypeError, ValueError):
            return None
        if kind not in LABELS:
            kind = "unclear"
        value = parse_speed_value(e.get("value")) if kind in VALUE_KINDS else None
        out[i] = (kind, value)
    if sorted(out) != list(range(1, n + 1)):
        return None
    return [out[i] for i in range(1, n + 1)]


def request_args(model: str) -> dict[str, Any]:
    """Model-specific request settings: temperature 0 on Haiku 4.5; low effort on the thinking-always models."""
    if model.startswith("claude-haiku-4-5"):
        return {"max_tokens": 2000, "extra_body": {"temperature": 0}}
    return {"max_tokens": 8000, "extra_body": {"output_config": {"effort": "low"}}}


class Labeler:
    """Sends grids to Claude and appends labels to a JSON-lines file (thread-safe)."""

    def __init__(self, model: str, api_key: str, out_path: Path, max_usd: float) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=4, timeout=180.0)
        self.model = model
        self.out_path = out_path
        self.max_usd = max_usd
        self.usd = 0.0
        self.lock = threading.Lock()
        self.failed = 0

    def label(self, crops: list[Crop]) -> int:
        """Label one grid of crops; returns how many labels were written."""
        if self.usd >= self.max_usd:
            return 0
        grid = make_grid(crops, {})
        buf = io.BytesIO()
        grid.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode()
        labels = None
        for _attempt in range(2):
            msg = self.client.messages.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                **request_args(self.model),
            )
            pin, pout = PRICES.get(self.model, (5e-6, 25e-6))
            with self.lock:
                self.usd += msg.usage.input_tokens * pin + msg.usage.output_tokens * pout
            if msg.stop_reason == "refusal":
                break
            text = "".join(b.text for b in msg.content if b.type == "text")
            labels = parse_reply(text, len(crops))
            if labels is not None:
                break
        if labels is None:
            with self.lock:
                self.failed += len(crops)
            return 0
        with self.lock, self.out_path.open("a", encoding="utf-8") as f:
            for c, (kind, value) in zip(crops, labels, strict=True):
                rec = {
                    "key": c.key,
                    "image": c.image.relative_to(BDD_ROOT).as_posix(),
                    "box": [round(v, 1) for v in c.box],
                }
                f.write(json.dumps({**rec, "kind": kind, "value": value, "model": self.model}) + "\n")
        return len(crops)


def read_key(env_file: Path | None) -> str:
    """ANTHROPIC_API_KEY from the environment or the given .env file (never printed)."""
    import os

    from dotenv import dotenv_values

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and env_file is not None and env_file.exists():
        key = dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (pass --env-file pointing at the project's .env)")
    return key


def main(argv: list[str] | None = None) -> int:
    """Label every unlabeled sign crop of a split."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "valid"], required=True)
    p.add_argument("--shards", nargs="*", help="limit to these shard folders, e.g. shard_000 shard_001")
    p.add_argument("--min-h", type=float, default=12.0, help="skip boxes shorter than this (pixels)")
    p.add_argument("--min-w", type=float, default=8.0)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0, help="label at most this many crops (0 = all)")
    p.add_argument("--keys", type=Path, default=None, help="only crops whose key is listed in this file")
    p.add_argument("--out", type=Path, default=None, help="output file (default data/signs/bdd_labels/<split>.jsonl)")
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Parallel runs (e.g. one per shard) must write to different files: data/signs/bdd_labels/<split>_<tag>.jsonl.
    out_path = args.out or OUT_DIR / f"{args.split}.jsonl"
    done: set[str] = set()
    for path in {out_path, *OUT_DIR.glob(f"{args.split}*.jsonl")}:
        if path.exists():
            done |= {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line}
    todo = [c for c in iter_crops(args.split, args.min_h, args.min_w, args.shards) if c.key not in done]
    if args.keys:
        wanted = set(args.keys.read_text(encoding="utf-8").split())
        todo = [c for c in todo if c.key in wanted]
    if args.limit:
        todo = todo[: args.limit]
    per = COLS * ROWS
    grids = [todo[i : i + per] for i in range(0, len(todo), per)]
    print(f"{len(done)} already labeled, {len(todo)} to label in {len(grids)} grids with {args.model}", flush=True)
    labeler = Labeler(args.model, read_key(args.env_file), out_path, args.max_usd)
    t0 = time.time()
    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(labeler.label, g) for g in grids]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                written += fut.result()
            except Exception as exc:  # noqa: BLE001 - one failed request must not stop the run
                print(f"grid failed: {type(exc).__name__}: {exc}", flush=True)
            if n % 20 == 0 or n == len(futures):
                rate = written / max(1e-6, time.time() - t0)
                print(f"{n}/{len(futures)} grids, {written} labels, ${labeler.usd:.2f}, {rate:.1f}/s", flush=True)
    print(f"done: {written} labels written, {labeler.failed} unparsed, ${labeler.usd:.2f} spent -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
