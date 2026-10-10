"""Download the development clips listed in eval/dev_clips.yaml from the BDD100K videos archive.

The archive is a single 1.97 TB zip split byte-for-byte into 100 GiB parts on Hugging Face
(`linxxx3/bdd100k_videos`). Its entries are stored uncompressed, so one clip can be fetched with HTTP range
requests: read the zip's central directory once (~13 MB, cached), then download exactly the clip's bytes and
verify their CRC-32.

    .venv/Scripts/python scripts/fetch_dev_clips.py            # all clips in the manifest
    .venv/Scripts/python scripts/fetch_dev_clips.py be986afd-012d020c
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "eval" / "dev_clips.yaml"
OUT_DIR = REPO_ROOT / "data" / "videos" / "dev"
INDEX_CACHE = REPO_ROOT / "data" / "videos" / ".bdd100k_index.json"  # every split (train, val, test)
SPLITS = ("train", "val", "test")

PART_URL = (
    "https://huggingface.co/datasets/linxxx3/bdd100k_videos/resolve/main/bdd100k_videos/bdd100k_videos.zip.part-%05d"
)
PART_SIZE = 107_374_182_400
TOTAL_SIZE = 1_969_155_915_058
USER_AGENT = "DrivePilot-dev/0.1 (hackathon dev-clip fetcher)"


def fetch_range(start: int, end: int) -> bytes:
    """Return bytes [start, end] (inclusive) of the logical zip, spanning part files as needed."""
    out = bytearray()
    while start <= end:
        part = start // PART_SIZE
        offset = start - part * PART_SIZE
        stop = min(end, (part + 1) * PART_SIZE - 1)
        request = urllib.request.Request(
            PART_URL % part, headers={"Range": f"bytes={offset}-{offset + stop - start}", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            out += response.read()
        start = stop + 1
    return bytes(out)


def read_index() -> dict[str, dict[str, int | str]]:
    """Map 'xxxxxxxx-xxxxxxxx' → {offset, size, crc, split} for every video, from the zip64 central directory."""
    tail = fetch_range(TOTAL_SIZE - 65_536, TOTAL_SIZE - 1)
    locator = tail.rfind(b"PK\x06\x07")
    if locator < 0:
        raise RuntimeError("zip64 end-of-central-directory locator not found")
    eocd64_offset = struct.unpack("<Q", tail[locator + 8 : locator + 16])[0]
    eocd64 = fetch_range(eocd64_offset, eocd64_offset + 55)
    if eocd64[:4] != b"PK\x06\x06":
        raise RuntimeError("zip64 end-of-central-directory record not found")
    cd_size, cd_offset = struct.unpack("<QQ", eocd64[40:56])
    cd = fetch_range(cd_offset, cd_offset + cd_size - 1)

    index: dict[str, dict[str, int | str]] = {}
    pos = 0
    while cd[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        crc, csize, usize = struct.unpack("<III", cd[pos + 16 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        local_offset = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + name_len].decode()
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        e = 0
        while e + 4 <= len(extra):  # zip64 extended information replaces 0xFFFFFFFF fields, in order
            header_id, size = struct.unpack("<HH", extra[e : e + 4])
            if header_id == 1:
                values, k = extra[e + 4 : e + 4 + size], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", values[k : k + 8])[0]
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", values[k : k + 8])[0]
                    k += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack("<Q", values[k : k + 8])[0]
            e += 4 + size
        split = next((s for s in SPLITS if f"/{s}/" in name), None)
        if split is not None and name.endswith(".mov") and method == 0:
            index[Path(name).stem] = {"offset": local_offset, "size": csize, "crc": crc, "split": split}
        pos += 46 + name_len + extra_len + comment_len
    return index


def load_index() -> dict[str, dict[str, int | str]]:
    """Return the cached index of every video, building it on first use."""
    if INDEX_CACHE.exists():
        return json.loads(INDEX_CACHE.read_text(encoding="utf-8"))
    print("reading the archive's central directory (one-time, ~13 MB)...")
    index = read_index()
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.write_text(json.dumps(index), encoding="utf-8")
    return index


def download_clip(entry: dict[str, int | str], dest: Path) -> None:
    """Download one stored zip entry and verify its CRC-32 before writing it."""
    offset, size = int(entry["offset"]), int(entry["size"])
    header = fetch_range(offset, offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("local file header not found")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = offset + 30 + name_len + extra_len
    data = fetch_range(start, start + size - 1)
    if zlib.crc32(data) != entry["crc"]:
        raise RuntimeError(f"CRC mismatch for {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    """Fetch the requested (default: all) manifest clips that are not already present."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="BDD ids to fetch (default: every clip in eval/dev_clips.yaml)")
    args = parser.parse_args(argv)

    clips = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["clips"]
    wanted = [c for c in clips if not args.ids or c["bdd_id"] in args.ids]
    missing = [c for c in wanted if not (OUT_DIR / f"{c['name']}__{c['bdd_id']}.mov").exists()]
    if not missing:
        print(f"all {len(wanted)} clips already in {OUT_DIR}")
        return 0
    index = load_index()
    for clip in missing:
        entry = index.get(clip["bdd_id"])
        if entry is None:
            print(f"not found in archive: {clip['bdd_id']}", file=sys.stderr)
            return 1
        dest = OUT_DIR / f"{clip['name']}__{clip['bdd_id']}.mov"
        print(f"downloading {dest.name} ({int(entry['size']) / 1e6:.1f} MB)")
        download_clip(entry, dest)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
