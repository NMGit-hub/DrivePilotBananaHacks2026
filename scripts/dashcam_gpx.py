"""The car dashcam's clips (D:/Dashcam, `dm_*.mp4`): read the burned-in GPS/speed overlay into GPX tracks.

    .venv/Scripts/python scripts/dashcam_gpx.py glyphs D:/Dashcam          # learn the overlay's glyphs (once)
    .venv/Scripts/python scripts/dashcam_gpx.py scan D:/Dashcam --jobs 8   # read every clip at 2 Hz (cached)
    .venv/Scripts/python scripts/dashcam_gpx.py gpx dm_20260919_152229 ...  # GPX for clips, into data/gps/

`glyphs`: every clip's date is known from its file name, which labels the date cells' digits (all ten digits over
the days recorded); the punctuation is fixed (date dashes, time colons, "mph", the coordinate symbols labeled from
one frame read by eye). Then a second pass keeps only frames whose whole line parses and adds their cells, so every
template is the mean of many samples. `scan` writes data/cache/dashcam/<clip>.json (readings at 2 Hz) and
data/cache/dashcam/scan.csv (one row per clip: how well it read, speed profile, where, day or night). `gpx` turns
cached readings into data/gps/<clip>.gpx (positions smoothed, speeds as read, a `video_start` waypoint at the clip's
first frame), which the app and the evals pick up like the BDD100K clips' GPX.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nav.dashcam_overlay import (  # noqa: E402
    LEFT_CELLS,
    LEFT_X0,
    OVERLAY_TOP,
    RIGHT_CELLS,
    RIGHT_X0,
    Glyphs,
    OverlayReader,
    TrackSample,
    cells,
    clip_start,
    parse_left,
    parse_right,
    sample_time,
    smooth_track,
    video_start_utc,
)
from nav.gps import TrackPoint, write_gpx  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache" / "dashcam"
GPS_DIR = REPO_ROOT / "data" / "gps"
MPH = 0.44704
# One frame read by eye (dm_20260914_075350.mp4 at 30 s), for the coordinate block's symbols.
EYE_FRAME = ("dm_20260914_075350.mp4", 30.0, "47°36'35\"N -122°-7'-16\"W")
SAMPLE_HZ = 1.0  # positions change once a second; speed twice (enough for selection and replay)


def frames_at(path: Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """Grayscale (luma) frames at the given video times, each the first frame at or after its time. Every frame of
    these clips is a keyframe, so PyAV seeks straight to it (~50 ms a frame; OpenCV's seek took ~560 ms)."""
    import av

    out: list[tuple[float, np.ndarray]] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            assert tb is not None
            for t in times:
                container.seek(int(t / tb), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is not None and float(frame.pts * tb) >= t - 0.021:
                        out.append((t, frame.to_ndarray(format="gray")))
                        break
    except Exception:  # noqa: BLE001 - a damaged clip: what was read so far
        pass
    return out


def duration_s(path: Path) -> float | None:
    """The clip's length, None if it can't be opened."""
    import av

    try:
        with av.open(str(path)) as c:
            return float(c.duration / av.time_base) if c.duration else None
    except Exception:  # noqa: BLE001 - a damaged file: skipped
        return None


# ------------------------------------------------------------------------------------------------ glyphs


def build_glyphs(folder: Path, n_clips: int, seed: int = 1) -> Glyphs:
    """Learn the templates (see the module docstring)."""
    clips = sorted(folder.glob("dm_*.mp4"))
    by_day: dict[str, list[Path]] = {}
    for p in clips:
        by_day.setdefault(p.name[3:11], []).append(p)
    rng = random.Random(seed)
    picks = [rng.choice(v) for v in by_day.values()] + rng.sample(clips, min(n_clips, len(clips)))
    labeled: list[tuple[str, np.ndarray]] = []
    frames: list[tuple[Path, float, np.ndarray]] = []
    for p in picks:
        d = duration_s(p)
        if not d or d < 6:
            continue
        for t, gray in frames_at(p, [2.0, d / 2, d - 3.0]):
            frames.append((p, t, gray))
            start = clip_start(p.name)
            assert start is not None
            date = start.strftime("%Y-%m-%d")
            if start.strftime("%H:%M") >= "23:50":
                continue  # (the date could roll over within the clip)
            left = cells(gray, LEFT_X0, LEFT_CELLS)
            labeled += [(c, left[k]) for k, c in enumerate(date)]
            labeled += [(":", left[14]), (":", left[17])]
            ink = [k for k in range(23, LEFT_CELLS) if int((left[k] > 45).sum()) >= 3]
            if len(ink) >= 6:  # "...d.d mph": the last three are m p h, the fifth from the end is "."
                labeled += [("m", left[ink[-3]]), ("p", left[ink[-2]]), ("h", left[ink[-1]]), (".", left[ink[-5]])]
    name, t_eye, right_text = EYE_FRAME
    eye = frames_at(folder / name, [t_eye])
    if eye:
        right = cells(eye[0][1], RIGHT_X0, RIGHT_CELLS)
        labeled += [(c, right[k]) for k, c in enumerate(right_text)]
    glyphs = Glyphs.learn(labeled)
    # Second pass: whole lines that parse add their cells (more samples per template, the symbols included).
    reader = OverlayReader(glyphs)
    more: list[tuple[str, np.ndarray]] = []
    for _, _, gray in frames:
        left_text, _ = reader.text(gray, LEFT_X0, LEFT_CELLS)
        right_text, _ = reader.text(gray, RIGHT_X0, RIGHT_CELLS)
        if parse_left(left_text) and parse_right(right_text):
            more += list(zip(left_text, cells(gray, LEFT_X0, LEFT_CELLS), strict=True))
            more += list(zip(right_text, cells(gray, RIGHT_X0, RIGHT_CELLS), strict=True))
    return Glyphs.learn(labeled + more)


# -------------------------------------------------------------------------------------------------- scan


def scan_clip(path_str: str) -> dict[str, Any]:
    """Read one clip at SAMPLE_HZ; cache the readings; return its summary row."""
    path = Path(path_str)
    row: dict[str, Any] = {"clip": path.stem}
    start = clip_start(path.name)
    d = duration_s(path)
    if start is None or not d:
        row["status"] = "unreadable"
        return row
    reader = OverlayReader()
    times = [round(0.5 + k / SAMPLE_HZ, 3) for k in range(int((d - 0.6) * SAMPLE_HZ))]
    readings: list[dict[str, Any]] = []
    time_ok = time_bad = failed = 0
    brightness: list[float] = []
    for t, gray in frames_at(path, times):
        if abs(t * SAMPLE_HZ - round(t * SAMPLE_HZ)) < 1e-6 or len(brightness) < 400:
            brightness.append(float(gray[: OVERLAY_TOP // 2].mean()))
        r = reader.read(gray)
        if r is None:
            failed += 1
            continue
        expected = (start + timedelta(seconds=t)).replace(tzinfo=None)
        if abs((r.local_time - expected).total_seconds()) <= 1.5:
            time_ok += 1
        else:
            time_bad += 1
        readings.append({"t": t, "mph": r.speed_mph, "lat": r.lat, "lon": r.lon, "clock": r.local_time.isoformat()})
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{path.stem}.json").write_text(json.dumps({"clip": path.name, "readings": readings}), encoding="utf-8")
    speeds = np.array([r["mph"] for r in readings]) if readings else np.zeros(0)
    n = len(times)
    row.update(
        status="ok",
        duration_s=round(d, 1),
        samples=n,
        read=len(readings),
        failed=failed,
        time_ok=time_ok,
        time_bad=time_bad,
        mean_mph=round(float(speeds.mean()), 1) if len(speeds) else None,
        max_mph=round(float(speeds.max()), 1) if len(speeds) else None,
        highway_s=round(float((speeds >= 45).sum()) / SAMPLE_HZ, 1),
        city_s=round(float(((speeds >= 15) & (speeds < 45)).sum()) / SAMPLE_HZ, 1),
        crawl_s=round(float(((speeds > 1) & (speeds < 10)).sum()) / SAMPLE_HZ, 1),
        stopped_s=round(float((speeds <= 0.5).sum()) / SAMPLE_HZ, 1),
        lat=round(float(readings[0]["lat"]), 5) if readings else None,
        lon=round(float(readings[0]["lon"]), 5) if readings else None,
        brightness=round(float(np.median(brightness)), 1) if brightness else None,
        local_start=start.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return row


def scan(folder: Path, jobs: int, only: list[str]) -> Path:
    """Scan every clip (or `only`) in parallel; write scan.csv; return its path."""
    clips = [p for p in sorted(folder.glob("dm_*.mp4")) if not only or any(o in p.stem for o in only)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for k, row in enumerate(pool.map(scan_clip, [str(p) for p in clips], chunksize=2), 1):
            rows.append(row)
            if k % 25 == 0 or k == len(clips):
                print(f"scanned {k}/{len(clips)}", flush=True)
    out = CACHE / "scan.csv"
    fields = sorted({key for row in rows for key in row}, key=lambda f: (f != "clip", f != "status", f))
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


# --------------------------------------------------------------------------------------------------- gpx


def write_clip_gpx(stem: str) -> Path:
    """data/gps/<clip>.gpx from the cached readings (`scan` first)."""
    data = json.loads((CACHE / f"{stem}.json").read_text(encoding="utf-8"))
    readings = data["readings"]
    if len(readings) < 3:
        raise SystemExit(f"{stem}: too few readings ({len(readings)})")
    start = clip_start(stem)
    video_start = video_start_utc(stem)
    assert start is not None and video_start is not None
    samples = smooth_track([TrackSample(r["t"], r["mph"] * MPH, r["lat"], r["lon"]) for r in readings])
    points = [TrackPoint(sample_time(start, s.t), s.lat, s.lon, s.speed_mps) for s in samples]
    GPS_DIR.mkdir(parents=True, exist_ok=True)
    out = GPS_DIR / f"{stem}.gpx"
    out.write_text(
        write_gpx([points], video_start, stem, "Mercedes dashcam overlay (scripts/dashcam_gpx.py)", stem),
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("glyphs")
    g.add_argument("folder", type=Path)
    g.add_argument("--clips", type=int, default=60)
    s = sub.add_parser("scan")
    s.add_argument("folder", type=Path)
    s.add_argument("--jobs", type=int, default=8)
    s.add_argument("--only", nargs="*", default=[])
    x = sub.add_parser("gpx")
    x.add_argument("clips", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "glyphs":
        glyphs = build_glyphs(args.folder, args.clips)
        glyphs.save()
        print(f"glyphs: {glyphs.chars!r} ({len(glyphs.chars)})")
    elif args.cmd == "scan":
        print(f"summary: {scan(args.folder, args.jobs, args.only)}")
    else:
        for clip in args.clips:
            print(write_clip_gpx(Path(clip).stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
