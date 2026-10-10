"""Frames from a video for labeling by eye: contact sheets with timestamps, or one full-resolution frame/crop.

    .venv/Scripts/python scripts/contact_sheet.py sheet <video> --every 2 --out out/          # grids of 12 frames
    .venv/Scripts/python scripts/contact_sheet.py frame <video> --t 13.5 --out out/f.jpg        # one full frame
    .venv/Scripts/python scripts/contact_sheet.py frame <video> --t 13.5 --crop 640,300,1280,720 --scale 2 --out c.jpg

Sheets: tiles of `--width` px, `--cols` per row, 12 tiles per sheet, each stamped with its video time (s). Times are
the video's own timestamps (the same clock as the app, the test sheet and GPX replay).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def read_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    """The frame at video time `t` (s)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def stamp(image: np.ndarray, text: str) -> np.ndarray:
    """Write `text` in the top-left corner, readable on any background."""
    out = image.copy()
    for color, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(out, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness, cv2.LINE_AA)
    return out


def sheets(video: Path, every: float, width: int, cols: int, out: Path, start: float, end: float | None) -> list[Path]:
    """Write contact sheets (12 tiles each); returns their paths."""
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
    end = duration - 0.05 if end is None else min(end, duration - 0.05)
    times = list(np.arange(start, end, every))
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(0, len(times), 12):
        tiles = []
        for t in times[k : k + 12]:
            frame = read_at(cap, float(t))
            if frame is None:
                continue
            h = round(frame.shape[0] * width / frame.shape[1])
            tiles.append(stamp(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA), f"{t:.1f}s"))
        if not tiles:
            continue
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
        path = out / f"{video.stem}_sheet{k // 12:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    cap.release()
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sheet", "frame"])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Folder (sheet) or .jpg file (frame).")
    parser.add_argument("--every", type=float, default=2.0, help="Seconds between tiles.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--width", type=int, default=480, help="Tile width (px).")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--t", type=float, default=0.0, help="frame: video time (s).")
    parser.add_argument("--crop", default=None, help="frame: x1,y1,x2,y2 in full-resolution pixels.")
    parser.add_argument("--scale", type=float, default=1.0, help="frame: resize factor after cropping.")
    args = parser.parse_args(argv)
    if not args.video.is_file():
        print(f"not found: {args.video}", file=sys.stderr)
        return 1
    if args.mode == "sheet":
        for path in sheets(args.video, args.every, args.width, args.cols, args.out, args.start, args.end):
            print(path)
        return 0
    cap = cv2.VideoCapture(str(args.video))
    frame = read_at(cap, args.t)
    cap.release()
    if frame is None:
        print(f"no frame at {args.t} s", file=sys.stderr)
        return 1
    if args.crop:
        x1, y1, x2, y2 = (int(v) for v in args.crop.split(","))
        frame = frame[y1:y2, x1:x2]
    if args.scale != 1.0:
        frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), stamp(frame, f"{args.t:.2f}s"), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
