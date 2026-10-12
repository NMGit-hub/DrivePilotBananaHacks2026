"""Join showcase videos (scripts/render_showcase.py) into one reel, each after a short title card.

    .venv/Scripts/python scripts/showcase_reel.py runs/showcase/final/DrivePilot_showcase.mp4 ^
        "runs/showcase/final/1_traffic_light_b.mp4=Red light|Tell me when it's green" ^
        "runs/showcase/final/4_rain.mp4=Rain|What did that yellow sign say?"

Each item is `<video>=<title>|<subtitle>` (subtitle optional). The reel takes the first video's size and frame rate
(others are scaled to fit, letterboxed; all must have its frame rate: frames are written one for one), re-encodes
H.264 (CRF 18) + AAC at 48 kHz mono, and opens and closes with a card (--intro, --outro). Title cards are silent;
each video keeps its own audio, in step with its frames.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ui.render import BG, MUTED, TEXT, font, rgba  # noqa: E402

AUDIO_RATE = 48000
CARD_S = 2.2


@dataclass(frozen=True, slots=True)
class Item:
    """One video of the reel and its title card."""

    path: Path
    title: str
    subtitle: str = ""
    segments: tuple[tuple[float, float | None], ...] = ((0.0, None),)  # (from, to) seconds of the video (to None:
    # its end), played in order: a slow stretch can be cut out


def parse_item(text: str) -> Item:
    """ "clip.mp4=Rain|What did that sign say?" -> Item; "clip.mp4@3.5-17=Rain" plays 3.5-17 s of the video,
    "clip.mp4@3-9,14-20=Rain" those two stretches one after the other."""
    path, sep, label = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected <video>[@start-end]=<title>[|<subtitle>], got {text!r}")
    title, _, subtitle = label.partition("|")
    path, at, spans = path.partition("@")
    segments: list[tuple[float, float | None]] = []
    for span in spans.split(",") if at else []:
        a, dash, b = span.partition("-")
        if not dash or float(b) <= float(a):
            raise argparse.ArgumentTypeError(f"expected @start-end[,start-end...], got {spans!r}")
        segments.append((float(a), float(b)))
    return Item(Path(path), title.strip(), subtitle.strip(), tuple(segments) or ((0.0, None),))


@dataclass(frozen=True, slots=True)
class Slide:
    """An image shown for `seconds` (at least as long as its narration), with narration spoken over it."""

    path: Path
    seconds: float
    narration: str = ""


def parse_slide(text: str) -> Slide:
    """ "intro_1.png=5=DrivePilot is ..." -> Slide."""
    path, sep, rest = text.partition("=")
    seconds, _, narration = rest.partition("=")
    if not sep or not seconds:
        raise argparse.ArgumentTypeError(f"expected <image>=<seconds>[=<narration>], got {text!r}")
    return Slide(Path(path), float(seconds), narration.strip())


def title_card(width: int, height: int, title: str, subtitle: str = "", brand: str = "DrivePilot") -> np.ndarray:
    """A BGR title card: the brand small, the title large, the subtitle under it, centered on the page's dark
    background."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), rgba(BG)[:3])
    d = ImageDraw.Draw(img)
    u = height / 720
    cx, cy = width / 2, height / 2
    d.text((cx, cy - 70 * u), brand, font=font(round(26 * u), 700), fill=rgba(MUTED)[:3], anchor="mm")
    d.text((cx, cy), title, font=font(round(54 * u), 700), fill=rgba(TEXT)[:3], anchor="mm")
    if subtitle:
        d.text((cx, cy + 62 * u), subtitle, font=font(round(28 * u), 400), fill=rgba("#7cc4ff")[:3], anchor="mm")
    return np.ascontiguousarray(np.asarray(img)[..., ::-1])


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The image scaled to fit width x height, letterboxed on black."""
    import cv2

    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    scale = min(width / w, height / h)
    nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
    out = np.zeros((height, width, 3), np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y : y + nh, x : x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def read_audio(path: Path) -> np.ndarray:
    """A video's audio as mono float32 at AUDIO_RATE (empty without an audio track)."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, np.float32)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def read_frames(path: Path) -> Iterator[np.ndarray]:
    """A video's frames (BGR), in order."""
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def probe(path: Path) -> tuple[int, int, Fraction]:
    """Width, height and frame rate of a video."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or Fraction(25)
        return stream.codec_context.width, stream.codec_context.height, Fraction(rate)


class Writer:
    """H.264 + AAC output: one video frame at a time, each with its slot of audio."""

    def __init__(self, path: Path, width: int, height: int, rate: Fraction, crf: int) -> None:
        import av

        self.av = av
        self.container = av.open(str(path), "w", container_options={"movflags": "+faststart"})
        self.v = self.container.add_stream("libx264", rate=rate)
        self.v.width, self.v.height, self.v.pix_fmt = width, height, "yuv420p"
        self.v.options = {"crf": str(crf), "preset": "medium", "tune": "film"}
        self.a = self.container.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        self.a.bit_rate = 160_000
        self.rate = rate
        self.frames = 0
        self.samples = 0

    def write(self, image: np.ndarray, audio: np.ndarray) -> int:
        """One frame and the audio of its time slot (padded with silence or trimmed to the slot); returns the slot's
        length in samples (the caller's next audio starts there, so a clip's audio stays contiguous)."""
        vf = self.av.VideoFrame.from_ndarray(image, format="bgr24")
        vf.pts = self.frames
        for packet in self.v.encode(vf):
            self.container.mux(packet)
        self.frames += 1
        want = round(self.frames * AUDIO_RATE / self.rate) - self.samples
        chunk = np.zeros(want, np.float32)
        n = min(want, len(audio))
        chunk[:n] = audio[:n]
        af = self.av.AudioFrame.from_ndarray(chunk[None, :], format="flt", layout="mono")
        af.sample_rate = AUDIO_RATE
        af.pts = self.samples
        self.samples += want
        for packet in self.a.encode(af):
            self.container.mux(packet)
        return want

    def close(self) -> None:
        """Flush both encoders and close the file."""
        for packet in self.v.encode():
            self.container.mux(packet)
        for packet in self.a.encode():
            self.container.mux(packet)
        self.container.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+", type=parse_item, metavar="VIDEO=TITLE[|SUBTITLE]")
    parser.add_argument("--intro", default="DrivePilot|A voice co-pilot that watches the road")
    parser.add_argument("--outro", default="DrivePilot|Banana Hacks 2026")
    parser.add_argument("--card-s", type=float, default=CARD_S)
    parser.add_argument("--no-cards", action="store_true", help="No title card before each video (titles drawn in).")
    parser.add_argument(
        "--slide",
        action="append",
        type=parse_slide,
        default=[],
        metavar="IMAGE=SECONDS[=NARRATION]",
        help="An image slide before the videos (after --intro), narrated in DrivePilot's voice (Piper).",
    )
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args(argv)

    for item in args.items:
        if not item.path.is_file():
            raise SystemExit(f"no such video: {item.path}")
    width, height, rate = probe(args.items[0].path)
    for item in args.items[1:]:  # (frames are written one for one: another rate would drift from its audio)
        other = probe(item.path)[2]
        if other != rate:
            raise SystemExit(
                f"{item.path.name}: {float(other):.3f} fps, the reel is {float(rate):.3f} fps: render it at that rate"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = Writer(args.out, width, height, rate, args.crf)
    step = 1.0 / float(rate)

    def card(text: str, subtitle: str = "", brand: str = "DrivePilot") -> None:
        image = title_card(width, height, text, subtitle, brand)
        for _ in range(round(args.card_s / step)):
            writer.write(image, np.zeros(0, np.float32))

    def slide(item: Slide, voice: Any) -> None:
        import cv2

        picture = cv2.imread(str(item.path))
        if picture is None:
            raise SystemExit(f"can't read slide {item.path}")
        image = fit(picture, width, height)
        speech = np.zeros(0, np.float32)
        if item.narration and voice is not None:
            from voice.audio import resample

            spoken = resample(voice.synthesize(item.narration).astype(np.float32), voice.rate, AUDIO_RATE)
            speech = np.concatenate((np.zeros(round(0.4 * AUDIO_RATE), np.float32), spoken))
        seconds = max(item.seconds, len(speech) / AUDIO_RATE + 0.6)
        used = 0
        for _ in range(round(seconds / step)):
            used += writer.write(image, speech[used:])

    try:
        if args.intro:
            title, _, subtitle = args.intro.partition("|")
            card(title, subtitle, brand="")
        voice = None
        if any(s.narration for s in args.slide):
            from core.config import load_config
            from voice.tts import Tts

            cfg = load_config(REPO_ROOT / "config.yaml")
            voice = Tts(cfg.voice.tts, cfg.root)
        for item_slide in args.slide:
            slide(item_slide, voice)
        for item in args.items:
            if not args.no_cards:
                card(item.title, item.subtitle)
            audio = read_audio(item.path)
            frames = 0
            pending = list(item.segments)
            seg_start, seg_end = pending.pop(0)
            used = round(seg_start * AUDIO_RATE)
            for k, image in enumerate(read_frames(item.path)):
                t = k * step
                if seg_end is not None and t >= seg_end - step / 2:
                    if not pending:
                        break
                    seg_start, seg_end = pending.pop(0)
                    used = round(seg_start * AUDIO_RATE)
                if t < seg_start - step / 2:
                    continue
                used += writer.write(fit(image, width, height), audio[used:])
                frames += 1
            print(f"{item.path.name}: {frames} frames ({frames * step:.1f} s)", flush=True)
        if args.outro:
            title, _, subtitle = args.outro.partition("|")
            card(title, subtitle, brand="")
    finally:
        writer.close()
    print(f"-> {args.out}: {writer.frames} frames ({writer.frames * step:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
