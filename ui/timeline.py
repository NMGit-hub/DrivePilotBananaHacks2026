"""Showcase timeline (scripts/render_showcase.py): when each spoken line plays in a rendered video, what the captions
show at each moment, and the audio track.

The renderer runs the app's perception and assistant in lock-step, so every answer is ready the instant its question
is asked. To look like the real app, each line is placed in video time the way the real speaker (voice/speaker.py)
would play it:

- the driver's question ("Hey DrivePilot, ...") is spoken so that it ends just before the moment the question is
  asked; the wake word stops an answer that is still playing (as `Assistant.interrupt` does);
- an answer starts after a realistic delay (the measured answer time) and its sentences follow one another;
- an alert starts at its event time; it interrupts an answer that is playing (that sentence is said again after it,
  unless the alert made answers stale: `drops_answers`), and a safety alert goes ahead of (and cuts off) one that
  isn't.

`schedule` places the lines on one speaker channel (the driver's voice is its own channel), `build_captions` turns
the placed lines into caption cards (a question with its whole answer, alerts on their own), `visible_captions` says
which cards show at a time and how opaque, and `mix` builds the audio track. All pure: tests/test_render_timeline.py.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

QUESTION, ANSWER, ALERT = "question", "answer", "alert"
SAFETY = 3  # answers.speech.Priority.SAFETY (kept as a number: this module needs no app imports)
GAP_S = 0.15  # silence between two lines played back to back
LOCAL_DELAY_S = 0.45  # a local answer (right of way, watches, controls): question asked -> first word
MODEL_DELAY_MIN_S = 0.8
MODEL_DELAY_MAX_S = 3.0
LOCAL_SOURCES = frozenset({"local", "control", "none", "parking", "error"})
HEADROOM = 0.95  # the mixed track's peak is kept at most this (AAC encoding overshoots a full-scale peak)


@dataclass(frozen=True, slots=True)
class Line:
    """One thing heard in the video.

    Attributes:
        kind: "question" (the driver), "answer" or "alert" (DrivePilot).
        text: what is said (and shown).
        want_t: video time it should start.
        duration: its audio length in seconds.
        priority: `answers.speech.Priority` value (alerts: SAFETY interrupts other alerts).
        drops_answers: an alert that makes answers stale ("Light's green"): the interrupted and queued answers are
            not said.
        group: the question an answer belongs to (its question has the same group); 0 = not part of a question.
        uid: tells apart two lines with the same text (a sentence said again after an alert keeps its uid).
        asked_t: an answer: when its question was asked (an alert that makes answers stale also drops the answers
            of questions asked before it that haven't arrived yet: in the app they would be cut while streaming).
    """

    kind: str
    text: str
    want_t: float
    duration: float
    priority: int = 1
    drops_answers: bool = False
    group: int = 0
    uid: int = 0
    asked_t: float = math.inf


@dataclass(frozen=True, slots=True)
class Placed:
    """A line as it plays: from `start` to `end` (video time); `cut` = interrupted before its end."""

    line: Line
    start: float
    end: float
    cut: bool = False


_QUESTION_WORDS = frozenset(
    {
        *("what", "where", "which", "who", "whose", "when", "why", "how", "any", "anything"),
        *("is", "are", "am", "can", "could", "should", "would", "will", "do", "does", "did", "was", "were"),
        *("has", "have"),
    }
)


def question_caption(text: str) -> str:
    """A question as the caption shows it: first letter capitalized, a question mark if it reads as a question
    ("what does that sign say" -> "What does that sign say?"; "tell me when it's green" stays as it is)."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    first = text.split()[0].lower().strip(",")
    if first in _QUESTION_WORDS and text[-1] not in ".?!":
        text += "?"
    return text


def answer_delay(
    source: str,
    first_ms: float | None,
    tts_s: float = 0.0,
    local_s: float = LOCAL_DELAY_S,
    min_s: float = MODEL_DELAY_MIN_S,
    max_s: float = MODEL_DELAY_MAX_S,
) -> float:
    """Question asked -> first word of the answer, as in the real app: `local_s` for local answers, else the measured
    time to the first sentence plus synthesizing it, kept within [min_s, max_s] (a slow API call must not leave the
    video silent for long)."""
    if source in LOCAL_SOURCES or first_ms is None:
        return local_s
    return min(max(first_ms / 1000.0 + tts_s, min_s), max_s)


def after_answers(t: float, answering: Sequence[tuple[float, float]]) -> float:
    """When an event caused by an answer really happens. In lock-step video time stands still while a question is
    answered, so what the answer set in motion (a route: its first guidance alerts) comes up while the answer is
    still "on its way"; in the app it can only come once the answer is ready. `answering`: (asked, ready) video
    times of the questions whose answers cause such events; an event inside one of them moves to its `ready`."""
    for asked, ready in sorted(answering):
        if asked <= t < ready:
            t = ready
    return t


def schedule(lines: Sequence[Line], gap: float = GAP_S) -> list[Placed]:
    """Place every line in time (see the module docstring), sorted by start."""
    order = {QUESTION: 0, ALERT: 1, ANSWER: 2}
    arrivals = sorted(lines, key=lambda line: (line.want_t, order.get(line.kind, 3)))
    placed: list[Placed] = []
    alerts: deque[Line] = deque()
    answers: deque[Line] = deque()
    playing: tuple[Line, float] | None = None  # (line, start)
    latest_question = 0
    dropped_at = -math.inf  # the latest alert that made answers stale
    i = 0

    def finish(at: float, cut: bool) -> None:
        assert playing is not None
        line, start = playing
        if at > start:  # (a line cut before it began, e.g. in the gap before it, was never heard)
            placed.append(Placed(line, start, at, cut))

    def next_queued() -> Line | None:
        if alerts:
            return alerts.popleft()
        if answers:
            return answers.popleft()
        return None

    while i < len(arrivals) or playing is not None:
        arrival = arrivals[i].want_t if i < len(arrivals) else math.inf
        if playing is not None and playing[1] + playing[0].duration <= arrival:
            end = playing[1] + playing[0].duration
            finish(end, False)
            following = next_queued()
            playing = (following, end + gap) if following is not None else None
            continue
        line = arrivals[i]
        i += 1
        t = line.want_t
        if line.kind == QUESTION:
            placed.append(Placed(line, t, t + line.duration))
            latest_question = max(latest_question, line.group)
            answers.clear()  # the wake word stops answering; alerts keep playing
            if playing is not None and playing[0].kind == ANSWER:
                finish(t, True)
                following = next_queued()
                playing = (following, t) if following is not None else None
            continue
        if line.kind == ALERT:
            if line.drops_answers:
                answers.clear()
                dropped_at = t
            if playing is None:
                playing = (line, t)
            elif playing[0].kind == ANSWER:
                interrupted = playing[0]
                finish(t, True)
                if not line.drops_answers:
                    answers.appendleft(interrupted)  # said again after the alert
                playing = (line, t)
            elif line.priority >= SAFETY > playing[0].priority:
                interrupted = playing[0]
                finish(t, True)
                alerts.appendleft(interrupted)  # said again right after the safety alert
                playing = (line, t)
            elif line.priority >= SAFETY:
                ahead = sum(1 for queued in alerts if queued.priority >= SAFETY)
                alerts.insert(ahead, line)
            else:
                alerts.append(line)
            continue
        if line.group and line.group < latest_question:
            continue  # the answer to an earlier question, overtaken by a newer one
        if line.group and line.asked_t <= dropped_at:
            continue  # its question was asked before an alert that made answers stale
        if playing is None:
            playing = (line, t)
        else:
            answers.append(line)
    return sorted(placed, key=lambda p: (p.start, order.get(p.line.kind, 3)))


@dataclass(slots=True)
class Caption:
    """One caption card: a question (with when its answer ends), an answer growing sentence by sentence, or an
    alert. `parts`: (video time it appears, text)."""

    kind: str
    parts: list[tuple[float, str]]
    start: float
    end: float  # when its audio ends (a question: when its answer ends)
    group: int = 0
    image_key: str | None = None  # a picture shown with the answer (the remembered frame it came from)

    def text_at(self, t: float) -> str:
        """The text shown at video time t (an answer shows the sentences started so far)."""
        return " ".join(text for at, text in self.parts if at <= t + 1e-9)


def build_captions(placed: Sequence[Placed]) -> list[Caption]:
    """Caption cards from the placed lines, sorted by start: one per question, one per question's answer (all its
    sentences, each appearing when it starts playing; a sentence said again after an alert appears once), one per
    alert or other line."""
    captions: list[Caption] = []
    answers: dict[int, Caption] = {}
    seen: set[tuple[int, int, str]] = set()
    for p in sorted(placed, key=lambda p: p.start):
        line = p.line
        if line.kind == ANSWER and line.group:
            key = (line.group, line.uid, line.text)
            card = answers.get(line.group)
            if card is None:
                card = Caption(ANSWER, [], p.start, p.end, line.group)
                answers[line.group] = card
                captions.append(card)
            if key not in seen:
                seen.add(key)
                card.parts.append((p.start, line.text))
            card.end = max(card.end, p.end)
            continue
        captions.append(Caption(line.kind, [(p.start, line.text)], p.start, p.end, line.group))
    for card in captions:  # a question stays up while its answer plays
        if card.kind == QUESTION and card.group in answers:
            card.end = max(card.end, answers[card.group].end)
    return sorted(captions, key=lambda c: c.start)


@dataclass(frozen=True, slots=True)
class Shown:
    """A caption card at one moment: its text then and its opacity (fade in / out)."""

    caption: Caption
    text: str
    alpha: float


def visible_captions(
    captions: Sequence[Caption],
    t: float,
    hold_s: float = 3.5,
    fade_s: float = 0.4,
    fade_in_s: float = 0.2,
    max_shown: int = 3,
) -> list[Shown]:
    """The cards on screen at video time t, oldest first: each from its start until `hold_s` after its audio ends
    (then fading out over `fade_s`), at most `max_shown` (the newest)."""
    shown: list[Shown] = []
    for card in captions:
        if card.start > t + 1e-9:
            continue
        gone = card.end + hold_s + fade_s
        if t >= gone:
            continue
        alpha = 1.0
        if fade_in_s > 0:
            alpha = min(alpha, (t - card.start) / fade_in_s)
        if t > card.end + hold_s:
            alpha = min(alpha, (gone - t) / fade_s)
        text = card.text_at(t)
        if text:
            shown.append(Shown(card, text, max(0.0, min(1.0, alpha))))
    return shown[-max_shown:] if max_shown > 0 else []


@dataclass(frozen=True, slots=True)
class Window:
    """A stretch of video time with a value (a highlighted track id, "listening")."""

    start: float
    end: float
    value: int = 0


def active(windows: Sequence[Window], t: float) -> Window | None:
    """The latest-starting window that contains t, if any."""
    found = [w for w in windows if w.start <= t < w.end]
    return max(found, key=lambda w: w.start) if found else None


T = TypeVar("T")


class FramePicker(Generic[T]):
    """Constant-frame-rate output from a source with irregular timestamps (as ffmpeg's -vsync cfr): `at(tau)` is the
    latest source frame at or before output time tau (within `half_slot`), so source frames are repeated or dropped
    as needed and the pictures stay on the video's clock. `frames`: (time, frame) in decoding order."""

    def __init__(self, frames: Iterator[tuple[float, T]], half_slot: float) -> None:
        self._frames = frames
        self.half_slot = half_slot
        self.current: tuple[float, T] | None = next(frames, None)
        self.upcoming: tuple[float, T] | None = next(frames, None)

    def at(self, tau: float) -> tuple[float, T] | None:
        """The frame to show at output time tau (times must not go back)."""
        while self.upcoming is not None and self.upcoming[0] <= tau + self.half_slot:
            self.current, self.upcoming = self.upcoming, next(self._frames, None)
        return self.current

    @property
    def exhausted(self) -> bool:
        """No source frames left after the current one."""
        return self.upcoming is None


@dataclass(slots=True)
class AudioTrack:
    """Mono float32 samples at `rate` covering video time from `t0`."""

    samples: np.ndarray
    rate: int
    t0: float
    placed_at: list[tuple[float, float]] = field(default_factory=list)  # (start, end) of each line mixed in

    def chunk(self, start_s: float, end_s: float) -> np.ndarray:
        """Samples from video time start_s to end_s (zeros beyond the track)."""
        a = round((start_s - self.t0) * self.rate)
        b = round((end_s - self.t0) * self.rate)
        out = np.zeros(max(0, b - a), np.float32)
        lo, hi = max(a, 0), min(b, len(self.samples))
        if hi > lo:
            out[lo - a : hi - a] = self.samples[lo:hi]
        return out


def mix(
    placed: Sequence[Placed],
    audio_for: Callable[[Line], np.ndarray],
    rate: int,
    t0: float,
    t1: float,
    volume: Callable[[Line], float] | None = None,
) -> AudioTrack:
    """The audio track from t0 to t1: each placed line's audio (`audio_for`, at `rate`) at its start, cut at its end
    if it was interrupted, summed, and scaled down if its peak exceeds HEADROOM."""
    total = max(0, round((t1 - t0) * rate))
    track = np.zeros(total, np.float32)
    spans: list[tuple[float, float]] = []
    for p in placed:
        audio = np.asarray(audio_for(p.line), dtype=np.float32)
        if p.cut:
            audio = audio[: max(0, round((p.end - p.start) * rate))]
        if len(audio) == 0:
            continue
        gain = volume(p.line) if volume is not None else 1.0
        at = round((p.start - t0) * rate)
        lo, hi = max(at, 0), min(at + len(audio), total)
        if hi <= lo:
            continue
        track[lo:hi] += gain * audio[lo - at : hi - at]
        spans.append((p.start, p.start + len(audio) / rate))
    peak = float(np.max(np.abs(track))) if total else 0.0
    if peak > HEADROOM:
        track *= HEADROOM / peak  # (Piper peaks near full scale; two voices at once add up)
    return AudioTrack(track, rate, t0, spans)
