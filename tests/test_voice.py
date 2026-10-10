"""M4 voice pipeline: endpointing, speculation, the pipeline's flow, the speaker's priorities, helpers.

No microphone or speaker needed: VAD, wake word, Whisper, TTS and the audio output are faked for the logic
tests; one GPU test runs the real models on a synthesized question.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from answers.speech import Priority, Utterance
from core.config import EndpointConfig, load_config
from voice.audio import CHUNK, SAMPLE_RATE, resample, to_float, to_int16, tone
from voice.endpoint import VAD_CHUNK, Endpointer
from voice.pipeline import HeardQuestion, VoicePipeline
from voice.speaker import AudioSpeaker
from voice.stt import strip_wake_fragment
from voice.tts import ClipCache

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Drive Pilot, can I go?", "can I go?"),
        ("It's a driver's pilot who got here first.", "who got here first."),
        ("drivepilot what does that sign say", "what does that sign say"),
        ("Who got here first?", "Who got here first?"),
        ("A lot of traffic ahead?", "A lot of traffic ahead?"),
        ("Hey DrivePilot.", ""),
        ("Pilot. Hey DrivePilot, is the intersection clear?", "is the intersection clear?"),  # said again
        ("Hey DrivePilot, hey Drive Pilot, can I go?", "can I go?"),
        ("Hey DrivePilot, can I go now, DrivePilot?", "can I go now, DrivePilot?"),  # the name at the end stays
        ("Can I go now, Drive Pilot?", "Can I go now, Drive Pilot?"),
    ],
)
def test_wake_fragment_is_removed(text: str, expected: str) -> None:
    assert strip_wake_fragment(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Resumed audio starts after the wake phrase: the loose pattern would eat "is it my turn," here.
        ("is it my turn, DrivePilot?", "is it my turn, DrivePilot?"),
        ("Can I go now, Drive Pilot?", "Can I go now, Drive Pilot?"),
        ("A driver is stopped ahead, why?", "A driver is stopped ahead, why?"),
        # A repeated wake phrase, as Whisper writes it (review finding: these were kept).
        ("Hey DrivePilot, is it my turn?", "is it my turn?"),
        ("It's a driver's pilot, who got here first?", "who got here first?"),
        ("It’s a driver’s pilot, who got here first?", "who got here first?"),
        ("Hey driver pilot, stop.", "stop."),
        ("A DrivePilot, can I go?", "can I go?"),
        ("Hey Dr. Pilot, can I go?", "can I go?"),
        ("Uh, hey DrivePilot, can I go?", "can I go?"),
    ],
)
def test_a_resumed_question_loses_only_a_repeated_wake_phrase(text: str, expected: str) -> None:
    assert strip_wake_fragment(text, pre_roll=False) == expected


def test_audio_helpers() -> None:
    audio = np.sin(np.linspace(0, 200, 22050)).astype(np.float32)
    assert abs(len(resample(audio, 22050, 16000)) - 16000) <= 1
    assert len(tone(16000, (440.0, 880.0), (0.1, 0.05), 0.5)) == 2400
    assert np.allclose(to_float(to_int16(audio)), audio, atol=1e-4)


# --------------------------------------------------------------------------------------- endpointing


class FakeVad:
    """Speech probability by stream time: `speech(t)` (t = seconds since the endpointer started)."""

    def __init__(self, speech: list[tuple[float, float]]) -> None:
        self.speech = speech
        self.reset()

    def reset(self) -> None:
        self.chunks = 0  # 32 ms chunks decided so far
        self.pending = 0  # samples waiting for a full chunk (like the real VAD)

    def process(self, audio: np.ndarray) -> list[float]:
        self.pending += len(audio)
        out = []
        while self.pending >= VAD_CHUNK:
            self.pending -= VAD_CHUNK
            t = self.chunks * VAD_CHUNK / SAMPLE_RATE
            self.chunks += 1
            out.append(0.9 if any(a <= t < b for a, b in self.speech) else 0.05)
        return out


def run_endpointer(speech: list[tuple[float, float]], seconds: float, cfg: EndpointConfig | None = None):
    ep = Endpointer(cfg or EndpointConfig(), FakeVad(speech))  # type: ignore[arg-type]
    ep.start(np.zeros(0, np.int16), 0.0)
    events = []
    chunk = np.zeros(CHUNK, np.int16)
    for i in range(1, round(seconds * SAMPLE_RATE / CHUNK) + 1):
        event = ep.feed(chunk, i * CHUNK / SAMPLE_RATE)
        if event is not None:
            events.append((round(i * CHUNK / SAMPLE_RATE, 2), event.kind, event.speech_end_t))
            if event.kind != "speculate":
                break
    return ep, events


def test_question_ends_after_the_silence_with_a_speculation_first() -> None:
    _, events = run_endpointer([(0.3, 2.0)], 5.0)
    kinds = [k for _, k, _ in events]
    assert kinds == ["speculate", "end"]
    (spec_t, _, spec_end), (end_t, _, end_speech) = events
    cfg = EndpointConfig()
    assert 2.0 + cfg.speculative_silence_s <= spec_t <= 2.0 + cfg.speculative_silence_s + 0.15
    assert 2.0 + cfg.end_silence_s <= end_t <= 2.0 + cfg.end_silence_s + 0.15
    assert end_speech == pytest.approx(2.0, abs=0.04) and spec_end == end_speech


def test_the_pause_after_the_wake_phrase_does_not_end_the_question() -> None:
    # "Hey DrivePilot, <pause> is it safe to pull out?": the tail of "...pilot" right after the wake fired, then
    # a pause longer than the end-of-question silence, then the question. It used to end at the pause.
    _, events = run_endpointer([(0.0, 0.2), (0.9, 2.5)], 6.0)
    kinds = [k for _, k, _ in events]
    assert kinds[-1] == "end" and events[-1][2] == pytest.approx(2.5, abs=0.04)


def test_a_one_word_command_right_after_the_wake_phrase_is_heard() -> None:
    # "Hey DrivePilot, stop": the word falls in the wake-phrase tail window; it still ends as a question.
    cfg = EndpointConfig()
    _, events = run_endpointer([(0.0, 0.35)], 5.0)
    assert events[-1][1] == "end" and events[-1][2] == pytest.approx(0.35, abs=0.04)
    assert events[-1][0] == pytest.approx(0.35 + cfg.wake_pause_s, abs=0.15)


def test_a_question_said_in_one_breath_with_the_wake_phrase_counts() -> None:
    _, events = run_endpointer([(0.0, 2.0)], 5.0)  # no pause between "DrivePilot" and the question
    assert events[-1][1] == "end" and events[-1][2] == pytest.approx(2.0, abs=0.04)


def test_speech_after_a_pause_voids_the_speculation() -> None:
    ep, events = run_endpointer([(0.3, 1.5), (1.9, 3.0)], 6.0)
    assert [k for _, k, _ in events] == ["speculate", "speculate", "end"]
    assert ep.speculation_still_valid()  # the second speculation covers everything that was said


def test_nothing_said_times_out() -> None:
    _, events = run_endpointer([], 6.0)
    assert events[-1][1] == "timeout" and events[-1][0] == pytest.approx(4.0, abs=0.1)


def test_a_question_that_never_ends_is_cut_at_the_maximum_length() -> None:
    _, events = run_endpointer([(0.2, 30.0)], 15.0, EndpointConfig(max_utterance_s=5.0))
    assert events[-1][1] == "end" and events[-1][0] == pytest.approx(5.0, abs=0.1)


def test_a_resumed_endpointer_has_no_wake_tail_and_keeps_the_wake_timeout() -> None:
    # Resumed at 2.0 s after a wake at 1.0 s: speech right away is the question (it ends after end_silence_s,
    # not wake_pause_s), and with nothing said it gives up 4 s after the wake, not 4 s after the resume.
    cfg = EndpointConfig()
    chunk = np.zeros(CHUNK, np.int16)

    def run(speech: list[tuple[float, float]]) -> tuple[float, str, float | None]:
        ep = Endpointer(cfg, FakeVad(speech))  # type: ignore[arg-type]
        ep.resume(2.0, wake_t=1.0)
        for i in range(1, 100):
            t = 2.0 + i * CHUNK / SAMPLE_RATE
            event = ep.feed(chunk, t)
            if event is not None and event.kind != "speculate":
                return t, event.kind, event.speech_end_t
        raise AssertionError("no end")

    end_t, kind, speech_end = run([(0.0, 0.3)])
    assert kind == "end" and speech_end == pytest.approx(2.3, abs=0.04)
    assert end_t == pytest.approx(2.3 + cfg.end_silence_s, abs=0.15)
    end_t, kind, _ = run([])
    assert kind == "timeout" and end_t == pytest.approx(1.0 + cfg.no_speech_timeout_s, abs=0.1)


# ------------------------------------------------------------------------------------------ pipeline


class FakeWake:
    def __init__(self, fire_at: list[float]) -> None:
        self.fire_at = list(fire_at)

    def process(self, chunk: np.ndarray, t: float) -> float | None:
        if self.fire_at and t >= self.fire_at[0]:
            self.fire_at.pop(0)
            return 0.9
        return None


class FakeStt:
    def __init__(self, text: str = "who got here first?", delay_s: float = 0.05) -> None:
        self.text = text
        self.delay_s = delay_s
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        self.calls.append(len(audio))
        time.sleep(self.delay_s)
        return self.text, self.delay_s * 1000


def feed_seconds(pipe: VoicePipeline, seconds: float) -> None:
    chunk = np.zeros(CHUNK, np.int16)
    for _ in range(round(seconds * SAMPLE_RATE / CHUNK)):
        pipe.feed(chunk)


def test_pipeline_wakes_collects_and_uses_the_speculative_transcript() -> None:
    cfg = load_config().voice
    heard: list[HeardQuestion] = []
    wakes: list[float] = []
    stt = FakeStt()
    pipe = VoicePipeline(
        cfg,
        FakeWake([1.0]),
        stt,
        heard.append,
        on_wake=lambda: wakes.append(time.perf_counter()),
        vad=FakeVad([(0.2, 1.8)]),
        answer_in_thread=False,
    )  # type: ignore[arg-type]
    feed_seconds(pipe, 5.0)
    pipe.flush()
    assert len(wakes) == 1 and pipe.wakes == 1
    assert len(heard) == 1 and heard[0].text == "who got here first?" and heard[0].speculative
    assert len(stt.calls) == 1  # the speculation was used: Whisper ran once
    assert stt.calls[0] >= round(cfg.endpoint.pre_roll_s * SAMPLE_RATE)  # the pre-roll is part of the question


def test_the_question_audio_has_no_gap_after_the_wake_word() -> None:
    # The chunk the wake word fired in used to be left out of the question's audio: an 80 ms hole right after
    # "Hey DrivePilot" that made Whisper mishear the first word on real voices ("tell me" -> "only one").
    cfg = load_config().voice
    received: list[np.ndarray] = []

    class RecordingStt(FakeStt):
        def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
            received.append(audio)
            return super().transcribe(audio)

    pipe = VoicePipeline(
        cfg, FakeWake([1.0]), RecordingStt(), lambda q: None, vad=FakeVad([(0.2, 1.8)]), answer_in_thread=False
    )  # type: ignore[arg-type]
    stream = (np.arange(5 * SAMPLE_RATE) % 20000).astype(np.int16)  # every sample different from its neighbour
    for k in range(0, len(stream), CHUNK):
        pipe.feed(stream[k : k + CHUNK])
    pipe.flush()
    audio = np.round(received[-1] * 32768.0).astype(np.int64)
    steps = np.diff(audio) % 20000
    assert np.all(steps == 1)  # contiguous: no chunk missing anywhere
    assert len(audio) > round(cfg.endpoint.pre_roll_s * SAMPLE_RATE)


def test_wake_during_a_question_is_ignored_and_empty_transcripts_are_dropped() -> None:
    # The question runs past the listening window (no_speech_timeout_s after the latest wake word, here the one
    # at 1.5 s inside the question): an empty transcript then ends it, instead of listening on.
    cfg = load_config().voice
    heard: list[HeardQuestion] = []
    pipe = VoicePipeline(
        cfg, FakeWake([1.0, 1.5]), FakeStt(text=""), heard.append, vad=FakeVad([(0.2, 6.0)]), answer_in_thread=False
    )  # type: ignore[arg-type]
    feed_seconds(pipe, 8.0)
    pipe.flush()
    assert pipe.wakes == 1 and heard == [] and pipe.empty_transcripts == 1 and pipe.resumes == 0


# The wake word fired early, inside "...pilot": the rest of the wake phrase passes for the question and the pause
# after it for its end, so Whisper hears only the wake phrase (stripped to ""). Seen on synthetic voices at
# threshold 0.15 ("Stop watching the light": woke, nothing heard).

WAKE_TAIL = 3000  # amplitude of the fake "...pilot" (int16)
QUESTION = 6000  # amplitude of the fake question: the fake Whisper hears the question only where it is


class LoudVad:
    """Speech wherever the audio is loud (the fake speech is a constant amplitude), on the real VAD's 32 ms grid."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pending = np.zeros(0, np.float32)

    def process(self, audio: np.ndarray) -> list[float]:
        self.pending = np.concatenate((self.pending, to_float(audio)))
        out = []
        while len(self.pending) >= VAD_CHUNK:
            chunk, self.pending = self.pending[:VAD_CHUNK], self.pending[VAD_CHUNK:]
            out.append(0.9 if float(np.abs(chunk).mean()) > 0.02 else 0.05)
        return out


class QuestionStt:
    """Hears "stop watching the light" in audio that contains the question; only the wake phrase gives ""."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate  # holds back the wake-phrase-only transcript until set
        self.calls: list[np.ndarray] = []
        self.pre_roll: list[bool] = []  # per call: does the audio start with the wake phrase's pre-roll?

    def transcribe(self, audio: np.ndarray, pre_roll: bool = True) -> tuple[str, float]:
        self.calls.append(audio)
        self.pre_roll.append(pre_roll)
        if float(np.abs(audio).max(initial=0.0)) > (WAKE_TAIL + QUESTION) / 2 / 32768:
            return "stop watching the light", 5.0
        if self.gate is not None:
            self.gate.wait(5.0)
        return "", 5.0


def speech_stream(seconds: float, *segments: tuple[float, float, int]) -> np.ndarray:
    """int16 audio, silent except for (start s, end s, amplitude) segments; whole chunks."""
    pcm = np.zeros(round(seconds * SAMPLE_RATE / CHUNK) * CHUNK, np.int16)
    for start, end, amplitude in segments:
        pcm[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)] = amplitude
    return pcm


def feed_pcm(pipe: VoicePipeline, pcm: np.ndarray, live: bool = False) -> None:
    """Feed whole chunks. `live`: every transcript is ready before the next chunk (a fast Whisper on a live mic),
    so a resume is picked up by the next chunk; otherwise chunks go in at once and `flush` picks it up (the
    benchmarks)."""
    for k in range(0, len(pcm), CHUNK):
        pipe.feed(pcm[k : k + CHUNK])
        if live:
            pipe._stt_pool.submit(lambda: None).result(timeout=5.0)


def wake_only_pipeline(wakes: list[float], stt: QuestionStt) -> tuple[VoicePipeline, list[HeardQuestion], list[str]]:
    heard: list[HeardQuestion] = []
    nothing: list[str] = []
    pipe = VoicePipeline(
        load_config().voice,
        FakeWake(wakes),  # type: ignore[arg-type]
        stt,  # type: ignore[arg-type]
        heard.append,
        vad=LoudVad(),  # type: ignore[arg-type]
        answer_in_thread=False,
        on_nothing=nothing.append,
    )
    return pipe, heard, nothing


@pytest.mark.parametrize("live", [True, False])
def test_a_question_after_a_wake_phrase_only_transcript_is_still_heard(live: bool) -> None:
    # Wake fires at 1.04 s; "...pilot" goes on to 1.5 s (past wake_tail_s, so it counts as the question); the pause
    # ends that "question" at ~2.0 s; the real question is 2.4-3.6 s. Live: the transcript is ready before the
    # next chunk, which resumes. Batch: the transcript is held back until all the audio is in, so only `flush`
    # can resume, and the end found while replaying must be dated when its audio arrived, not at the replay.
    gate = None if live else threading.Event()
    stt = QuestionStt(gate)
    pipe, heard, nothing = wake_only_pipeline([1.0], stt)
    feed_pcm(pipe, speech_stream(7.0, (0.6, 1.5, WAKE_TAIL), (2.4, 3.6, QUESTION)), live=live)
    fed_at = time.perf_counter()
    if gate is not None:
        time.sleep(0.2)
        gate.set()
    pipe.flush()
    assert [q.text for q in heard] == ["stop watching the light"] and nothing == []
    if not live:
        assert heard[0].end_detected_at <= fed_at  # (the replay ran after the 0.2 s wait)
    assert pipe.wakes == 1 and pipe.resumes == 1 and heard[0].resumed
    assert heard[0].speech_end_t == pytest.approx(3.6, abs=0.05)
    assert heard[0].end_t == pytest.approx(3.6 + pipe.cfg.endpoint.end_silence_s, abs=0.15)
    question = np.abs(stt.calls[-1]) > (WAKE_TAIL + QUESTION) / 2 / 32768
    assert int(question.sum()) == round(1.2 * SAMPLE_RATE)  # all of the question went to Whisper
    assert stt.pre_roll[0] and not stt.pre_roll[-1]  # the resumed audio has no wake phrase to strip loosely
    assert not pipe.listening


def test_a_repeated_wake_phrase_restarts_the_listening_window() -> None:
    # No answer after an early wake, so the driver says "Hey DrivePilot" again (3.5-4.2 s, the wake fires at 4.24),
    # pauses, and asks at 5.3 s: more than no_speech_timeout_s after the first wake, but within it of the repeat.
    # (Review finding: the repeat was swallowed and the window still ran from the first wake: question lost.)
    wakes: list[float] = []
    pipe, heard, nothing = wake_only_pipeline([1.0, 4.2], QuestionStt())
    pipe.on_wake = lambda: wakes.append(pipe.t)
    feed_pcm(
        pipe,
        speech_stream(10.0, (0.6, 1.5, WAKE_TAIL), (3.5, 4.2, WAKE_TAIL), (5.3, 6.3, QUESTION)),
        live=True,
    )
    pipe.flush()
    assert [q.text for q in heard] == ["stop watching the light"] and nothing == []
    assert len(wakes) == 1 and pipe.wakes == 1  # one question: the repeat doesn't start (or supersede) another


def test_the_name_at_the_end_of_a_resumed_question_does_not_lose_it() -> None:
    # "...is it safe, DrivePilot?" after a wake-phrase-only end: the wake word fires at the question's end (3.7 s).
    # Restarting the question there would drop it; it must be heard, without a second on_wake (which would make
    # the app count it as superseded by a newer question).
    wakes: list[float] = []
    pipe, heard, nothing = wake_only_pipeline([1.0, 3.6], QuestionStt())
    pipe.on_wake = lambda: wakes.append(pipe.t)
    feed_pcm(pipe, speech_stream(8.0, (0.6, 1.5, WAKE_TAIL), (2.4, 3.6, QUESTION)), live=True)
    pipe.flush()
    assert [q.text for q in heard] == ["stop watching the light"] and nothing == [] and len(wakes) == 1


def test_after_a_wake_phrase_only_transcript_listening_still_ends_at_the_wake_timeout() -> None:
    pipe, heard, nothing = wake_only_pipeline([1.0], QuestionStt())
    feed_pcm(pipe, speech_stream(8.0, (0.6, 1.5, WAKE_TAIL)), live=True)
    pipe.flush()
    assert heard == [] and nothing == ["timeout"] and pipe.resumes == 1
    assert pipe.empty_transcripts == 0 and pipe.timeouts == 1


def test_a_new_wake_word_replaces_a_pending_resume() -> None:
    # The wake-phrase-only transcript is still being made when "Hey DrivePilot" is said again (2.24 s): the new
    # wake takes over, and the old empty transcript neither resumes nor reports "nothing" (which would switch
    # off the page's listening light during the new question).
    gate = threading.Event()
    pipe, heard, nothing = wake_only_pipeline([1.0, 2.2], QuestionStt(gate))
    pcm = speech_stream(7.0, (0.6, 1.5, WAKE_TAIL), (2.6, 3.6, QUESTION))
    split = round(2.4 * SAMPLE_RATE / CHUNK) * CHUNK
    feed_pcm(pipe, pcm[:split])
    gate.set()
    feed_pcm(pipe, pcm[split:])
    pipe.flush()
    assert [q.text for q in heard] == ["stop watching the light"] and not heard[0].resumed
    assert nothing == [] and pipe.wakes == 2 and pipe.resumes == 0


# -------------------------------------------------------------------------------------------- speaker


class FakeOutput:
    """Plays instantly-starting sounds whose length is real time / 20; interruptible."""

    def __init__(self) -> None:
        self.played: list[int] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def play(
        self, audio: np.ndarray, rate: int, timeout: float | None = None, should_start: Any = None
    ) -> float | None:
        with self._lock:  # like AudioOutput: the start check and stop() share a lock
            if should_start is not None and not should_start():
                return None
            self._stop.clear()
        started = time.perf_counter()
        self.played.append(len(audio))
        self._stop.wait(len(audio) / rate / 20)
        return started

    def stop(self) -> None:
        with self._lock:
            self._stop.set()


class FakeTts:
    rate = 16000
    name = "fake"

    class cfg:
        length_scale = 1.0

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(len(text) * 1600, np.float32)  # 0.1 s per character (5 ms when played by FakeOutput)


def make_speaker(tmp_path: Path) -> tuple[AudioSpeaker, FakeOutput]:
    tts, output = FakeTts(), FakeOutput()
    speaker = AudioSpeaker(tts, output, ClipCache(tts, tmp_path), chime_volume=0.2)  # type: ignore[arg-type]
    return speaker.start(), output


def test_alerts_interrupt_answers_and_the_answer_resumes(tmp_path: Path) -> None:
    speaker, _ = make_speaker(tmp_path)
    long_answer = "x" * 400  # 2 s when played
    speaker.say(Utterance(long_answer, "answer", Priority.ANSWER))
    time.sleep(0.3)
    speaker.say(Utterance("You're up, check and go.", "alert", Priority.REQUESTED, clip="your_turn"))
    assert speaker.wait_idle(10.0)
    records = list(speaker.records)
    assert [(r.kind, r.interrupted) for r in records] == [("answer", True), ("alert", False), ("answer", False)]
    assert records[1].latency_ms is not None and records[1].latency_ms < 100  # pre-rendered: plays at once
    speaker.close()


def test_an_alert_that_changes_the_situation_drops_stale_answers(tmp_path: Path) -> None:
    speaker, _ = make_speaker(tmp_path)
    speaker.say(Utterance("Not yet, the light's red. " * 8, "answer", Priority.ANSWER))
    speaker.say(Utterance("I'll tell you when it turns green.", "answer", Priority.ANSWER))
    time.sleep(0.3)
    green = Utterance("Light's green, check the intersection and go.", "alert", Priority.REQUESTED, drops_answers=True)
    speaker.say(green)
    speaker.say(Utterance("A newer answer.", "answer", Priority.ANSWER))  # said after the alert: kept
    assert speaker.wait_idle(10.0)
    said = sorted((r.kind, r.text[:15], r.interrupted, r.sound_at is not None) for r in speaker.records)
    assert said == [
        ("alert", "Light's green, ", False, True),
        ("answer", "A newer answer.", False, True),
        ("answer", "I'll tell you w", True, False),  # queued: dropped, never played
        ("answer", "Not yet, the li", True, True),  # playing: cut off, not resumed
    ]
    speaker.close()


def test_stop_silences_everything(tmp_path: Path) -> None:
    speaker, _ = make_speaker(tmp_path)
    for text in ("x" * 400, "y" * 400, "z" * 400):
        speaker.say(Utterance(text, "answer", Priority.ANSWER))
    time.sleep(0.2)
    speaker.stop()
    assert speaker.wait_idle(2.0)
    records = list(speaker.records)
    assert len(records) == 3 and all(r.interrupted for r in records)  # every utterance gets a record
    assert sum(r.sound_at is not None for r in records) == 1  # only the one that was playing made a sound
    speaker.close()


def test_clip_cache_renders_once_and_reloads(tmp_path: Path) -> None:
    tts = FakeTts()
    first = ClipCache(tts, tmp_path).get("Light's green.")  # type: ignore[arg-type]
    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 1
    again = ClipCache(tts, tmp_path).get("Light's green.")  # type: ignore[arg-type]
    assert len(again) == len(first)


def test_listening_stops_speech_and_blips(tmp_path: Path) -> None:
    speaker, output = make_speaker(tmp_path)
    speaker.say(Utterance("x" * 400, "answer", Priority.ANSWER))
    time.sleep(0.2)
    speaker.listening()  # "Hey DrivePilot" mid-answer
    assert speaker.wait_idle(2.0)
    kinds = [(r.kind, r.interrupted) for r in speaker.records]
    assert kinds == [("answer", True), ("cue", False)]  # the answer is dropped, not resumed
    assert output.played[-1] < 0.1 * FakeTts.rate  # a short blip
    speaker.close()


def test_navigation_alerts_are_synthesized_not_saved(tmp_path: Path) -> None:
    tts, output = FakeTts(), FakeOutput()
    clips = ClipCache(tts, tmp_path)  # type: ignore[arg-type]
    clips.prepare(["Not this one."])
    played: list[tuple[str, float | None]] = []
    speaker = AudioSpeaker(
        tts,  # type: ignore[arg-type]
        output,  # type: ignore[arg-type]
        clips,
        on_record=lambda r: played.append((r.text, r.sound_at)),
    ).start()
    speaker.say(Utterance("In half a mile, use the left lane.", "alert", Priority.REQUESTED, clip="nav_lanes"))
    speaker.say(Utterance("Not this one.", "alert", Priority.REQUESTED, clip="nav_not_this_exit"))
    assert speaker.wait_idle(5.0)
    speaker.close()
    assert len(list(tmp_path.glob("*.wav"))) == 1  # only the prepared fixed text is kept on disk
    assert [text for text, _ in played] == ["In half a mile, use the left lane.", "Not this one."]
    assert all(sound_at is not None for _, sound_at in played)


def test_pipeline_reports_when_no_question_follows_the_wake_word() -> None:
    # Nothing said: "timeout". Only noise after the wake word (an empty transcript): it listens on, then "timeout".
    # Sound that outlasts the listening window and transcribes to nothing: "empty".
    reports = []
    for segments in ([], [(1.2, 1.8, WAKE_TAIL)], [(1.2, 6.0, WAKE_TAIL)]):
        pipe, heard, nothing = wake_only_pipeline([1.0], QuestionStt())
        feed_pcm(pipe, speech_stream(9.0, *segments), live=True)
        pipe.flush()
        assert heard == []
        reports.append(nothing)
    assert reports == [["timeout"], ["timeout"], ["empty"]]


# --------------------------------------------------------------------------------- real models (GPU)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_pipeline_hears_a_synthetic_question() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    voice_path = ROOT / "data" / "wakeword" / "piper_voices" / "en_US-joe-medium.onnx"
    if not voice_path.is_file():
        pytest.skip("held-out test voice not downloaded (scripts/wakeword)")
    from piper import PiperVoice

    from voice.router import route
    from voice.stt import Transcriber
    from voice.wakeword import WakeWordDetector

    cfg = load_config()
    voice = PiperVoice.load(str(voice_path))
    speech = resample(
        np.concatenate([c.audio_float_array for c in voice.synthesize("Hey Drive Pilot, can I go?")]),
        voice.config.sample_rate,
        SAMPLE_RATE,
    )
    stream = to_int16(
        np.concatenate([np.zeros(SAMPLE_RATE, np.float32), speech, np.zeros(2 * SAMPLE_RATE, np.float32)])
    )
    stt = Transcriber(cfg.voice.stt, cfg.root)
    assert stt.device == "cuda"
    heard: list[HeardQuestion] = []
    pipe = VoicePipeline(
        cfg.voice, WakeWordDetector(cfg.voice.wakeword, cfg.root), stt, heard.append, answer_in_thread=False
    )
    for k in range(0, len(stream) - CHUNK + 1, CHUNK):
        pipe.feed(stream[k : k + CHUNK])
    pipe.flush(timeout=30.0)
    assert len(heard) == 1
    result = route(heard[0].text)
    assert (result.kind.value, result.intent) == ("right_of_way", "can_i_go"), heard[0].text


# ------------------------------------------------------------------- M4/M7 review regressions (speaker)


class SlowStartOutput(FakeOutput):
    """Takes 50 ms to prepare a sound (resampling) before the start check, like a busy AudioOutput."""

    def play(
        self, audio: np.ndarray, rate: int, timeout: float | None = None, should_start: Any = None
    ) -> float | None:
        time.sleep(0.05)
        return super().play(audio, rate, timeout, should_start)


def test_a_stop_while_a_sound_is_being_prepared_prevents_it(tmp_path: Path) -> None:
    tts, output = FakeTts(), SlowStartOutput()
    speaker = AudioSpeaker(tts, output, ClipCache(tts, tmp_path)).start()  # type: ignore[arg-type]
    for _ in range(20):
        speaker.say(Utterance("x" * 400, "answer", Priority.ANSWER))  # 2 s when played
        time.sleep(0.01)  # rendered; now inside the 50 ms preparation window
        speaker.stop()
        assert speaker.wait_idle(2.0)
    speaker.close()
    assert all(r.sound_at is None for r in speaker.records)  # never started, not "stopped the previous sound"


def test_an_alert_plays_at_once_while_a_long_answer_is_synthesized(tmp_path: Path) -> None:
    class SlowTts(FakeTts):
        def synthesize(self, text: str) -> np.ndarray:
            if text.startswith("long"):
                time.sleep(0.6)  # Piper on a busy CPU, a long sentence
            return super().synthesize(text)

    tts = SlowTts()
    clips = ClipCache(tts, tmp_path)  # type: ignore[arg-type]
    clips.prepare(["You're up, check and go."])
    speaker = AudioSpeaker(tts, FakeOutput(), clips).start()  # type: ignore[arg-type]
    speaker.say(Utterance("long answer " * 10, "answer", Priority.ANSWER))
    time.sleep(0.05)  # synthesis has started
    speaker.say(Utterance("You're up, check and go.", "alert", Priority.REQUESTED, clip="your_turn"))
    assert speaker.wait_idle(5.0)
    speaker.close()
    alert = next(r for r in speaker.records if r.kind == "alert")
    assert alert.latency_ms is not None and alert.latency_ms < 100  # didn't wait for the 0.6 s synthesis


def test_the_wake_word_stops_answers_but_not_alerts(tmp_path: Path) -> None:
    speaker, _ = make_speaker(tmp_path)
    speaker.say(Utterance("Light's green, check the intersection and go." * 4, "alert", Priority.REQUESTED))
    speaker.say(Utterance("an answer " * 20, "answer", Priority.ANSWER))
    time.sleep(0.1)  # the alert is playing, the answer waits
    speaker.listening()
    assert speaker.wait_idle(5.0)
    speaker.close()
    played = [(r.kind, r.sound_at is not None) for r in speaker.records]
    assert ("alert", True) in played and ("answer", False) in played and ("cue", True) in played
    alert = next(r for r in speaker.records if r.kind == "alert")
    assert not alert.interrupted  # the alert finished
