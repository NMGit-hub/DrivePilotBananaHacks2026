"""M4 voice pipeline: endpointing, speculation, the pipeline's flow, the speaker's priorities, helpers.

No microphone or speaker needed: VAD, wake word, Whisper, TTS and the audio output are faked for the logic
tests; one GPU test runs the real models on a synthesized question.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

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
    ],
)
def test_wake_fragment_is_removed(text: str, expected: str) -> None:
    assert strip_wake_fragment(text) == expected


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
    assert 2.0 + 0.25 <= spec_t <= 2.0 + 0.25 + 0.15
    assert 2.0 + 0.6 <= end_t <= 2.0 + 0.6 + 0.15
    assert end_speech == pytest.approx(2.0, abs=0.04) and spec_end == end_speech


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


def test_wake_during_a_question_is_ignored_and_empty_transcripts_are_dropped() -> None:
    cfg = load_config().voice
    heard: list[HeardQuestion] = []
    pipe = VoicePipeline(
        cfg, FakeWake([1.0, 1.5]), FakeStt(text=""), heard.append, vad=FakeVad([(0.2, 1.8)]), answer_in_thread=False
    )  # type: ignore[arg-type]
    feed_seconds(pipe, 5.0)
    pipe.flush()
    assert pipe.wakes == 1 and heard == [] and pipe.empty_transcripts == 1


# -------------------------------------------------------------------------------------------- speaker


class FakeOutput:
    """Plays instantly-starting sounds whose length is real time / 20; interruptible."""

    def __init__(self) -> None:
        self.played: list[int] = []
        self._stop = threading.Event()

    def play(self, audio: np.ndarray, rate: int, timeout: float | None = None) -> float | None:
        self._stop.clear()
        started = time.perf_counter()
        self.played.append(len(audio))
        self._stop.wait(len(audio) / rate / 20)
        return started

    def stop(self) -> None:
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
    said = [(r.kind, r.text[:15], r.interrupted) for r in speaker.records]
    assert said == [
        ("answer", "Not yet, the li", True),
        ("alert", "Light's green, ", False),
        ("answer", "A newer answer.", False),
    ]
    speaker.close()


def test_stop_silences_everything(tmp_path: Path) -> None:
    speaker, _ = make_speaker(tmp_path)
    for text in ("x" * 400, "y" * 400, "z" * 400):
        speaker.say(Utterance(text, "answer", Priority.ANSWER))
    time.sleep(0.2)
    speaker.stop()
    assert speaker.wait_idle(2.0)
    assert len(speaker.records) == 1 and speaker.records[0].interrupted
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
    cfg = load_config().voice
    nothing: list[str] = []
    silent = VoicePipeline(
        cfg,
        FakeWake([1.0]),
        FakeStt(),
        lambda q: None,
        vad=FakeVad([]),
        answer_in_thread=False,
        on_nothing=nothing.append,
    )  # type: ignore[arg-type]
    feed_seconds(silent, 8.0)
    empty = VoicePipeline(
        cfg,
        FakeWake([1.0]),
        FakeStt(text=""),
        lambda q: None,
        vad=FakeVad([(0.2, 1.8)]),
        answer_in_thread=False,
        on_nothing=nothing.append,
    )  # type: ignore[arg-type]
    feed_seconds(empty, 5.0)
    empty.flush()
    assert nothing == ["timeout", "empty"]


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
