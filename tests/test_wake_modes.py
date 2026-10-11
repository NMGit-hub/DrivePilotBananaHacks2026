"""Wake modes (`voice.wake_mode`): "hey_drivepilot", "hey" and "none" (no wake word).

Speech is faked as constant-amplitude segments; the fake Whisper "hears" a sentence chosen by the loudest segment
in the audio it gets, so each test scripts what was said. VAD (endpointer and onset) is the amplitude-based
`LoudVad` from test_voice.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_config
from tests.test_voice import FakeWake, LoudVad, feed_pcm, speech_stream
from voice.pipeline import HeardQuestion, VoicePipeline
from voice.router import meant_for_us, strip_hey

HEY_ONLY, HEY_QUESTION, CHAT, QUESTION, WAKE_PHRASE = 3000, 5000, 7000, 9000, 11000
SAID = {
    HEY_ONLY: "Hey.",
    HEY_QUESTION: "Hey, what does that sign say?",
    CHAT: "We're almost there, I think.",
    QUESTION: "What does that sign say?",
    WAKE_PHRASE: "Hey DrivePilot, we're almost there.",
}


class ScriptedStt:
    """Hears the sentence of the loudest scripted segment in the audio ("" for silence)."""

    def __init__(self) -> None:
        self.pre_roll: list[bool] = []

    def transcribe(self, audio: np.ndarray, pre_roll: bool = True) -> tuple[str, float]:
        self.pre_roll.append(pre_roll)
        peak = float(np.abs(audio).max(initial=0.0)) * 32768
        level = max((a for a in SAID if a <= peak + 1), default=None)
        return (SAID[level] if level is not None else ""), 5.0


def make(mode: str, wakes: list[float] | None = None, speaking=None):  # type: ignore[no-untyped-def]
    cfg = load_config().voice.model_copy(update={"wake_mode": mode})
    heard: list[HeardQuestion] = []
    nothing: list[str] = []
    woke: list[float] = []
    pipe = VoicePipeline(
        cfg,
        FakeWake(wakes or []),  # type: ignore[arg-type]
        ScriptedStt(),  # type: ignore[arg-type]
        heard.append,
        on_wake=lambda: woke.append(0.0),
        vad=LoudVad(),  # type: ignore[arg-type]
        answer_in_thread=False,
        on_nothing=nothing.append,
        speaking=speaking,
        onset_vad=LoudVad(),  # type: ignore[arg-type]
    )
    return pipe, heard, nothing, woke


def run(pipe: VoicePipeline, seconds: float, *segments: tuple[float, float, int]) -> None:
    feed_pcm(pipe, speech_stream(seconds, *segments), live=True)
    pipe.flush()


def test_the_default_mode_ignores_speech_without_the_wake_word() -> None:
    pipe, heard, nothing, woke = make("hey_drivepilot")
    run(pipe, 5.0, (1.0, 2.5, QUESTION))
    assert heard == [] and nothing == [] and woke == [] and pipe.open_starts == 0


def test_hey_mode_hears_a_question_that_starts_with_hey() -> None:
    pipe, heard, nothing, woke = make("hey")
    run(pipe, 5.0, (1.0, 2.5, HEY_QUESTION))
    assert [(q.text, q.open) for q in heard] == [("what does that sign say?", True)]
    assert nothing == [] and woke == [] and pipe.open_starts == 1
    assert pipe.stt.pre_roll == [False]  # type: ignore[attr-defined]  # (no wake phrase to strip loosely)


def test_hey_mode_ignores_talk_without_hey_silently() -> None:
    pipe, heard, nothing, woke = make("hey")
    run(pipe, 5.0, (1.0, 2.5, QUESTION))
    assert heard == [] and nothing == [] and woke == [] and pipe.ignored == 1


def test_a_lone_hey_opens_the_listening_window_for_the_question() -> None:
    # "Hey." (1.0-1.4 s), a pause, then the question: the blip/listening light comes after the "hey", and the
    # question is taken without needing a second "hey".
    pipe, heard, nothing, woke = make("hey")
    run(pipe, 7.0, (1.0, 1.4, HEY_ONLY), (2.4, 3.6, QUESTION))
    assert [(q.text, q.open, q.resumed) for q in heard] == [("What does that sign say?", True, True)]
    assert nothing == [] and len(woke) == 1 and pipe.resumes == 1 and not pipe.listening


def test_a_lone_hey_and_then_silence_says_nothing() -> None:
    pipe, heard, nothing, woke = make("hey")
    run(pipe, 9.0, (1.0, 1.4, HEY_ONLY))
    assert heard == [] and nothing == [] and len(woke) == 1 and not pipe.listening


def test_no_wake_word_mode_answers_questions_and_ignores_remarks() -> None:
    pipe, heard, nothing, woke = make("none")
    run(pipe, 8.0, (1.0, 2.0, CHAT), (4.0, 5.2, QUESTION))
    assert [q.text for q in heard] == ["What does that sign say?"] and pipe.ignored == 1
    assert nothing == [] and woke == [] and pipe.open_starts == 2


def test_no_wake_word_mode_can_take_everything() -> None:
    pipe, heard, *_ = make("none")
    pipe.cfg = pipe.cfg.model_copy(update={"open_mic": pipe.cfg.open_mic.model_copy(update={"only_questions": False})})
    run(pipe, 5.0, (1.0, 2.0, CHAT))
    assert [q.text for q in heard] == [SAID[CHAT]]


def test_our_own_voice_does_not_start_a_question() -> None:
    # DrivePilot talks until 3.0 s (the mic hears it as speech from 1.0 s); speech within echo_tail_s after that
    # is still ignored; a question after the tail is heard.
    pipe, heard, *_ = make("none", speaking=lambda: pipe.t < 3.0)
    tail = pipe.cfg.open_mic.echo_tail_s
    run(pipe, 9.0, (1.0, 3.0 + tail / 2, QUESTION), (5.0, 6.2, QUESTION))
    assert [q.speech_end_t for q in heard] == [pytest.approx(6.2, abs=0.05)] and pipe.open_starts == 1


def test_the_wake_word_during_open_speech_makes_it_a_question_with_the_blip() -> None:
    # "hey" mode, but the full "Hey DrivePilot, ..." is said: the wake model fires while the speech is being
    # collected; the question counts even though Whisper's text doesn't pass the "hey" check as a lone "hey".
    pipe, heard, nothing, woke = make("hey", wakes=[1.6])
    run(pipe, 6.0, (1.0, 2.8, WAKE_PHRASE))
    # (the real Transcriber strips the repeated wake phrase, the router the name; the fake hears it all)
    assert [q.text for q in heard] == [SAID[WAKE_PHRASE]] and not heard[0].open
    assert len(woke) == 1 and pipe.wakes == 1


def test_the_mode_switches_while_running() -> None:
    pipe, heard, *_ = make("hey_drivepilot")
    run(pipe, 3.0, (1.0, 2.0, QUESTION))
    pipe.set_mode("none")
    run(pipe, 3.0, (1.0, 2.0, QUESTION))
    assert [q.text for q in heard] == ["What does that sign say?"] and pipe.mode == "none"
    with pytest.raises(ValueError):
        pipe.set_mode("hello")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey, what does that sign say?", "what does that sign say?"),
        ("hey can I go", "can I go"),
        ("Hay, who got here first?", "who got here first?"),
        ("Hey.", ""),
        ("Hey DrivePilot.", ""),
        ("Hey DrivePilot, can I go?", "DrivePilot, can I go?"),  # (the router strips the name)
        ("What does that sign say?", None),
        ("Heyday was a good album.", None),
        ("They said turn left", None),
    ],
)
def test_strip_hey(text: str, expected: str | None) -> None:
    assert strip_hey(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What does that sign say?", True),
        ("which lane for 520 east", True),
        ("who got here first", True),
        ("tell me when it's green", True),
        ("take me to Costco", True),
        ("find me a parking spot", True),
        ("quiet mode", True),
        ("stop", True),
        ("repeat that", True),
        ("can I merge now", True),
        ("Is that a school bus?", True),
        ("describe the road ahead", True),
        ("What?", False),  # a passenger to another: not "repeat that"
        ("Huh?", False),
        ("We're almost there, I think.", False),
        ("I love this song", False),
        ("That was a long meeting.", False),
        ("", False),
        ("Um.", False),
    ],
)
def test_meant_for_us(text: str, expected: bool) -> None:
    assert meant_for_us(text) is expected
