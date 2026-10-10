"""M5 review regressions: model answers never vouch for traffic; requests stop when nobody listens."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
import pytest

from answers.assistant import Assistant
from answers.scene import NO_VOUCH, PERMISSION, VOUCHING, SceneHandler, describe_world
from answers.sentences import sentences
from answers.speech import TextSpeaker
from answers.vision import (
    ClaudeVision,
    GeminiVision,
    MockVision,
    OllamaVision,
    VisionCancelled,
    VisionError,
    VisionRequest,
    VisionTimeout,
    guarded_stream,
)
from core.config import ConfigError, load_config
from state.stop_queue import EGO_KEY, Approach, QueueEntry
from state.traffic_light import LightColor, LightReading
from tests.test_vision import cfg, frame, handler, snapshot
from voice.router import route

VOUCHING_REPLIES = [
    "Yes, it looks clear both ways.",
    "Looks clear, go ahead.",
    "You're good to go.",
    "Yes, you can merge now.",
    "All clear, go ahead.",
    "Go for it.",
    "Safe to go.",
    "Okay, go ahead.",
    "You can go now.",
    "Nothing's coming, you can pull out.",
    "You're clear to turn.",
    "yes — go ahead.",
    "Go. Nothing is coming from either side.",
    "It's clear. Go.",
    "Merge now.",
]


def spoken_by_assistant(reply: str, question: str, **snap: Any) -> list[str]:
    speaker = TextSpeaker()
    scene = handler(MockVision(reply, chunk_chars=5))
    a = Assistant(load_config(env_file=None, environ={}), lambda: snapshot(**snap), speaker, model=scene)
    a.ask(question)
    return speaker.texts()


@pytest.mark.parametrize("reply", VOUCHING_REPLIES)
def test_go_questions_never_hear_the_model_vouch(reply: str) -> None:
    # Stopped, no stop-sign queue and no light (a driveway, a parking-lot exit, a missed sign): the model answers.
    said = spoken_by_assistant(reply, "Is it safe to pull out?", ego_stopped=True, ego_stopped_since=10.0)
    assert said and said[-1] == NO_VOUCH
    for sentence in said:
        assert sentence == NO_VOUCH or not (PERMISSION.search(sentence) or VOUCHING.search(sentence)), sentence


@pytest.mark.parametrize("reply", ["You're good to go.", "Go ahead.", "Yes, you can go now.", "Go for it."])
def test_permission_is_removed_whatever_the_question(reply: str) -> None:
    said = spoken_by_assistant(reply, "What does that sign say?", ego_stopped=True, ego_stopped_since=10.0)
    assert said == [NO_VOUCH]


def test_descriptions_with_clear_are_fine_when_not_about_going() -> None:
    reply = "The road ahead is clear of construction; the right lane reopens after the bridge."
    assert spoken_by_assistant(reply, "Is the road closed ahead?", ego_stopped=False) == [reply]


def test_negated_safety_words_are_not_vouching() -> None:
    assert not VOUCHING.search("The light is red, so it's not safe to go yet.")
    assert not VOUCHING.search("I can't tell; the left side isn't clear from here.")


# --------------------------------------------------------------------------------------------- context


def test_deadlock_context_never_says_we_go_first() -> None:
    ours = QueueEntry(EGO_KEY, Approach.US, 10.0, label="you")
    others = [
        QueueEntry(f"track:{i}", a, 10.1 * i, label="car")
        for i, a in enumerate((Approach.LEFT, Approach.AHEAD, Approach.RIGHT), 1)
    ]
    queue = replace(
        snapshot().queue,
        active=True,
        order=(ours, *others),
        deadlock=True,
        your_turn=False,
        tie_groups=((EGO_KEY, "track:1", "track:2", "track:3"),),
    )
    text = " ".join(describe_world(snapshot(queue=queue, ego_stopped=True, ego_stopped_since=10.0)))
    assert "nobody has a clear right of way" in text and "us," not in text and "then us" not in text
    assert "can't tell whether it is an all-way stop" in text


def test_context_mentions_cars_we_lost() -> None:
    lost = QueueEntry("track:4", Approach.LEFT, 8.0, label="white car", drop_reason="lost from view")
    queue = replace(
        snapshot().queue, active=True, order=(QueueEntry(EGO_KEY, Approach.US, 10.0),), your_turn=True, dropped=(lost,)
    )
    text = " ".join(describe_world(snapshot(queue=queue, ego_stopped=True, ego_stopped_since=10.0)))
    assert "Lost track of the white car on your left" in text


# ----------------------------------------------------------------------------------------- cancellation


def test_closing_the_answer_closes_the_provider_stream() -> None:
    model = MockVision("First sentence is here. " + "More words. " * 30, chunk_chars=6, delay_s=0.05)
    stream = handler(model).answer(route("what's ahead?"), snapshot())
    assert next(stream) == "First sentence is here."
    stream.close()
    deadline = time.perf_counter() + 1.0
    while model.closed_streams == 0 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert model.closed_streams == 1  # the worker stopped and closed the provider generator


def test_cancel_stops_waiting_immediately() -> None:
    model = MockVision("Too slow.", first_delay_s=2.0)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    start = time.perf_counter()
    with pytest.raises(VisionCancelled):
        list(guarded_stream(model, VisionRequest(system="s", text="q"), 3.0, 8.0, cancel))
    assert time.perf_counter() - start < 0.5


def test_stop_cuts_a_streamed_answer_off_at_once() -> None:
    model = MockVision("The sign says no parking. It's a tow-away zone after 4 PM on weekdays.", chunk_chars=200)
    model.stream = _two_part_stream  # type: ignore[method-assign]
    speaker = TextSpeaker()
    scene = handler(model)
    a = Assistant(load_config(env_file=None, environ={}), lambda: snapshot(), speaker, model=scene)
    finished = threading.Event()
    threading.Thread(target=lambda: (a.ask("What does that sign say?"), finished.set()), daemon=True).start()
    deadline = time.perf_counter() + 2.0
    while not speaker.texts() and time.perf_counter() < deadline:
        time.sleep(0.01)
    stop_at = time.perf_counter()
    a.ask("Stop")
    assert finished.wait(1.0) and time.perf_counter() - stop_at < 0.5  # not after the 3 s second sentence
    assert speaker.texts() == ["The sign says no parking."]


def _two_part_stream(request: VisionRequest) -> Iterator[str]:
    yield "The sign says no parking. "
    time.sleep(3.0)
    yield "It's a tow-away zone after 4 PM on weekdays."


def sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def test_real_claude_client_stream_is_closed_on_cancel_and_errors_are_wrapped() -> None:
    import httpx2

    closed: list[float] = []

    class SlowBody(httpx2.SyncByteStream):
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def __iter__(self) -> Iterator[bytes]:
            start = {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "x",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
            yield sse("message_start", start)
            yield sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            for i in range(20):
                if self.fail and i == 1:
                    raise httpx2.ReadTimeout("simulated")
                yield sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"Word {i}. "}},
                )
                time.sleep(0.1)

        def close(self) -> None:
            closed.append(time.perf_counter())

    def client(fail: bool) -> ClaudeVision:
        def respond(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowBody(fail))

        http = httpx2.Client(transport=httpx2.MockTransport(respond))
        return ClaudeVision("sk-ant-test", "claude-haiku-4-5", total_s=5.0, read_s=3.0, http_client=http)

    cancel = threading.Event()
    stream = guarded_stream(client(fail=False), VisionRequest(system="s", text="q"), 3.0, 8.0, cancel)
    assert next(stream) == "Word 0. "
    cancel.set()
    cancelled_at = time.perf_counter()
    with pytest.raises(VisionCancelled):
        next(stream)
    deadline = time.perf_counter() + 1.0
    while not closed and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert closed and closed[0] - cancelled_at < 0.5  # the HTTP body was closed right away, not at the end

    with pytest.raises(VisionError):
        client(fail=True).complete(VisionRequest(system="s", text="q"))  # a mid-stream httpx2 error is wrapped


def test_complete_json_has_deadlines() -> None:
    model = MockVision('{"lanes": 3}', first_delay_s=1.0)
    start = time.perf_counter()
    with pytest.raises(VisionTimeout):
        model.complete_json(VisionRequest(system="s", text="q"), first_token_s=0.2, total_s=0.5)
    assert time.perf_counter() - start < 0.6


def test_ollama_tolerates_a_message_without_content() -> None:
    import httpx

    def respond(request: httpx.Request) -> httpx.Response:
        lines = [{"message": None}, {"message": {"content": "Three lanes."}}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    model = OllamaVision("m", "http://localhost:11434", 5.0, 5.0, transport=httpx.MockTransport(respond))
    assert model.complete(VisionRequest(system="s", text="q")) == "Three lanes."


def test_gemini_streaming_loop() -> None:
    model = GeminiVision("gm-test", "gemini-3.8-flash", total_s=5.0)
    chunks = [type("Chunk", (), {"text": t})() for t in ("It's a ", None, "school zone.")]

    class Models:
        def generate_content_stream(self, **kwargs: Any) -> Iterator[Any]:
            assert kwargs["model"] == "gemini-3.8-flash"
            return iter(chunks)

    model._client = type("Client", (), {"models": Models()})()  # type: ignore[assignment]
    assert model.complete(VisionRequest(system="s", text="q")) == "It's a school zone."


# ----------------------------------------------------------------------------------------- the request


def test_light_questions_get_up_to_three_labeled_close_ups() -> None:
    model = MockVision("The left arrow is red; the through lights are green.")
    boxes = (
        (200.0, 100.0, 220.0, 160.0),
        (620.0, 100.0, 640.0, 160.0),
        (1000.0, 100.0, 1020.0, 160.0),
        (1100.0, 100.0, 1110.0, 130.0),
    )
    snap = snapshot(light=LightReading(20.0, LightColor.GREEN, True, 4, boxes))
    list(handler(model).answer(route("Is the left arrow green?"), snap))
    labels = [i.label for i in model.requests[0].images]
    assert labels[-3:] == [
        "Close-up of the traffic light on the left",
        "Close-up of the traffic light in the middle",
        "Close-up of the traffic light on the right",
    ]
    assert "cannot read arrows" in model.requests[0].system


def test_sign_questions_reach_further_back_at_full_resolution() -> None:
    model = MockVision("It said speed limit 30.")
    asked: list[float] = []
    shots = [frame(17.6), frame(18.8), frame(20.0)]
    h = SceneHandler(
        load_config(env_file=None, environ={}), model, lambda count, spacing: (asked.append(spacing), shots[-count:])[1]
    )
    list(h.answer(route("What did that sign say?"), snapshot()))
    assert asked == [1.2]
    widths = [
        cv2.imdecode(np.frombuffer(i.jpeg, np.uint8), cv2.IMREAD_COLOR).shape[1] for i in model.requests[0].images
    ]
    assert widths == [1280, 1280, 1280]


def test_time_override_and_navigation_context() -> None:
    model = MockVision("Two-hour parking until 6 PM.")
    c = cfg(time_override="Saturday 17:40")
    h = SceneHandler(
        c,
        model,
        lambda count, spacing: [frame(20.0)],
        extra_context=lambda: ["Speed: 0 mph.", "Next: turn left on 148th Ave in 0.3 miles."],
    )
    list(h.answer(route("Can I park here?"), snapshot()))
    text = model.requests[0].text
    assert "Local time: Saturday 17:40." in text and "Next: turn left on 148th Ave in 0.3 miles." in text


def test_deadlines_must_be_ordered() -> None:
    with pytest.raises(ConfigError, match="first_token_timeout_s"):
        load_config(overrides=["vision.first_token_timeout_s=9", "vision.total_timeout_s=8"], env_file=None, environ={})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Speed limit is 35 mph. You're doing about 40.", ["Speed limit is 35 mph.", "You're doing about 40."]),
        (
            "Take I-405 N. Then use exit 14B. It's on the right.",
            ["Take I-405 N.", "Then use exit 14B.", "It's on the right."],
        ),
        ("Not that one, no. The next exit is yours.", ["Not that one, no.", "The next exit is yours."]),
        ("Go. Nothing is coming from either side.", ["Go. Nothing is coming from either side."]),
        ("Parking ends at 6 p.m. ", ["Parking ends at 6 p.m."]),
    ],
)
def test_sentence_ends(text: str, expected: list[str]) -> None:
    assert list(sentences([text])) == expected
