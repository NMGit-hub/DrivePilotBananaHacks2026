"""M5: the vision-model interface (mocked providers), scene prompts, streaming sentences, timeouts, safety net."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from answers.scene import (
    FALLBACK_EMPTY,
    FALLBACK_ERROR,
    FALLBACK_TIMEOUT,
    GREEN_CHECK,
    LOST_REST,
    NO_VOUCH,
    SceneHandler,
    describe_world,
    spoken_time,
)
from answers.sentences import SentenceSplitter, sentences
from answers.vision import (
    ClaudeVision,
    GeminiVision,
    ImagePart,
    MockVision,
    OllamaVision,
    VisionError,
    VisionRequest,
    create_vision_model,
)
from core.config import AppConfig, load_config
from core.types import Frame
from state.stop_queue import EGO_KEY, Approach, QueueEntry
from state.traffic_light import LightColor, LightReading
from state.world import VehicleView, WorldSnapshot, empty_snapshot
from voice.router import route

SATURDAY_540PM = datetime(2026, 10, 10, 17, 40)
JPEG = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()


def cfg(**vision: str) -> AppConfig:
    return load_config(overrides=[f"vision.{k}={v}" for k, v in vision.items()], env_file=None, environ={})


def frame(t: float, w: int = 1280, h: int = 720) -> Frame:
    image = np.full((h, w, 3), 90, np.uint8)
    cv2.rectangle(image, (600, 100), (620, 160), (30, 30, 30), -1)
    return Frame(image=image, t=t, index=round(t * 30), captured_at=0.0)


def snapshot(t: float = 20.0, **kw: Any) -> WorldSnapshot:
    base = replace(empty_snapshot(t), frame=frame(t), ego_motion_confident=True)
    return replace(base, **kw)


def handler(model: MockVision, c: AppConfig | None = None, frames: list[Frame] | None = None) -> SceneHandler:
    shots = frames if frames is not None else [frame(18.6), frame(19.3), frame(20.0)]
    return SceneHandler(c or cfg(), model, lambda count, spacing: shots[-count:], now=lambda: SATURDAY_540PM)


def ask(h: SceneHandler, question: str, snap: WorldSnapshot | None = None) -> list[str]:
    return list(h.answer(route(question), snap or snapshot()))


# ------------------------------------------------------------------------------------ provider factory


def test_providers_are_selected_by_config() -> None:
    keys = {"ANTHROPIC_API_KEY": "sk-ant-test", "GEMINI_API_KEY": "gm-test"}
    for name, kind in (
        ("claude", ClaudeVision),
        ("gemini", GeminiVision),
        ("ollama", OllamaVision),
        ("mock", MockVision),
    ):
        c = load_config(overrides=[f"vision_model={name}"], env_file=None, environ=keys)
        model = create_vision_model(c)
        assert isinstance(model, kind) and model.name == name
        model.close()


def test_default_provider_is_claude_haiku() -> None:
    c = load_config(env_file=None, environ={})
    assert c.vision_model == "claude" and c.vision.claude.model == "claude-haiku-4-5"


@pytest.mark.parametrize("name", ["claude", "gemini"])
def test_missing_key_is_a_clear_error(name: str) -> None:
    c = load_config(overrides=[f"vision_model={name}"], env_file=None, environ={})
    with pytest.raises(VisionError, match="_API_KEY is not set"):
        create_vision_model(c)


# ----------------------------------------------------------------------------------- the interface


def test_mock_streams_in_pieces_and_records_requests() -> None:
    model = MockVision("The sign says no parking.", chunk_chars=5)
    request = VisionRequest(system="s", text="q", images=(ImagePart(JPEG, "Now"),))
    pieces = list(model.stream(request))
    assert len(pieces) > 1 and "".join(pieces) == "The sign says no parking."
    assert model.requests == [request]


def test_complete_json() -> None:
    model = MockVision('Sure: {"lanes": 4, "my_lane": 2}')
    assert model.complete_json(VisionRequest(system="s", text="count lanes")) == {"lanes": 4, "my_lane": 2}
    assert "JSON object only" in model.requests[0].system
    with pytest.raises(VisionError):
        MockVision("four lanes").complete_json(VisionRequest(system="s", text="q"))
    with pytest.raises(VisionError):
        MockVision("{lanes: four}").complete_json(VisionRequest(system="s", text="q"))


# -------------------------------------------------------------------------------------------- Claude


class FakeStream:
    def __init__(self, pieces: list[str], stop_reason: str = "end_turn") -> None:
        self.text_stream = iter(pieces)
        self.stop_reason = stop_reason

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return type("Message", (), {"stop_reason": self.stop_reason})()


class FakeMessages:
    def __init__(self, stream: FakeStream | Exception) -> None:
        self._stream = stream
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if isinstance(self._stream, Exception):
            raise self._stream
        return self._stream


def claude_with(stream: FakeStream | Exception) -> tuple[ClaudeVision, FakeMessages]:
    model = ClaudeVision("sk-ant-test", "claude-haiku-4-5", total_s=5.0, read_s=3.0)
    messages = FakeMessages(stream)
    model._client = type("Client", (), {"messages": messages, "close": lambda self: None})()  # type: ignore[assignment]
    return model, messages


def test_claude_request_has_labeled_images_then_the_question() -> None:
    model, messages = claude_with(FakeStream(["It says ", "no parking."]))
    request = VisionRequest(
        system="SYS",
        text="Context... question",
        images=(ImagePart(JPEG, "1.0 s ago"), ImagePart(JPEG, "Now")),
        max_tokens=123,
    )
    assert "".join(model.stream(request)) == "It says no parking."
    call = messages.calls[0]
    assert call["model"] == "claude-haiku-4-5" and call["system"] == "SYS" and call["max_tokens"] == 123
    content = call["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == "1.0 s ago:" and content[2]["text"] == "Now:"
    assert content[1]["source"]["media_type"] == "image/jpeg" and content[-1]["text"] == "Context... question"


def test_claude_errors_and_refusals_become_vision_errors() -> None:
    import anthropic
    import httpx2

    model, _ = claude_with(anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")))
    with pytest.raises(VisionError, match="APIConnectionError"):
        list(model.stream(VisionRequest(system="s", text="q")))
    model, _ = claude_with(FakeStream(["I can't help with that."], stop_reason="refusal"))
    with pytest.raises(VisionError, match="refused"):
        list(model.stream(VisionRequest(system="s", text="q")))


# -------------------------------------------------------------------------------------- Gemini / Ollama


def test_gemini_request() -> None:
    model = GeminiVision("gm-test", "gemini-3.8-flash", total_s=5.0, thinking_level="low")
    contents, config = model.build(
        VisionRequest(system="SYS", text="Q", images=(ImagePart(JPEG, "Now"),), max_tokens=77)
    )
    assert contents[0] == "Now:" and contents[1].inline_data.mime_type == "image/jpeg" and contents[-1] == "Q"
    assert config.system_instruction == "SYS" and config.max_output_tokens == 77
    assert str(config.thinking_config.thinking_level).lower().endswith("low")


def ollama_with(handler_fn: Any) -> OllamaVision:
    return OllamaVision("qwen2.5vl:7b", "http://localhost:11434", 5.0, 5.0, transport=httpx.MockTransport(handler_fn))


def test_ollama_streams_ndjson() -> None:
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        lines = [{"message": {"content": "It's a "}}, {"message": {"content": "school zone."}}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    model = ollama_with(respond)
    text = "".join(model.stream(VisionRequest(system="SYS", text="Q", images=(ImagePart(JPEG, "Now"),))))
    assert text == "It's a school zone."
    body = seen[0]
    assert body["model"] == "qwen2.5vl:7b" and body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert len(body["messages"][1]["images"]) == 1 and body["messages"][1]["content"].endswith("Q")


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, content=b"oops"), httpx.Response(200, content=b'{"error": "model not found"}')],
)
def test_ollama_errors(response: httpx.Response) -> None:
    with pytest.raises(VisionError):
        list(ollama_with(lambda request: response).stream(VisionRequest(system="s", text="q")))


# ------------------------------------------------------------------------------------------ the prompt


def test_prompt_has_frames_time_motion_and_tracker_state() -> None:
    model = MockVision("Two-hour parking until 6 PM.")
    queue = replace(
        empty_snapshot().queue,
        active=True,
        order=(
            QueueEntry("track:3", Approach.LEFT, 8.0, label="white car"),
            QueueEntry(EGO_KEY, Approach.US, 10.0, label="you"),
        ),
    )
    lead = VehicleView(5, "car", "silver", None, True, True, 9.0, (500.0, 300.0, 780.0, 600.0), False)
    snap = snapshot(
        ego_stopped=True,
        ego_stopped_since=8.0,
        queue=queue,
        lead_vehicle=lead,
        people_ahead=1,
        light=LightReading(20.0, LightColor.RED, True, 1, ((600.0, 100.0, 620.0, 160.0),)),
    )
    ask(handler(model), "Hey DrivePilot, can I park here?", snap)
    request = model.requests[0]
    assert [i.label for i in request.images] == ["1.4 s ago", "0.7 s ago", "Now"]  # no light close-up here
    widths = [cv2.imdecode(np.frombuffer(i.jpeg, np.uint8), cv2.IMREAD_COLOR).shape[1] for i in request.images]
    assert widths == [640, 640, 1280]
    text = request.text
    assert "Local time: Saturday 5:40 PM." in text
    assert "We are stopped (for 12 s)." in text
    assert "silver car directly ahead of us (stopped)" in text
    assert "Right of way: not our turn yet; 1 car(s) go before us: the white car on your left." in text
    assert "Traffic light facing us looks red." in text and "1 pedestrian ahead." in text
    assert text.endswith("Driver's question: can I park here?")
    assert "cannot judge moving traffic" in request.system and "check both ways" in request.system


def test_light_questions_add_a_close_up_of_the_light() -> None:
    model = MockVision("Green arrow for the left lane.")
    snap = snapshot(light=LightReading(20.0, LightColor.GREEN, True, 1, ((600.0, 100.0, 620.0, 160.0),)))
    ask(handler(model), "Is the left arrow green?", snap)
    labels = [i.label for i in model.requests[0].images]
    assert labels[-1] == "Close-up of the traffic light in the middle"
    crop = cv2.imdecode(np.frombuffer(model.requests[0].images[-1].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[0] == 256  # upscaled so the lamps are visible


def test_without_frames_the_model_is_told() -> None:
    model = MockVision("I can't see the road right now.")
    h = handler(model, frames=[])
    ask(h, "What does that sign say?", replace(snapshot(), frame=None))
    assert model.requests[0].images == () and "No camera image is available" in model.requests[0].text


def test_describe_world_when_moving_and_unsure() -> None:
    assert describe_world(snapshot(ego_stopped=False))[0] == "We are moving."
    assert describe_world(snapshot(ego_motion_confident=False))[0] == "Unsure whether we are moving."


def test_spoken_time() -> None:
    assert spoken_time(datetime(2026, 10, 12, 0, 5)) == "Monday 12:05 AM"
    assert spoken_time(datetime(2026, 10, 12, 12, 30)) == "Monday 12:30 PM"


# --------------------------------------------------------------------------------- streaming + failures


def test_answers_stream_sentence_by_sentence() -> None:
    model = MockVision(
        "That sign says no parking from 7 to 9 AM. It's 5:40 PM, so you can park.", chunk_chars=4, delay_s=0.01
    )
    h = handler(model)
    start = time.perf_counter()
    stream = h.answer(route("can I park here?"), snapshot())
    first = next(stream)
    first_at = time.perf_counter() - start
    rest = list(stream)
    total = time.perf_counter() - start
    assert first == "That sign says no parking from 7 to 9 AM."
    assert rest == ["It's 5:40 PM, so you can park."]
    assert first_at < total * 0.75  # speech can start before the model has finished


def test_no_first_token_in_time_gives_a_spoken_fallback() -> None:
    model = MockVision("Too late.", first_delay_s=0.5)
    assert ask(handler(model, cfg(first_token_timeout_s="0.15")), "what does that sign say?") == [FALLBACK_TIMEOUT]


def test_a_slow_answer_is_cut_off_after_what_was_said() -> None:
    model = MockVision("First sentence is here. Second one comes far too slowly.", chunk_chars=6, delay_s=0.1)
    said = ask(handler(model, cfg(first_token_timeout_s="0.4", total_timeout_s="0.5")), "what's ahead?")
    assert said == ["First sentence is here.", LOST_REST]  # says the rest was lost (a caveat may have been in it)


@pytest.mark.parametrize("error", [VisionError("claude: APIConnectionError"), RuntimeError("socket closed")])
def test_provider_errors_give_a_spoken_fallback(error: Exception) -> None:
    class Broken(MockVision):
        def stream(self, request: VisionRequest) -> Any:
            raise error
            yield ""  # pragma: no cover

    assert ask(handler(Broken()), "what does that sign say?") == [FALLBACK_ERROR]


def test_empty_answer() -> None:
    assert ask(handler(MockVision("")), "what's that?") == [FALLBACK_EMPTY]


@pytest.mark.parametrize("bare", ["Go.", "Yes, go ahead.", "You're clear!", "Merge now.", "go now"])
def test_a_bare_go_from_the_model_is_rewritten(bare: str) -> None:
    assert ask(handler(MockVision(bare)), "is it clear?") == [NO_VOUCH]


def test_normal_answers_are_not_rewritten() -> None:
    text = "The light is green, but a pedestrian is still crossing on your right."
    assert ask(handler(MockVision(text)), "is the light green?") == [text]  # (driving: no reminder)
    stopped = snapshot(ego_stopped=True)
    assert ask(handler(MockVision(text)), "is the light green?", stopped) == [text, GREEN_CHECK]  # (it follows)
    text = "The bus ahead is stopped with its hazards on."
    assert ask(handler(MockVision(text)), "why is the bus stopped?") == [text]


# ------------------------------------------------------------------------------------- sentence split


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (
            ["Turn right onto Main St. in 500 feet. Then keep left."],
            ["Turn right onto Main St. in 500 feet.", "Then keep left."],
        ),
        (["Your exit is in 2.5 miles. ", "It's exit 14."], ["Your exit is in 2.5 miles.", "It's exit 14."]),
        (["Yes. The light is green for your lane."], ["Yes. The light is green for your lane."]),  # no tiny fragment
        (
            ["Parking is allowed until 6 p.m. today. ", "After that it's tow-away."],
            ["Parking is allowed until 6 p.m. today.", "After that it's tow-away."],
        ),
        (["No sentence end"], ["No sentence end"]),
    ],
)
def test_sentence_splitting(pieces: list[str], expected: list[str]) -> None:
    assert list(sentences(pieces)) == expected


def test_splitter_emits_as_soon_as_a_sentence_is_complete() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("The road is closed ahead") == []
    assert splitter.feed(". Use the left ") == ["The road is closed ahead."]
    assert splitter.flush() == "Use the left"
