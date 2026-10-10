"""The vision-language model behind one interface; `vision_model` in config.yaml picks the provider.

`VisionModel.stream(request)` yields text as it is generated (so speech can start on the first sentence);
`complete_json` asks for a JSON object (e.g. lane counts for navigation). Providers:
- `claude` (default): Anthropic Claude Haiku via the official SDK, streaming;
- `gemini`: Google Gemini Flash via google-genai, low thinking, streaming;
- `ollama`: a local model through Ollama's HTTP chat API, streaming;
- `mock`: scripted replies for tests and offline runs (records every request).
Failures (missing key, network, timeouts, refusals) raise `VisionError`; callers turn that into a spoken
fallback. Images are JPEG bytes; prompts are built in `answers/scene.py`.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from core.config import AppConfig
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from anthropic.types import ImageBlockParam, TextBlockParam

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ImagePart:
    """One image in a request, with a short label the prompt refers to ("Now", "1 s ago", ...)."""

    jpeg: bytes
    label: str


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """Everything a provider needs for one question."""

    system: str
    text: str  # context + question, after the images
    images: tuple[ImagePart, ...] = ()
    max_tokens: int = 300


class VisionError(RuntimeError):
    """The model could not answer (no key, network, timeout, refusal, bad output)."""


class VisionTimeout(VisionError):
    """The model did not answer in time."""


class VisionModel(ABC):
    """A vision-language model provider."""

    name: str = "base"

    @abstractmethod
    def stream(self, request: VisionRequest) -> Iterator[str]:
        """Yield the answer's text in pieces as it is generated. Raises VisionError."""

    def complete(self, request: VisionRequest) -> str:
        """The whole answer."""
        return "".join(self.stream(request))

    def complete_json(self, request: VisionRequest) -> dict[str, Any]:
        """Ask for a JSON object and parse it (the request text must describe the object). Raises VisionError."""
        text = self.complete(
            VisionRequest(
                system=request.system + "\nReply with a single JSON object only, no prose and no code fences.",
                text=request.text,
                images=request.images,
                max_tokens=request.max_tokens,
            )
        )
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise VisionError(f"{self.name}: no JSON object in the reply")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VisionError(f"{self.name}: invalid JSON in the reply") from exc
        if not isinstance(value, dict):
            raise VisionError(f"{self.name}: JSON reply is not an object")
        return value

    def close(self) -> None:  # noqa: B027 - optional hook: providers without connections keep the default
        """Release network resources."""


# ------------------------------------------------------------------------------------------- Claude


class ClaudeVision(VisionModel):
    """Anthropic Claude (Haiku by default), streamed with the official SDK."""

    name = "claude"

    def __init__(self, api_key: str, model: str, timeout_s: float, max_retries: int = 1) -> None:
        import anthropic  # deferred: only the selected provider's SDK is imported

        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=anthropic.Timeout(timeout_s, connect=3.0),
            max_retries=max_retries,
        )

    @staticmethod
    def content(request: VisionRequest) -> list[TextBlockParam | ImageBlockParam]:
        """The user message: each image after its label, then the context and question."""
        content: list[TextBlockParam | ImageBlockParam] = []
        for image in request.images:
            content.append({"type": "text", "text": f"{image.label}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(image.jpeg).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": request.text})
        return content

    def stream(self, request: VisionRequest) -> Iterator[str]:
        """Stream text deltas from the Messages API."""
        content = self.content(request)
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": content}],
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except self._anthropic.APIError as exc:
            raise VisionError(f"claude: {type(exc).__name__}: {exc}") from exc
        if final.stop_reason == "refusal":
            raise VisionError("claude: refused")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


# ------------------------------------------------------------------------------------------- Gemini


class GeminiVision(VisionModel):
    """Google Gemini Flash via google-genai, low thinking, streamed."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float, thinking_level: str = "low") -> None:
        from google import genai  # deferred import
        from google.genai import types

        self._types = types
        self.model = model
        self.thinking_level = thinking_level
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def build(self, request: VisionRequest) -> tuple[list[Any], Any]:
        """Contents and config for generate_content_stream (separate for testing)."""
        types = self._types
        contents: list[Any] = []
        for image in request.images:
            contents.append(f"{image.label}:")
            contents.append(types.Part.from_bytes(data=image.jpeg, mime_type="image/jpeg"))
        contents.append(request.text)
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            max_output_tokens=request.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )
        return contents, config

    def stream(self, request: VisionRequest) -> Iterator[str]:
        """Stream text chunks."""
        contents, config = self.build(request)
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # the SDK raises several unrelated error types
            raise VisionError(f"gemini: {type(exc).__name__}: {exc}") from exc


# ------------------------------------------------------------------------------------------- Ollama


class OllamaVision(VisionModel):
    """A local vision model served by Ollama (HTTP chat API, NDJSON streaming)."""

    name = "ollama"

    def __init__(self, model: str, host: str, timeout_s: float, transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self._client = httpx.Client(base_url=host, timeout=httpx.Timeout(timeout_s, connect=2.0), transport=transport)

    def stream(self, request: VisionRequest) -> Iterator[str]:
        """Stream message content from /api/chat."""
        labels = "\n".join(f"Image {i + 1}: {image.label}" for i, image in enumerate(request.images))
        body = {
            "model": self.model,
            "stream": True,
            "options": {"num_predict": request.max_tokens},
            "messages": [
                {"role": "system", "content": request.system},
                {
                    "role": "user",
                    "content": f"{labels}\n\n{request.text}" if labels else request.text,
                    "images": [base64.standard_b64encode(image.jpeg).decode("ascii") for image in request.images],
                },
            ],
        }
        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code != 200:
                    raise VisionError(f"ollama: HTTP {response.status_code}")
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise VisionError(f"ollama: {data['error']}")
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        return
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise VisionError(f"ollama: {type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


# --------------------------------------------------------------------------------------------- Mock


class MockVision(VisionModel):
    """Scripted provider for tests and offline runs. Records every request it receives."""

    name = "mock"

    def __init__(
        self,
        reply: str | Callable[[VisionRequest], str] = "I can't see anything unusual ahead.",
        chunk_chars: int = 7,
        delay_s: float = 0.0,
        first_delay_s: float = 0.0,
        error: VisionError | None = None,
    ) -> None:
        self.reply = reply
        self.chunk_chars = chunk_chars
        self.delay_s = delay_s
        self.first_delay_s = first_delay_s
        self.error = error
        self.requests: list[VisionRequest] = []
        self._lock = threading.Lock()

    def stream(self, request: VisionRequest) -> Iterator[str]:
        """Yield the scripted reply in small chunks, with optional delays."""
        with self._lock:
            self.requests.append(request)
        if self.first_delay_s:
            time.sleep(self.first_delay_s)
        if self.error is not None:
            raise self.error
        text = self.reply(request) if callable(self.reply) else self.reply
        for i in range(0, len(text), self.chunk_chars):
            if self.delay_s and i:
                time.sleep(self.delay_s)
            yield text[i : i + self.chunk_chars]


# ------------------------------------------------------------------------------------------ factory


def create_vision_model(cfg: AppConfig, name: str | None = None) -> VisionModel:
    """The provider selected by `vision_model` (or `name`). Raises VisionError if its key is missing."""
    name = name or cfg.vision_model
    secrets = cfg.secrets
    if name == "claude":
        if secrets.anthropic_api_key is None:
            raise VisionError("claude: ANTHROPIC_API_KEY is not set (.env)")
        c = cfg.vision.claude
        return ClaudeVision(secrets.anthropic_api_key.get_secret_value(), c.model, c.timeout_s)
    if name == "gemini":
        if secrets.gemini_api_key is None:
            raise VisionError("gemini: GEMINI_API_KEY is not set (.env)")
        g = cfg.vision.gemini
        return GeminiVision(secrets.gemini_api_key.get_secret_value(), g.model, g.timeout_s, g.thinking_level)
    if name == "ollama":
        o = cfg.vision.ollama
        return OllamaVision(o.model, o.host, o.timeout_s)
    if name == "mock":
        return MockVision()
    raise VisionError(f"unknown vision model {name!r}")


def max_tokens_for(cfg: AppConfig, name: str | None = None) -> int:
    """The configured answer length for a provider."""
    name = name or cfg.vision_model
    by_name: dict[str, int] = {"claude": cfg.vision.claude.max_tokens, "gemini": cfg.vision.gemini.max_tokens}
    return by_name.get(name, 300)


def describe_providers() -> Sequence[str]:
    """Provider names accepted by `vision_model`."""
    return ("claude", "gemini", "ollama", "mock")
