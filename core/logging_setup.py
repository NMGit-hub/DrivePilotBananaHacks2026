"""Structured logging: readable console lines plus JSON-lines files, with secret redaction.

Usage:
    log = get_logger(__name__)
    log.info("frame processed", fps=29.7, gpu_mb=812)

Keyword arguments become structured fields (any name works, including `level` or `msg`): the console shows them
as `key=value`, the JSON file stores them as real JSON values (numpy scalars converted, NaN/inf → null). Every
registered secret value that appears in a log line is replaced with "***".
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_secrets_lock = threading.Lock()
_secrets: tuple[str, ...] = ()
# Keys the JSON formatter writes itself; colliding field names are prefixed so nothing is silently dropped.
_JSON_RESERVED = frozenset({"ts", "level", "logger", "msg", "exc"})


def register_secrets(values: Iterable[str]) -> None:
    """Set the secret values to redact from every log line and run record (replaces the previous set)."""
    global _secrets
    with _secrets_lock:
        # Longest first so a secret that contains another is fully redacted. Ignore trivially short values.
        _secrets = tuple(sorted({v for v in values if v and len(v) >= 8}, key=len, reverse=True))


def redact(text: str) -> str:
    """Replace every registered secret value in `text` with '***'."""
    for secret in _secrets:
        if secret in text:
            text = text.replace(secret, "***")
    return text


def to_jsonable(value: Any) -> Any:
    """Convert a value to plain JSON types: numpy → Python, NaN/inf → None, Path/other → str."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    return str(value)


def dumps_json_line(record: Mapping[str, Any]) -> str:
    """Serialize a record as one redacted JSON line (strict JSON: no NaN)."""
    return redact(json.dumps(to_jsonable(record), ensure_ascii=False, allow_nan=False))


class StructLogger:
    """A thin wrapper over `logging.Logger` whose keyword arguments become structured fields.

    The message is positional-only, so fields may use any name (`level=0.8`, `msg="..."`).
    `exc_info` and `stack_info` keep their standard meaning.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        """The underlying logger name."""
        return self._logger.name

    def isEnabledFor(self, level: int) -> bool:
        """True if a record at `level` would be emitted."""
        return self._logger.isEnabledFor(level)

    def _log(
        self, level: int, msg: str, fields: dict[str, Any], exc_info: Any = None, stack_info: bool = False
    ) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(
                level, msg, exc_info=exc_info, stack_info=stack_info, stacklevel=3, extra={"fields": fields}
            )

    def debug(self, msg: str, /, *, exc_info: Any = None, stack_info: Any = False, **fields: Any) -> None:
        """Log at DEBUG with structured fields."""
        self._log(logging.DEBUG, msg, fields, exc_info, stack_info)

    def info(self, msg: str, /, *, exc_info: Any = None, stack_info: Any = False, **fields: Any) -> None:
        """Log at INFO with structured fields."""
        self._log(logging.INFO, msg, fields, exc_info, stack_info)

    def warning(self, msg: str, /, *, exc_info: Any = None, stack_info: Any = False, **fields: Any) -> None:
        """Log at WARNING with structured fields."""
        self._log(logging.WARNING, msg, fields, exc_info, stack_info)

    def error(self, msg: str, /, *, exc_info: Any = None, stack_info: Any = False, **fields: Any) -> None:
        """Log at ERROR with structured fields."""
        self._log(logging.ERROR, msg, fields, exc_info, stack_info)

    def exception(self, msg: str, /, *, stack_info: Any = False, **fields: Any) -> None:
        """Log at ERROR with the current exception's traceback."""
        self._log(logging.ERROR, msg, fields, True, stack_info)


def get_logger(name: str) -> StructLogger:
    """Return a structured logger for `name` (usually `__name__`)."""
    return StructLogger(logging.getLogger(name))


def _fmt_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


class ConsoleFormatter(logging.Formatter):
    """`12:03:04.123 INFO  perception: msg key=value ...`"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        fields = getattr(record, "fields", None) or {}
        parts = [f"{ts} {record.levelname:<5} {record.name}: {record.getMessage()}"]
        parts += [f"{k}={_fmt_value(v)}" for k, v in fields.items()]
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return redact(line)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts (UTC ISO), level, logger, msg, plus the structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in (getattr(record, "fields", None) or {}).items():
            payload[f"f_{key}" if key in _JSON_RESERVED else key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return dumps_json_line(payload)


def setup_logging(level: str = "INFO", log_file: Path | None = None, secrets: Iterable[str] | None = None) -> None:
    """Configure the root logger: console (stderr) and optionally a JSON-lines file.

    Safe to call again (e.g. once a run folder exists); handlers are replaced. `secrets=None` keeps the
    currently registered secrets, so a second call never turns redaction off.
    """
    if secrets is not None:
        register_secrets(secrets)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Third-party libraries are chatty at INFO; keep them at WARNING unless we are debugging.
    noisy_level = logging.DEBUG if level == "DEBUG" else logging.WARNING
    for name in ("ultralytics", "httpx", "httpcore", "anthropic", "urllib3", "PIL", "matplotlib", "faster_whisper"):
        logging.getLogger(name).setLevel(noisy_level)
