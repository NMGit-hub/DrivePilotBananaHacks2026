"""Split streamed model text into sentences, so speech starts after the first sentence instead of the last.

Splits after ".", "!" or "?" followed by whitespace, except after common abbreviations in driving answers
("St.", "Ave.", "a.m.", "approx.") and single initials; decimals ("2.5 miles") have no space and never split.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_ABBREVIATIONS = {
    "st", "ave", "av", "dr", "rd", "blvd", "hwy", "pkwy", "ln", "ct", "pl", "sq", "mt", "ft", "mr", "mrs", "ms",
    "no", "vs", "approx", "e.g", "i.e", "a.m", "p.m", "u.s", "etc", "jr", "sr", "n", "s", "e", "w", "mph", "min",
}  # fmt: skip
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")


def _is_abbreviation(text_before: str) -> bool:
    word = re.search(r"([A-Za-z.]+)$", text_before)
    if word is None:
        return False
    token = word.group(1).rstrip(".").lower()
    return token in _ABBREVIATIONS or len(token) == 1


class SentenceSplitter:
    """Feed text pieces; get complete sentences back. `flush()` returns the remainder at the end."""

    def __init__(self, min_chars: int = 12) -> None:
        self.min_chars = min_chars  # don't emit tiny fragments like "Yes." on their own (joined with the next)
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        """Add streamed text; return the sentences it completed."""
        self._buffer += piece
        out: list[str] = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            end = match.end()
            candidate = self._buffer[start:end].strip()
            punct_at = match.start()
            if self._buffer[punct_at] == "." and _is_abbreviation(self._buffer[start:punct_at]):
                continue
            if len(candidate) < self.min_chars:
                continue
            out.append(candidate)
            start = end
        self._buffer = self._buffer[start:]
        return out

    def flush(self) -> str | None:
        """The rest of the text (end of stream)."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None


def sentences(pieces: Iterable[str], min_chars: int = 12) -> Iterator[str]:
    """Turn a stream of text pieces into a stream of sentences."""
    splitter = SentenceSplitter(min_chars)
    for piece in pieces:
        yield from splitter.feed(piece)
    rest = splitter.flush()
    if rest:
        yield rest
