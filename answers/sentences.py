"""Split streamed model text into sentences, so speech starts after the first sentence instead of the last.

A sentence ends at ".", "!" or "?" followed by whitespace and then a capital letter, digit or quote
("...35 mph. You're doing 40." splits; "until 6 p.m. today" doesn't). A few abbreviations that are normally
followed by a capitalized name never end a sentence ("Main St. Paul", "Dr. King"). Decimals ("2.5 miles") have
no space and never split. Very short sentences ("Yes.") are joined with the next one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_ABBREVIATIONS = {
    "st", "ave", "av", "dr", "rd", "blvd", "hwy", "pkwy", "ln", "ct", "pl", "sq", "mt", "mr", "mrs", "ms",
    "vs", "e.g", "i.e", "approx", "jr", "sr", "u.s", "u.s.a",
}  # fmt: skip
# Tokens that may or may not end a sentence; at the very end of the text so far, wait for the next word.
_AMBIGUOUS = {"a.m", "p.m", "mph", "min", "no", "approx"}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?:\s+(?=[A-Z0-9\"'(])|\s+$)")


def _last_word(text_before: str) -> str:
    word = re.search(r"([A-Za-z.]+)$", text_before)
    return word.group(1).rstrip(".").lower() if word is not None else ""


def _is_abbreviation(text_before: str) -> bool:
    return _last_word(text_before) in _ABBREVIATIONS


def _is_ambiguous(text_before: str) -> bool:
    word = _last_word(text_before)
    return word in _AMBIGUOUS or len(word) == 1


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
            punct_at = match.start()
            if self._buffer[punct_at] == "." and _is_abbreviation(self._buffer[start:punct_at]):
                continue
            at_end = match.end() == len(self._buffer)
            if at_end and self._buffer[punct_at] == "." and _is_ambiguous(self._buffer[start:punct_at]):
                continue  # "...until 6 p.m. " — wait for the next word to decide
            candidate = self._buffer[start : match.end()].strip()
            if len(candidate) < self.min_chars:
                continue
            out.append(candidate)
            start = match.end()
        self._buffer = self._buffer[start:]
        return out

    def flush(self) -> str | None:
        """The rest of the text (end of stream)."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None


def sentences(pieces: Iterable[str], min_chars: int = 12) -> Iterator[str]:
    """Turn a stream of text pieces into a stream of sentences (closing it closes `pieces`)."""
    splitter = SentenceSplitter(min_chars)
    try:
        for piece in pieces:
            yield from splitter.feed(piece)
    finally:
        close = getattr(pieces, "close", None)
        if close is not None:
            close()
    rest = splitter.flush()
    if rest:
        yield rest
