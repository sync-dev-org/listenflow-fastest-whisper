"""Immutable, opt-in rejection rules shared by all candidate generators."""

from __future__ import annotations

import re
import unicodedata
from codecs import getincrementaldecoder
from dataclasses import dataclass

from tokenizers.decoders import ByteLevel


def _byte_values() -> dict[str, int]:
    # ByteLevel keeps printable bytes and maps the others consecutively above U+00FF.
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    values = {chr(value): value for value in printable}
    remaining = [value for value in range(256) if value not in printable]
    values.update((chr(256 + index), value) for index, value in enumerate(remaining))
    return values


_BYTE_VALUES = _byte_values()


def decode_candidate_text(tokenizer, tokens: list[int], *, final: bool) -> str:
    """Decode confirmed characters, keeping incomplete trailing UTF-8 bytes pending."""
    if not isinstance(getattr(tokenizer, "decoder", None), ByteLevel):
        return tokenizer.decode(tokens, skip_special_tokens=False)
    data = bytes(_BYTE_VALUES[char] for token in tokens for char in tokenizer.id_to_token(token))
    return getincrementaldecoder("utf-8")("replace").decode(data, final=final)


def _normalized(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        "".join(char for char in text.lower() if not unicodedata.category(char).startswith("P")),
    )


def _word(char: str) -> bool:
    return char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")


@dataclass(frozen=True)
class RejectionSettings:
    patterns: tuple[str, ...] = ()
    reject_repetition: bool = False
    on_exhausted: str = "pass"
    phrase_hit_no_speech: bool = True
    phrases: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.patterns or self.reject_repetition)

    def matched_phrase(self, text: str, *, final: bool) -> str | None:
        if self.patterns:
            normalized = _normalized(text)
            for pattern, phrase in zip(self.patterns, self.phrases, strict=True):
                start = normalized.find(pattern)
                while start >= 0:
                    end = start + len(pattern)
                    left = start == 0 or not _word(normalized[start - 1])
                    right = end == len(normalized) or not _word(normalized[end])
                    if left and right and (final or end < len(normalized)):
                        return phrase
                    start = normalized.find(pattern, start + 1)
        return None

    def reason(self, tokens: list[int], text: str, *, final: bool) -> str | None:
        if self.matched_phrase(text, final=final) is not None:
            return "phrase"
        if self.reject_repetition:
            # Three consecutive copies of a 1–15 token period in the generated tail.
            for period in range(1, min(15, len(tokens) // 3) + 1):
                if tokens[-period * 3 :] == tokens[-period:] * 3:
                    return "repetition"
        return None


NO_REJECTION = RejectionSettings()


def rejection_settings(
    *,
    reject_phrases: list[str] = [],  # noqa: B006 -- copied into immutable patterns, never mutated
    reject_repetition: bool = False,
    on_rejected_exhausted: str = "pass",
    phrase_hit_no_speech: bool = True,
) -> RejectionSettings:
    if not isinstance(reject_phrases, list) or any(
        not isinstance(phrase, str) for phrase in reject_phrases
    ):
        raise TypeError("reject_phrases must be a list of strings")
    patterns = tuple(_normalized(phrase).strip() for phrase in reject_phrases)
    if any(not pattern for pattern in patterns):
        raise ValueError("reject_phrases must not contain an empty normalized phrase")
    if type(reject_repetition) is not bool:
        raise TypeError("reject_repetition must be a boolean")
    if type(phrase_hit_no_speech) is not bool:
        raise TypeError("phrase_hit_no_speech must be a boolean")
    if not isinstance(on_rejected_exhausted, str) or on_rejected_exhausted not in ("pass", "drop"):
        raise ValueError("on_rejected_exhausted must be 'pass' or 'drop'")
    return RejectionSettings(
        patterns,
        reject_repetition,
        on_rejected_exhausted,
        phrase_hit_no_speech,
        tuple(reject_phrases),
    )
