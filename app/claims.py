"""What the answer actually claims, and which source each claim rests on.

An answer is not one thing that is either grounded or not. It is several
sentences, each resting on a different source or on none, and the difference
matters to a reader deciding whether to trust it. Until now the only structure
was the `[1]` markers the model happened to write, read back with a regex.

Two ways to get the structure, in order of preference:

- The model returns it: `{"answer": ..., "claims": [{"text": ..., "source_ids": [1]}]}`.
  Asked for when the provider can hold to a shape, and never trusted blindly.
- The markers are read back from prose. Always available, and the fallback
  whenever the JSON is missing, malformed, or disagrees with its own answer.

The verification of these claims lives in `app.verify`; this module only says
what was claimed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# The model is told to write this exact token when the sources cannot answer.
# Detecting abstention by matching prose meant maintaining a list of the phrases
# five models happen to use, in two languages, and being silently wrong for the
# sixth.
ABSTAIN_MARKER = "[[NO_ANSWER]]"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Claim:
    """One statement in the answer, and the citations it points at."""

    text: str
    source_ids: list[int] = field(default_factory=list)
    supported: bool | None = None  # filled in by app.verify; None means unchecked

    def as_dict(self) -> dict:
        return {"text": self.text, "source_ids": list(self.source_ids), "supported": self.supported}


def strip_markers(text: str) -> str:
    """The claim without its citation markers, which are not part of it."""
    return re.sub(r"\s+", " ", _MARKER.sub(" ", text)).strip()


def claims_from_markers(answer: str) -> list[Claim]:
    """Read `[1]`-style citations back out of prose, one claim per sentence.

    A marker written after the full stop lands in its own fragment when the
    text is split into sentences. That fragment is not a claim - it is the
    citation of the sentence before it, and treating it as one produced a claim
    whose entire text was "[1]", cited and unverifiable.
    """
    claims: list[Claim] = []
    for sentence in _SENTENCE_SPLIT.split(answer):
        text = sentence.strip()
        if not text:
            continue
        ids = list(dict.fromkeys(int(found) for found in _MARKER.findall(text)))
        if not strip_markers(text) and claims:
            claims[-1].source_ids = list(dict.fromkeys(claims[-1].source_ids + ids))
            claims[-1].text = f"{claims[-1].text} {text}".strip()
            continue
        claims.append(Claim(text=text, source_ids=ids))
    return claims


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_structured(raw: str, source_count: int) -> tuple[str, list[Claim]] | None:
    """Read `{answer, claims}` from a model that was asked for it.

    Returns None whenever the payload cannot be trusted as a whole - no answer
    text, no usable claims, or claims citing sources that were never supplied -
    so the caller can fall back to reading the prose instead of half-believing
    a malformed structure.
    """
    payload = _extract_json(raw)
    if payload is None:
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    entries = payload.get("claims")
    claims: list[Claim] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            raw_ids = entry.get("source_ids")
            ids: list[int] = []
            if isinstance(raw_ids, list):
                for value in raw_ids:
                    try:
                        number = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= number <= source_count:
                        ids.append(number)
            claims.append(Claim(text=text.strip(), source_ids=list(dict.fromkeys(ids))))
    if not claims:
        claims = claims_from_markers(answer)
    return answer.strip(), claims


def with_markers(answer: str, claims: list[Claim]) -> str:
    """Put `[1]` markers back into an answer whose citations arrived as JSON.

    The reader, the highlighter and every saved conversation expect inline
    markers, so a structured answer is rendered into the same shape rather than
    becoming a second format everything downstream has to understand.
    """
    if _MARKER.search(answer):
        return answer
    rendered = answer
    for claim in claims:
        if not claim.source_ids or claim.text not in rendered:
            continue
        markers = "".join(f"[{number}]" for number in claim.source_ids)
        cited = claim.text.rstrip()
        punctuation = ""
        if cited and cited[-1] in ".!?؟؛":
            punctuation, cited = cited[-1], cited[:-1].rstrip()
        rendered = rendered.replace(claim.text, f"{cited} {markers}{punctuation}", 1)
    return rendered
