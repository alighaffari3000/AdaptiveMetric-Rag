"""Does the cited source actually say what the claim says?

A citation is a promise, and until now nothing checked it. A model that cites
[2] for a number that appears nowhere in source 2 produces an answer that looks
more trustworthy than an uncited one and is worse.

The check is deliberately shallow, and shallow in a specific direction: it can
say "this claim is not supported by the text it cites" with reasonable
confidence, and it cannot say "this claim is true". Two signals, both computed
offline:

- every number and date in the claim must appear in the cited source. This is
  what catches the failure that matters most in this corpus - an amount, a
  percentage or a contract date carried over from the wrong document, or
  invented outright.
- enough of the claim's content words must appear in the cited source, which
  catches a sentence attached to a passage about something else.

An LLM judge is available for the same job when a provider is configured, and
neither is on the answer's critical path: verification never rewrites or
withholds an answer, it labels it.
"""

from __future__ import annotations

import json
import logging
import re

from .claims import Claim, strip_markers
from .models import AppSettings
from .net import redact
from .text import date_terms, number_terms, tokenize

logger = logging.getLogger("adaptive_metric_rag.verify")

# Swept on the golden corpus' extractive answers, which are quotations and
# therefore fully supported by construction: 0.5 keeps every one of them and
# still rejects a sentence sharing only its function words with the source.
OVERLAP_FLOOR = .5
# Below this many content words there is not enough signal to judge overlap,
# so only the numeric check applies.
MIN_TOKENS_FOR_OVERLAP = 4


def support_score(claim: str, source: str) -> float:
    """Share of the claim's content words the source also uses."""
    claim_terms = set(tokenize(claim))
    if not claim_terms:
        return 1.0
    source_terms = set(tokenize(source))
    return len(claim_terms & source_terms) / len(claim_terms)


def unsupported_figures(claim: str, source: str) -> list[str]:
    """Numbers and dates the claim states and the cited source does not."""
    source_numbers, source_dates = set(number_terms(source)), set(date_terms(source))
    missing = [value for value in number_terms(claim) if value not in source_numbers]
    missing += [value for value in date_terms(claim)
                if value not in source_dates and value not in source_numbers]
    return list(dict.fromkeys(missing))


def check_claim(claim: Claim, sources: dict[int, str]) -> bool | None:
    """Whether the cited sources support the claim. None when nothing is cited."""
    if not claim.source_ids:
        return None
    cited = " \n".join(sources.get(number, "") for number in claim.source_ids)
    if not cited.strip():
        return False
    # The markers are the citation, not part of what is claimed: "[1]" would
    # otherwise read as the number 1 and fail against a source that never
    # mentions it.
    statement = strip_markers(claim.text)
    if not statement:
        return None
    if unsupported_figures(statement, cited):
        return False
    terms = tokenize(statement)
    if len(terms) < MIN_TOKENS_FOR_OVERLAP:
        return True
    return support_score(statement, cited) >= OVERLAP_FLOOR


def verify_lexically(claims: list[Claim], sources: dict[int, str]) -> list[Claim]:
    for claim in claims:
        claim.supported = check_claim(claim, sources)
    return claims


_JUDGE_SYSTEM = (
    "You check whether a source passage supports a statement. Answer only about support: a "
    "statement is supported when the passage states it or directly implies it, and unsupported "
    "when the passage is silent about it or contradicts it. A statement in a different language "
    "from the passage can still be supported. Reply with JSON only, in the form "
    '{"verdicts": [{"id": 1, "supported": true}, ...]}.'
)


async def verify_with_model(settings: AppSettings, claims: list[Claim],
                            sources: dict[int, str]) -> list[Claim]:
    """Ask the configured model the same question, one call for the whole answer."""
    from .providers import complete

    cited = [(index, claim) for index, claim in enumerate(claims, 1) if claim.source_ids]
    if not cited:
        return verify_lexically(claims, sources)
    blocks = []
    for index, claim in cited:
        passage = "\n".join(sources.get(number, "") for number in claim.source_ids)
        blocks.append(f"[{index}] statement: {claim.text}\n[{index}] passage: {passage[:1500]}")
    user = "\n\n".join(blocks) + "\n\nJSON:"
    raw = await complete(settings, _JUDGE_SYSTEM, user, temperature=0.0, max_tokens=600)
    verdicts = _parse_verdicts(raw)
    for index, claim in cited:
        if index in verdicts:
            claim.supported = verdicts[index]
        else:
            claim.supported = check_claim(claim, sources)
    for claim in claims:
        if not claim.source_ids:
            claim.supported = None
    return claims


def _parse_verdicts(raw: str) -> dict[int, bool]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    verdicts: dict[int, bool] = {}
    for entry in payload.get("verdicts", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        value = entry.get("supported")
        if isinstance(value, bool):
            verdicts[number] = value
    return verdicts


async def verify(settings: AppSettings, claims: list[Claim], sources: dict[int, str]) -> list[Claim]:
    """Label each claim, by the configured method, without ever failing the answer."""
    if not settings.verify_claims or not claims:
        return claims
    if settings.verify_backend == "llm" and settings.provider != "local":
        try:
            return await verify_with_model(settings, claims, sources)
        except Exception as exc:  # a check must never take down the answer
            logger.warning("claim verification fell back to the lexical check: %s", redact(exc))
    return verify_lexically(claims, sources)
