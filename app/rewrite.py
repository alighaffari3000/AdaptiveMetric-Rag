"""Turn what a person typed into what retrieval should look for.

«و مبلغش چقدر بود؟» is a complete question to anyone who read the previous turn
and meaningless to a retriever: it names no contract, no document and no amount,
so it matches whichever chunk happens to contain the word «مبلغ». The question
has to be made standalone before it is embedded.

Two ways to do that, and the cheap one is not a fallback:

- The rule-based plan carries the topic forward from the conversation. It is
  offline, costs nothing, and is what the `local` provider and the evaluation
  harness use.
- The LLM plan asks the configured model for a standalone question plus two or
  three alternative phrasings, whose results are fused by rank in `search`.
  Anything the model gets wrong - malformed JSON, an empty rewrite, a refusal -
  falls back to the rule-based plan rather than to the raw question.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from .models import AppSettings
from .net import redact
from .providers import complete
from .text import tokenize

logger = logging.getLogger("adaptive_metric_rag.rewrite")

# A question this short cannot stand on its own once its stop words are gone:
# «چرا؟» leaves nothing at all, «و مبلغش چقدر بود؟» leaves three tokens, while
# every self-contained question in the golden set leaves more.
STANDALONE_TOKEN_FLOOR = 4
CONTEXT_TERMS = 8
MAX_VARIANTS = 2
CACHE_LIMIT = 256


@dataclass(frozen=True)
class QueryPlan:
    """What to retrieve with, and where it came from."""

    question: str
    original: str
    variants: list[str] = field(default_factory=list)
    source: str = "verbatim"

    @property
    def rewritten(self) -> bool:
        return self.question.strip() != self.original.strip()


def needs_context(message: str) -> bool:
    """Whether the message leans on the conversation to mean anything."""
    return len(tokenize(message)) <= STANDALONE_TOKEN_FLOOR


def context_terms(history: list[dict[str, str]], message: str, turns: int) -> list[str]:
    """Topic words from the recent turns that the message does not already carry.

    Questions come first and newest first: the previous question names the topic
    more reliably than the answer to it, which is padded with prose.
    """
    recent = [turn for turn in history if turn.get("content")][-turns:] if turns > 0 else []
    ordered = ([turn for turn in reversed(recent) if turn.get("role") == "user"]
               + [turn for turn in reversed(recent) if turn.get("role") != "user"])
    seen = set(tokenize(message))
    terms: list[str] = []
    for turn in ordered:
        for token in tokenize(turn["content"]):
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return terms[:CONTEXT_TERMS]


def rule_plan(message: str, history: list[dict[str, str]], turns: int) -> QueryPlan:
    """Carry the conversation's topic into a question that dropped it.

    The topic is appended rather than substituted, so the words the person
    actually chose still dominate both the lexical and the dense signal.
    """
    if not history or not needs_context(message):
        return QueryPlan(question=message, original=message)
    terms = context_terms(history, message, turns)
    if not terms:
        return QueryPlan(question=message, original=message)
    return QueryPlan(question=f"{message.strip()} {' '.join(terms)}", original=message, source="rules")


_REWRITE_SYSTEM = (
    "You rewrite the last user message of a conversation into a standalone search query "
    "for a document retrieval system. Resolve every pronoun and ellipsis from the history. "
    "Keep the language of the user's message, keep names, numbers and dates exactly as written, "
    "and never answer the question.\n"
    "Reply with JSON only, in this shape:\n"
    '{"standalone": "<the self-contained question>", '
    '"queries": ["<alternative phrasing>", "<another one>"], '
    '"intents": {"<intent>": <0..1>}, "entities": ["<name>"], "dates": ["<date>"]}'
)


def _history_prompt(history: list[dict[str, str]], turns: int) -> str:
    recent = history[-turns:] if turns > 0 else []
    lines = [f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}" for turn in recent]
    return "\n".join(lines) if lines else "(no previous turns)"


def parse_rewrite(raw: str, message: str) -> QueryPlan | None:
    """Read the model's JSON defensively; anything unusable returns None.

    Models wrap JSON in prose or in a fenced block often enough that insisting
    on a clean body would send most answers to the fallback for no reason.
    """
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
    if not isinstance(payload, dict):
        return None
    standalone = payload.get("standalone")
    if not isinstance(standalone, str) or not standalone.strip():
        return None
    raw_queries = payload.get("queries")
    variants: list[str] = []
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if isinstance(item, str) and item.strip() and item.strip() != standalone.strip():
                variants.append(item.strip()[:400])
    return QueryPlan(question=standalone.strip()[:400], original=message,
                     variants=list(dict.fromkeys(variants))[:MAX_VARIANTS], source="llm")


async def llm_plan(settings: AppSettings, message: str, history: list[dict[str, str]],
                   turns: int) -> QueryPlan | None:
    user = (f"Conversation so far:\n{_history_prompt(history, turns)}\n\n"
            f"Last user message:\n{message.strip()}\n\nJSON:")
    raw = await complete(settings, _REWRITE_SYSTEM, user, temperature=0.0, max_tokens=400)
    return parse_rewrite(raw, message)


# Keyed by conversation, message and the turns the rewrite was derived from. The
# phase plan asks for (conversation_id, message); the recent history belongs in
# the key as well, because the same «و مبلغش چقدر بود؟» asked twice in one
# conversation is a different question each time, and the cached answer to the
# first would be silently wrong for the second.
_CACHE: OrderedDict[tuple[str, str, str], QueryPlan] = OrderedDict()


def _cache_key(conversation_id: str, message: str, history: list[dict[str, str]], turns: int) -> tuple[str, str, str]:
    return conversation_id, message, _history_prompt(history, turns)


def clear_cache() -> None:
    _CACHE.clear()


async def plan_query(settings: AppSettings, message: str, history: list[dict[str, str]] | None = None,
                     conversation_id: str = "") -> QueryPlan:
    """The question retrieval should actually run, with its alternatives."""
    history = history or []
    if not settings.enable_query_rewrite:
        return QueryPlan(question=message, original=message)

    turns = settings.history_turns
    key = _cache_key(conversation_id, message, history, turns)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return replace(cached, source=f"{cached.source}-cached")

    plan = rule_plan(message, history, turns)
    if settings.provider != "local":
        try:
            from_model = await llm_plan(settings, message, history, turns)
            if from_model is not None:
                plan = from_model if settings.enable_multi_query else replace(from_model, variants=[])
        except Exception as exc:  # a rewrite must never take down the answer
            logger.warning("query rewrite failed, keeping the rule-based plan: %s", redact(exc))

    _CACHE[key] = plan
    while len(_CACHE) > CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return plan
