from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .index import index
from .models import QueryAnalysis


TOKEN_RE = re.compile(r"[\w\u0600-\u06FF.-]+", re.UNICODE)
DATE_RE = re.compile(r"(?:\b(?:19|20)\d{2}\b|\b1[34]\d{2}\b|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
DIMENSION = 384

# Measured on eval/golden.jsonl; see eval/BASELINE.md. A top chunk about four
# standard deviations above the corpus mean is as distinctive as retrieval gets.
STANDOUT_REFERENCE = 4.0
EARLY_EXIT_STANDOUT = 3.6
EARLY_EXIT_MARGIN = .04
# Under rank fusion a chunk is cited when its raw similarity sits within this
# many standard deviations of the best chunk's. Swept on the golden set: 1.5
# roughly doubles citation precision over citing every returned chunk while
# keeping 98 percent of the relevant ones (eval/BASELINE.md).
CITATION_STANDOUT_GAP = 1.5

PERSIAN_STOP = {"از", "به", "در", "با", "برای", "که", "این", "آن", "را", "و", "یا", "چه", "چرا", "چگونه", "است", "شد", "می"}
ENGLISH_STOP = {"the", "a", "an", "of", "to", "in", "for", "is", "was", "and", "or", "what", "why", "how", "does"}

KEYWORD_EXPANSIONS = {
    "کتاب": ["book", "title"], "متن": ["text", "passage", "content"],
    "موضوع": ["topic", "subject", "about"], "خلاصه": ["summary", "overview"],
    "نویسنده": ["author", "written"], "ناشر": ["publisher", "publication"],
    "انتشارات": ["publisher", "publication"], "سال": ["year", "published"],
    "وبسایت": ["website", "web", "url"], "وب‌سایت": ["website", "web", "url"],
    "قرارداد": ["contract", "agreement"], "مبلغ": ["amount", "price", "cost"],
    "تاریخ": ["date", "year", "time"], "پایان": ["end", "expiry", "expiration"],
    "خلاق": ["creative", "creativity", "innovation", "imagination"],
    "خلاقیت": ["creativity", "creative thinking", "innovation"],
    "چطور": ["how", "ways", "methods"], "چگونه": ["how", "ways", "methods"],
    "باشیم": ["become", "being"],
}


def normalize_query(query: str) -> str:
    normalized = query.strip().lower().replace("ي", "ی").replace("ك", "ک")
    normalized = re.sub(r"^(?:خوب|خب|خب،|خوب،|لطفاً|لطفا|ببین|راستی)\s+", "", normalized)
    normalized = normalized.replace("چطوری", "چطور").replace("چه جوری", "چطور").replace("چجوری", "چطور")
    return re.sub(r"\s+", " ", normalized).strip()


def expanded_keywords(query: str) -> list[str]:
    """Extract multiple distinct keywords and add compact cross-language variants."""
    query = normalize_query(query)
    keywords = list(dict.fromkeys(tokenize(query)))
    lowered = query.lower()
    for source, expansions in KEYWORD_EXPANSIONS.items():
        if source in lowered:
            keywords.extend(expansions)
    quoted = re.findall(r'["«](.*?)["»]', query)
    keywords.extend(item.strip().lower() for item in quoted if item.strip())
    return list(dict.fromkeys(keywords))[:24]


def query_variants(query: str) -> list[str]:
    query = normalize_query(query)
    keywords = expanded_keywords(query)
    translated = [term for term in keywords if term.isascii()]
    variants = [query.strip()]
    if translated:
        variants.append(" ".join(translated))
        variants.append(query.strip() + " | " + " ".join(translated))
    return list(dict.fromkeys(variant for variant in variants if variant))


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("ي", "ی").replace("ك", "ک")
    return [t for t in TOKEN_RE.findall(normalized) if len(t) > 1 and t not in PERSIAN_STOP and t not in ENGLISH_STOP]


def best_evidence(query: str, content: str, answer: str = "") -> str:
    """Pick the sentence most responsible for a chunk matching the query."""
    query_terms = set(tokenize(query))
    answer_terms = set(tokenize(re.sub(r"\[\d+\]", "", answer)))
    important_terms = query_terms | answer_terms
    important_numbers = set(NUMBER_RE.findall(query + " " + answer))
    important_dates = set(DATE_RE.findall(query + " " + answer))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?؟؛])\s+|\n+", content) if part.strip()]
    if not sentences:
        return content[:320]

    def evidence_score(sentence: str) -> tuple[float, int]:
        terms = set(tokenize(sentence))
        answer_overlap = len(answer_terms & terms) / max(len(answer_terms), 1)
        query_overlap = len(query_terms & terms) / max(len(query_terms), 1)
        combined_overlap = len(important_terms & terms) / max(len(important_terms), 1)
        sentence_numbers = set(NUMBER_RE.findall(sentence))
        sentence_dates = set(DATE_RE.findall(sentence))
        number_bonus = .55 if important_numbers and important_numbers & sentence_numbers else 0
        date_bonus = .45 if important_dates and important_dates & sentence_dates else 0
        return .55 * answer_overlap + .30 * query_overlap + .15 * combined_overlap + number_bonus + date_bonus, -len(sentence)

    selected = max(sentences, key=evidence_score)
    return selected[:600]


def embed(text: str) -> list[float]:
    """Fast multilingual feature hashing; deterministic and requires no model download."""
    vector = [0.0] * DIMENSION
    tokens = tokenize(text)
    features = tokens + [f"{tokens[i]}::{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[idx] += sign * (1.0 if "::" not in feature else 0.55)
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return max(0.0, sum(x * y for x, y in zip(a, b)))


def analyze_query(query: str) -> QueryAnalysis:
    query = normalize_query(query)
    q = query.lower()
    language = "fa" if re.search(r"[\u0600-\u06FF]", query) else "en"
    numbers = NUMBER_RE.findall(query)
    temporal = DATE_RE.findall(query)
    quoted = re.findall(r'["«](.*?)["»]', query)
    tokens = tokenize(query)
    caps = re.findall(r"\b[A-Z][A-Za-z0-9_.-]+\b", query)
    entities = list(dict.fromkeys(quoted + caps + [t for t in tokens if any(c.isdigit() for c in t)]))[:8]
    keywords = expanded_keywords(query)
    query_tokens = list(dict.fromkeys(tokens))

    temporal_words = ("when", "date", "expire", "released", "زمان", "تاریخ", "پایان", "منقضی", "منتشر")
    causal_words = ("why", "cause", "reason", "چرا", "علت", "دلیل")
    code_words = ("error", "exception", "stack", "cuda", "api", "function", "خطا", "کد")
    numeric_words = ("how much", "price", "dose", "count", "قیمت", "دوز", "مقدار", "چقدر")
    conceptual_words = ("how does", "explain", "concept", "topic", "subject", "summary",
                        "چگونه", "چطور", "مفهوم", "توضیح", "موضوع", "درباره", "خلاصه", "چی هست", "چیه", "خلاق")
    browse_words = ("from the book", "book passage", "from the text", "quote from", "متن کتاب",
                    "از متن", "از کتاب", "بخشی از", "برام بنویس", "برایم بنویس")

    if any(w in q for w in browse_words):
        intent = "document_browse"
        weights = {"dense": .52, "bm25": .08, "keyword": .18, "entity": .03, "numeric": .02, "temporal": .02, "metadata": .15}
    elif temporal or any(w in q for w in temporal_words):
        intent = "temporal_fact"
        weights = {"dense": .16, "bm25": .10, "keyword": .10, "entity": .22, "numeric": .07, "temporal": .28, "metadata": .07}
    elif numbers or any(w in q for w in numeric_words):
        intent = "numeric_fact"
        weights = {"dense": .15, "bm25": .11, "keyword": .10, "entity": .22, "numeric": .28, "temporal": .06, "metadata": .08}
    elif any(w in q for w in causal_words):
        intent = "causal"
        weights = {"dense": .41, "bm25": .15, "keyword": .12, "entity": .15, "numeric": .03, "temporal": .05, "metadata": .09}
    elif any(w in q for w in code_words):
        intent = "technical_code"
        weights = {"dense": .27, "bm25": .27, "keyword": .11, "entity": .21, "numeric": .05, "temporal": .02, "metadata": .07}
    elif any(w in q for w in conceptual_words):
        intent = "conceptual"
        weights = {"dense": .52, "bm25": .13, "keyword": .12, "entity": .07, "numeric": .02, "temporal": .02, "metadata": .12}
    else:
        intent = "exact_fact"
        weights = {"dense": .25, "bm25": .19, "keyword": .11, "entity": .23, "numeric": .07, "temporal": .06, "metadata": .09}
    return QueryAnalysis(intent=intent, language=language, weights=weights, entities=entities, numbers=numbers,
                         temporal_terms=temporal, keywords=keywords, query_tokens=query_tokens)


def _overlap(needles: list[str], haystack: str) -> float:
    if not needles:
        return 0.0
    low = haystack.lower()
    return sum(1 for item in needles if item.lower() in low) / len(needles)


def _numeric_signal(numbers: list[str], content: str, numeric_intent: bool) -> float:
    if numbers:
        return _overlap(numbers, content)
    return 1.0 if numeric_intent and NUMBER_RE.search(content) else 0.0


def _temporal_signal(terms: list[str], content: str, temporal_intent: bool) -> float:
    if terms:
        return _overlap(terms, content)
    return 1.0 if temporal_intent and DATE_RE.search(content) else 0.0


def _multi_keyword_signal(keywords: list[str], content: str) -> float:
    if not keywords:
        return 0.0
    lowered = content.lower()
    matches = sum(1 for keyword in set(keywords) if keyword.lower() in lowered)
    # Requiring up to four distinct terms rewards true multi-keyword matches
    # without penalizing concise queries against a different source language.
    return min(1.0, matches / min(4, len(set(keywords))))



RRF_K = 60


def _weighted_rrf(signals: dict[str, np.ndarray], weights: dict[str, float], k: int = RRF_K) -> np.ndarray:
    """Fuse heterogeneous signals by rank instead of by value.

    Summing the raw signals weighted by intent looks reasonable but silently
    favours whichever signal happens to have the widest numeric spread. A
    semantic cosine varies over a narrow high band while BM25 is min-max
    normalised across the full [0, 1] range, so the lexical term decided almost
    every ranking no matter what the intent weights said. Reciprocal rank fusion
    only reads the ordering each signal produces, so the weights mean what they
    claim to mean.

    A signal that separates nothing - all zeros, or one constant value - carries
    no ordering, so it is skipped and its weight is not spent.
    """
    size = len(next(iter(signals.values()))) if signals else 0
    fused = np.zeros(size, dtype=np.float64)
    spent = 0.0
    for name, values in signals.items():
        weight = weights.get(name, 0.0)
        if weight <= 0 or not size:
            continue
        ranked = np.flatnonzero(values > 0)
        if ranked.size == 0:
            continue
        if ranked.size == size and np.ptp(values[ranked]) == 0:
            continue  # every candidate scores the same: no ordering to contribute
        order = ranked[np.argsort(-values[ranked], kind="stable")]
        # Equal values share a rank. Numbering ties 1, 2, 3 in corpus order
        # would let insertion order decide between chunks a signal cannot
        # tell apart, which is exactly what sparse signals like entity or
        # numeric produce: many chunks at 1.0, the rest at 0.
        _, first_index, inverse = np.unique(-values[order], return_index=True, return_inverse=True)
        ranks = first_index[inverse] + 1
        fused[order] += weight / (k + ranks)
        spent += weight
    if spent <= 0:
        # Nothing separated these candidates - a single candidate, or every
        # signal flat across all of them. They tie at the top rather than all
        # scoring zero, which would make the evidence gate refuse a good answer.
        return np.ones(size, dtype=np.float64)
    # Rescale so that ranking first on every contributing signal reads as 1.0.
    return fused * (k + 1) / spent


def _weighted_sum(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """The original scoring: each signal's value times its intent weight.

    Kept for the feature-hashing embedding, where it still edges out rank fusion.
    Its weights were tuned against exactly that signal mix, and rank fusion gives
    a meaningless dense signal a fairer share of the vote than it deserves.
    """
    size = len(next(iter(signals.values()))) if signals else 0
    total = np.zeros(size, dtype=np.float64)
    for name, values in signals.items():
        total += weights.get(name, 0.0) * values
    return total


def score_confidence(selected: list[dict[str, Any]], query_tokens: list[str],
                     standout: float = 0.0) -> float:
    """How much the retrieved evidence should be trusted, in [0, 1].

    Two ingredients the reader can reason about: how strongly the best source
    stands out from everything else in the library, and how much of the question
    the selected text actually covers.

    The fused score cannot serve as the first ingredient. Rank fusion puts the
    best chunk at or near 1.0 for almost every query, including ones the library
    cannot answer, because it measures ordering rather than quality. `standout`
    is measured on the raw similarities instead, as how many standard deviations
    the best chunk sits above the corpus mean, which does not depend on the
    embedding model's own scale. When a reranker has run, its judgement of the
    top chunk replaces it, being a direct answer to "does this passage answer
    this question".
    """
    if not selected:
        return 0.0
    terms = set(query_tokens)
    haystack = " ".join(chunk["content"].lower() for chunk in selected)
    coverage = min(1.0, sum(1 for term in terms if term in haystack) / max(len(terms), 1))
    reranked = selected[0].get("rerank_score")
    relevance = float(reranked) if reranked is not None else min(1.0, standout / STANDOUT_REFERENCE)
    return max(0.0, min(1.0, .55 * relevance + .45 * coverage))


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    analysis: QueryAnalysis
    confidence: float
    early_exit: bool
    standout: float = 0.0
    fusion: str = "linear"


def apply_semantic_profile(weights: dict[str, float], dense_weight: float) -> dict[str, float]:
    """Raise the dense signal to its deserved share when the embedding is semantic.

    The stock intent weights were tuned when the dense signal was feature hashing,
    which carries no meaning: giving it 0.16 of the vote was correct then. A real
    multilingual embedding ranks the right chunk first on its own, and the five
    lexical signals combined were outvoting it. This keeps the adaptive idea - the
    remaining share is still distributed by intent - while letting the strongest
    signal lead.
    """
    if weights.get("dense", 0) >= dense_weight:
        return weights
    others = {name: value for name, value in weights.items() if name != "dense"}
    total = sum(others.values()) or 1.0
    remaining = 1.0 - dense_weight
    adjusted = {name: value / total * remaining for name, value in others.items()}
    adjusted["dense"] = dense_weight
    return adjusted


def retrieve(query: str, candidate_count: int = 100, context_count: int = 5, filters: dict[str, Any] | None = None,
             query_vector: list[float] | None = None, dense_weight: float | None = None,
             fusion: str = "linear") -> RetrievalResult:
    analysis = analyze_query(query)
    if dense_weight is not None:
        analysis = analysis.model_copy(
            update={"weights": apply_semantic_profile(analysis.weights, dense_weight)}
        )
    snapshot = index.snapshot()
    if not snapshot.size:
        return RetrievalResult([], analysis, 0.0, False)

    document_id = (filters or {}).get("document_id")
    if document_id:
        subset = np.fromiter(
            (i for i, row in enumerate(snapshot.rows) if row["document_id"] == document_id),
            dtype=np.int64,
        )
        if not subset.size:
            return RetrievalResult([], analysis, 0.0, False)
    else:
        subset = None

    qvec = np.asarray(query_vector if query_vector is not None else embed(query), dtype=np.float32)
    qtokens = analysis.keywords

    dense = snapshot.dense_scores(qvec)
    lexical = snapshot.bm25_scores(qtokens, subset)
    universe = subset if subset is not None else np.arange(snapshot.size, dtype=np.int64)

    # First-stage hybrid retrieval: union of the strongest dense and lexical
    # candidates. Ranking ties keep corpus order, matching the previous scorer.
    dense_limit = max(1, round(candidate_count * .65))
    lexical_limit = max(1, candidate_count - dense_limit)
    dense_ranked = universe[np.argsort(-dense[universe], kind="stable")][:dense_limit]
    lexical_ranked = universe[np.argsort(-lexical[universe], kind="stable")][:lexical_limit]
    candidate_set = set(dense_ranked.tolist()) | set(lexical_ranked.tolist())
    candidates = [position for position in universe.tolist() if position in candidate_set]

    bm_max = max((lexical[position] for position in candidates), default=1) or 1
    scored: list[dict[str, Any]] = []
    signals: dict[str, list[float]] = {name: [] for name in analysis.weights}
    for position in candidates:
        row = snapshot.rows[position]
        content = row["content"]
        metadata_text = f'{row.get("document_name", "")} {row.get("section") or ""} {row.get("document_type", "")}'
        features = {
            "dense": float(dense[position]),
            "bm25": float(lexical[position] / bm_max),
            "keyword": _multi_keyword_signal(analysis.keywords, content + " " + metadata_text),
            "entity": _overlap(analysis.entities, content),
            "numeric": _numeric_signal(analysis.numbers, content, analysis.intent == "numeric_fact"),
            "temporal": _temporal_signal(analysis.temporal_terms, content, analysis.intent == "temporal_fact"),
            "metadata": _overlap(qtokens[:6], metadata_text),
        }
        for name, value in features.items():
            signals[name].append(value)
        chunk = dict(row)
        chunk["features"] = {name: round(value, 4) for name, value in features.items()}
        scored.append(chunk)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in signals.items()}
    fused = (_weighted_rrf(arrays, analysis.weights) if fusion == "rank"
             else _weighted_sum(arrays, analysis.weights))
    # How far each chunk stands above the corpus, in standard deviations of the
    # raw similarity; the best chunk's value is the result-level `standout`.
    pool = dense[universe]
    pool_mean, pool_std = (float(pool.mean()), float(pool.std()) + 1e-9) if pool.size > 1 else (0.0, 1.0)
    standout = float((pool.max() - pool_mean) / pool_std) if pool.size > 1 else 0.0
    for chunk, value in zip(scored, fused):
        chunk["score"] = round(float(value), 6)
        chunk["standout"] = round((chunk["features"]["dense"] - pool_mean) / pool_std, 3) if pool.size > 1 else 0.0

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[:context_count]
    confidence = score_confidence(selected, analysis.query_tokens or qtokens, standout)
    top = selected[0]["score"] if selected else 0.0
    second = selected[1]["score"] if len(selected) > 1 else 0.0
    early = standout > EARLY_EXIT_STANDOUT and top - second > EARLY_EXIT_MARGIN
    return RetrievalResult(selected, analysis, round(confidence, 3), early, round(standout, 3), fusion)


def select_grounded(result: RetrievalResult) -> tuple[bool, list[dict[str, Any]]]:
    """Decide whether retrieval produced usable evidence, and which chunks to cite.

    Evidence existence is based on retrieval signals, not the user-facing
    confidence preference. Raising that preference must never hide known facts.
    """
    top_features = result.chunks[0]["features"] if result.chunks else {}
    browse_intent = result.analysis.intent == "document_browse"
    semantic_intent = result.analysis.intent in {"conceptual", "causal", "document_browse"}
    dense_floor = .24 if semantic_intent else .30
    confidence_floor = .08 if semantic_intent else .24
    substantive_match = bool(top_features) and (
        top_features.get("bm25", 0) >= .08
        or top_features.get("entity", 0) >= .50
        or top_features.get("dense", 0) >= dense_floor
        or (result.analysis.intent == "numeric_fact" and top_features.get("numeric", 0) >= .80)
        or (result.analysis.intent == "temporal_fact" and top_features.get("temporal", 0) >= .80)
    )
    evidence_found = bool(result.chunks) and (
        browse_intent or (result.confidence >= confidence_floor and substantive_match)
    )
    if not evidence_found:
        return False, []
    top = result.chunks[0]
    if result.fusion == "rank" and top.get("rerank_score") is None:
        # Fused rank scores sit near 1.0 for every returned chunk, so a
        # fraction of the top score would cite all of them. Compare raw
        # similarities instead, on a scale that does not depend on the model.
        floor = top.get("standout", 0.0) - CITATION_STANDOUT_GAP
        grounded = [chunk for chunk in result.chunks if chunk.get("standout", 0.0) >= floor]
    else:
        citation_floor = max(.10, top["score"] * .45)
        grounded = [chunk for chunk in result.chunks if chunk["score"] >= citation_floor]
    return bool(grounded), grounded
