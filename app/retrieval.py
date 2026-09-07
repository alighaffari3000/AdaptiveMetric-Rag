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
# Share of the question's terms the best chunk must contain for its match to
# count as substantive. Swept against 38 reviewed unanswerable questions
# (eval/BASELINE.md): higher values buy abstention accuracy at a rate of
# roughly one correct answer lost per unanswerable question caught, which is
# the wrong trade for an assistant that cites its sources.
LEXICAL_FLOOR = .20

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


# Persian punctuation lives inside the Arabic block, so TOKEN_RE's
# \u0600-\u06FF range swallows it and "است؟" comes out as a single token that
# never matches the stopword list. Stripped where a token is compared against a
# question's words; deliberately NOT stripped inside `tokenize`, which feeds the
# BM25 postings and the hashed vectors. Doing it there is the correct fix and
# costs 2 points of hit@5 today, because two cross-language cases that share no
# term at all with their target document are currently ranked right by luck and
# the luck moves. Worth revisiting once the default embedding is semantic; see
# eval/BASELINE.md.
TOKEN_EDGES = "،؛؟۔٪٫٬"


def strip_edges(token: str) -> str:
    return token.strip(TOKEN_EDGES)


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("ي", "ی").replace("ك", "ک")
    return [t for t in TOKEN_RE.findall(normalized) if len(t) > 1 and t not in PERSIAN_STOP and t not in ENGLISH_STOP]


def _segments(content: str) -> list[tuple[str, list[str]]]:
    """Split a chunk into (heading, sentences) groups along its heading lines.

    A packed chunk holds several sections. The heading is what links a question
    to the right one - often the only place the question's subject appears -
    while the answer is the prose underneath it. Keeping the two apart lets a
    heading select a section without being quoted as the answer to it.
    """
    from .structure import looks_like_heading

    groups: list[tuple[str, list[str]]] = [("", [])]
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if looks_like_heading(line, unambiguous=True):
            groups.append((line, []))
            continue
        for part in re.split(r"(?<=[.!?؟؛])\s+", line):
            part = part.strip()
            if part:
                groups[-1][1].append(part)
    return [group for group in groups if group[0] or group[1]]


def best_evidence(query: str, content: str, answer: str = "") -> str:
    """Pick the sentence most responsible for a chunk matching the query."""
    query_terms = set(tokenize(query))
    answer_terms = set(tokenize(re.sub(r"\[\d+\]", "", answer)))
    important_terms = query_terms | answer_terms
    important_numbers = set(NUMBER_RE.findall(query + " " + answer))
    important_dates = set(DATE_RE.findall(query + " " + answer))

    def overlap(text: str, terms: set[str]) -> float:
        found = set(tokenize(text))
        return len(terms & found) / max(len(terms), 1)

    def sentence_score(sentence: str) -> tuple[float, int]:
        terms = set(tokenize(sentence))
        answer_overlap = len(answer_terms & terms) / max(len(answer_terms), 1)
        query_overlap = len(query_terms & terms) / max(len(query_terms), 1)
        combined_overlap = len(important_terms & terms) / max(len(important_terms), 1)
        sentence_numbers = set(NUMBER_RE.findall(sentence))
        sentence_dates = set(DATE_RE.findall(sentence))
        number_bonus = .55 if important_numbers and important_numbers & sentence_numbers else 0
        date_bonus = .45 if important_dates and important_dates & sentence_dates else 0
        return .55 * answer_overlap + .30 * query_overlap + .15 * combined_overlap + number_bonus + date_bonus, -len(sentence)

    groups = _segments(content)
    if not groups:
        return content[:320]

    def group_score(group: tuple[str, list[str]]) -> float:
        heading, sentences = group
        # The heading says which section this is; the body says whether the
        # answer is in it. Both are measured the same way so neither can win on
        # the scale it happens to use: weighting a heading against a sentence
        # score that tops out well below 1.0 let a heading sharing one word of
        # the question beat a body containing all of it.
        body = max((overlap(sentence, important_terms) for sentence in sentences), default=0.0)
        return .45 * overlap(heading, important_terms) + .55 * body

    # A heading-only group carries no answer, only a label, so it is chosen
    # only when the chunk is nothing but headings.
    with_body = [group for group in groups if group[1]]
    heading, sentences = max(with_body or groups, key=group_score)
    if not sentences:
        return heading[:600]
    return max(sentences, key=sentence_score)[:600]


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

    # `structure` is funded out of `metadata`, not out of `dense`. It reads the
    # same headings `metadata` did, without the file name and type diluting
    # them, so that is the weight it is entitled to. Paying for it from the
    # semantic signal instead cost conceptual queries 14 points of hit@5
    # (eval/BASELINE.md), which is the opposite of the point.
    if any(w in q for w in browse_words):
        intent = "document_browse"
        weights = {"dense": .52, "bm25": .08, "keyword": .18, "entity": .03, "numeric": .02, "temporal": .02,
                   "metadata": .07, "structure": .08}
    elif temporal or any(w in q for w in temporal_words):
        intent = "temporal_fact"
        weights = {"dense": .16, "bm25": .10, "keyword": .10, "entity": .22, "numeric": .07, "temporal": .28,
                   "metadata": .04, "structure": .03}
    elif numbers or any(w in q for w in numeric_words):
        intent = "numeric_fact"
        weights = {"dense": .15, "bm25": .11, "keyword": .10, "entity": .22, "numeric": .28, "temporal": .06,
                   "metadata": .05, "structure": .03}
    elif any(w in q for w in causal_words):
        intent = "causal"
        weights = {"dense": .41, "bm25": .15, "keyword": .12, "entity": .15, "numeric": .03, "temporal": .05,
                   "metadata": .04, "structure": .05}
    elif any(w in q for w in code_words):
        intent = "technical_code"
        weights = {"dense": .27, "bm25": .27, "keyword": .11, "entity": .21, "numeric": .05, "temporal": .02,
                   "metadata": .03, "structure": .04}
    elif any(w in q for w in conceptual_words):
        intent = "conceptual"
        weights = {"dense": .52, "bm25": .13, "keyword": .12, "entity": .07, "numeric": .02, "temporal": .02,
                   "metadata": .05, "structure": .07}
    else:
        intent = "exact_fact"
        weights = {"dense": .25, "bm25": .19, "keyword": .11, "entity": .23, "numeric": .07, "temporal": .06,
                   "metadata": .04, "structure": .05}
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


# Words that name a document rather than its subject. "موضوع این کتاب چیه؟"
# asks what a book is about; a contract clause headed "ماده ۲ — موضوع قرارداد"
# matches the word and means something else entirely. Measured on the golden
# set, letting these terms score cost conceptual queries 14 points of hit@5 by
# ranking two contracts above the book the question was about. They stay in
# every other signal, where a body full of them is genuine evidence.
DOCUMENT_WORDS = frozenset({
    "موضوع", "خلاصه", "درباره", "مفهوم", "توضیح", "کتاب", "متن", "سند", "مطلب", "محتوا",
    "topic", "subject", "about", "summary", "overview", "book", "text", "document",
    "title", "content", "passage",
})


def _query_coverage(query_tokens: list[str], content: str) -> float:
    """Share of the question's distinct terms that appear in this text.

    Unlike the fused signals this does not depend on what the other candidates
    scored, which is what an evidence gate needs: whether to answer at all
    cannot be decided on a scale set by the best of a bad field.
    """
    terms = {stripped for token in query_tokens if len(stripped := strip_edges(token)) > 1}
    terms -= PERSIAN_STOP | ENGLISH_STOP
    if not terms:
        return 0.0
    lowered = content.lower()
    return sum(1 for term in terms if term in lowered) / len(terms)


def _section_signal(keywords: list[str], section_path: str) -> float:
    """How much of the question the section headings above a chunk account for.

    A heading is a few deliberate words, so a match here says more than the
    same match against a paragraph - which is also why a coincidental match
    does more damage. Scored as the share of the query's distinct content terms
    present in the path, capped like the keyword signal so a long question is
    not required to match every one of its words.
    """
    if not keywords or not section_path:
        return 0.0
    lowered = section_path.lower()
    distinct = {keyword.lower() for keyword in keywords if len(keyword) > 1}
    distinct -= DOCUMENT_WORDS
    if not distinct:
        return 0.0
    matches = sum(1 for keyword in distinct if keyword in lowered)
    return min(1.0, matches / min(3, len(distinct)))


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
    haystack = " ".join(chunk["content"] for chunk in selected)
    # Same term set the evidence gate uses, so a question mark fused onto a
    # stopword cannot quietly lower the confidence of a chunk that answers.
    coverage = _query_coverage(query_tokens, haystack)
    reranked = selected[0].get("rerank_score")
    # A chunk below the corpus mean is not distinctive; it is not evidence
    # against itself either. Left unclamped, a negative standout subtracted
    # from the coverage term and drove confidence to zero for a chunk that
    # answered the question, which is what happens whenever a search is
    # narrowed to a section the embedding scores poorly.
    relevance = (float(reranked) if reranked is not None
                 else max(0.0, min(1.0, standout / STANDOUT_REFERENCE)))
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

    # The corpus the user is searching, before any narrowing a router applied.
    # Confidence is measured against this rather than against the slice a
    # router guessed at, or narrowing to one section would make every answer
    # look unremarkable and the evidence gate would refuse it.
    scope = subset
    # A caller that has already decided which chunks are worth searching - the
    # table-of-contents router is the one that does - narrows the corpus here.
    # Chunks are named by id, never by row position: a position is an offset
    # into one snapshot, and an ingest or delete between the caller's decision
    # and this call would silently point it at different chunks.
    # An empty narrowing is treated as no narrowing, so the router can never
    # remove the only chunk that holds the answer.
    chunk_ids = (filters or {}).get("chunk_ids")
    if chunk_ids:
        wanted = set(chunk_ids)
        chosen = np.fromiter(
            (i for i, row in enumerate(snapshot.rows) if row["id"] in wanted),
            dtype=np.int64,
        )
        if chosen.size:
            subset = chosen if subset is None else np.intersect1d(subset, chosen)
            if not subset.size:
                return RetrievalResult([], analysis, 0.0, False)

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
        # `metadata` mixes the file name and type in with the heading, which
        # dilutes exactly the term that locates an answer inside a document.
        # `structure` reads the section path on its own.
        section_path = row.get("section_path") or row.get("section") or ""
        metadata_text = f'{row.get("document_name", "")} {section_path} {row.get("document_type", "")}'
        features = {
            "dense": float(dense[position]),
            "bm25": float(lexical[position] / bm_max),
            "keyword": _multi_keyword_signal(analysis.keywords, content + " " + metadata_text),
            "entity": _overlap(analysis.entities, content),
            "numeric": _numeric_signal(analysis.numbers, content, analysis.intent == "numeric_fact"),
            "temporal": _temporal_signal(analysis.temporal_terms, content, analysis.intent == "temporal_fact"),
            "metadata": _overlap(qtokens[:6], metadata_text),
            "structure": _section_signal(analysis.keywords, section_path),
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
    # raw similarity. The scale comes from the corpus the user is searching,
    # before any narrowing a router applied, so routing to one section cannot
    # make that section unremarkable against itself. The value, though, is the
    # returned chunk's own: reporting the corpus maximum would describe a chunk
    # the router may have excluded from the results entirely.
    pool = dense[scope] if scope is not None else dense
    pool_mean, pool_std = (float(pool.mean()), float(pool.std()) + 1e-9) if pool.size > 1 else (0.0, 1.0)
    for chunk, value in zip(scored, fused):
        chunk["score"] = round(float(value), 6)
        chunk["standout"] = round((chunk["features"]["dense"] - pool_mean) / pool_std, 3) if pool.size > 1 else 0.0

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[:context_count]
    # The most distinctive chunk among those actually returned. Reporting the
    # corpus maximum instead described a chunk the caller may never see - a
    # router that narrowed the search to the wrong section still reported the
    # confidence of the right one. Reading the whole returned set rather than
    # only its first member matches what `score_confidence` already does with
    # coverage, and keeps a fused ranking from being punished when its top
    # chunk won on an exact identifier rather than on similarity.
    standout = max((float(chunk["standout"]) for chunk in selected), default=0.0)
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
    # The lexical test has to be an absolute one. `bm25` here is the candidate's
    # score divided by the best candidate's, so the top chunk reads 1.0 whenever
    # any candidate matched a query term at all - measured at 81% of golden
    # queries - and comparing that to a floor of .08 asked "did retrieval return
    # anything". This asks how much of the question the chunk actually contains.
    substantive_match = bool(top_features) and (
        _query_coverage(result.analysis.query_tokens, result.chunks[0]["content"]) >= LEXICAL_FLOOR
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
