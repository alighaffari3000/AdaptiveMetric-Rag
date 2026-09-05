"""In-memory retrieval index: a dense vector matrix plus an inverted token index.

The previous implementation re-read every chunk from SQLite on every query,
decoded each embedding from JSON, scored it in a Python loop, and rebuilt the
BM25 corpus statistics from scratch. That is linear work per query with a very
large constant. This module does the same arithmetic once at load time and keeps
it in memory, so a query costs one matrix product plus a walk over the postings
of the query terms.

The BM25 formula is reproduced exactly as `retrieval._bm25` computed it, so
swapping the storage layer does not move retrieval quality.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import database
from .text import date_terms, normalize, number_terms

logger = logging.getLogger("adaptive_metric_rag.index")

K1 = 2.5
B = .75
SATURATION = 1.5  # the `tf + 1.5 * (...)` denominator term of the original formula

CHUNK_QUERY = (
    "SELECT c.id,c.document_id,c.position,c.page,c.page_end,c.section,c.section_path,c.parent_id,"
    "c.content,c.vector,c.tokens,c.metadata,d.name document_name,d.type document_type "
    "FROM chunks c JOIN documents d ON d.id=c.document_id "
    "ORDER BY c.rowid"
)


@dataclass
class Snapshot:
    """An immutable view a query can score against without holding the lock.

    Alongside the vectors and postings it carries what the lexical signals need
    in comparable form: each chunk's folded text and metadata, and the canonical
    numbers and dates found in it. Folding a chunk costs a few regex passes, and
    doing it per query would repeat that work for every candidate on every
    question; here it happens once when the index is built.
    """

    rows: list[dict[str, Any]]
    matrix: np.ndarray
    lengths: np.ndarray
    postings: dict[str, tuple[np.ndarray, np.ndarray]]
    doc_freq: dict[str, int]
    avg_len: float
    stale_vectors: int
    folded: list[str] = field(default_factory=list)
    folded_meta: list[str] = field(default_factory=list)
    numbers: list[frozenset[str]] = field(default_factory=list)
    dates: list[frozenset[str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.rows)

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity, clamped at zero exactly as the previous scorer did."""
        if not self.size or self.matrix.shape[1] != query_vector.shape[0]:
            return np.zeros(self.size, dtype=np.float32)
        return np.maximum(self.matrix @ query_vector, 0.0)

    def bm25_scores(self, query_tokens: list[str], subset: np.ndarray | None = None) -> np.ndarray:
        """Okapi BM25 over the postings of the query terms only.

        `subset` restricts the corpus statistics to the given row indices, which
        is what a document filter needs: term rarity is relative to the corpus
        actually being searched.
        """
        scores = np.zeros(self.size, dtype=np.float64)
        if not self.size or not query_tokens:
            return scores
        if subset is None:
            total = self.size
            avg_len = self.avg_len
            doc_freq = self.doc_freq
            allowed = None
        else:
            total = int(subset.size)
            if not total:
                return scores
            avg_len = float(self.lengths[subset].mean()) if total else 0.0
            doc_freq = _subset_doc_freq(self, subset)
            allowed = np.zeros(self.size, dtype=bool)
            allowed[subset] = True

        divisor = max(avg_len, 1)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, term_freq = posting
            if allowed is not None:
                keep = allowed[indices]
                indices, term_freq = indices[keep], term_freq[keep]
                if not indices.size:
                    continue
            df = doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (total - df + .5) / (df + .5))
            denominator = term_freq + SATURATION * (1 - B + B * self.lengths[indices] / divisor)
            scores[indices] += idf * (term_freq * K1 / denominator)
        return scores


def _subset_doc_freq(snapshot: Snapshot, subset: np.ndarray) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for index in subset:
        counter.update(set(snapshot.rows[index]["tokens"]))
    return counter


class ChunkIndex:
    """Thread-safe owner of the current snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = _empty_snapshot()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def snapshot(self) -> Snapshot:
        with self._lock:
            if not self._loaded:
                self.reload()
            return self._snapshot

    def reload(self) -> Snapshot:
        """Rebuild the whole index from the database."""
        rows = database.rows(CHUNK_QUERY)
        snapshot = _build(rows)
        with self._lock:
            self._snapshot = snapshot
            self._loaded = True
        logger.info(
            "index loaded: %d chunks, %d dimensions, %d distinct tokens%s",
            snapshot.size,
            snapshot.matrix.shape[1] if snapshot.size else 0,
            len(snapshot.postings),
            f", {snapshot.stale_vectors} chunks need re-indexing" if snapshot.stale_vectors else "",
        )
        return snapshot

    def invalidate(self) -> None:
        with self._lock:
            self._loaded = False


def _empty_snapshot() -> Snapshot:
    return Snapshot(
        rows=[],
        matrix=np.zeros((0, 0), dtype=np.float32),
        lengths=np.zeros(0, dtype=np.float64),
        postings={},
        doc_freq={},
        avg_len=0.0,
        stale_vectors=0,
    )


def _build(raw_rows: list[dict[str, Any]]) -> Snapshot:
    if not raw_rows:
        return _empty_snapshot()

    vectors = [database.decode_vector(row["vector"]) for row in raw_rows]
    dimensions = Counter(vector.shape[0] for vector in vectors)
    dimension = dimensions.most_common(1)[0][0]
    stale = sum(count for size, count in dimensions.items() if size != dimension)
    if stale:
        logger.warning(
            "%d chunks were embedded with a different model (dimension mismatch) and score zero "
            "on the dense signal until they are re-indexed",
            stale,
        )

    matrix = np.zeros((len(raw_rows), dimension), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    lengths = np.zeros(len(raw_rows), dtype=np.float64)
    postings_build: dict[str, list[tuple[int, int]]] = defaultdict(list)
    folded: list[str] = []
    folded_meta: list[str] = []
    numbers: list[frozenset[str]] = []
    dates: list[frozenset[str]] = []

    for position, (raw, vector) in enumerate(zip(raw_rows, vectors)):
        if vector.shape[0] == dimension:
            matrix[position] = vector
        tokens = json.loads(raw["tokens"]) if isinstance(raw["tokens"], str) else list(raw["tokens"])
        lengths[position] = len(tokens)
        for term, frequency in Counter(tokens).items():
            postings_build[term].append((position, frequency))
        row = {key: raw[key] for key in raw if key not in {"vector", "tokens"}}
        row["tokens"] = tokens
        rows.append(row)
        content = raw["content"]
        folded.append(normalize(content))
        folded_meta.append(normalize(
            f'{raw.get("document_name") or ""} {raw.get("section_path") or raw.get("section") or ""} '
            f'{raw.get("document_type") or ""}'
        ))
        numbers.append(frozenset(number_terms(content)))
        dates.append(frozenset(date_terms(content)))

    postings = {
        term: (
            np.fromiter((index for index, _ in entries), dtype=np.int64, count=len(entries)),
            np.fromiter((count for _, count in entries), dtype=np.float64, count=len(entries)),
        )
        for term, entries in postings_build.items()
    }
    doc_freq = {term: len(entries) for term, entries in postings_build.items()}
    return Snapshot(
        rows=rows,
        matrix=matrix,
        lengths=lengths,
        postings=postings,
        doc_freq=doc_freq,
        avg_len=float(lengths.mean()),
        stale_vectors=stale,
        folded=folded,
        folded_meta=folded_meta,
        numbers=numbers,
        dates=dates,
    )


index = ChunkIndex()
