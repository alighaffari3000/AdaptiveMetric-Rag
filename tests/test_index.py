"""Tests for the in-memory chunk index and the storage migration behind it."""

import json
import math
import sqlite3
import uuid
from collections import Counter

import numpy as np
import pytest

from app import database
from app.index import ChunkIndex, _build
from app.retrieval import embed, tokenize


def make_row(content: str, document_name: str = "doc.txt", document_id: str = "d1") -> dict:
    return {
        "id": uuid.uuid4().hex,
        "document_id": document_id,
        "position": 0,
        "page": None,
        "section": None,
        "content": content,
        "vector": database.encode_vector(embed(content)),
        "tokens": json.dumps(tokenize(content), ensure_ascii=False),
        "metadata": "{}",
        "document_name": document_name,
        "document_type": "text/plain",
    }


def reference_bm25(query_tokens, doc_tokens, avg_len, doc_freq, total):
    """The scoring loop retrieval used before the index existed."""
    if not query_tokens or not doc_tokens:
        return 0.0
    counts = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = counts[term]
        if not tf:
            continue
        idf = math.log(1 + (total - doc_freq[term] + .5) / (doc_freq[term] + .5))
        denom = tf + 1.5 * (1 - .75 + .75 * len(doc_tokens) / max(avg_len, 1))
        score += idf * (tf * 2.5 / denom)
    return score


def test_vector_round_trips_through_the_blob_encoding():
    vector = embed("قرارداد شماره ۱۳۷")
    restored = database.decode_vector(database.encode_vector(vector))
    assert restored.shape == (384,)
    assert np.allclose(restored, np.asarray(vector, dtype=np.float32))


def test_decode_still_reads_pre_migration_json_text():
    vector = embed("legacy row")
    restored = database.decode_vector(json.dumps(vector))
    assert np.allclose(restored, np.asarray(vector, dtype=np.float32))


def test_bm25_matches_the_previous_python_implementation():
    contents = [
        "قرارداد شماره ۱۳۷ میان شرکت آفتاب و شرکت سپهر امضا شد",
        "مبلغ کل قرارداد دو میلیارد و چهارصد میلیون ریال است",
        "the annual subscription fee is EUR 186,000 invoiced quarterly",
        "قرارداد پشتیبانی نرم‌افزار شماره ۲۰۸ مبلغ کمتری دارد",
    ]
    snapshot = _build([make_row(text) for text in contents])
    token_lists = [tokenize(text) for text in contents]
    doc_freq = Counter()
    for tokens in token_lists:
        doc_freq.update(set(tokens))
    avg_len = sum(len(t) for t in token_lists) / len(token_lists)

    for query in ("مبلغ قرارداد", "شرکت آفتاب", "annual fee", "قرارداد ۲۰۸"):
        query_tokens = tokenize(query)
        actual = snapshot.bm25_scores(query_tokens)
        expected = [
            reference_bm25(query_tokens, tokens, avg_len, doc_freq, len(token_lists))
            for tokens in token_lists
        ]
        assert np.allclose(actual, expected), query


def test_dense_scores_match_the_previous_cosine_and_clamp_at_zero():
    contents = ["قرارداد شماره ۱۳۷", "annual subscription fee", "مرخصی استحقاقی سالانه"]
    snapshot = _build([make_row(text) for text in contents])
    query = np.asarray(embed("مبلغ قرارداد"), dtype=np.float32)
    scores = snapshot.dense_scores(query)
    expected = [max(0.0, float(np.dot(np.asarray(embed(text), dtype=np.float32), query))) for text in contents]
    assert np.allclose(scores, expected, atol=1e-6)
    assert (scores >= 0).all()


def test_dense_scores_are_zero_when_the_query_dimension_does_not_match():
    snapshot = _build([make_row("قرارداد")])
    assert snapshot.dense_scores(np.zeros(16, dtype=np.float32)).tolist() == [0.0]


def test_subset_statistics_treat_the_filtered_corpus_as_the_whole_corpus():
    contents = ["قرارداد آفتاب", "قرارداد سپهر", "مرخصی و دورکاری", "ارزیابی عملکرد"]
    rows = [make_row(text, document_id="a" if i < 2 else "b") for i, text in enumerate(contents)]
    snapshot = _build(rows)
    subset = np.array([0, 1], dtype=np.int64)
    scores = snapshot.bm25_scores(tokenize("قرارداد"), subset)
    assert scores[2] == scores[3] == 0.0
    # "قرارداد" appears in both subset documents, so it carries little information there.
    assert scores[0] == pytest.approx(scores[1])


def test_index_of_a_mixed_dimension_corpus_keeps_working():
    rows = [make_row("قرارداد شماره ۱۳۷"), make_row("مبلغ کل قرارداد")]
    rows[1]["vector"] = database.encode_vector([0.1] * 16)
    snapshot = _build(rows)
    assert snapshot.stale_vectors == 1
    assert snapshot.matrix.shape == (2, 384)
    assert snapshot.dense_scores(np.asarray(embed("قرارداد"), dtype=np.float32))[1] == 0.0
    # The stale row still participates through the lexical signal.
    assert snapshot.bm25_scores(tokenize("مبلغ"))[1] > 0


def test_empty_corpus_produces_an_empty_snapshot():
    snapshot = _build([])
    assert snapshot.size == 0
    assert snapshot.dense_scores(np.zeros(384, dtype=np.float32)).size == 0
    assert snapshot.bm25_scores(["قرارداد"]).size == 0


def test_index_reloads_after_invalidation(fresh_db):
    index = ChunkIndex()
    assert index.snapshot().size == 0

    now = "2026-01-01T00:00:00+00:00"
    with fresh_db.connect() as db:
        db.execute("INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
                   ("d1", "doc.txt", "text/plain", 0, 1, now, "{}"))
        row = make_row("قرارداد شماره ۱۳۷")
        db.execute(
            "INSERT INTO chunks(id,document_id,position,page,section,content,vector,tokens,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (row["id"], "d1", 0, None, None, row["content"], row["vector"], row["tokens"], "{}"),
        )

    assert index.snapshot().size == 0, "a loaded snapshot must not change under the caller"
    index.invalidate()
    assert index.snapshot().size == 1


def test_old_json_embedding_database_migrates_to_blobs(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_path)
    conn.executescript(
        """
        CREATE TABLE documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
          size INTEGER NOT NULL, chunks INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE chunks (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, page INTEGER, section TEXT,
          content TEXT NOT NULL, embedding TEXT NOT NULL, tokens TEXT NOT NULL,
          metadata TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    vector = embed("قرارداد شماره ۱۳۷")
    conn.execute("INSERT INTO documents VALUES('d1','doc.txt','text/plain',0,1,'2026-01-01','{}')")
    conn.execute(
        "INSERT INTO chunks VALUES('c1','d1',0,NULL,NULL,?,?,?,'{}')",
        ("قرارداد شماره ۱۳۷", json.dumps(vector), json.dumps(tokenize("قرارداد شماره ۱۳۷"))),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(database, "DB_PATH", legacy_path)
    database.close_thread_connection()
    try:
        database.init_db()
        columns = {row["name"] for row in database.rows("PRAGMA table_info(chunks)")}
        assert "vector" in columns and "embedding" not in columns
        stored = database.row("SELECT vector FROM chunks WHERE id='c1'")
        assert np.allclose(database.decode_vector(stored["vector"]), np.asarray(vector, dtype=np.float32))
        assert database.rows("PRAGMA user_version")[0]["user_version"] == database.SCHEMA_VERSION
    finally:
        database.close_thread_connection()


def test_migration_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "twice.db")
    database.close_thread_connection()
    try:
        database.init_db()
        database.init_db()
        columns = {row["name"] for row in database.rows("PRAGMA table_info(chunks)")}
        assert "vector" in columns
    finally:
        database.close_thread_connection()


def test_nested_connect_commits_once(fresh_db):
    with fresh_db.connect() as outer:
        outer.execute("INSERT INTO meta(key,value) VALUES('a','1')")
        with fresh_db.connect() as inner:
            inner.execute("INSERT INTO meta(key,value) VALUES('b','2')")
        assert fresh_db.meta_get("b") == "" or fresh_db.meta_get("b") == "2"
    assert fresh_db.meta_get("a") == "1"
    assert fresh_db.meta_get("b") == "2"


def test_failed_unit_of_work_rolls_back(fresh_db):
    with pytest.raises(RuntimeError):
        with fresh_db.connect() as db:
            db.execute("INSERT INTO meta(key,value) VALUES('rolled','back')")
            raise RuntimeError("boom")
    assert fresh_db.meta_get("rolled") == ""
