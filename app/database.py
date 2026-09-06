from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


logger = logging.getLogger("adaptive_metric_rag.database")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "adaptive_rag.db"

SCHEMA_VERSION = 2

_local = threading.local()
_write_lock = threading.RLock()

VECTOR_DTYPE = np.float32


def encode_vector(vector: Any) -> bytes:
    """Store an embedding as raw float32 rather than a JSON string."""
    return np.asarray(vector, dtype=VECTOR_DTYPE).tobytes()


def decode_vector(blob: bytes | str) -> np.ndarray:
    if isinstance(blob, str):  # a row written before the BLOB migration
        return np.asarray(json.loads(blob), dtype=VECTOR_DTYPE)
    return np.frombuffer(blob, dtype=VECTOR_DTYPE)


def _open() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def connection() -> sqlite3.Connection:
    """One long-lived connection per thread; SQLite connections are not shareable."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _open()
        _local.conn = conn
        _local.depth = 0
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Reentrant unit of work. Only the outermost block commits."""
    conn = connection()
    _local.depth = getattr(_local, "depth", 0) + 1
    outermost = _local.depth == 1
    try:
        if outermost:
            _write_lock.acquire()
        yield conn
        if outermost:
            conn.commit()
    except BaseException:
        if outermost:
            conn.rollback()
        raise
    finally:
        _local.depth -= 1
        if outermost:
            _write_lock.release()


def close_thread_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
    _local.depth = 0


def _create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
          size INTEGER NOT NULL, chunks INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, page INTEGER, section TEXT,
          section_path TEXT NOT NULL DEFAULT '',
          content TEXT NOT NULL, vector BLOB NOT NULL, tokens TEXT NOT NULL,
          metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE TABLE IF NOT EXISTS settings (
          id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
          role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        """
    )


def _migrate_embeddings_to_blob(db: sqlite3.Connection) -> None:
    """Schema v1: replace the JSON `embedding` column with a float32 `vector` BLOB.

    A database created by this version already has the BLOB column and is left
    alone; only databases from before the migration are rebuilt.
    """
    columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)")}
    if "vector" in columns or "embedding" not in columns:
        return
    db.executescript(
        """
        CREATE TABLE chunks_v1 (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, page INTEGER, section TEXT,
          content TEXT NOT NULL, vector BLOB NOT NULL, tokens TEXT NOT NULL,
          metadata TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    if "embedding" in columns:
        rows = db.execute(
            "SELECT id,document_id,position,page,section,content,embedding,tokens,metadata FROM chunks"
        ).fetchall()
        db.executemany(
            "INSERT INTO chunks_v1(id,document_id,position,page,section,content,vector,tokens,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (r["id"], r["document_id"], r["position"], r["page"], r["section"], r["content"],
                 encode_vector(json.loads(r["embedding"])), r["tokens"], r["metadata"])
                for r in rows
            ],
        )
        if rows:
            logger.info("migrated %d chunk embeddings from JSON text to float32 blobs", len(rows))
    db.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        ALTER TABLE chunks_v1 RENAME TO chunks;
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        """
    )


def _migrate_add_section_path(db: sqlite3.Connection) -> None:
    """Schema v2: chunks gain the section path they sit under.

    Existing rows keep an empty path rather than a guessed one. They score zero
    on the structure signal, which is correct: the document was chunked before
    structure was extracted, and re-ingesting it is the only honest way to fill
    the column in.
    """
    columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)")}
    if "section_path" in columns:
        return
    db.execute("ALTER TABLE chunks ADD COLUMN section_path TEXT NOT NULL DEFAULT ''")
    logger.info("added chunks.section_path; existing chunks carry an empty path until re-indexed")


def init_db() -> None:
    with connect() as db:
        _create_schema(db)
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            _migrate_embeddings_to_blob(db)
        if version < 2:
            _migrate_add_section_path(db)
        if version < SCHEMA_VERSION:
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def meta_get(key: str, default: str = "") -> str:
    found = row("SELECT value FROM meta WHERE key=?", (key,))
    return found["value"] if found else default


def meta_set(key: str, value: str) -> None:
    execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))


def rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()]


def row(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    found = rows(sql, params)
    return found[0] if found else None


def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    with connect() as db:
        db.execute(sql, params)


def json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
