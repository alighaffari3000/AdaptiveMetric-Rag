"""Where chunks come from, stated as an interface instead of assumed.

Retrieval never touches storage: it scores an in-memory snapshot that the index
builds once. But the index built that snapshot with a SQL string of its own, and
`parent_windows` wrote another, so "swap SQLite for pgvector" meant editing
retrieval-adjacent code and hoping nothing else had its own query.

This is the seam, and it is deliberately narrow - three reads, no writes, no
vector search. A store is asked for rows; ranking stays in this application,
where it can be measured. A future Postgres or Qdrant backing would implement
`ChunkStore` and change nothing above it. Nothing else implements it today, and
this file does not pretend otherwise: it exists so the boundary is somewhere
specific rather than spread across three modules.
"""

from __future__ import annotations

from typing import Any, Protocol

from . import database

CHUNK_QUERY = (
    "SELECT c.id,c.document_id,c.position,c.page,c.page_end,c.section,c.section_path,c.parent_id,"
    "c.content,c.vector,c.tokens,c.metadata,d.name document_name,d.type document_type "
    "FROM chunks c JOIN documents d ON d.id=c.document_id "
    "ORDER BY c.rowid"
)


class ChunkStore(Protocol):
    """The whole of what retrieval needs from storage."""

    def chunk_rows(self) -> list[dict[str, Any]]:
        """Every chunk, with its vector, tokens and document metadata."""

    def parents(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        """The parent windows for these ids, keyed by id."""

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """One chunk with the fields a reader needs, or None."""


class SqliteStore:
    """The store this application ships with."""

    def chunk_rows(self) -> list[dict[str, Any]]:
        return database.rows(CHUNK_QUERY)

    def parents(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not parent_ids:
            return {}
        placeholders = ",".join("?" * len(parent_ids))
        found = database.rows(
            f"SELECT id,content,page_start,page_end,section_path FROM parents WHERE id IN ({placeholders})",
            tuple(parent_ids),
        )
        return {row["id"]: row for row in found}

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return database.row(
            "SELECT c.id,c.document_id,c.page,c.page_end,c.section,c.section_path,c.content,"
            "d.name document_name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",
            (chunk_id,),
        )


store: ChunkStore = SqliteStore()
