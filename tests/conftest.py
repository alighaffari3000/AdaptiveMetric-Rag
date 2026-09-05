"""Point every test at a throwaway database before the app package is imported."""

import os
import tempfile
from pathlib import Path

import pytest

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="amr-tests-"))
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """A private database file for one test, with the index reset around it."""
    from app import database
    from app.index import index

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.close_thread_connection()
    database.init_db()
    index.invalidate()
    try:
        yield database
    finally:
        database.close_thread_connection()
        index.invalidate()
