"""Tests for the lightweight SQLite persistence layer.

The tests are intentionally minimal – they verify that the helper
functions behave as expected and that the database schema is created
correctly.  No external dependencies beyond the standard library are
required.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

# Import the module under test
import app.db as db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary database file used for a single test.

    ``app.db`` uses a module‑level constant :data:`DB_PATH`.  The tests
    monkeypatch that constant so that each test gets an isolated
    database.
    """

    path = tmp_path / "chat_history.db"
    original_path = db.DB_PATH
    db.DB_PATH = path
    yield path
    # Restore the original value so other tests are not affected
    db.DB_PATH = original_path


def test_init_db_creates_table(db_path: Path) -> None:
    """Verify that :func:`init_db` creates the expected table.

    The test checks the table name and a few column names.  It also
    verifies that the operation is idempotent by calling ``init_db``
    twice.
    """

    # First call – should create the table
    db.init_db()
    # Second call – should be no‑op
    db.init_db()

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("PRAGMA table_info(chat_log)")
        columns = [row[1] for row in cur.fetchall()]
    expected = [
        "id",
        "session_id",
        "role",
        "content",
        "tool_id",
        "tool_name",
        "tool_args",
        "ts",
    ]
    assert columns == expected, f"Unexpected table columns: {columns}"


def test_log_and_load_history(db_path: Path) -> None:
    """Test that messages are persisted and can be retrieved in order."""

    db.init_db()
    session = "test-session"
    db.log_message(session, "user", "Hello")
    db.log_message(session, "assistant", "Hi there!")
    history = db.load_history(session)
    assert len(history) == 2
    assert history[0][:2] == ("user", "Hello")
    assert history[1][:2] == ("assistant", "Hi there!")


def test_log_tool_msg(db_path: Path) -> None:
    """Ensure tool messages are stored in the correct rows."""

    db.init_db()
    session = "tool-session"
    tool_id = "tool-1"
    tool_name = "echo"
    tool_args = "{}"
    content = "Tool executed"
    db.log_tool_msg(session, tool_id, tool_name, tool_args, content)
    history = db.load_history(session)
    # The helper creates two rows: one for the tool invocation and one for the result
    assert len(history) == 2
    # First row – assistant with tool metadata
    assert history[0] == (
        "assistant",
        "",
        tool_id,
        tool_name,
        tool_args,
    )
    # Second row – tool output
    assert history[1] == ("tool", content, tool_id, None, None)
