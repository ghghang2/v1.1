"""
All tests for the apply_patch tool consolidated into a single file.
"""

import json
import os
import sys
from pathlib import Path
import pytest

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.apply_patch import apply_patch


def _relative_path(relative: str) -> str:
    """Return a path relative to the repository root."""
    repo_root = Path(__file__).resolve().parents[1]
    return str(Path(relative).relative_to(repo_root))


def test_apply_patch_update_with_context(tmp_path: Path):
    """Update a file using a diff that includes context lines."""
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "tests" / "tmp_context_test.txt"
    target_path.write_text("Hello\nWorld\nFoo", encoding="utf-8")
    relative = _relative_path(target_path)
    # Diff that changes World -> Universe with context
    diff = "@@ -2,1 +2,1 @@\n-World\n+Universe\n"
    result_json = apply_patch(relative, "update", diff)
    result = json.loads(result_json)
    assert "result" in result
    assert result["result"] == f"File updated: {relative}"
    assert target_path.read_text(encoding="utf-8") == "Hello\nUniverse\nFoo"
    target_path.unlink()


def test_apply_patch_create_nested_dirs(tmp_path: Path):
    """Creating a file in nested directories should create the directories."""
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "tests" / "nested" / "dir" / "test_nested.txt"
    relative = _relative_path(target_path)
    diff = "+Nested line 1\n+Nested line 2\n"
    result_json = apply_patch(relative, "create", diff)
    result = json.loads(result_json)
    assert "result" in result
    assert result["result"] == f"File created: {relative}"
    assert target_path.read_text(encoding="utf-8") == "Nested line 1\nNested line 2"
    target_path.unlink()
    # Clean up directories
    target_path.parent.rmdir()
    target_path.parent.parent.rmdir()


def test_apply_patch_error_invalid_diff(tmp_path: Path):
    """Providing a create diff that does not start with '+' should error."""
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "tests" / "tmp_invalid_diff.txt"
    relative = _relative_path(target_path)
    diff = "Hello without plus"
    result_json = apply_patch(relative, "create", diff)
    result = json.loads(result_json)
    assert "error" in result
    assert "Invalid Add Line" in result["error"]
    if target_path.exists():
        target_path.unlink()


def test_apply_patch_update_no_change(tmp_path: Path):
    """Updating with a diff that results in no changes should succeed."""
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "tests" / "tmp_no_change.txt"
    content = "Line1\nLine2"
    target_path.write_text(content, encoding="utf-8")
    relative = _relative_path(target_path)
    # Diff that matches but makes no change
    diff = "@@ -1,2 +1,2 @@\n-Line1\n-Line2\n+Line1\n+Line2\n"
    result_json = apply_patch(relative, "update", diff)
    result = json.loads(result_json)
    assert "result" in result
    assert result["result"] == f"File updated: {relative}"
    assert target_path.read_text(encoding="utf-8") == content
    target_path.unlink()


def test_apply_patch_delete_nonexistent(tmp_path: Path):
    """Deleting a non\u2011existent file should succeed without error."""
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "tests" / "tmp_nonexistent.txt"
    relative = _relative_path(target_path)
    # Ensure file does not exist
    if target_path.exists():
        target_path.unlink()
    result_json = apply_patch(relative, "delete", "")
    result = json.loads(result_json)
    assert "result" in result
    assert result["result"] == f"File deleted: {relative}"
    assert not target_path.exists()
