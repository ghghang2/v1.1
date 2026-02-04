import json
from pathlib import Path
import sys, os

# Ensure repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.apply_patch import func as apply_patch
from app.tools.create_file import func as create_file


def test_apply_patch_success(tmp_path):
    """Apply a patch to an existing file and verify the content changes."""
    # Path relative to repository root
    relative_path = "tests/tmp_apply_patch.txt"
    original_content = "Line 1\nLine 2\n"
    # Create the file using the create_file tool
    create_file(relative_path, original_content)

    # Build a unified diff that replaces "Line 2" with "Line 2 modified"
    diff = (
        f"--- a/{relative_path}\n"
        f"+++ b/{relative_path}\n"
        "@@ -1,2 +1,2 @@\n"
        " Line 1\n"
        "-Line 2\n"
        "+Line 2 modified\n"
    )

    # Apply the patch
    result_json = apply_patch(relative_path, diff)
    result = json.loads(result_json)

    # Verify success
    assert "result" in result, f"Tool returned error: {result.get('error')}"
    assert result["result"] == f"Patch applied to {relative_path}"

    # Verify the file content
    repo_root = Path(__file__).resolve().parents[1]
    target_file = repo_root / relative_path
    assert target_file.exists(), "Target file missing after patch"
    expected_content = "Line 1\nLine 2 modified\n"
    assert target_file.read_text(encoding="utf-8") == expected_content


def test_apply_patch_nonexistent_file(tmp_path):
    """Applying a patch to a file that does not exist should return an error."""
    relative_path = "tests/nonexistent_file.txt"
    diff = "--- a/nonexistent_file.txt\n+++ b/nonexistent_file.txt\n@@\n"
    result_json = apply_patch(relative_path, diff)
    result = json.loads(result_json)
    assert "error" in result, "Expected an error for nonexistent file"
    assert "does not exist" in result["error"], f"Unexpected error message: {result["error"]}"