# app/tools/apply_patch.py
"""Tool that applies a unified diff to a target file.

The tool is intentionally lightweight and mirrors the interface used by
other tools in :mod:`app.tools`.  It exposes a single callable named
``func`` together with ``name`` and ``description`` attributes so that
``app.tools.__init__`` can automatically discover it.

The function expects two string arguments:

``target_path``
    The path (relative to the repository root) of the file to patch.

``patch_text``
    A unified diff that will be applied to ``target_path``.

The function returns a JSON string.  On success a ``result`` key is
provided; on failure an ``error`` key is returned.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_repo_root() -> Path:
    """Return the repository root.

    The module lives in ``app/tools``.  The repo root is two levels up
    from this file.
    """
    return Path(__file__).resolve().parents[2]


def _safe_resolve(repo_root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` against ``repo_root`` and ensure it does not
    escape the repository root.
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target

# ---------------------------------------------------------------------------
# The actual tool implementation
# ---------------------------------------------------------------------------

def _apply_patch(target_path: str, patch_text: str) -> str:
    """Apply a unified diff to ``target_path``.

    Parameters
    ----------
    target_path
        Path relative to the repository root.
    patch_text
        Unified diff to apply.

    Returns
    -------
    str
        JSON string.  On success:

        .. code-block:: json

            { "result": "Patch applied to <target_path>" }

        On failure:

        .. code-block:: json

            { "error": "<exception message>" }
    """
    try:
        repo_root = _resolve_repo_root()
        target = _safe_resolve(repo_root, target_path)

        if not target.is_file():
            raise FileNotFoundError(f"Target file does not exist: {target_path}")

        # Write patch text to a temporary file
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp_patch:
            tmp_patch.write(patch_text)
            patch_file_path = tmp_patch.name

        # Run the external patch command
        cmd = [
            "patch",
            "-p0",
            "-i", patch_file_path,
            str(target),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up the temporary patch file
        os.remove(patch_file_path)

        if result.returncode == 0:
            return json.dumps({"result": f"Patch applied to {target_path}"})
        else:
            # Prefer stderr over stdout for error messages
            err_msg = result.stderr.strip() or result.stdout.strip()
            if not err_msg:
                err_msg = f"Patch command failed with exit code {result.returncode}"
            return json.dumps({"error": err_msg})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# Public attributes for auto-discovery
# ---------------------------------------------------------------------------
func = _apply_patch
name = "apply_patch"
description = (
    "Apply a unified diff to a target file under the repository root. "
    "Returns a JSON string with either a `result` key on success or an `error` "
    "key on failure."
)

__all__ = ["func", "name", "description"]

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
