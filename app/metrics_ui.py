"""Streamlit UI for displaying Llama metrics.

The original implementation used an infinite `while True` loop which blocks
Streamlit's execution, preventing the app from rendering.  The updated
version performs a single fetch and displays the data.  For real‑time
updates users can call :func:`display_metrics_panel` again or use a
Streamlit timer.
"""

import streamlit as st
import requests
import time
import argparse
import datetime
import re
import sys
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# Regular‑expression helpers
# --------------------------------------------------------------------------- #
TOKENS_PER_SEC_RE = re.compile(
    r'(?P<value>\d+(?:\.\d+)?)\s+tokens per second',
    re.IGNORECASE
)

# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def parse_log(path_string) -> Tuple[bool, float]:
    """
    Returns (processing, prediction_tps)

    *processing* – True if the server is still busy with the latest request.
    *prediction_tps* – tokens per second for the prediction phase of the
                       most recent eval that we encounter.
    """
    log_path = Path(path_string)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # We'll walk the file from the end to the start.
    # Python's `reversed(list(fp))` is simplest for small-to‑medium logs.
    # For very large logs you might want to use `mmap` or a custom tailer.
    with log_path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()

    is_processing = None
    tps = None

    # Scan from bottom to top
    for raw_line in reversed(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        if is_processing is None:

            if re.search(r'slot update_slots:', raw_line, re.IGNORECASE):
                is_processing = True

            if raw_line.lower() == 'srv  update_slots: all slots are idle':
                is_processing = False
        
        if re.search(r'eval time', raw_line, re.IGNORECASE):
            m = TOKENS_PER_SEC_RE.search(raw_line)
            if m:
                tps = float(m.group('value'))
                break
                
    return is_processing, tps


@st.fragment(run_every=1.0)
def display_metrics_panel() -> None:
    """Display a sidebar panel with the latest metrics.

    The function fetches metrics once and renders them.  It does not
    block the UI, making it safe to call from a Streamlit app.
    """
    processing, prediction_tps = parse_log("llama_server.log")
    # Use emojis to indicate processing state: ⚙️ for running, ⏸️ for idle
    emoji = "⚙️" if processing else "⏸️"
    st.metric(label="processing", value=emoji)
    st.metric(label="tps", value=prediction_tps)
    st.write(f"Updated: {time.strftime('%H:%M:%S')}")
