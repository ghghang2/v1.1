"""Integration test for the real Playwright browser tool.

This test is deliberately kept minimal – it simply starts a headless browser,
navigates to a public page, takes a screenshot and verifies that the file
exists.  It runs only if Playwright is available in the environment.
"""

import os
import tempfile

import pytest

from app.tools.browser import BrowserManager


def test_browser_real_basic_flow():
    """Start the browser, navigate, screenshot and stop."""
    tool = BrowserManager(headless=True)
    try:
        tool.start()
    except Exception as e:
        pytest.skip(f"Skipping real browser test due to environment issue: {e}")
    # Navigate to a public site; use example.com which is always available.
    result = tool.navigate("https://example.com")
    assert result["url"] == "https://example.com"

    # Take a screenshot.
    screenshot_path = os.path.join(tempfile.gettempdir(), "screenshot.png")
    result = tool.screenshot(screenshot_path)
    assert os.path.exists(result["path"])

    tool.stop()
