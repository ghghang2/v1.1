"""Tests for the Python browser tool."""

import json
import os
import sys
from pathlib import Path
import tempfile
from unittest.mock import patch

# Ensure repository root is on sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.tools.browser as browser_tool

# Helper to parse JSON result

def _parse(result: str):
    return json.loads(result)

# Dummy classes to mock Playwright behavior
class DummyPage:
    def __init__(self):
        self.url = None
    def goto(self, url, timeout=None):
        self.url = url
    def screenshot(self, full_page=True, **kwargs):
        return b"dummy-image"
    def click(self, selector, **kwargs):
        pass
    def fill(self, selector, text, **kwargs):
        pass

class DummyContext:
    def new_page(self):
        return DummyPage()
    def close(self):
        pass

class DummyBrowser:
    def launch(self, **kwargs):
        # In real Playwright, launch returns a Browser object.
        return self
    def close(self):
        pass
    def new_context(self, **kwargs):
        return DummyContext()

class DummyPlaywright:
    def __init__(self):
        self.firefox = DummyBrowser()
    def stop(self):
        pass

# Patch sync_playwright to return a dummy that mimics the real API
class DummySyncPlaywright:
    def start(self):
        return DummyPlaywright()

@patch('app.tools.browser.sync_playwright', return_value=DummySyncPlaywright())
class TestBrowserTool:
    def test_start_and_stop(self, mock_sync):
        # Ensure start returns ok
        res = browser_tool.browser("start")
        data = _parse(res)
        assert "result" in data
        assert data["result"]["action"] == "start"
        assert data["result"]["status"] == "ok"

        # Stop should work
        res = browser_tool.browser("stop")
        data = _parse(res)
        assert data["result"]["action"] == "stop"

    def test_navigate_without_start(self, mock_sync):
        # Reset _mgr to None
        browser_tool._mgr = None
        res = browser_tool.browser("navigate", url="https://example.com")
        data = _parse(res)
        assert "error" in data

    def test_full_flow(self, mock_sync):
        # Start
        browser_tool.browser("start")
        # Navigate
        res = browser_tool.browser("navigate", url="https://example.com")
        data = _parse(res)
        assert data["result"]["url"] == "https://example.com"

        # Screenshot to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            res = browser_tool.browser("screenshot", path=tmp_path)
            data = _parse(res)
            assert data["result"]["path"] == tmp_path
            # Verify file written
            assert os.path.exists(tmp_path)
        finally:
            os.remove(tmp_path)

        # Click action
        res = browser_tool.browser("click", selector="#btn")
        data = _parse(res)
        assert data["result"]["selector"] == "#btn"

        # Type action
        res = browser_tool.browser("type", selector="#inp", text="hello")
        data = _parse(res)
        assert data["result"]["selector"] == "#inp"
        assert data["result"]["text"] == "hello"

        # Stop at end
        browser_tool.browser("stop")
