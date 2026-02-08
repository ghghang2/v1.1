# Browser tool for OpenAI function calling

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

# ---------------------------------------------------------------------------
# BrowserManager – thin wrapper around Playwright
# ---------------------------------------------------------------------------

class BrowserManager:
    """Manage a single Firefox browser session.

    Parameters
    ----------
    headless: bool, default ``True``
        Whether to run Firefox headlessly.
    user_data_dir: str | None
        Path to a persistent user data directory.
    proxy: str | None
        Proxy URL in the form ``http://host:port``.
    """

    def __init__(self, *, headless: bool = True, user_data_dir: Optional[str] = None, proxy: Optional[str] = None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.proxy = proxy
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start(self) -> None:
        if self.browser:
            return  # already started
        self.playwright = sync_playwright().start()
        launch_args: dict = {
            "headless": self.headless,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        if self.proxy:
            launch_args["proxy"] = {"server": self.proxy}
        self.browser = self.playwright.firefox.launch(**launch_args)
        context_args: dict = {}
        if self.user_data_dir:
            context_args["user_data_dir"] = str(Path(self.user_data_dir).expanduser().resolve())
        self.context = self.browser.new_context(**context_args)
        self.page = self.context.new_page()

    def stop(self) -> None:
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        self.page = None

    # ---------------------------------------------------------------------
    # Browser actions
    # ---------------------------------------------------------------------
    def navigate(self, url: str, timeout: int = 30_000) -> None:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.goto(url, timeout=timeout)

    def screenshot(self, path: str, full_page: bool = True, **kwargs) -> bytes:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        img = self.page.screenshot(full_page=full_page, **kwargs)
        Path(path).write_bytes(img)
        return img

    def click(self, selector: str, **kwargs) -> None:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.click(selector, **kwargs)

    def type_text(self, selector: str, text: str, **kwargs) -> None:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.fill(selector, text, **kwargs)

# ---------------------------------------------------------------------------
# Public function for OpenAI function calling
# ---------------------------------------------------------------------------

_mgr: BrowserManager | None = None


def browser(action: str, *, url: str | None = None, path: str | None = None, selector: str | None = None, text: str | None = None, headless: bool | None = None, user_data_dir: str | None = None, proxy: str | None = None, timeout: int | None = None) -> str:
    """Perform a browser action.

    Parameters
    ----------
    action:
        One of ``start``, ``stop``, ``navigate``, ``screenshot``, ``click`` or ``type``.
    url:
        Target URL for navigation.
    path:
        File path for screenshots.
    selector:
        CSS selector for click or type actions.
    text:
        Text to type.
    headless, user_data_dir, proxy:
        Browser configuration used only on ``start``.
    timeout:
        Navigation timeout in ms.
    """
    global _mgr
    try:
        if action == "start":
            if _mgr is None:
                _mgr = BrowserManager(
                    headless=headless if headless is not None else True,
                    user_data_dir=user_data_dir,
                    proxy=proxy,
                )
            _mgr.start()
            return json.dumps({"result": {"action": "start", "status": "ok"}})
        if _mgr is None:
            return json.dumps({"error": "Browser not started. Call start first."})
        if action == "stop":
            _mgr.stop()
            _mgr = None
            return json.dumps({"result": {"action": "stop", "status": "ok"}})
        if action == "navigate":
            if not url:
                raise ValueError("url is required for navigate")
            _mgr.navigate(url, timeout=timeout or 30_000)
            return json.dumps({"result": {"action": "navigate", "url": url}})
        if action == "screenshot":
            if not path:
                raise ValueError("path is required for screenshot")
            _mgr.screenshot(path)
            return json.dumps({"result": {"action": "screenshot", "path": path}})
        if action == "click":
            if not selector:
                raise ValueError("selector is required for click")
            _mgr.click(selector)
            return json.dumps({"result": {"action": "click", "selector": selector}})
        if action == "type":
            if not selector or text is None:
                raise ValueError("selector and text are required for type")
            _mgr.type_text(selector, text)
            return json.dumps({"result": {"action": "type", "selector": selector, "text": text}})
        return json.dumps({"error": f"Unknown action '{action}'"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# OpenAI function call metadata – auto‑discovery reads these
# ---------------------------------------------------------------------------

name = "browser"
func = browser
description = "Control a browser: start, stop, navigate, screenshot, click, type."
# Provide a custom schema so optional fields are optional
schema = {
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "navigate", "screenshot", "click", "type"]},
            "url": {"type": "string"},
            "path": {"type": "string"},
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "headless": {"type": "boolean"},
            "user_data_dir": {"type": "string"},
            "proxy": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["action"],
    }
}

# ---------------------------------------------------------------------------
# CLI wrapper for manual testing – optional
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser tool CLI")
    parser.add_argument("action", choices=["start", "stop", "navigate", "screenshot", "click", "type"], help="Action to perform")
    parser.add_argument("--url")
    parser.add_argument("--path")
    parser.add_argument("--selector")
    parser.add_argument("--text")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--user-data-dir")
    parser.add_argument("--proxy")
    parser.add_argument("--timeout", type=int)
    ns = parser.parse_args()
    print(browser(ns.action, url=ns.url, path=ns.path, selector=ns.selector, text=ns.text, headless=ns.headless, user_data_dir=ns.user_data_dir, proxy=ns.proxy, timeout=ns.timeout))
