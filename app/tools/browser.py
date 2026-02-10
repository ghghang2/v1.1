"""Browser tool for OpenAI function calling.

This module provides a thin wrapper around Playwright's Firefox
browser.  It exposes a :func:`browser` function that can be used by the
ChatGPT agent to perform navigation, screenshots, clicking, typing and
extraction.

The original implementation was copied from the repository.  A small
``get_source`` helper has been added to return the raw HTML of the
currently loaded page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Lazy import of Playwright; the module-level variable will be patched in tests.
sync_playwright: Any | None = None

# ---------------------------------------------------------------------------
# BrowserManager - thin wrapper around Playwright
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
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None

    def start(self) -> None:
        if self.browser:
            return  # already started
        global sync_playwright
        if sync_playwright is None:
            from playwright.sync_api import sync_playwright as _sp
            sync_playwright = _sp
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
    def navigate(self, url: str, timeout: int = 30_000) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        self.page.goto(url, timeout=timeout)
        return {"url": url}

    def screenshot(self, path: str, full_page: bool = True, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        img = self.page.screenshot(full_page=full_page, **kwargs)
        Path(path).write_bytes(img)
        return {"path": path}

    def click(self, selector: str, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        self.page.click(selector, **kwargs)
        return {"selector": selector}

    def type_text(self, selector: str, text: str, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        self.page.fill(selector, text, **kwargs)
        return {"selector": selector, "text": text}

    # ---------------------------------------------------------------------
    # Helper utilities for generic extraction
    # ---------------------------------------------------------------------
    def _elements(self, selector: str):
        """Return Playwright element handles for a selector.

        Raises RuntimeError if no elements found.
        """
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        handles = self.page.locator(selector).element_handles()
        if not handles:
            raise RuntimeError(f"No elements found for selector: {selector}")
        return handles

    def extract(self, selector: str, *, mode: str = "text", multiple: bool = False, attr: str | None = None):
        """Extract content from the page.

        Parameters
        ----------
        selector:
            CSS selector for the elements to extract.
        mode:
            ``text`` (default) returns ``innerText``, ``html`` returns
            ``innerHTML`` and ``attribute`` returns the named attribute.
        multiple:
            If True return a list of all matches.
        attr:
            The attribute name when ``mode == "attribute"``.
        """
        handles = self._elements(selector)
        def _value(h):
            if mode == "text":
                return h.text_content() or ""
            if mode == "html":
                return h.inner_html() or ""
            if mode == "attribute":
                if not attr:
                    raise ValueError("attr must be supplied for mode='attribute'")
                return h.get_attribute(attr) or ""
            raise ValueError(f"Unsupported mode: {mode}")
        values = [_value(h) for h in handles]
        return values if multiple else values[0]

    def evaluate(self, script: str, args: list | None = None):
        """Run arbitrary JS in the page context and return the result."""
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        if args is None:
            args = []
        return self.page.evaluate(script, *args)

    def wait_for(self, selector: str | None = None, timeout: int = 30_000):
        """Wait until an element matching *selector* appears or until *timeout*.

        If *selector* is None, waits for network idle.
        """
        if not self.page:
            raise RuntimeError("Browser not started \u2013 call start() first")
        if selector:
            self.page.wait_for_selector(selector, timeout=timeout)
        else:
            self.page.wait_for_load_state("networkidle", timeout=timeout)

    # ---------------------------------------------------------------------
    # New helper method
    # ---------------------------------------------------------------------
    def get_source(self) -> str:
        """Return the full HTML source of the currently loaded page.

        This method is a thin wrapper around the underlying Playwright
        ``page.content()`` call. It is useful when you need the raw HTML
        for debugging or archival.
        """
        if not self.page:
            raise RuntimeError("Browser not started or page not available.")
        return self.page.content()

# ---------------------------------------------------------------------------
# Public function for OpenAI function calling
# ---------------------------------------------------------------------------

_mgr: BrowserManager | None = None


def browser(action: str, *, url: str | None = None, path: str | None = None, selector: str | None = None, text: str | None = None, headless: bool | None = None, user_data_dir: str | None = None, proxy: str | None = None, timeout: int | None = None, **kwargs) -> str:
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
                _mgr = BrowserManager(headless=headless if headless is not None else True, user_data_dir=user_data_dir, proxy=proxy)
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

        if action == "wait_for":
            sel = kwargs.get("selector")
            to = kwargs.get("timeout", 30_000)
            _mgr.wait_for(selector=sel, timeout=to)
            return json.dumps({"result": {"action": "wait_for", "selector": sel}})

        if action == "extract":
            sel = kwargs.get("selector")
            mode = kwargs.get("mode", "text")
            multiple = kwargs.get("multiple", False)
            attr = kwargs.get("attr")
            if not sel:
                raise ValueError("selector is required for extract")
            res = _mgr.extract(sel, mode=mode, multiple=multiple, attr=attr)
            return json.dumps({"result": {"action": "extract", "result": res}})

        if action == "evaluate":
            script = kwargs.get("script")
            args = kwargs.get("args", [])
            if not script:
                raise ValueError("script is required for evaluate")
            res = _mgr.evaluate(script, args=args)
            return json.dumps({"result": {"action": "evaluate", "result": res}})

        if action == "screenshot":
            if not path:
                raise ValueError("path is required for screenshot")
            full_page = kwargs.get("full_page", True)
            _mgr.screenshot(path, full_page=full_page)
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
