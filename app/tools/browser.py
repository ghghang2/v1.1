"""Stateless Browser Tool for OpenAI Function Calling.

This module provides a robust, stateless wrapper around Playwright.
Each call launches a fresh browser instance, performs a sequence of actions,
returns the result, and cleans up immediately. This prevents threading
crashes and "protocol" errors common in long-running sessions.
"""

from __future__ import annotations

import json
from typing import Any, List, Dict, Optional
from playwright.sync_api import sync_playwright

def browser(url: str, actions: Optional[List[Dict[str, Any]]] = None, selector: Optional[str] = None, **kwargs) -> str:
    """
    Visit a webpage, perform optional interactions, and extract content.
    
    This tool is STATELESS: It opens a browser, runs your commands, and closes.
    You cannot "keep" the browser open between calls.
    
    Parameters
    ----------
    url : str
        The URL to visit.
    actions : List[Dict], optional
        A list of interactions to perform before extracting data.
        Supported action types:
        - {"type": "click", "selector": "..."}
        - {"type": "type", "selector": "...", "text": "..."}
        - {"type": "wait", "selector": "..."} (or "timeout": ms)
        - {"type": "screenshot", "path": "..."}
    selector : str, optional
        A specific CSS selector to extract text from. If None, returns the full page text.
    **kwargs :
        Handles "hallucinated" nested JSON arguments from some LLMs.

    Returns
    -------
    str
        JSON string containing the extracted text, source, or operation results.
    """

    # --- 1. ARGUMENT UNPACKING (Fixes the "url is required" error) ---
    # LLMs sometimes wrap args inside a 'kwargs' string. We unpack them here.
    if kwargs.get("kwargs") and isinstance(kwargs["kwargs"], str):
        try:
            extra_args = json.loads(kwargs["kwargs"])
            if isinstance(extra_args, dict):
                if not url: url = extra_args.get("url")
                if not actions: actions = extra_args.get("actions")
                if not selector: selector = extra_args.get("selector")
        except json.JSONDecodeError:
            pass

    if not url:
        return json.dumps({"error": "URL is required."})

    # --- 2. STATELESS EXECUTION (Fixes the "thread" error) ---
    try:
        with sync_playwright() as p:
            # Launch fresh for every single call
            browser = p.firefox.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()

            # Navigate
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
            except Exception as e:
                return json.dumps({"error": f"Navigation failed: {str(e)}"})

            # Perform Actions (if any)
            interaction_log = []
            if actions:
                for act in actions:
                    act_type = act.get("type")
                    try:
                        if act_type == "click":
                            page.click(act["selector"], timeout=5000)
                            interaction_log.append(f"Clicked {act['selector']}")
                        elif act_type == "type":
                            page.fill(act["selector"], act["text"], timeout=5000)
                            interaction_log.append(f"Typed into {act['selector']}")
                        elif act_type == "wait":
                            if "selector" in act:
                                page.wait_for_selector(act["selector"], timeout=10000)
                            elif "timeout" in act:
                                page.wait_for_timeout(act["timeout"])
                            interaction_log.append(f"Waited for {act.get('selector') or act.get('timeout')}")
                        elif act_type == "screenshot":
                            path = act.get("path", "screenshot.png")
                            page.screenshot(path=path)
                            interaction_log.append(f"Screenshot saved to {path}")
                    except Exception as e:
                        interaction_log.append(f"Error during {act_type}: {str(e)}")

            # Extract Content
            content = ""
            if selector:
                try:
                    # Try to get specific element
                    page.wait_for_selector(selector, timeout=5000)
                    elements = page.locator(selector).all_inner_texts()
                    content = "\n".join(elements)
                except:
                    content = f"Element '{selector}' not found."
            else:
                # Default: Get the main readable text
                content = page.evaluate("() => document.body.innerText")

            browser.close()
            
            return json.dumps({
                "status": "success",
                "url": url,
                "interactions": interaction_log,
                "content": content[:5000]  # Truncate to avoid context limit overflow
            })

    except Exception as global_ex:
        return json.dumps({"error": f"Browser tool error: {str(global_ex)}"})

# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------
func = browser
name = "browser"
description = """
Safe, stateless browser tool. Use this to visit a website and extract content.
This tool CANNOT maintain a session. Every call is a fresh visit.

Inputs:
- url (required): The website to visit.
- selector (optional): A CSS selector to extract specific text. If omitted, returns full page text.
- actions (optional): A list of actions to perform BEFORE extraction. Use this to click buttons or log in.
  Example actions: 
  [
    {"type": "click", "selector": "#cookie-accept"}, 
    {"type": "type", "selector": "#search", "text": "AI News"},
    {"type": "wait", "selector": "#results"}
  ]
"""

__all__ = ["browser", "func", "name", "description"]

# """Browser tool for OpenAI function calling.

# This module provides a thin wrapper around Playwright's Firefox
# browser.  It exposes a :func:`browser` function that can be used by the
# ChatGPT agent to perform navigation, screenshots, clicking, typing and
# extraction.

# The original implementation was copied from the repository.  A small
# ``get_source`` helper has been added to return the raw HTML of the
# currently loaded page.
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any, Optional

# # Lazy import of Playwright; the module-level variable will be patched in tests.
# sync_playwright: Any | None = None

# # ---------------------------------------------------------------------------
# # BrowserManager - thin wrapper around Playwright
# # ---------------------------------------------------------------------------

# class BrowserManager:
#     """Manage a single Firefox browser session.

#     Parameters
#     ----------
#     headless: bool, default ``True``
#         Whether to run Firefox headlessly.
#     user_data_dir: str | None
#         Path to a persistent user data directory.
#     proxy: str | None
#         Proxy URL in the form ``http://host:port``.
#     """

#     def __init__(self, *, headless: bool = True, user_data_dir: Optional[str] = None, proxy: Optional[str] = None):
#         self.headless = headless
#         self.user_data_dir = user_data_dir
#         self.proxy = proxy
#         self.playwright: Any | None = None
#         self.browser: Any | None = None
#         self.context: Any | None = None
#         self.page: Any | None = None

#     def start(self) -> None:
#         if self.browser:
#             return  # already started
#         global sync_playwright
#         if sync_playwright is None:
#             from playwright.sync_api import sync_playwright as _sp
#             sync_playwright = _sp
#         self.playwright = sync_playwright().start()
#         launch_args: dict = {
#             "headless": self.headless,
#             "args": ["--no-sandbox", "--disable-dev-shm-usage"],
#         }
#         if self.proxy:
#             launch_args["proxy"] = {"server": self.proxy}
#         self.browser = self.playwright.firefox.launch(**launch_args)
#         context_args: dict = {}
#         if self.user_data_dir:
#             context_args["user_data_dir"] = str(Path(self.user_data_dir).expanduser().resolve())
#         self.context = self.browser.new_context(**context_args)
#         self.page = self.context.new_page()

#     def stop(self) -> None:
#         if self.context:
#             self.context.close()
#             self.context = None
#         if self.browser:
#             self.browser.close()
#             self.browser = None
#         if self.playwright:
#             self.playwright.stop()
#             self.playwright = None
#         self.page = None

#     # ---------------------------------------------------------------------
#     # Browser actions
#     # ---------------------------------------------------------------------
#     def navigate(self, url: str, timeout: int = 30_000) -> dict:
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         self.page.goto(url, timeout=timeout)
#         return {"url": url}

#     def screenshot(self, path: str, full_page: bool = True, **kwargs) -> dict:
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         img = self.page.screenshot(full_page=full_page, **kwargs)
#         Path(path).write_bytes(img)
#         return {"path": path}

#     def click(self, selector: str, **kwargs) -> dict:
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         self.page.click(selector, **kwargs)
#         return {"selector": selector}

#     def type_text(self, selector: str, text: str, **kwargs) -> dict:
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         self.page.fill(selector, text, **kwargs)
#         return {"selector": selector, "text": text}

#     # ---------------------------------------------------------------------
#     # Helper utilities for generic extraction
#     # ---------------------------------------------------------------------
#     def _elements(self, selector: str):
#         """Return Playwright element handles for a selector.

#         Raises RuntimeError if no elements found.
#         """
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         handles = self.page.locator(selector).element_handles()
#         if not handles:
#             raise RuntimeError(f"No elements found for selector: {selector}")
#         return handles

#     def extract(self, selector: str, *, mode: str = "text", multiple: bool = False, attr: str | None = None):
#         """Extract content from the page.

#         Parameters
#         ----------
#         selector:
#             CSS selector for the elements to extract.
#         mode:
#             ``text`` (default) returns ``innerText``, ``html`` returns
#             ``innerHTML`` and ``attribute`` returns the named attribute.
#         multiple:
#             If True return a list of all matches.
#         attr:
#             The attribute name when ``mode == "attribute"``.
#         """
#         handles = self._elements(selector)
#         def _value(h):
#             if mode == "text":
#                 return h.text_content() or ""
#             if mode == "html":
#                 return h.inner_html() or ""
#             if mode == "attribute":
#                 if not attr:
#                     raise ValueError("attr must be supplied for mode='attribute'")
#                 return h.get_attribute(attr) or ""
#             raise ValueError(f"Unsupported mode: {mode}")
#         values = [_value(h) for h in handles]
#         return values if multiple else values[0]

#     def evaluate(self, script: str, args: list | None = None):
#         """Run arbitrary JS in the page context and return the result."""
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         if args is None:
#             args = []
#         return self.page.evaluate(script, *args)

#     def wait_for(self, selector: str | None = None, timeout: int = 30_000):
#         """Wait until an element matching *selector* appears or until *timeout*.

#         If *selector* is None, waits for network idle.
#         """
#         if not self.page:
#             raise RuntimeError("Browser not started \u2013 call start() first")
#         if selector:
#             self.page.wait_for_selector(selector, timeout=timeout)
#         else:
#             self.page.wait_for_load_state("networkidle", timeout=timeout)

#     # ---------------------------------------------------------------------
#     # New helper method
#     # ---------------------------------------------------------------------
#     def get_source(self) -> str:
#         """Return the full HTML source of the currently loaded page.

#         This method is a thin wrapper around the underlying Playwright
#         ``page.content()`` call. It is useful when you need the raw HTML
#         for debugging or archival.
#         """
#         if not self.page:
#             raise RuntimeError("Browser not started or page not available.")
#         return self.page.content()

# # ---------------------------------------------------------------------------
# # Public function for OpenAI function calling
# # ---------------------------------------------------------------------------

# _mgr: BrowserManager | None = None

# def browser(action: str, *, url: str | None = None, path: str | None = None, selector: str | None = None, text: str | None = None, headless: bool | None = None, user_data_dir: str | None = None, proxy: str | None = None, timeout: int | None = None, **kwargs) -> str:
#     """Perform a browser action using Playwright.

#     This function is designed for OpenAI function calling.  The caller
#     passes an ``action`` keyword that determines which operation to
#     perform.  All other keyword arguments are interpreted based on the
#     chosen action.

#     Supported actions and their required/optional parameters:

#     ``start``
#         Initialise a new Firefox browser instance.
#         Parameters:
#             * ``headless`` (bool, optional) – Run the browser headlessly.
#             * ``user_data_dir`` (str, optional) – Path to a persistent user
#               data directory.
#             * ``proxy`` (str, optional) – Proxy URL in the form
#               ``http://host:port``.

#     ``stop``
#         Close the current browser session.

#     ``navigate``
#         Navigate to a URL.
#         Parameters:
#             * ``url`` (str, required) – Target URL.
#             * ``timeout`` (int, optional) – Navigation timeout in milliseconds
#               (default 30 000 ms).

#     ``wait_for``
#         Wait for a selector to appear or for the page to become idle.
#         Parameters:
#             * ``selector`` (str, optional) – CSS selector to wait for.
#             * ``timeout`` (int, optional) – Timeout in ms.

#     ``extract``
#         Retrieve text, HTML or an attribute from matched elements.
#         Parameters:
#             * ``selector`` (str, required) – CSS selector.
#             * ``mode`` (str, optional) – ``text`` (default), ``html`` or
#               ``attribute``.
#             * ``multiple`` (bool, optional) – Return a list of all matches.
#             * ``attr`` (str, optional) – Attribute name when ``mode`` is
#               ``attribute``.

#     ``evaluate``
#         Execute arbitrary JavaScript in the page context.
#         Parameters:
#             * ``script`` (str, required) – JavaScript source.
#             * ``args`` (list, optional) – Arguments to pass to the script.

#     ``screenshot``
#         Capture a screenshot of the current page.
#         Parameters:
#             * ``path`` (str, required) – File path to save the image.
#             * ``full_page`` (bool, optional) – Capture the full scrollable
#               page (default ``True``).

#     ``click``
#         Click an element identified by a CSS selector.
#         Parameters:
#             * ``selector`` (str, required) – CSS selector.

#     ``type``
#         Type text into an input field.
#         Parameters:
#             * ``selector`` (str, required) – CSS selector.
#             * ``text`` (str, required) – Text to type.

#     The function returns a JSON string containing either a ``result``
#     object or an ``error`` message.  All internal exceptions are caught
#     and converted to an error JSON payload.
#     """
#     global _mgr
#     try:
#         if action == "start":
#             if _mgr is None:
#                 _mgr = BrowserManager(headless=headless if headless is not None else True, user_data_dir=user_data_dir, proxy=proxy)
#             _mgr.start()
#             return json.dumps({"result": {"action": "start", "status": "ok"}})

#         if _mgr is None:
#             return json.dumps({"error": "Browser not started. Call start first."})

#         if action == "stop":
#             _mgr.stop()
#             _mgr = None
#             return json.dumps({"result": {"action": "stop", "status": "ok"}})

#         if action == "navigate":
#             if not url:
#                 raise ValueError("url is required for navigate")
#             _mgr.navigate(url, timeout=timeout or 30_000)
#             return json.dumps({"result": {"action": "navigate", "url": url}})

#         if action == "wait_for":
#             sel = selector
#             to = timeout or 30_000
#             _mgr.wait_for(selector=sel, timeout=to)
#             return json.dumps({"result": {"action": "wait_for", "selector": sel}})

#         if action == "extract":
#             sel = selector
#             mode = kwargs.get("mode", "text")
#             multiple = kwargs.get("multiple", False)
#             attr = kwargs.get("attr")
#             if not sel:
#                 raise ValueError("selector is required for extract")
#             res = _mgr.extract(sel, mode=mode, multiple=multiple, attr=attr)
#             return json.dumps({"result": {"action": "extract", "result": res}})

#         if action == "evaluate":
#             script = kwargs.get("script")
#             args = kwargs.get("args", [])
#             if not script:
#                 raise ValueError("script is required for evaluate")
#             res = _mgr.evaluate(script, args=args)
#             return json.dumps({"result": {"action": "evaluate", "result": res}})

#         if action == "screenshot":
#             if not path:
#                 raise ValueError("path is required for screenshot")
#             full_page = kwargs.get("full_page", True)
#             _mgr.screenshot(path, full_page=full_page)
#             return json.dumps({"result": {"action": "screenshot", "path": path}})

#         if action == "click":
#             if not selector:
#                 raise ValueError("selector is required for click")
#             _mgr.click(selector)
#             return json.dumps({"result": {"action": "click", "selector": selector}})

#         if action == "type":
#             if not selector or text is None:
#                 raise ValueError("selector and text are required for type")
#             _mgr.type_text(selector, text)
#             return json.dumps({"result": {"action": "type", "selector": selector, "text": text}})

#         return json.dumps({"error": f"Unknown action '{action}'"})
#     except Exception as exc:
#         return json.dumps({"error": str(exc)})

# # ---------------------------------------------------------------------------
# # Expose attributes for tool discovery
# # ---------------------------------------------------------------------------
# func = browser
# name = "browser"
# description = browser.__doc__ or "Perform browser actions via Playwright."

# __all__ = ["browser", "func", "name", "description"]
