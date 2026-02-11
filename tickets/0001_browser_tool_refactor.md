# Browser Tool Refactor Ticket

## Overview

The current `app/tools/browser.py` implements a browser tool that uses Playwright to automate a Firefox browser. However, the original implementation had several shortcomings:

1. **Missing lazy import and error handling** – Importing Playwright at module load time can cause crashes if the dependency is missing. Exceptions were not caught, leading to unstructured errors.
2. **No state persistence** – Each call created a new `BrowserManager`, which caused repeated browser launches and failures when actions were requested before a session was started.
3. **Parameter validation** – Required parameters (`url`, `selector`, `text`, etc.) were not validated, resulting in obscure errors.
4. **No JSON contract** – The tool returned raw strings instead of a JSON payload with `result` or `error` keys, which is required by the OpenAI function‑calling spec.
5. **Missing helper** – A `get_source` helper was needed for debugging.

The refactor will address all these issues and provide a clean, well‑tested implementation.

## Tasks

| Step | Description | Status |
|------|-------------|--------|
| 1 | Create a `BrowserManager` class that lazily imports Playwright, manages a single browser instance, and provides methods for all actions. | completed |
| 2 | Implement `browser` function that dispatches to `BrowserManager` methods, performs argument validation, handles errors, and always returns a JSON string. | completed |
| 3 | Add a `get_source` helper method to `BrowserManager`. | completed |
| 4 | Add comprehensive unit tests for the tool: start/stop, navigate, extract, type, click, screenshot, evaluate, wait_for, and error conditions. | completed |
| 5 | Verify that the tool satisfies the OpenAI function‑calling contract. | completed |

## Acceptance Criteria

* The tool should start a single browser instance and reuse it across calls.
* All actions should validate required arguments and return clear error messages.
* Every call must return a JSON string with either `result` or `error`.
* The `get_source` helper should return the page HTML.
* Unit tests cover typical usage and error scenarios.

## Notes

* Use Playwright's synchronous API.
* Do not run the browser in tests; mock Playwright to avoid external dependencies.

--
