"""Get the top headlines from the New York Times.

This script demonstrates how to use the :pyfunc:`app.tools.browser.browser`
function to control a headless browser, navigate to the NYTimes front page,
capture the raw HTML and then extract the headline strings.

The extraction logic is intentionally simple – it looks for ``<h2>`` tags
which are used for article titles on the NYTimes home page.  If you want a
more robust solution you can replace the regex with ``BeautifulSoup`` or a
more specific CSS selector.

Usage:
    python -m scripts.get_nyt_headlines

The script will print the headlines to stdout.
"""

from __future__ import annotations

import json
import html
import re
from typing import List

# Import the browser helper from the repo tools.  The function returns a
# JSON string that contains a ``result`` dict.
from app.tools.browser import browser


def get_nyt_headlines(url: str = "https://www.nytimes.com/") -> List[str]:
    """Return a list of headline strings from the NYTimes home page.

    Parameters
    ----------
    url:
        The URL to navigate to – by default the NYTimes front page.
    """

    # Start a fresh browser session.
    browser("start")
    try:
        # Navigate to the target URL.
        browser("navigate", url=url)
        # Grab the raw page content.
        html_resp = browser("get_html")
        data = json.loads(html_resp)
        page_html = data["result"]["html"]

        # NYTimes uses <h2> tags for article headlines.
        raw_headlines = re.findall(r"<h2[^>]*>(.*?)</h2>", page_html, flags=re.DOTALL)
        # Clean up any nested tags and HTML entities.
        clean_headlines = [html.unescape(h.strip()).strip() for h in raw_headlines]
        # Remove empty or duplicate entries while preserving order.
        seen = set()
        unique: List[str] = []
        for h in clean_headlines:
            if h and h not in seen:
                seen.add(h)
                unique.append(h)
        return unique
    finally:
        # Ensure the browser is closed even if an error occurs.
        browser("stop")


if __name__ == "__main__":
    headlines = get_nyt_headlines()
    for idx, headline in enumerate(headlines, 1):
        print(f"{idx}. {headline}")
