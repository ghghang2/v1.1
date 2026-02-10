"""Utility script to fetch the NYTimes homepage and save its full HTML to a file.

Usage
-----
    python -m scripts.save_nytimes_html

The script uses the :class:`app.tools.browser.BrowserManager` in headless mode.
It writes the resulting HTML to ``nytimes.html`` in the current working
directory.  The file is overwritten if it already exists.
"""

import os
from app.tools.browser import BrowserManager

NYTIMES_URL = "https://www.nytimes.com"
OUTPUT_FILE = "nytimes.html"


def main() -> None:
    # Start a headless browser instance
    manager = BrowserManager(headless=True)
    try:
        manager.start()
        # Navigate to the NYTimes homepage
        print(f"Navigating to {NYTIMES_URL}")
        manager.navigate(NYTIMES_URL)
        # Grab the full page source
        print("Extracting full page source...")
        html = manager.get_source()
        # Write to disk
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
            fp.write(html)
        print(f"Full HTML written to {os.path.abspath(OUTPUT_FILE)}")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
