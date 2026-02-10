# Apply Patch Review Ticket

This ticket documents the problems that were discovered while applying a series of `apply_patch` edits to the data‑fetching script (`fetch_johnnyanmac.sh`). The goal is to provide a clear, concise list of issues so that future contributors can understand what went wrong and how to address it.

---

## 1. Incorrect CSV to line conversion
* **Patch**: Replaced `tr ',' '\n'` but did not handle the case where the CSV string is empty.
* **Issue**: Empty line was created, leading to an unnecessary API request for an empty ID.
* **Fix**: Skip empty lines with `[[ -z "$ID" ]] && continue`.

## 2. Wrong Algolia query format
* **Patch**: Multiple attempts used `?query=user:%s` or `?tags=comment&hitsPerPage=1000&user=$USERNAME`.
* **Issue**: These URLs are malformed; Algolia expects `query=user:<username>`.
* **Fix**: Use `?query=user:${USERNAME}&tags=comment&hitsPerPage=1000&page=${PAGE}`.

## 3. Inadequate extraction of comment IDs
* **Patch**: Used `.hits[] | .text // empty` to get IDs.
* **Issue**: `text` does not exist in the Algolia response; the correct field is `objectID`.
* **Fix**: Extract with `.hits[]? | .objectID`.

## 4. Missing comment text extraction
* **Patch**: After fetching comment JSON, sometimes the script used the wrong jq expression.
* **Issue**: Resulting file contained `null` or empty strings.
* **Fix**: Use `jq -r '.text // empty'` on the comment JSON.

## 5. No pagination or limit handling for comments
* **Patch**: Looped over pages but never respected `MAX_COMMENTS`.
* **Issue**: Potentially fetched thousands of comments, causing long runtimes and rate‑limit hits.
* **Fix**: Keep a counter and break out of the loop when the limit is reached.

## 6. No rate‑limit throttling
* **Patch**: No delays between API calls.
* **Issue**: Risk of being throttled or blocked by the HN/Algolia APIs.
* **Fix**: Add a configurable sleep (e.g., `sleep 0.1` after each request).

## 7. Lack of error handling
* **Patch**: No checks for HTTP status or JSON parsing errors.
* **Issue**: Silent failures may produce incomplete data.
* **Fix**: Use `curl -f` and check the exit status; validate JSON with `jq`.

## 8. Repetitive patching and code duplication
* **Patch**: Multiple small patches applied sequentially.
* **Issue**: Script became fragmented, hard to read, and more error‑prone.
* **Fix**: Refactor into functions (`fetch_user`, `fetch_story_ids`, `fetch_story`, `fetch_comment_ids`, `fetch_comment_text`).

## 9. Environment variable misuse
* **Patch**: Set `MAX_STORIES` and `MAX_COMMENTS` but did not provide defaults or validate them.
* **Issue**: Could lead to negative values or unintended large numbers.
* **Fix**: Set sensible defaults and validate input.

---

This ticket serves as a record of the challenges encountered during the patching process and provides actionable fixes for each issue.