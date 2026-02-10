# Issue and Improvement Tickets

The following tickets capture the known problems and potential improvements identified during the development of the data‑fetching script (`fetch_johnnyanmac.sh`). Each ticket is described in detail so that future contributors understand the context, why it matters, and how it can be addressed.

---

1. **Incorrect Algolia query syntax**
   * **Problem**: The script attempted to query Algolia using `?tags=comment&hitsPerPage=1000&user=$USERNAME` or `?query=user:%s…`. These URLs are malformed or use the wrong parameter (`user` is not a supported query field).
   * **Impact**: No comments were retrieved, causing an empty `*_comments.json` file.
   * **Resolution**: Use `?query=user:${USERNAME}&tags=comment&hitsPerPage=1000&page=${PAGE}`.

2. **Broken URL interpolation**
   * **Problem**: Several attempts used `printf` or `sed` to inject the username into the URL, which either escaped incorrectly or produced an empty string.
   * **Impact**: API requests failed silently.
   * **Resolution**: Construct URLs directly with shell variable expansion, e.g. `curl "https://…?query=user:${USERNAME}&…"`.

3. **Wrong jq filter for comment IDs**
   * **Problem**: The script used `.hits[] | .text // empty` to pull IDs, but Algolia’s response contains `objectID` instead of `text`.
   * **Impact**: The ID list was empty, so subsequent HN API calls never ran.
   * **Resolution**: Extract IDs with `.hits[]? | .objectID` and then fetch each comment via the HN API.

4. **Missing extraction of comment text**
   * **Problem**: Even after fetching the comment JSON, the script sometimes used the wrong jq expression to pull the `text` field.
   * **Impact**: The resulting `*_comments.json` contained `null` or empty strings.
   * **Resolution**: Use `jq -r '.text // empty'` on the comment JSON.

5. **No pagination handling for comments**
   * **Problem**: The script looped over pages but never respected the `MAX_COMMENTS` limit, potentially fetching thousands of comments.
   * **Impact**: Long runtimes and excessive API usage.
   * **Resolution**: Keep a counter and break out of the page loop when `COUNT >= MAX_COMMENTS`.

6. **Empty line handling for story IDs**
   * **Problem**: Converting the CSV list of story IDs to a file introduced blank lines that were processed as empty IDs.
   * **Impact**: Extra API calls and possible errors.
   * **Resolution**: Skip empty lines with `[[ -z "$ID" ]] && continue`.

7. **Insufficient rate‑limit throttling**
   * **Problem**: The script did not pause between requests, risking throttling by HN or Algolia.
   * **Impact**: Potential request failures or bans.
   * **Resolution**: Add a small sleep (`sleep 0.1` or configurable `SLEEP`) after each API call.

8. **Script maintainability and readability**
   * **Problem**: Multiple patches were applied piecemeal, leading to duplicated logic and a harder‑to‑read script.
   * **Impact**: Future modifications are error‑prone.
   * **Resolution**: Refactor into functions (`fetch_user`, `fetch_story_ids`, etc.) and centralise constants.

9. **Lack of error handling**
   * **Problem**: No checks for HTTP status codes or JSON parsing errors.
   * **Impact**: Silent failures and incomplete data.
   * **Resolution**: Use `curl -f` and check exit status; validate JSON with `jq`.

10. **No unit tests**
    * **Problem**: The script has no automated tests.
    * **Impact**: Hard to guarantee correctness after changes.
    * **Resolution**: Add pytest tests covering each function.

---

These tickets should serve as a starting point for addressing the current limitations of the data‑fetching workflow. Each item includes enough context for a new contributor to understand the root cause and the recommended path forward.