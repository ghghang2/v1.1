"""A thin wrapper around the local `llama-server`.

The wrapper exposes a simple async ``chat`` method that streams tokens
from the server.  It is intentionally minimal – the goal for Phase 2 is to
have a testable component that can be used by agents and the HTTP proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import requests
from typing import AsyncGenerator, Dict, Optional

from .config import SERVER_URL

@dataclass
class LlamaClient:
    """A very small wrapper around the llama‑server API.

    Parameters
    ----------
    server_url:
        Base URL for the llama‑server.  Defaults to :data:`app.config.SERVER_URL`.
    """

    server_url: str = SERVER_URL

    def _post(self, path: str, payload: Dict, stream: bool = False) -> requests.Response:
        url = f"{self.server_url}{path}"
        return requests.post(url, json=payload, stream=stream, timeout=10)

    async def chat(self, prompt: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Yield tokens from the LLM.

        The function performs a blocking ``requests.post`` inside a threadpool
        via :func:`asyncio.to_thread` to keep the API async without adding
        external dependencies.
        """

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if session_id:
            payload["session_id"] = session_id

        def _request() -> requests.Response:
            return self._post("/v1/chat/completions", payload, stream=True)

        response = await asyncio.to_thread(_request)
        if response.status_code != 200:
            raise RuntimeError(f"LLM request failed: {response.status_code} {response.text}")

        # The llama-server streams one line JSON per token.  We yield the
        # ``content`` field of the token.
        for line in response.iter_lines():
            if not line:
                continue
            # llama-server streams lines like "data: {"..."}"
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                decoded = decoded[6:]
            # Skip keep-alive or empty messages
            if decoded in ("", "[DONE]"):
                continue
            data = json.loads(decoded)
            token = data.get("choices", [{}])[0].get("delta", {}).get("content")
            if token:
                yield token
        # Final token (stop) may not be yielded; the client can decide
        # when the stream ends.