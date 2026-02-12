"""Test the FastAPI proxy that forwards requests to the LLM client.

The test monkeypatches the :class:`app.llama_client.LlamaClient` so that
no external network calls are made.  It also patches the persistence
layer so that the database is not touched.
"""

import asyncio
import pytest
from httpx import AsyncClient

# Import the module under test
import app.server as server


def test_proxy_forwards_and_streams(monkeypatch):
    """Verify that the proxy forwards a prompt and streams tokens back.

    The test ensures that:
    * The LLM client is called with the correct arguments.
    * The response is streamed token‑by‑token.
    * The chat history persistence is invoked for both the user and
      assistant tokens.
    """

    # Keep track of insert calls so we can assert on them later
    inserted_calls = []
    def fake_insert(session_id: str, role: str, content: str, *_, **__):  # noqa: ANN001
        inserted_calls.append((session_id, role, content))

    monkeypatch.setattr(server.chat_history, "insert", fake_insert)

    # Dummy LLM that yields two tokens
    async def dummy_chat(prompt: str, session_id=None):  # noqa: ANN001
        assert prompt == "Hello"
        assert session_id == "session-1"
        for token in ("Hi", " there!"):
            await asyncio.sleep(0)  # yield control to the event loop
            yield token

    monkeypatch.setattr(server.llama_client, "chat", dummy_chat)

    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    resp = client.post(
        "/chat/agent1",
        json={"prompt": "Hello", "session_id": "session-1"},
    )
    assert resp.status_code == 200
    # The response body should be the concatenated tokens
    assert resp.text == "Hi there!"

    # The history should have one user message and two assistant messages
    expected = [
        ("agent1", "user", "Hello"),
        ("agent1", "assistant", "Hi"),
        ("agent1", "assistant", " there!"),
    ]
    assert inserted_calls == expected
