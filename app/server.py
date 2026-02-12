"""Minimal FastAPI proxy that forwards chat requests to llama-server.

The proxy accepts a JSON payload with ``prompt`` and optional ``session_id``.
It streams the LLM response back to the client while persisting the chat
history in SQLite.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

from .llama_client import LlamaClient
from .db import ChatHistory

app = FastAPI()

llama_client = LlamaClient()
chat_history = ChatHistory()


@app.post("/chat/{chat_id}")
async def chat(chat_id: str, request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    session_id = body.get("session_id")
    # Persist user message
    chat_history.insert(chat_id, "user", prompt)

    async def token_generator():
        async for token in llama_client.chat(prompt, session_id=session_id):
            chat_history.insert(chat_id, "assistant", token)
            yield token
    return StreamingResponse(token_generator(), media_type="text/plain")