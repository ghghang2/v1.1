import os
import json
import requests
import pytest
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from app.llama_client import LlamaClient
from app import config

SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8000")


def _is_server_available():
    try:
        return requests.get(SERVER_URL + "/health", timeout=1).ok
    except Exception:
        return False


@pytest.mark.skipif(not _is_server_available(), reason="LLAMA server not running, skipping tests")
def test_llama_client_chat_stream():
    client = LlamaClient(server_url=SERVER_URL)
    # Use a very short prompt to avoid long responses
    prompt = "Say hi"
    tokens = []
    async def collect():
        async for token in client.chat(prompt):
            tokens.append(token)
    import asyncio
    asyncio.run(collect())
    assert len(tokens) > 0