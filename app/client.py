from openai import OpenAI
from .config import SERVER_URL

def get_client() -> OpenAI:
    """Return a client that talks to the local OpenAI‑compatible server."""
    return OpenAI(base_url=f"{SERVER_URL}/v1", api_key="token")