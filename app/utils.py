# app/utils.py  (only the added/modified parts)
from typing import List, Tuple, Dict, Optional, Any
from .config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME
from .client import get_client
from openai import OpenAI
from .tools import get_tools

def build_api_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """
    Convert local chat history into the format expected by the OpenAI API,
    optionally adding a tool list.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": repo_docs})
    for user_msg, bot_msg in history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    # The client will pass `tools=tools` when calling chat.completions.create
    return msgs

def stream_response(
    history: List[Tuple[str, str]],
    user_msg: str,
    client: OpenAI,
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
):
    """
    Yield the cumulative assistant reply while streaming.
    Also returns any tool call(s) that the model requested.
    """
    new_hist = history + [(user_msg, "")]
    api_msgs = build_api_messages(new_hist, system_prompt, repo_docs, tools)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_msgs,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls = None
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_resp += token
        yield full_resp

        # Capture tool calls once the model finishes sending them
        if chunk.choices[0].delta.tool_calls:
            tool_calls = chunk.choices[0].delta.tool_calls

    return full_resp, tool_calls