"""A minimal **ipywidgets** based chat client.

This module is intentionally lightweight – it only depends on the
``openai`` package and standard ``ipywidgets``.  It can be imported and
executed inside a Jupyter notebook to provide a chat‑like UI that talks
directly to the local server defined in :mod:`app.server`.

Typical usage inside a notebook::

    >>> from nbchat import run_chat
    >>> run_chat()  # doctest: +SKIP

The function creates the widgets, attaches a click handler and displays
the interface.  The conversation is stored in a local ``list`` and
persisted to the server via the OpenAI compatible endpoint.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import ipywidgets as widgets
from IPython.display import clear_output, display

from app.client import get_client
from app.tools import get_tools
from app.db import log_tool_msg
from app.config import DEFAULT_SYSTEM_PROMPT
import streamlit as st  # only used for st.empty() placeholder in tool handling

__all__ = ["run_chat"]


def _stream_and_display(client, user_input: str, history: List[tuple], chat_history, conversation):
    """Send *user_input* to the server and stream the assistant reply.

    Parameters
    ----------
    client:
        The :class:`openai.OpenAI` client returned by :func:`app.client.get_client`.
    user_input:
        The message typed by the user.
    history:
        The conversation history as a list of ``(role, content)`` tuples.

    Returns
    -------
    str
        The full assistant reply.
    """

    # Build the list of messages to send
    messages = []
    # System message – the repo uses a default prompt; we keep it simple
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_input})

    assistant_text = ""
    tool_calls: List[dict] = []
    for chunk in client.chat.completions.create(
        model="unsloth/gpt-oss-20b-GGUF:F16",
        messages=messages,
        stream=True,
    ):
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            assistant_text += content
            # Update the chat history display live
            with chat_history:
                clear_output(wait=True)
                for role, msg in conversation:
                    print(f"{role}: {msg}")
                print(f"Bot: {assistant_text}")
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc in delta.tool_calls:
                tool_calls.append(tc)
    return assistant_text, tool_calls


def run_chat() -> None:
    """Instantiate and display an ipywidgets chat UI.

    The function creates the widgets, attaches a click handler and
    displays the interface.  It is intended to be executed inside a
    Jupyter notebook cell.
    """

    # Create widgets
    chat_history = widgets.Output(layout=widgets.Layout(width="100%"))
    message_input = widgets.Text(
        placeholder="Type your message...",
        description="You:",
        layout=widgets.Layout(width="80%"),
    )
    send_button = widgets.Button(description="Send", button_style="primary")
    input_box = widgets.HBox([message_input, send_button])

    # Conversation history stored as a list of (role, content)
    conversation: List[tuple] = []
    # Unique session ID for this notebook instance
    session_id = str(uuid.uuid4())
    # Persistent SQLite DB for history
    DB_PATH = Path("chat_history.db")
    # Ensure table exists
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, created_at TIMESTAMP)")
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM history ORDER BY created_at ASC")
        rows = cur.fetchall()
        conversation.extend(rows)

    # Helper to refresh the output area
    def _refresh_display():
        with chat_history:
            clear_output(wait=True)
            for role, msg in conversation:
                print(f"{role}: {msg}")

    # Click handler
    def _handle_tool_calls(client, messages, session_id, tool_calls, conversation):
        """Execute tool calls returned by the model.

        The function iterates over the list of tool call objects, calls the
        corresponding Python function, appends the result to ``conversation``
        and logs the interaction to the SQLite DB.
        """
        tools = get_tools()
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            # Find matching tool
            tool_def = next((t for t in tools if t['function']['name'] == name), None)
            if not tool_def:
                result = f"⚠️ Unknown tool {name}"
            else:
                func = tool_def.get('function').get('name')
                # Use the function object from the module
                module_name = name
                mod = __import__(f"app.tools.{module_name}", fromlist=["func"])
                tool_func = getattr(mod, "func")
                try:
                    result = tool_func(**args)
                except Exception as exc:
                    result = f"❌ Tool error: {exc}"
            # Append to conversation and DB
            conversation.append(("Tool", result))
            log_tool_msg(session_id, tc.id, name, tc.function.arguments, result)
            # Display result in chat history output
            with chat_history:
                clear_output(wait=True)
                for role, msg in conversation:
                    print(f"{role}: {msg}")

    def _send_message(_):
        user_msg = message_input.value.strip()
        if not user_msg:
            return
        # Append user message
        conversation.append(("You", user_msg))
        _refresh_display()
        # Persist user message
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO history (role, content, created_at) VALUES (?,?,?)",
                         ("You", user_msg, datetime.utcnow()))
        # Get a client pointing at the local server
        client = get_client()
        # Build messages for the model (system + history) – this is used by
        # ``_stream_and_display``.  We keep this explicit construction to
        # make the intent clear to readers.
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]
        for role, msg in conversation[:-1]:
            messages.append({"role": role.lower(), "content": msg})
        messages.append({"role": "user", "content": user_msg})
        # Stream assistant reply
        assistant_reply, tool_calls = _stream_and_display(client, user_msg, conversation[:-1], chat_history, conversation)
        conversation.append(("Bot", assistant_reply))
        _refresh_display()
        # Persist assistant reply
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO history (role, content, created_at) VALUES (?,?,?)",
                         ("Bot", assistant_reply, datetime.utcnow()))
        # Handle any tool calls
        if tool_calls:
            _handle_tool_calls(client, messages, session_id, tool_calls, conversation)
        # Clear input
        message_input.value = ""

    send_button.on_click(_send_message)

    # Display the UI
    # New chat button
    new_chat_button = widgets.Button(description="New chat", button_style="danger")

    def _new_chat(_):
        global conversation
        conversation = []
        # Delete DB history
        if DB_PATH.exists():
            DB_PATH.unlink()
        _refresh_display()

    new_chat_button.on_click(_new_chat)

    display(chat_history)
    display(input_box)
    display(new_chat_button)
