#!/usr/bin/env python3
"""
app.py – Streamlit UI that talks to a local llama‑server and can push the repo
to GitHub.  The code now supports **multiple** tool calls per request.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import streamlit.components.v1 as components
from git import InvalidGitRepositoryError, Repo

from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.tools import get_tools, TOOLS
from app.docs_extractor import extract


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def refresh_docs() -> str:
    """Run the extractor and return the Markdown content."""
    return extract().read_text(encoding="utf-8")


def is_repo_up_to_date(repo_path: Path) -> bool:
    """Return True iff the local HEAD matches origin/main and the working tree
    has no dirty files."""
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        return False

    if not repo.remotes:
        return False

    try:
        repo.remotes.origin.fetch()
    except Exception:
        return False

    for branch_name in ("main", "master"):
        try:
            remote_branch = repo.remotes.origin.refs[branch_name]
            break
        except IndexError:
            continue
    else:
        return False

    return (
        repo.head.commit.hexsha == remote_branch.commit.hexsha
        and not repo.is_dirty(untracked_files=True)
    )


def build_messages(
    history,
    system_prompt,
    repo_docs,
    user_input=None,
):
    """
    Build the list of messages expected by the OpenAI chat API.
    """
    msgs = [{"role": "system", "content": str(system_prompt)}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": str(repo_docs)})

    for u, a in history:
        msgs.append({"role": "user", "content": str(u)})
        msgs.append({"role": "assistant", "content": str(a)})

    if user_input is not None:
        msgs.append({"role": "user", "content": str(user_input)})

    return msgs


def stream_and_collect(client, messages, tools, placeholder):
    """
    Stream a response from the model, updating `placeholder` live, and
    collect any tool calls that are emitted.
    """
    stream = client.chat.completions.create(
        model="unsloth/gpt-oss-20b-GGUF:F16",
        messages=messages,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls_buffer = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Regular text
        if delta.content:
            full_resp += delta.content
            placeholder.markdown(full_resp, unsafe_allow_html=True)

        # Tool calls
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tc_delta.id,
                        "name": tc_delta.function.name,
                        "arguments": "",
                    }
                if tc_delta.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

    final_tool_calls = list(tool_calls_buffer.values()) if tool_calls_buffer else None
    return full_resp, final_tool_calls


def process_tool_calls(client, messages, tools, placeholder, tool_calls):
    """
    For every tool call in `tool_calls`:
        • call the tool
        • append assistant_tool_call_msg + tool_msg to `messages`
        • stream a new assistant reply
    Return the final assistant text and the next list of tool calls.
    """
    if not tool_calls:
        return "", None

    full_text = ""
    for tc in tool_calls:
        args = json.loads(tc.get("arguments") or "{}")
        func = next((t.func for t in TOOLS if t.name == tc.get("name")), None)

        if func:
            try:
                result = func(**args)
            except Exception as exc:
                result = f"❌  Tool error: {exc}"
        else:
            result = f"⚠️  Unknown tool '{tc.get('name')}'"

        # --------------------------------------------------------------------
        #  Render the tool‑call *result* in the UI before the next assistant turn
        # --------------------------------------------------------------------
        tool_output_str = (
            f"**Tool call**: `{tc.get('name')}`"
            f"({', '.join(f'{k}={v}' for k, v in args.items())}) → `{result}`"
        )
        placeholder.markdown(tool_output_str, unsafe_allow_html=True)

        # Build messages to send back to the model
        assistant_tool_call_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments") or "{}",
                    },
                }
            ],
        }

        tool_msg = {
            "role": "tool",
            "tool_call_id": str(tc.get("id") or ""),
            "content": str(result or ""),
        }

        messages.append(assistant_tool_call_msg)
        messages.append(tool_msg)

        # Stream the next assistant reply – each iteration gets a fresh placeholder
        placeholder2 = st.empty()
        new_text, new_tool_calls = stream_and_collect(
            client, messages, tools, placeholder2
        )
        full_text += new_text
        tool_calls = new_tool_calls or []

        if not tool_calls:  # no more calls – break early
            break

    return full_text, tool_calls


# --------------------------------------------------------------------------- #
#  Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Chat with GPT‑OSS", layout="wide")
    REPO_PATH = Path(__file__).parent

    # Session state
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("repo_docs", "")
    st.session_state.has_pushed = is_repo_up_to_date(REPO_PATH)

    # -------------------------------------------------------------------- #
    #  Sidebar
    # -------------------------------------------------------------------- #
    with st.sidebar:
        st.header("Settings")

        # System prompt editor
        prompt = st.text_area(
            "System prompt",
            st.session_state.system_prompt,
            height=120,
        )
        if prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = prompt

        # New chat button
        if st.button("New Chat"):
            st.session_state.history = []
            st.session_state.repo_docs = ""
            st.success("Chat history cleared. Start fresh!")

        # Refresh docs button
        if st.button("Refresh Docs"):
            st.session_state.repo_docs = refresh_docs()
            st.success("Codebase docs updated!")

        # Push to GitHub button
        if st.button("Push to GitHub"):
            with st.spinner("Pushing to GitHub…"):
                try:
                    from app.push_to_github import main as push_main

                    push_main()
                    st.session_state.has_pushed = True
                    st.success("✅  Repository pushed to GitHub.")
                except Exception as exc:
                    st.error(f"❌  Push failed: {exc}")

        # Push status
        status = "✅  Pushed" if st.session_state.has_pushed else "⚠️  Not pushed"
        st.markdown(f"**Push status:** {status}")

        # Available tools
        st.subheader("Available tools")
        for t in TOOLS:
            st.markdown(f"- **{t.name}**: {t.description}")

    # -------------------------------------------------------------------- #
    #  Chat history
    # -------------------------------------------------------------------- #
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # -------------------------------------------------------------------- #
    #  User input
    # -------------------------------------------------------------------- #
    if user_input := st.chat_input("Enter request…"):
        with st.chat_message("user"):
            st.markdown(user_input)

        client = get_client()
        tools = get_tools()
        msgs = build_messages(
            st.session_state.history,
            st.session_state.system_prompt,
            st.session_state.repo_docs,
            user_input,
        )
        with st.chat_message("assistant"):
            placeholder = st.empty()

        # First assistant turn – may contain tool calls
        final_text, tool_calls = stream_and_collect(client, msgs, tools, placeholder)

        # --------------------------------------------------------------------
        #  Show the tool calls that were detected
        # --------------------------------------------------------------------
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc.get("arguments") or "{}")
                st.markdown(
                    f"**Tool call**: `{tc.get('name')}`({', '.join(f'{k}={v}' for k, v in args.items())})",
                    unsafe_allow_html=True,
                )

        # Add the partial assistant reply to the message history
        msgs.append({"role": "assistant", "content": final_text})

        # If the model invoked tools, let the helper process all of them
        if tool_calls:
            full_text, remaining_calls = process_tool_calls(
                client, msgs, tools, placeholder, tool_calls
            )
            st.session_state.history.append((user_input, full_text))
            # Process any nested calls that might have been triggered by a tool
            while remaining_calls:
                full_text, remaining_calls = process_tool_calls(
                    client, msgs, tools, placeholder, remaining_calls
                )
                st.session_state.history[-1] = (user_input, full_text)
        else:
            # No tool calls – just store what we already got
            st.session_state.history.append((user_input, final_text))

    # -------------------------------------------------------------------- #
    #  Browser‑leaving guard
    # -------------------------------------------------------------------- #
    has_pushed = st.session_state.get("has_pushed", False)
    components.html(
        f"""
        <script>
        window.top.hasPushed = {str(has_pushed).lower()};
        window.top.onbeforeunload = function (e) {{
            if (!window.top.hasPushed) {{
                e.preventDefault(); e.returnValue = '';
                return 'You have not pushed to GitHub yet.\\nDo you really want to leave?';
            }}
        }};
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()