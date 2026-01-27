# app.py
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.tools import get_tools, TOOLS          # new registry
from app.docs_extractor import extract
import json

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def refresh_docs() -> str:
    """Run the extractor once (same folder as app.py)."""
    return extract().read_text(encoding="utf-8")

def is_repo_up_to_date(repo_path: Path) -> bool:
    """Return True iff local HEAD == remote `origin/main` AND no dirty files."""
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        return False

    if not repo.remotes:
        return False

    origin = repo.remotes.origin
    try:
        origin.fetch()
    except Exception:
        return False

    for branch_name in ("main", "master"):
        try:
            remote_branch = origin.refs[branch_name]
            break
        except IndexError:
            continue
    else:
        return False

    return (
        repo.head.commit.hexsha == remote_branch.commit.hexsha
        and not repo.is_dirty(untracked_files=True)
    )

# --------------------------------------------------------------------------- #
#  Message building & streaming (needed for function calling)
# --------------------------------------------------------------------------- #
def build_messages(history, system_prompt, repo_docs, user_input=None):
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
    stream = client.chat.completions.create(
        model="unsloth/gpt-oss-20b-GGUF:F16",
        messages=messages,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    # We use a dictionary to track multiple potential tool calls by their index
    tool_calls_buffer = {}

    for chunk in stream:
        delta = chunk.choices[0].delta
        
        # 1. Handle regular text content
        if delta.content:
            full_resp += delta.content
            placeholder.markdown(full_resp, unsafe_allow_html=True)

        # 2. Handle tool call deltas
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                
                if idx not in tool_calls_buffer:
                    # Initialize the entry for this specific tool call
                    tool_calls_buffer[idx] = {
                        "id": tc_delta.id,
                        "name": tc_delta.function.name,
                        "arguments": ""
                    }
                
                # Append the new fragment of the arguments JSON string
                if tc_delta.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments
                    
    # Convert the buffer back into a list format for your return statement
    final_tool_calls = list(tool_calls_buffer.values()) if tool_calls_buffer else None
    return full_resp, final_tool_calls


# --------------------------------------------------------------------------- #
#  Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Chat with GPT‑OSS", layout="wide")
    REPO_PATH = Path(__file__).parent

    # session state
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("repo_docs", "")
    st.session_state.has_pushed = is_repo_up_to_date(REPO_PATH)

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

        # Show available tools
        st.subheader("Available tools")
        for t in TOOLS:
            st.markdown(f"- **{t.name}**: {t.description}")

    # Render chat history
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # User input
    if user_input := st.chat_input("Enter request…"):
        st.chat_message("user").markdown(user_input)

        client = get_client()
        tools = get_tools()

        # Build messages for the first call
        msgs = build_messages(
            st.session_state.history,
            st.session_state.system_prompt,
            st.session_state.repo_docs,
            user_input,
        )

        placeholder = st.empty()
        final_text, tool_calls = stream_and_collect(client, msgs, tools, placeholder)

        # Append assistant reply to history
        st.session_state.history.append((user_input, final_text))

        # If the model wanted to call a tool
        if tool_calls:
            tool_call = tool_calls[0]
            try:
                args = json.loads(tool_call.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            func = next((t.func for t in TOOLS if t.name == tool_call.get("name")), None)

            if func is None:
                tool_result = f"⚠️  Tool '{tool_call.get("name")}' not registered."
            else:
                try:
                    tool_result = func(**args)
                except Exception as exc:
                    tool_result = f"❌  Tool error: {exc}"

            # Show the tool call & its result
            st.chat_message("assistant").markdown(
                f"**Tool call**: `{tool_call.get("name")}({', '.join(f'{k}={v}' for k, v in args.items())})` → `{tool_result}`"
            )

            assistant_tool_call_msg = {
                "role": "assistant",
                "content": None,                              # no assistant text for a tool call
                "tool_calls": [
                    {
                        "id": tool_call.get("id"),
                        "type": "function",                   # <-- required
                        "function": {
                            "name": tool_call.get("name"),
                            "arguments": tool_call.get("arguments") or "{}"
                        }
                    }
                ]
            }

            # Send the tool result back to the model for the final answer
            tool_msg = {
                "role": "tool",
                "tool_call_id": str(tool_call.get("id") or ""),   # <-- guard against None
                "content": str(tool_result or ""),         # <-- guard against None
            }
            msgs2 = build_messages(
                st.session_state.history,
                st.session_state.system_prompt,
                st.session_state.repo_docs,
            )
            msgs2.append(assistant_tool_call_msg)
            msgs2.append(tool_msg)

            with st.chat_message("assistant") as assistant_msg2:
                placeholder2 = st.empty()
                final_text2, _ = stream_and_collect(
                    client, msgs2, tools, placeholder2
                )

            # Replace the assistant reply with the final answer
            st.session_state.history[-1] = (user_input, final_text2)

    # Browser‑leaving guard
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