from __future__ import annotations

import asyncio
import ipywidgets as widgets
import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from IPython.display import display

# ----------------------------------------------------------------------
# Lazy imports (only what is needed for the minimal version)
# ----------------------------------------------------------------------
_client = None
_db_module = None
_config_module = None


def lazy_import(module_name: str):
    """Import a module only when needed."""
    global _client, _db_module, _config_module

    if module_name == "app.client":
        if _client is None:
            from app.client import get_client
            _client = get_client
        return _client()
    elif module_name == "app.db":
        if _db_module is None:
            import app.db as db_module
            _db_module = db_module
        return _db_module
    elif module_name == "app.config":
        if _config_module is None:
            import app.config as config_module
            _config_module = config_module
        return _config_module
    else:
        raise ValueError(f"Unknown module {module_name}")


# ----------------------------------------------------------------------
# Main Chat UI class
# ----------------------------------------------------------------------
class ChatUI:
    """Minimal chat UI with streaming, reasoning, and persistent history."""

    def __init__(self):
        # Initialize database
        db = lazy_import("app.db")
        db.init_db()

        # Load configuration
        config = lazy_import("app.config")
        self.default_system_prompt = config.DEFAULT_SYSTEM_PROMPT
        self.model_name = config.MODEL_NAME

        # State
        self.session_id = str(uuid.uuid4())
        self.history: List[Tuple[str, str, str, str, str]] = []  # (role, content, tool_id, tool_name, tool_args)
        self.system_prompt = self.default_system_prompt

        # Load existing session IDs
        self.session_ids = db.get_session_ids()

        # Build UI widgets
        self._create_widgets()

        # Start background metrics updater
        self._start_metrics_updater()

        # Load initial chat history for the current session
        self._load_history()

        # Display the UI
        display(self.layout)

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------
    def _create_widgets(self):
        """Create all ipywidgets components."""
        # ----- Sidebar -----
        self.metrics_output = widgets.HTML(
            value="<i>Loading server status...</i>",
            layout=widgets.Layout(width="100%", border="1px solid gray", padding="10px")
        )

        self.new_chat_btn = widgets.Button(
            description="New Chat",
            button_style="primary",
            layout=widgets.Layout(width="100%")
        )
        self.new_chat_btn.on_click(self._on_new_chat)

        # Session selector
        options = list(self.session_ids)
        if self.session_id not in options:
            options.append(self.session_id)
        self.session_dropdown = widgets.Dropdown(
            options=options,
            value=self.session_id,
            description="Session:",
            layout=widgets.Layout(width="100%")
        )
        self.session_dropdown.observe(self._on_session_change, names="value")

        sidebar = widgets.VBox([
            widgets.HTML("<h3>Chat Controls</h3>"),
            self.metrics_output,
            widgets.HTML("<hr>"),
            self.new_chat_btn,
            widgets.HTML("<hr>"),
            self.session_dropdown,
        ], layout=widgets.Layout(width="25%", border="1px solid black", padding="10px"))

        # ----- Main area -----
        self.chat_history = widgets.VBox(
            [],
            layout=widgets.Layout(
                width="100%",
                height="400px",
                overflow_y="auto",
                border="1px solid #ccc",
                padding="10px"
            )
        )

        self.input_text = widgets.Text(
            placeholder="Enter your message...",
            layout=widgets.Layout(width="80%")
        )
        self.send_btn = widgets.Button(
            description="Send",
            button_style="success",
            layout=widgets.Layout(width="18%", margin_left="2%")
        )
        self.input_box = widgets.HBox([self.input_text, self.send_btn])

        self.input_text.on_submit(self._on_send)
        self.send_btn.on_click(self._on_send)

        main = widgets.VBox([
            widgets.HTML("<h2>Chat</h2>"),
            self.chat_history,
            self.input_box,
        ], layout=widgets.Layout(width="75%", padding="10px"))

        # Overall layout
        self.layout = widgets.HBox([sidebar, main])

    # ------------------------------------------------------------------
    # Metrics updater (simplified version from original)
    # ------------------------------------------------------------------
    def _start_metrics_updater(self):
        """Background thread to update server metrics every 2 seconds."""
        def update_loop():
            while True:
                try:
                    log_path = Path("llama_server.log")
                    if log_path.exists():
                        with open(log_path, "rb") as f:
                            f.seek(0, 2)
                            f.seek(max(0, f.tell() - 4000))
                            lines = f.read().decode("utf-8", errors="ignore").splitlines()
                        # Check if server is processing
                        proc = any("slot update_slots:" in l.lower() for l in lines[-10:])
                        if any("all slots are idle" in l.lower() for l in lines[-5:]):
                            proc = False
                        # Extract tokens per second
                        tps = 0.0
                        for line in reversed(lines):
                            if "eval time" in line.lower():
                                m = re.search(r"(?P<value>\d+(?:\.\d+)?)\s+tokens per second", line, re.IGNORECASE)
                                if m:
                                    tps = float(m.group("value"))
                                    break
                        emoji = "🟢" if proc else "⚫"
                        content = f"<b>Server:</b> {emoji}<br><b>TPS:</b> <code>{tps}</code><br><i>{time.strftime('%H:%M:%S')}</i>"
                    else:
                        content = "<i>Log file not found</i>"
                except Exception as e:
                    content = f"<i>Error: {e}</i>"

                # Update widget in main thread
                self.metrics_output.value = content
                time.sleep(2)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _load_history(self):
        """Load messages for the current session from the database."""
        db = lazy_import("app.db")
        rows = db.load_history(self.session_id)
        # Convert (role, content) to our 5‑tuple format (tool fields empty)
        self.history = [(role, content, "", "", "") for role, content in rows]
        self._render_history()

    def _render_history(self):
        """Rebuild the chat_history widget from self.history."""
        children = []
        for role, content, tool_id, tool_name, tool_args in self.history:
            if role == "user":
                children.append(self._render_user_message(content))
            elif role == "analysis":
                children.append(self._render_analysis_message(content))
            elif role == "assistant":
                children.append(self._render_assistant_message(content))
            # tool messages are not used in this minimal version
        self.chat_history.children = children
        # Scroll to bottom (disabled due to HTML widget limitation)

    def _render_user_message(self, content: str) -> widgets.HTML:
        return widgets.HTML(
            value=f'<div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px;"><b>User:</b> {content}</div>',
            layout=widgets.Layout(width="100%", margin="5px 0")
        )

    def _render_analysis_message(self, content: str) -> widgets.HTML:
        return widgets.HTML(
            value=f'''
            <div style="background-color: #fff3e0; padding: 10px; border-radius: 10px; margin: 5px;">
                <details open>
                    <summary><b>Reasoning</b></summary>
                    {content}
                </details>
            </div>
            ''',
            layout=widgets.Layout(width="100%", margin="5px 0")
        )

    def _render_assistant_message(self, content: str) -> widgets.HTML:
        return widgets.HTML(
            value=f'<div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px;"><b>Assistant:</b> {content}</div>',
            layout=widgets.Layout(width="100%", margin="5px 0")
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_new_chat(self, btn):
        """Create a new chat session."""
        self.session_id = str(uuid.uuid4())
        self.history = []
        # Update dropdown options
        db = lazy_import("app.db")
        options = list(db.get_session_ids())
        if self.session_id not in options:
            options.append(self.session_id)
        self.session_dropdown.options = options
        self.session_dropdown.value = self.session_id
        self._render_history()

    def _on_session_change(self, change):
        """Switch to a different session."""
        if change["new"]:
            self.session_id = change["new"]
            self._load_history()

    def _on_send(self, *args):
        """Handle user message submission."""
        user_input = self.input_text.value.strip()
        if not user_input:
            return

        # Clear input field
        self.input_text.value = ""

        # Add user message to history and database
        self.history.append(("user", user_input, "", "", ""))
        self._render_history()

        db = lazy_import("app.db")
        db.log_message(self.session_id, "user", user_input)

        # Stream assistant response
        self._stream_assistant_response(user_input)

    # ------------------------------------------------------------------
    # Streaming logic
    # ------------------------------------------------------------------
    def _stream_assistant_response(self, user_input: str):
        """Stream the assistant's response, showing reasoning and final answer."""
        # Prepare the messages list for the API call
        client = lazy_import("app.client")
        config = lazy_import("app.config")

        # Build messages from history + new user input
        # (Simplified version of app.chat.build_messages)
        messages = [{"role": "system", "content": self.system_prompt}]
        for role, content, *_ in self.history:
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
            # analysis messages are not part of the conversation for the API
        messages.append({"role": "user", "content": user_input})

        # Create placeholders in the UI
        # We will add them one by one as they appear
        reasoning_placeholder = None
        assistant_placeholder = None

        # Start streaming
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            max_tokens=2048,
            # No tools in this minimal version
        )

        reasoning_accum = ""
        assistant_accum = ""

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Reasoning content (if present)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if reasoning_placeholder is None:
                    # Create reasoning placeholder and add it to the chat
                    reasoning_placeholder = widgets.HTML(
                        value='<div style="background-color: #fff3e0; padding: 10px; border-radius: 10px; margin: 5px;"><details open><summary><b>Reasoning</b></summary></div>',
                        layout=widgets.Layout(width="100%", margin="5px 0")
                    )
                    self.chat_history.children = list(self.chat_history.children) + [reasoning_placeholder]
                reasoning_accum += delta.reasoning_content
                # Update the placeholder content
                reasoning_placeholder.value = f'''
                <div style="background-color: #fff3e0; padding: 10px; border-radius: 10px; margin: 5px;">
                    <details open>
                        <summary><b>Reasoning</b></summary>
                        {reasoning_accum}
                    </details>
                </div>
                '''

            # Assistant content
            if delta.content:
                if assistant_placeholder is None:
                    # Create assistant placeholder and add it after reasoning (if any)
                    assistant_placeholder = widgets.HTML(
                        value='<div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px;"><b>Assistant:</b> </div>',
                        layout=widgets.Layout(width="100%", margin="5px 0")
                    )
                    # Insert after reasoning if it exists, else at the end
                    children = list(self.chat_history.children)
                    if reasoning_placeholder is not None:
                        # reasoning is already the last element; we want assistant after it
                        children.append(assistant_placeholder)
                    else:
                        children.append(assistant_placeholder)
                    self.chat_history.children = children
                assistant_accum += delta.content
                assistant_placeholder.value = f'''
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px;">
                    <b>Assistant:</b> {assistant_accum}
                </div>
                '''

        # Streaming finished – now permanently add messages to history and DB
        db = lazy_import("app.db")
        if reasoning_accum:
            self.history.append(("analysis", reasoning_accum, "", "", ""))
            db.log_message(self.session_id, "analysis", reasoning_accum)
        if assistant_accum:
            self.history.append(("assistant", assistant_accum, "", "", ""))
            db.log_message(self.session_id, "assistant", assistant_accum)

        # Final render to ensure everything is correct (though placeholders are already there)
        # This also handles the case where no placeholders were created (e.g., empty response)
        self._render_history()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_chat() -> None:
    """Launch the chat interface."""
    ChatUI()