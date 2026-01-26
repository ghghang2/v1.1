## app/__init__.py

```python
__all__ = ["client", "config", "utils", "docs_extractor"]
```

## app/client.py

```python
from openai import OpenAI
from .config import NGROK_URL

def get_client() -> OpenAI:
    """Return a client that talks to the local OpenAI‑compatible server."""
    return OpenAI(base_url=f"{NGROK_URL}/v1", api_key="token")
```

## app/config.py

```python
# Configuration – tweak these values as needed
NGROK_URL = "http://localhost:8000"
MODEL_NAME = "unsloth/gpt-oss-20b-GGUF:F16"
DEFAULT_SYSTEM_PROMPT = "Be concise and accurate at all times"

USER_NAME = "ghghang2"          # GitHub user / org name
REPO_NAME = "v1.2"              # Repository to push to
IGNORED_ITEMS = [
    ".*",
    "sample_data",
    "llama-server",
    "__pycache__",
    "*.log",
    "*.yml",
    "*.json",
    "*.out",
]
```

## app/docs_extractor.py

```python
# app/docs_extractor.py
"""
extract_docs.py → docs_extractor.py
-----------------------------------
Walk a directory tree and write a single Markdown file that contains:

* The relative path of each file (as a level‑2 heading)
* The raw source code of that file (inside a fenced code block)
"""

from __future__ import annotations

import pathlib
import sys
import logging

log = logging.getLogger(__name__)

def walk_python_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return all *.py files sorted alphabetically."""
    return sorted(root.rglob("*.py"))

def write_docs(root: pathlib.Path, out: pathlib.Path) -> None:
    """Append file path + code to *out*."""
    with out.open("w", encoding="utf-8") as f_out:
        for p in walk_python_files(root):
            rel = p.relative_to(root)
            f_out.write(f"## {rel}\n\n")
            f_out.write("```python\n")
            f_out.write(p.read_text(encoding="utf-8"))
            f_out.write("\n```\n\n")

def extract(repo_root: pathlib.Path | str = ".", out_file: pathlib.Path | str | None = None) -> pathlib.Path:
    """
    Extract the repo into a Markdown file and return the path.

    Parameters
    ----------
    repo_root : pathlib.Path | str
        Root of the repo to walk.  Defaults to the current dir.
    out_file : pathlib.Path | str | None
        Path to write the Markdown.  If ``None`` uses ``repo_docs.md``.
    """
    root = pathlib.Path(repo_root).resolve()
    out = pathlib.Path(out_file or "repo_docs.md").resolve()

    log.info("Extracting docs from %s → %s", root, out)
    write_docs(root, out)
    log.info("✅  Wrote docs to %s", out)
    return out

def main() -> None:  # CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Extract a repo into Markdown")
    parser.add_argument("repo_root", nargs="?", default=".", help="Root of the repo")
    parser.add_argument("output", nargs="?", default="repo_docs.md", help="Output Markdown file")
    args = parser.parse_args()

    extract(args.repo_root, args.output)

if __name__ == "__main__":
    main()
```

## app/push_to_github.py

```python
# scripts/push_to_github.py
"""
Entry point that wires the `RemoteClient` together.
"""

from pathlib import Path
from .remote import RemoteClient, REPO_NAME

def main() -> None:
    """Create/attach the remote, pull, commit and push."""
    client = RemoteClient(Path(__file__).resolve().parent.parent)  # repo root

    # 1️⃣  Ensure the GitHub repo exists
    client.ensure_repo(REPO_NAME)

    # 2️⃣  Attach (or re‑attach) the HTTPS remote
    client.attach_remote()

    # 3️⃣  Pull the latest changes
    client.fetch()
    client.pull()

    # 4️⃣  Write .gitignore (idempotent)
    client.write_gitignore()

    # 5️⃣  Commit everything that is new / changed
    client.commit_all("Initial commit")

    # 6️⃣  Make sure we are on the main branch
    if "main" not in [b.name for b in client.repo.branches]:
        client.repo.git.checkout("-b", "main")
        client.repo.git.reset("--hard")
    else:
        client.repo.git.checkout("main")
        client.repo.git.reset("--hard")

    # 7️⃣  Push to GitHub
    client.push()

if __name__ == "__main__":
    main()
```

## app/remote.py

```python
# remote/remote.py
"""
A single, self‑contained adapter that knows how to talk to:
  * a local Git repository (via gitpython)
  * GitHub (via PyGithub)
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import logging
from typing import Optional

from git import Repo, GitCommandError, InvalidGitRepositoryError
from github import Github, GithubException
from github.Auth import Token
from github.Repository import Repository

from .config import USER_NAME, REPO_NAME, IGNORED_ITEMS

log = logging.getLogger(__name__)

def _token() -> str:
    """Return the GitHub PAT from the environment."""
    t = os.getenv("GITHUB_TOKEN")
    if not t:
        raise RuntimeError("GITHUB_TOKEN env variable not set")
    return t

def _remote_url() -> str:
    """HTTPS URL that contains the PAT – used only for git push."""
    return f"https://{USER_NAME}:{_token()}@github.com/{USER_NAME}/{REPO_NAME}.git"

class RemoteClient:
    """Thin wrapper around gitpython + PyGithub that knows how to create
    a repo, fetch/pull/push and keep the local repo up‑to‑date.
    """

    def __init__(self, local_path: Path | str):
        self.local_path = Path(local_path).resolve()
        try:
            self.repo = Repo(self.local_path)
            if self.repo.bare:
                raise InvalidGitRepositoryError(self.local_path)
        except (InvalidGitRepositoryError, GitCommandError):
            log.info("Initializing a fresh git repo at %s", self.local_path)
            self.repo = Repo.init(self.local_path)

        self.github = Github(auth=Token(_token()))
        self.user = self.github.get_user()

    # ------------------------------------------------------------------ #
    #  Local‑repo helpers
    # ------------------------------------------------------------------ #
    def is_clean(self) -> bool:
        """Return True if there are no uncommitted changes."""
        return not self.repo.is_dirty(untracked_files=True)

    def fetch(self) -> None:
        """Fetch from the remote (if it exists)."""
        if "origin" in self.repo.remotes:
            log.info("Fetching from origin…")
            self.repo.remotes.origin.fetch()
        else:
            log.info("No remote configured – skipping fetch")

    def pull(self, rebase: bool = True) -> None:
        """Pull the `main` branch from origin, optionally rebasing.

        If the repo has uncommitted changes we commit them first with a
        deterministic message.  This guarantees that ``git pull`` (with or
        without rebase) will succeed.
        """
        if "origin" not in self.repo.remotes:
            raise RuntimeError("No remote named 'origin' configured")

        branch = "main"
        log.info("Pulling %s%s…", branch, " (rebase)" if rebase else "")

        # 1️⃣  Commit any dirty work
        if self.repo.is_dirty(untracked_files=True):
            log.info("Committing local changes before pull")
            self.commit_all("Auto‑commit before pull")

        # 2️⃣  Pull
        try:
            if rebase:
                self.repo.remotes.origin.pull(
                    refspec=branch, rebase=True, progress=None
                )
            else:
                self.repo.remotes.origin.pull(branch)
        except GitCommandError as exc:
            log.warning("Rebase failed: %s – falling back to merge", exc)
            self.repo.git.merge(f"origin/{branch}")

    def push(self, remote: str = "origin") -> None:
        """Push the local `main` branch to the given remote."""
        if remote not in self.repo.remotes:
            raise RuntimeError(f"No remote named '{remote}'")
        log.info("Pushing to %s…", remote)
        self.repo.remotes[remote].push("main")

    def reset_hard(self) -> None:
        """Discard any uncommitted or stale merge‑conflict data."""
        self.repo.git.reset("--hard")

    # ------------------------------------------------------------------ #
    #  GitHub helpers
    # ------------------------------------------------------------------ #
    def ensure_repo(self, name: str = REPO_NAME) -> Repository:
        """Create the GitHub repo if it does not exist and return it."""
        try:
            repo = self.user.get_repo(name)
            log.info("Repo '%s' already exists on GitHub", name)
        except GithubException:
            log.info("Creating new repo '%s' on GitHub", name)
            repo = self.user.create_repo(name, private=False)
        return repo

    def attach_remote(self, url: Optional[str] = None) -> None:
        """Delete any existing `origin` remote and add a fresh one."""
        if url is None:
            url = _remote_url()
        if "origin" in self.repo.remotes:
            log.info("Removing old origin remote")
            self.repo.delete_remote("origin")
        log.info("Adding new origin remote: %s", url)
        self.repo.create_remote("origin", url)

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    def write_gitignore(self) -> None:
        """Create a .gitignore that matches the constants in config.py."""
        path = self.local_path / ".gitignore"
        content = "\n".join(IGNORED_ITEMS) + "\n"
        path.write_text(content, encoding="utf-8")
        log.info("Wrote %s", path)

    def commit_all(self, message: str = "Initial commit") -> None:
        """Stage everything and commit (ignoring the 'nothing to commit' error)."""
        self.repo.git.add(A=True)
        try:
            self.repo.index.commit(message)
            log.info("Committed: %s", message)
        except GitCommandError as exc:
            if "nothing to commit" in str(exc):
                log.info("Nothing new to commit")
            else:
                raise
```

## app/utils.py

```python
# app/utils.py  (only the added/modified parts)
from typing import List, Tuple, Dict, Optional
from .config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME
from .client import get_client
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Build the messages list that the OpenAI API expects
# --------------------------------------------------------------------------- #
def build_api_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str] = None,
) -> List[Dict]:
    """
    Convert local chat history into the format expected by the OpenAI API.

    Parameters
    ----------
    history : List[Tuple[str, str]]
        (user, assistant) pairs.
    system_prompt : str
        Prompt given to the model.
    repo_docs : str | None
        Full code‑base text.  If supplied it is sent as the *first* assistant
        message so the model can read it before answering.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": repo_docs})
    for user_msg, bot_msg in history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    return msgs

# --------------------------------------------------------------------------- #
# Stream the assistant reply token‑by‑token
# --------------------------------------------------------------------------- #
def stream_response(
    history: List[Tuple[str, str]],
    user_msg: str,
    client: OpenAI,
    system_prompt: str,
    repo_docs: Optional[str] = None,
):
    """Yield the cumulative assistant reply while streaming."""
    new_hist = history + [(user_msg, "")]
    api_msgs = build_api_messages(new_hist, system_prompt, repo_docs)

    stream = client.chat.completions.create(
        model=MODEL_NAME, messages=api_msgs, stream=True
    )

    full_resp = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_resp += token
        yield full_resp
```

## app.py

```python
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.utils import stream_response
from app.docs_extractor import extract

# --------------------------------------------------------------------------- #
# Helpers
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

    # try common branch names
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
# Streamlit UI
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
        bot_output = ""

        with st.chat_message("assistant") as assistant_msg:
            placeholder = st.empty()
            for partial in stream_response(
                st.session_state.history,
                user_input,
                client,
                st.session_state.system_prompt,
                st.session_state.repo_docs,
            ):
                bot_output = partial
                placeholder.markdown(bot_output, unsafe_allow_html=True)

        st.session_state.history.append((user_input, bot_output))

    # Browser‑leaving guard
    has_pushed = st.session_state.get("has_pushed", False)
    components.html(
        f"""
        <script>
        // Make the flag visible to the outer window
        window.top.hasPushed = {str(has_pushed).lower()};

        // Attach the unload guard to the outer window
        window.top.onbeforeunload = function (e) {{
            if (!window.top.hasPushed) {{
            // Modern browsers require e.preventDefault() + e.returnValue
            e.preventDefault();
            e.returnValue = '';
            return 'You have not pushed to GitHub yet.\\nDo you really want to leave?';
            }}
        }};
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()
```

## run.py

```python
#!/usr/bin/env python3
"""
Launch the llama‑server demo in true head‑less mode.
Optimized for Google Colab notebooks with persistent ngrok tunnels.
"""
import os
import subprocess
import sys
import time
import socket
import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------#
#  Utility functions
# ---------------------------------------------------------------------------#


def run(cmd, *, shell=False, cwd=None, env=None, capture=False):
    """Run a command and return its stdout (if capture=True)."""
    if env is None:
        env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip() if capture else None
    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] Command failed: {exc.cmd}\n{exc.stdout or exc.stderr}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_service(url, timeout=30, interval=1):
    """Wait for a service to become available at the given URL."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def save_service_info(tunnel_url, llama_pid, streamlit_pid, ngrok_pid):
    """Save service information to a JSON file for later retrieval."""
    info = {
        "tunnel_url": tunnel_url,
        "llama_server_pid": llama_pid,
        "streamlit_pid": streamlit_pid,
        "ngrok_pid": ngrok_pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("service_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Also save just the URL to a simple text file
    with open("tunnel_url.txt", "w") as f:
        f.write(tunnel_url)


def get_ngrok_tunnel_url(max_attempts=10, interval=2):
    """Get the tunnel URL from ngrok's local API."""
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as response:
                data = json.loads(response.read())
                if data.get('tunnels'):
                    # Get the public URL (prefer https)
                    for tunnel in data['tunnels']:
                        if tunnel.get('public_url', '').startswith('https'):
                            return tunnel['public_url']
                    # Fallback to first tunnel
                    return data['tunnels'][0]['public_url']
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(interval)
            else:
                raise Exception(f"Failed to get ngrok tunnel URL after {max_attempts} attempts: {e}")
    return None


# ---------------------------------------------------------------------------#
#  Main routine
# ---------------------------------------------------------------------------#


def main():
    # 1. Check prerequisites
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        sys.exit("[ERROR] GITHUB_TOKEN not set")

    NGROK_TOKEN = os.getenv("NGROK_TOKEN")
    if not NGROK_TOKEN:
        sys.exit("[ERROR] NGROK_TOKEN not set")

    # Check for port conflicts
    if is_port_in_use(4040):
        sys.exit("[ERROR] Port 4040 (ngrok API) is already in use")
    if is_port_in_use(8000):
        sys.exit("[ERROR] Port 8000 (llama-server) is already in use")
    if is_port_in_use(8002):
        sys.exit("[ERROR] Port 8002 (Streamlit) is already in use")

    # 2. Download the pre‑built llama‑server binary
    REPO = "ghghang2/llamacpp_t4_v1"
    FILES = ["llama-server"]
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = GITHUB_TOKEN

    for f in FILES:
        run(
            f"gh release download --repo {REPO} --pattern {f}",
            shell=True,
            env=env,
        )
        run(f"chmod +x ./{f}", shell=True)

    # 3. Start llama‑server
    # Keep the file handle open so the subprocess can write to it
    llama_log = Path("llama_server.log").open("w", encoding="utf-8", buffering=1)
    llama_proc = subprocess.Popen(
        [
            "./llama-server",
            "-hf",
            "unsloth/gpt-oss-20b-GGUF:F16",
            "--port",
            "8000",
        ],
        stdout=llama_log,
        stderr=llama_log,
        start_new_session=True,
    )
    
    print("✅ llama-server started (PID: {}), waiting for it to be ready...".format(llama_proc.pid))
    
    # Wait for llama-server to be ready
    if not wait_for_service("http://localhost:8000/health", timeout=240):
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] llama-server failed to start within 60 seconds")
    
    print("✅ llama-server is ready on port 8000")

    # 4. Install required Python packages
    print("📦 Installing Python packages...")
    run("pip install -q streamlit pygithub", shell=True)

    # 5. Start the Streamlit UI
    streamlit_log = Path("streamlit.log").open("w", encoding="utf-8", buffering=1)
    streamlit_proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8002",
            "--server.headless",
            "true",
        ],
        stdout=streamlit_log,
        stderr=streamlit_log,
        start_new_session=True,
    )

    print("✅ Streamlit started (PID: {}), waiting for it to be ready...".format(streamlit_proc.pid))
    
    # Wait for Streamlit to be ready
    if not wait_for_service("http://localhost:8002", timeout=30):
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] Streamlit failed to start within 30 seconds")
    
    print("✅ Streamlit is ready on port 8002")

    # 6. Start ngrok as a background process
    print("🌐 Setting up ngrok tunnel...")
    
    # Create ngrok config file
    ngrok_config = f"""version: 2
authtoken: {NGROK_TOKEN}
tunnels:
  streamlit:
    proto: http
    addr: 8002
"""
    with open("ngrok.yml", "w") as f:
        f.write(ngrok_config)
    
    # Start ngrok in background
    ngrok_log = Path("ngrok.log").open("w", encoding="utf-8", buffering=1)
    ngrok_proc = subprocess.Popen(
        ["ngrok", "start", "--all", "--config", "ngrok.yml", "--log", "stdout"],
        stdout=ngrok_log,
        stderr=ngrok_log,
        start_new_session=True,
    )
    
    print("✅ ngrok started (PID: {}), waiting for tunnel to establish...".format(ngrok_proc.pid))
    
    # Wait for ngrok API to be available
    if not wait_for_service("http://localhost:4040/api/tunnels", timeout=15):
        ngrok_proc.terminate()
        ngrok_log.close()
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] ngrok API did not become available")
    
    # Get tunnel URL from ngrok API
    try:
        tunnel_url = get_ngrok_tunnel_url()
        if not tunnel_url:
            raise Exception("No tunnel URL found")
    except Exception as e:
        print(f"[ERROR] Could not get tunnel URL: {e}")
        print("Check ngrok.log for details:")
        run("tail -20 ngrok.log", shell=True)
        ngrok_proc.terminate()
        ngrok_log.close()
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit(1)
    
    print("✅ ngrok tunnel established")

    # Save service information for later retrieval
    save_service_info(tunnel_url, llama_proc.pid, streamlit_proc.pid, ngrok_proc.pid)

    print("\n" + "="*70)
    print("🎉 ALL SERVICES RUNNING SUCCESSFULLY!")
    print("="*70)
    print(f"🌐 Public URL: {tunnel_url}")
    print(f"🦙 llama-server PID: {llama_proc.pid}")
    print(f"📊 Streamlit PID: {streamlit_proc.pid}")
    print(f"🔌 ngrok PID: {ngrok_proc.pid}")
    print("="*70)
    print("\n📝 Service info saved to: service_info.json")
    print("📝 Tunnel URL saved to: tunnel_url.txt")
    print("📋 Logs available at: llama_server.log, streamlit.log, ngrok.log")
    print("\n💡 To check status later, run: !python launch_demo.py --status")
    print("💡 To get tunnel URL, run: !cat tunnel_url.txt")
    print("💡 To stop services, run: !python launch_demo.py --stop")
    print("\nℹ️  Services will continue running in the background.")
    print("ℹ️  The ngrok tunnel will remain active until stopped.")


def status():
    """Check the status of running services."""
    if not Path("service_info.json").exists():
        print("❌ No service info found. Services may not be running.")
        return
    
    with open("service_info.json", "r") as f:
        info = json.load(f)
    
    print("\n" + "="*70)
    print("SERVICE STATUS")
    print("="*70)
    print(f"Started at: {info['started_at']}")
    print(f"Public URL: {info['tunnel_url']}")
    print(f"llama-server PID: {info['llama_server_pid']}")
    print(f"Streamlit PID: {info['streamlit_pid']}")
    print(f"ngrok PID: {info['ngrok_pid']}")
    print("="*70)
    
    # Check if processes are still running
    for name, pid in [("llama-server", info['llama_server_pid']), 
                       ("Streamlit", info['streamlit_pid']),
                       ("ngrok", info['ngrok_pid'])]:
        try:
            os.kill(pid, 0)  # Check if process exists
            print(f"✅ {name} is running (PID: {pid})")
        except OSError:
            print(f"❌ {name} is NOT running (PID: {pid})")
    
    # Verify tunnel is still active
    print("\n🔍 Checking ngrok tunnel status...")
    try:
        tunnel_url = get_ngrok_tunnel_url(max_attempts=2, interval=1)
        if tunnel_url:
            print(f"✅ Tunnel is active: {tunnel_url}")
        else:
            print("⚠️  Could not verify tunnel status")
    except Exception as e:
        print(f"⚠️  Tunnel check failed: {e}")
    
    print("\n📋 Recent log entries:")
    print("\n--- llama_server.log (last 5 lines) ---")
    if Path("llama_server.log").exists():
        run("tail -5 llama_server.log", shell=True)
    
    print("\n--- streamlit.log (last 5 lines) ---")
    if Path("streamlit.log").exists():
        run("tail -5 streamlit.log", shell=True)
    
    print("\n--- ngrok.log (last 5 lines) ---")
    if Path("ngrok.log").exists():
        run("tail -5 ngrok.log", shell=True)


def stop():
    """Stop all running services."""
    if not Path("service_info.json").exists():
        print("❌ No service info found. Services may not be running.")
        return
    
    with open("service_info.json", "r") as f:
        info = json.load(f)
    
    print("🛑 Stopping services...")
    
    for name, pid in [("llama-server", info['llama_server_pid']), 
                       ("Streamlit", info['streamlit_pid']),
                       ("ngrok", info['ngrok_pid'])]:
        try:
            os.kill(pid, 15)  # SIGTERM
            print(f"✅ Stopped {name} (PID: {pid})")
            time.sleep(0.5)  # Give process time to terminate
        except OSError:
            print(f"⚠️  {name} (PID: {pid}) was not running")
    
    print("\n✅ All services stopped")
    
    # Clean up service info file
    try:
        os.remove("service_info.json")
        os.remove("tunnel_url.txt")
        print("🧹 Cleaned up service info files")
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            status()
        elif sys.argv[1] == "--stop":
            stop()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python launch_demo.py [--status|--stop]")
            sys.exit(1)
    else:
        main()
```

