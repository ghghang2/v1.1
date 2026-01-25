#!/usr/bin/env python3
"""
Push the contents of a local folder to a GitHub repo
while respecting a .gitignore.

* Works in Colab, Jupyter, or a local terminal.
* Uses a token for authentication.
* Creates the repo on GitHub if it does not exist.
* Always pushes to a branch called `main`.
"""

import sys
from pathlib import Path

from github import Github, GithubException
from github.Auth import Token
from git import Repo, GitCommandError, InvalidGitRepositoryError

# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------
LOCAL_DIR      = Path(__file__).parent          # <-- folder you want to push
REPO_NAME      = "v0"                      # GitHub repo name
USER_NAME      = "ghghang2"         # e.g. ghghang2
TOKEN          = "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"     # personal access token

IGNORED_ITEMS = [
    ".config",
    ".ipynb_checkpoints",
    "sample_data",
    "llama-server",
    "nohup.out",
]

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def remote_url() -> str:
    """HTTPS URL that contains the token (for git push)."""
    return f"https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{REPO_NAME}.git"

def ensure_remote(repo: Repo, url: str) -> None:
    """Attach or replace the 'origin' remote."""
    if "origin" in repo.remotes:
        repo.delete_remote("origin")
    repo.create_remote("origin", url)

def create_repo_if_missing(g: Github) -> None:
    """Create the GitHub repo if it does not exist yet."""
    user = g.get_user()
    try:
        user.get_repo(REPO_NAME)
        print(f"Repo '{REPO_NAME}' already exists on GitHub.")
    except GithubException:
        user.create_repo(REPO_NAME, private=False)
        print(f"Created repo '{REPO_NAME}' on GitHub.")

def write_gitignore(repo_path: Path, items: list[str]) -> None:
    """Create a .gitignore that excludes the supplied items."""
    gitignore_path = repo_path / ".gitignore"
    gitignore_path.write_text("\n".join(items) + "\n")
    print(f"Created .gitignore at {gitignore_path}")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main() -> None:
    if not LOCAL_DIR.is_dir():
        print(f"Local directory '{LOCAL_DIR}' not found.")
        sys.exit(1)

    repo_path = LOCAL_DIR

    # Open an existing repo or create a new one
    try:
        repo = Repo(repo_path)
        if repo.bare:
            raise InvalidGitRepositoryError(repo_path)
        print(f"Using existing git repo at {repo_path}")
    except (InvalidGitRepositoryError, GitCommandError):
        repo = Repo.init(repo_path)
        print(f"Initialised new git repo at {repo_path}")

    # Create the remote repo on GitHub (if needed)
    g = Github(auth=Token(TOKEN))
    create_repo_if_missing(g)

    # Attach the remote URL
    ensure_remote(repo, remote_url())

    # Create .gitignore
    write_gitignore(repo_path, IGNORED_ITEMS)

    # Stage everything (ignores applied)
    repo.git.add(A=True)

    # Commit – ignore "nothing to commit" error
    try:
        repo.index.commit("Initial commit with .gitignore")
        print("Committed changes.")
    except GitCommandError as e:
        if "nothing to commit" in str(e):
            print("Nothing new to commit.")
        else:
            raise

    # Switch to / create the local 'main' branch
    if "main" not in [b.name for b in repo.branches]:
        repo.git.checkout("-b", "main")
        print("Created local branch 'main'.")
    else:
        repo.git.checkout("main")
        print("Switched to existing branch 'main'.")

    # Push 'main' and set upstream
    try:
        # -u sets the upstream branch for future `git push` / `git pull`
        repo.git.push("-u", "origin", "main")
        print("Push complete. Remote 'main' is now tracked.")
    except GitCommandError as exc:
        print(f"Git operation failed: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()