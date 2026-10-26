"""Thread-scoped LangSmith sandboxes.

Every conversation thread gets its own sandbox named `chatlc-<thread_id>`. The
first turn creates it; later turns on the same thread find it by name, so repos
cloned on turn 1 are still there on turn 5. Idle sandboxes stop after 10 minutes
and wake on the next command; stopped ones are deleted after a day.

No secrets ever go into the sandbox: it only sees the user's code and public repos.
"""

import functools
import os
from pathlib import Path

from deepagents.backends.langsmith import LangSmithSandbox
from langsmith.sandbox import ResourceNotFoundError, Sandbox, SandboxClient

from chat_langchain.settings import (
    SANDBOX_DELETE_AFTER_STOP_SECONDS,
    SANDBOX_IDLE_TTL_SECONDS,
    SANDBOX_PREFIX,
    SANDBOX_WORKDIR,
)

SKILLS_DIR = Path(__file__).parent / "skills"


@functools.cache
def client() -> SandboxClient:
    # Defaults to LANGSMITH_API_KEY; a deployment can pass a dedicated key instead.
    return SandboxClient(api_key=os.getenv("LANGSMITH_SANDBOX_API_KEY") or None)


def sandbox_name(thread_id: str) -> str:
    return f"{SANDBOX_PREFIX}{thread_id}"


def get_or_create(thread_id: str) -> Sandbox:
    name = sandbox_name(thread_id)
    try:
        return client().get_sandbox(name)
    except ResourceNotFoundError:
        sandbox = client().create_sandbox(
            name=name,
            idle_ttl_seconds=SANDBOX_IDLE_TTL_SECONDS,
            delete_after_stop_seconds=SANDBOX_DELETE_AFTER_STOP_SECONDS,
            timeout=120,
        )
        _seed(LangSmithSandbox(sandbox=sandbox))
        return sandbox


def _seed(backend: LangSmithSandbox) -> None:
    """Create the work dirs and copy the bundled skills in, so skill scripts run here."""
    w = SANDBOX_WORKDIR
    backend.execute(f"mkdir -p {w}/repos {w}/snippets {w}/skills")
    backend.upload_files(
        [
            (f"{w}/skills/{path.relative_to(SKILLS_DIR)}", path.read_bytes())
            for path in SKILLS_DIR.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
    )
