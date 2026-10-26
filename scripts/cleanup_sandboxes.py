"""Delete this demo's sandboxes. Defaults to the ones evals create.

    python scripts/cleanup_sandboxes.py               # chatlc-eval-*
    python scripts/cleanup_sandboxes.py chatlc-       # every chat-langchain thread sandbox
"""

import sys

from chat_langchain.sandbox import client


def cleanup(prefix: str = "chatlc-eval-") -> int:
    doomed = [sb.name for sb in client().list_sandboxes() if sb.name.startswith(prefix)]
    for name in doomed:
        client().delete_sandbox(name)
        print(f"deleted {name}")
    print(f"{len(doomed)} sandbox(es) matching {prefix!r} deleted")
    return len(doomed)


if __name__ == "__main__":
    cleanup(sys.argv[1] if len(sys.argv) > 1 else "chatlc-eval-")
