"""Eval judges. Each value can be overridden with an env var.

The offline judge grades both sides of the model comparison, so it stays fixed.
The online judge is the name of a model configuration; LangSmith runs it, not this code.
Keep it outside the Claude family: a different family avoids self-preference, and the
evaluator runtime forces a tool call, which Claude Sonnet/Opus 5.5 reject.
With CHATLC_ONLINE_JUDGE_API_KEY and CHATLC_ONLINE_JUDGE_BASE_URL set (in .env.judge), the
setup script creates a workspace configuration named chat-langchain-judge for that
OpenAI-compatible endpoint. Without them it expects an existing one named gpt-5.5.
"""

import os

JUDGE_MODEL = os.getenv("CHATLC_JUDGE_MODEL", "langsmith:custom/claude-sonnet-5-5")
ONLINE_JUDGE_CONFIG = os.getenv(
    "CHATLC_ONLINE_JUDGE_CONFIG",
    "chat-langchain-judge" if os.getenv("CHATLC_ONLINE_JUDGE_API_KEY") else "gpt-5.5",
)
ONLINE_JUDGE_MODEL = os.getenv("CHATLC_ONLINE_JUDGE_MODEL", "gpt-5.5")  # for a created configuration
