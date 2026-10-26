"""Every tunable in one place. Each value can be overridden with an env var.

Models are org-level saved model configurations, called through the LangSmith LLM
Gateway. `langsmith:custom/<name>` makes init_chat_model build a ChatOpenAI client
on gateway.smith.langchain.com/v1, authenticated with LANGSMITH_GATEWAY_API_KEY.
The gateway looks the configuration up by name, attaches the provider credential
the org admin stored with it, and translates the request to the provider's API.
No provider key ever reaches this code.
"""

import os

MAIN_MODEL = os.getenv("CHATLC_MAIN_MODEL", "langsmith:custom/claude-sonnet-5-5")
# Classifier guardrail and the source-diver subagent.
FAST_MODEL = os.getenv("CHATLC_FAST_MODEL", "langsmith:custom/claude-haiku-5-5")

# Eval judges. The offline judge grades both sides of the model comparison, so it stays fixed.
# The online judge is the name of a model configuration; LangSmith runs it, not this code.
# Keep it outside the Claude family: a different family avoids self-preference, and the
# evaluator runtime forces a tool call, which Claude Sonnet/Opus 5.5 reject.
# With CHATLC_ONLINE_JUDGE_API_KEY and CHATLC_ONLINE_JUDGE_BASE_URL set (in .env.judge), the
# setup script creates a workspace configuration named chat-langchain-judge for that
# OpenAI-compatible endpoint. Without them it expects an existing one named gpt-5.5.
JUDGE_MODEL = os.getenv("CHATLC_JUDGE_MODEL", "langsmith:custom/claude-sonnet-5-5")
ONLINE_JUDGE_CONFIG = os.getenv(
    "CHATLC_ONLINE_JUDGE_CONFIG",
    "chat-langchain-judge" if os.getenv("CHATLC_ONLINE_JUDGE_API_KEY") else "gpt-5.5",
)
ONLINE_JUDGE_MODEL = os.getenv("CHATLC_ONLINE_JUDGE_MODEL", "gpt-5.5")  # for a created configuration

DOCS_MCP_URL ="https://docs.langchain.com/mcp"
REFERENCE_MCP_URL = "https://reference.langchain.com/mcp"

# One LangSmith sandbox per conversation thread, named after the thread.
SANDBOX_PREFIX = "chatlc-"
SANDBOX_IDLE_TTL_SECONDS = 600  # stop after 10 idle minutes; wakes on next command
SANDBOX_DELETE_AFTER_STOP_SECONDS = 86_400  # delete a day after it stops
SANDBOX_WORKDIR = os.getenv("CHATLC_SANDBOX_WORKDIR", "/work")
