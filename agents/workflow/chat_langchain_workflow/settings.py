"""Every tunable for the workflow agent. Each value can be overridden with an env var.

Models are org-level saved model configurations, called through the LangSmith LLM
Gateway. `langsmith:custom/<name>` makes init_chat_model build a ChatOpenAI client
on gateway.smith.langchain.com/v1, authenticated with LANGSMITH_GATEWAY_API_KEY.
The gateway looks the configuration up by name, attaches the provider credential
the org admin stored with it, and translates the request to the provider's API.
No provider key ever reaches this code.
"""

import os

# The `respond` node.
MAIN_MODEL = os.getenv("CHATLC_MAIN_MODEL", "langsmith:custom/claude-sonnet-5-5")
# The `classify` node. It forces a tool call, so it must be a model that accepts one.
FAST_MODEL = os.getenv("CHATLC_FAST_MODEL", "langsmith:custom/claude-haiku-5-5")

DOCS_MCP_URL = "https://docs.langchain.com/mcp"
REFERENCE_MCP_URL = "https://reference.langchain.com/mcp"
