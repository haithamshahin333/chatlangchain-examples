# chat-langchain: Deep Agent

A `create_deep_agent` loop that answers LangChain-ecosystem questions. It searches the
public docs MCP servers, reads upstream source in a sandbox, and checks code before it
returns it.

This folder is a complete LangGraph project. `langgraph deploy` uploads only this folder,
so it holds everything the deployment runs. It shares no code with `../workflow`.

| File | What it does |
|---|---|
| `chat_langchain_deepagent/agent.py` | Async graph factory `make_graph(config)` (graph id `chat_langchain_deepagent`): system prompt, `source-diver` subagent, `create_deep_agent(...)` |
| `chat_langchain_deepagent/guardrail.py` | `RelevanceGuardrail` middleware: `before_agent` refuses off-topic turns and jumps to the end |
| `chat_langchain_deepagent/classifier.py` | The guardrail's classifier: `Relevance` schema, prompt, `classify()`/`aclassify()`, the refusal text |
| `chat_langchain_deepagent/sandbox.py` | Gets or creates the thread's LangSmith sandbox (`chatlc-<thread_id>`) and seeds the skill into it |
| `chat_langchain_deepagent/docs_tools.py` | Loads the docs and API-reference MCP tools with `MCPAdapter` |
| `chat_langchain_deepagent/skills/validate-snippet/` | The skill: `SKILL.md` plus `check_snippet.py`, which runs inside the sandbox |
| `chat_langchain_deepagent/settings.py` | Models (`CHATLC_MAIN_MODEL`, `CHATLC_FAST_MODEL`), MCP URLs, sandbox naming and TTLs |
| `langgraph.json` | Deployment config. Secrets come from the repo-root `.env` (`"env": "../../.env"`). |
| `pyproject.toml` | This agent's dependencies |

Run, deploy and evaluate it from the repo root: `make dev-deepagent`, `make deploy-deepagent-dev`,
`make eval-deepagent MODEL=…`. See the root README.
