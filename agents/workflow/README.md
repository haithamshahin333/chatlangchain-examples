# chat-langchain: LangGraph workflow

A fixed graph that answers LangChain-ecosystem questions from the public docs MCP servers.

```
START -> classify --relevant--> retrieve -> respond -> END
                 \--off-topic--> refuse ------------> END
```

This folder is a complete LangGraph project. `langgraph deploy` uploads only this folder,
so it holds everything the deployment runs. It shares no code with `../deepagent`.

| File | What it does |
|---|---|
| `chat_langchain_workflow/graph.py` | The `StateGraph` (graph id `chat_langchain_workflow`): `classify`, `refuse`, `retrieve`, `respond` |
| `chat_langchain_workflow/classifier.py` | The relevance guardrail: `Relevance` schema, prompt, `aclassify()`, the refusal text |
| `chat_langchain_workflow/docs_tools.py` | Loads the docs and API-reference MCP tools with `MCPAdapter` |
| `chat_langchain_workflow/settings.py` | Models (`CHATLC_MAIN_MODEL`, `CHATLC_FAST_MODEL`) and MCP URLs |
| `langgraph.json` | Deployment config. Secrets come from the repo-root `.env` (`"env": "../../.env"`). |
| `pyproject.toml` | This agent's dependencies (no `deepagents`) |

Run, deploy and evaluate it from the repo root: `make dev-workflow`, `make deploy-workflow-dev`,
`make eval-classifier`. See the root README.
