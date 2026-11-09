---
name: validate-snippet
description: Use before returning any Python code that imports LangChain, LangGraph, Deep Agents or LangSmith, and whenever the user asks you to check or fix such code. Confirms the code compiles and that every imported module and name really exists in the current released packages.
---

# Validate a Python snippet

Hallucinated imports are the most common way a LangChain answer goes wrong. This
skill checks a snippet against the real, current packages from PyPI inside your
sandbox. It never runs the snippet itself, so no API keys are needed.

## Steps

1. Write the exact code you plan to show to `/work/snippets/<short-name>.py`.
2. Run the checker, adding a `--with` for every integration package the snippet
   imports (for example `--with langchain-anthropic`):

   ```bash
   cd /work/snippets && uv run --quiet --no-project \
     --with langchain --with langchain-core --with langgraph \
     --with deepagents --with langsmith \
     python /work/skills/validate-snippet/check_snippet.py <short-name>.py
   ```

3. Read the report:
   - `OK` lines are verified.
   - `MISSING` lines name an import that does not exist. Find the real name with the
     `search_api` or `get_symbol` tool, fix the code, and run the checker again.
   - A `SyntaxError` means the code does not compile. Fix it and run again.
4. Only return the code once the checker exits `PASS`. Mention in your answer that
   the imports were checked, and list the package versions it reports.
