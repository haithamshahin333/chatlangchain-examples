"""Online code evaluator: does a production answer cite its sources?

Uploaded verbatim as the code of the `cites_sources` run rule. LangSmith runs it in a
sandbox (stdlib only, no network) on each root run of an agent's Production environment.
`run` arrives as a plain dict. The returned score becomes feedback named after the rule.

An answer without a docs link, an API-reference link or a source permalink is the
cheapest signal that the agent answered from memory instead of from the docs.
"""


def perform_eval(run):
    import re

    messages = (run.get("outputs") or {}).get("messages") or []
    if not messages:
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):  # content blocks (Responses API)
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    text = content or ""
    if text.startswith("I can only help with the LangChain ecosystem"):
        return None  # a guardrail refusal has nothing to cite; guardrail_audit grades it
    links = re.findall(
        r"https?://(?:docs\.langchain\.com|reference\.langchain\.com|github\.com/langchain-ai)/\S+",
        text,
    )
    return {"score": 1 if links else 0, "comment": f"{len(links)} source link(s) in the answer"}
