"""Does the validate-snippet skill make the deep agent's code better? A with/without ablation.

    python -m evals.skill_eval                    # both arms, 3 repetitions each
    python -m evals.skill_eval --repetitions 1    # a quicker pass

Two experiments over the same questions (evals/data/skill.jsonl), same model, same judge:
  off    no skill, and no system-prompt line about it. The baseline.
  skill  the skill is loaded, but there is no system-prompt line, so only the skill's own
         description can trigger it.

Evaluators:
  imports_valid  the Python in the final answer passes the skill's own checker. It runs in a
                 separate grader sandbox, whether or not the agent ran the check itself.
  correctness    the deep agent eval's LLM judge, against the reference answer.
  skill_used     (skill arm only) the agent read SKILL.md on rows that need Python, and
                 didn't on rows that don't.

In LangSmith: dataset chat-langchain-skill → Experiments → Group by `variant`, or select
both experiments → Compare. Every example runs on its own sandbox, and so does the grader
(chatlc-eval-grader); `make eval-skill` deletes them afterwards.
"""

import argparse
import ast
import asyncio
import hashlib
import re
import sys
import uuid
from functools import cache

from chat_langchain_deepagent.agent import make_graph
from chat_langchain_deepagent.sandbox import SKILLS_DIR
from chat_langchain_deepagent.sandbox import client as sandbox_client
from chat_langchain_deepagent.settings import MAIN_MODEL
from deepagents.backends.langsmith import LangSmithSandbox
from langchain_core.messages import convert_to_messages
from langsmith import Client, aevaluate
from langsmith.sandbox import ResourceNotFoundError

from evals.common import DEEPAGENT_AGENT, ensure_dataset, short_name, tag_experiment
from evals.deep_agent_eval import correctness
from evals.settings import JUDGE_MODEL

DATASET = "chat-langchain-skill"
VARIANTS = ("off", "skill")
SKILL_DIR = SKILLS_DIR / "validate-snippet"


def skill_version() -> str:
    """Short hash of the skill folder, so experiments on different skill versions don't mix."""
    digest = hashlib.sha256()
    for path in sorted(SKILL_DIR.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(SKILL_DIR)).encode() + path.read_bytes())
    return digest.hexdigest()[:12]


# --- target ------------------------------------------------------------------------------

def make_target(variant: str):
    async def target(inputs: dict) -> dict:
        configurable = {"thread_id": f"eval-{uuid.uuid4()}", "model": MAIN_MODEL, "skill_variant": variant}
        graph = await make_graph({"configurable": configurable})
        result = await graph.ainvoke(
            {"messages": convert_to_messages(inputs["messages"])},
            {"configurable": configurable, "recursion_limit": 150},
        )
        calls = [tc for m in result["messages"] if m.type == "ai" for tc in m.tool_calls]
        return {
            "answer": result["messages"][-1].text,
            "skill_read": any(
                tc["name"] == "read_file"
                and str(tc["args"].get("file_path", "")).endswith("validate-snippet/SKILL.md")
                for tc in calls
            ),
            "skill_ran": any(
                tc["name"] == "execute" and "check_snippet.py" in str(tc["args"].get("command", ""))
                for tc in calls
            ),
            "tools_called": [tc["name"] for tc in calls],
        }

    return target


# --- grader: the skill's checker, run on the final answer in its own sandbox -------------

GRADER = "chatlc-eval-grader"  # the chatlc-eval- prefix puts it in `make cleanup`'s scope
BASE_PACKAGES = ("langchain", "langchain-core", "langgraph", "deepagents", "langsmith")
COVERED = {"langchain", "langchain_core", "langgraph", "deepagents", "langsmith", "pydantic", "typing_extensions"}
# Modules whose PyPI name isn't the module name with "_" → "-".
SPECIAL = {
    "langchain.mcp": "fastmcp",  # the langchain[mcp] extra
    "langgraph.checkpoint.postgres": "langgraph-checkpoint-postgres",
    "langgraph.checkpoint.sqlite": "langgraph-checkpoint-sqlite",
    "langgraph.store.postgres": "langgraph-checkpoint-postgres",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
}
SAFE_PACKAGE = re.compile(r"^[A-Za-z0-9_-]+$")  # the only text from the answer that reaches the shell
CODE_BLOCK = re.compile(r"```(\w*)[^\n]*\n(.*?)```", re.DOTALL)


def python_blocks(answer: str) -> list[str]:
    """Fenced blocks tagged python, plus untagged ones that import something."""
    return [
        code
        for lang, code in CODE_BLOCK.findall(answer)
        if lang.lower() in ("python", "py")
        or (not lang and re.search(r"^\s*(from|import)\s+\w", code, re.MULTILINE))
    ]


def extra_packages(code: str) -> list[str]:
    """PyPI packages the snippet needs beyond the base set, from its top-level imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # the checker reports the SyntaxError itself
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    packages = set()
    for module in modules:
        special = next((pkg for prefix, pkg in SPECIAL.items() if module.startswith(prefix)), None)
        top = module.split(".")[0]
        if special:
            packages.add(special)
        elif top not in COVERED and top not in sys.stdlib_module_names:
            packages.add(top.replace("_", "-"))
    return sorted(p for p in packages if SAFE_PACKAGE.match(p))


@cache
def grader() -> LangSmithSandbox:
    """The grader sandbox. Built once in main() before the evals start, never concurrently."""
    try:
        sandbox = sandbox_client().get_sandbox(GRADER)
    except ResourceNotFoundError:
        sandbox = sandbox_client().create_sandbox(
            name=GRADER, idle_ttl_seconds=600, delete_after_stop_seconds=3600, timeout=120
        )
    backend = LangSmithSandbox(sandbox=sandbox)
    backend.execute("mkdir -p /grader/snippets")
    backend.upload_files([("/grader/check_snippet.py", (SKILL_DIR / "check_snippet.py").read_bytes())])
    return backend


def check(code: str) -> tuple[bool, str]:
    """Run check_snippet.py on one block. Returns (passed, the MISSING/SyntaxError lines)."""
    path = f"/grader/snippets/{uuid.uuid4().hex}.py"
    grader().upload_files([(path, code.encode())])
    withs = " ".join(f"--with {pkg}" for pkg in (*BASE_PACKAGES, *extra_packages(code)))
    result = grader().execute(
        f"cd /grader && uv run --quiet --no-project {withs} python check_snippet.py {path}", timeout=300
    )
    problems = [line for line in result.output.splitlines() if line.startswith(("MISSING", "SyntaxError"))]
    return result.exit_code == 0, "; ".join(problems) or result.output.strip()[-300:]


# --- evaluators --------------------------------------------------------------------------

def imports_valid(outputs: dict) -> dict:
    blocks = python_blocks(outputs["answer"])
    if not blocks:
        return {"key": "imports_valid", "score": None, "comment": "no Python code in the answer"}
    failures = [detail for ok, detail in map(check, blocks) if not ok]
    return {
        "key": "imports_valid",
        "score": int(not failures),
        "comment": " | ".join(failures) or f"{len(blocks)} Python block(s) pass",
    }


def skill_used(outputs: dict, reference_outputs: dict) -> dict:
    expected, read = reference_outputs["needs_python"], outputs["skill_read"]
    return {
        "key": "skill_used",
        "score": int(read == expected),
        "comment": f"read SKILL.md: {read}, ran the checker: {outputs['skill_ran']}, needs Python: {expected}",
    }


# --- runs ----------------------------------------------------------------------------------

async def run(variant: str, repetitions: int) -> str:
    client = Client()
    ensure_dataset(
        client,
        DATASET,
        "skill.jsonl",
        "Questions whose answers need Python (write, fix) or don't (no_code), for the "
        "validate-snippet skill ablation.",
        agents=[DEEPAGENT_AGENT],
    )
    results = await aevaluate(
        make_target(variant),
        data=DATASET,
        evaluators=[imports_valid, correctness] + ([skill_used] if variant == "skill" else []),
        experiment_prefix=f"skill-{variant}-{short_name(MAIN_MODEL)}",
        metadata={
            "models": MAIN_MODEL,
            "variant": variant,
            "skill_version": skill_version(),
            "judge": JUDGE_MODEL,
        },
        num_repetitions=repetitions,
        max_concurrency=4,
        client=client,
    )
    tag_experiment(client, results.experiment_name, [DEEPAGENT_AGENT])
    print(f"experiment: {results.experiment_name} (variant {variant})")
    return results.experiment_name


async def main(repetitions: int) -> list[str]:
    ok, detail = check("import langchain\n")  # creates the grader and warms its uv cache
    if not ok:
        raise SystemExit(f"grader sandbox is not working: {detail}")
    return [await run(variant, repetitions) for variant in VARIANTS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    asyncio.run(main(parser.parse_args().repetitions))
