"""Stand the whole demo up in one go: both agents deployed, online evals on, experiments run.

    python -m scripts.provision                        # everything (about 20 minutes)
    python -m scripts.provision --skip deploy          # reuse the deployments already there
    python -m scripts.provision --skip online offline  # deploy and smoke-test only

Stages, in this order:
  deploy   Both agents → development in parallel, then a smoke test of each. Production
           follows the same way once the online stage is done. A failed smoke test stops
           the run, so nothing reaches production after it.
  online   Evaluators, the review automation and the annotation queue on each agent's
           Production environment (scripts/setup_online_evals.py). It runs before the
           production deploy, so the production smoke traffic is the first thing graded.
  offline  The classify node on the guardrail (fast) model, the deep agent on the main and
           fast models, a pairwise judge over those two, then cleanup of eval sandboxes.

The first deploy creates each agent and its Local/Development/Staging/Production
environments, which the other stages need. Rerunning is safe: deploys update the same
deployments, datasets and rules are reused, and each offline stage adds new experiments.
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import deque
from pathlib import Path

from langsmith import Client

from chat_langchain.settings import FAST_MODEL, JUDGE_MODEL, MAIN_MODEL, ONLINE_JUDGE_CONFIG
from evals import classifier_eval, deep_agent_eval
from evals.common import DEEPAGENT_AGENT, WORKFLOW_AGENT
from scripts import setup_online_evals, smoke
from scripts.cleanup_sandboxes import cleanup

ROOT = Path(__file__).resolve().parent.parent
AGENTS = {  # agent id → (langgraph config, smoke-test name)
    WORKFLOW_AGENT: ("langgraph.workflow.json", "workflow"),
    DEEPAGENT_AGENT: ("langgraph.deepagent.json", "deepagent"),
}
STAGES = ("deploy", "online", "offline")
REQUIRED_ENV = ("LANGSMITH_API_KEY", "LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_SANDBOX_API_KEY")
CUSTOM = "langsmith:custom/"


class ProvisionError(Exception):
    pass


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def preflight(skip: set[str]) -> None:
    """Fail in seconds, not after a 10-minute deploy. Only names are checked, never values."""
    problems = []
    if not (ROOT / ".env").exists():
        problems.append(".env is missing: cp .env.example .env and fill it in")
    problems += [
        f"{name} is not set (see .env.example)"
        for name in REQUIRED_ENV
        if os.getenv(name, "").endswith("...") or not os.getenv(name)  # "..." = the placeholder
    ]
    creates_judge = "online" not in skip and bool(os.getenv("CHATLC_ONLINE_JUDGE_API_KEY"))
    if creates_judge and not os.getenv("CHATLC_ONLINE_JUDGE_BASE_URL"):
        problems.append("CHATLC_ONLINE_JUDGE_BASE_URL is not set (needed with CHATLC_ONLINE_JUDGE_API_KEY)")
    if problems:
        raise ProvisionError("\n  - ".join(["preflight failed:", *problems]))

    client = Client()
    models = {MAIN_MODEL, FAST_MODEL} | ({JUDGE_MODEL} if "offline" not in skip else set())
    wanted = {m.removeprefix(CUSTOM) for m in models if m.startswith(CUSTOM)}
    if "online" not in skip and not creates_judge:  # otherwise the online stage creates it
        wanted.add(ONLINE_JUDGE_CONFIG)
    try:
        workspaces = client.request_with_retries("GET", "/api/v1/workspaces").json()
        configs = client.request_with_retries("GET", "/playground-settings").json()
    except Exception as error:  # the error carries the status and URL, never the key
        raise ProvisionError(
            f"LangSmith rejected LANGSMITH_API_KEY ({str(error)[:160]}). Use a service key "
            "from the target workspace with the Workspace Admin role."
        ) from None
    if len(workspaces) != 1:  # a personal token or org key: deploys could land anywhere
        names = ", ".join(repr(w["display_name"]) for w in workspaces)
        raise ProvisionError(
            f"LANGSMITH_API_KEY reaches {len(workspaces)} workspaces ({names}). Use a service "
            "key created in the one workspace to provision, with the Workspace Admin role."
        )
    have = {c["name"] for c in configs}
    problems += [f"no model configuration named {name!r}" for name in sorted(wanted - have)]
    if "deploy" in skip:  # the other stages need the agents the first deploy creates
        for agent in AGENTS:
            try:
                client.request_with_retries("GET", f"/api/v1/agents/{agent}")
            except Exception:
                problems.append(f"agent {agent!r} doesn't exist yet: run the deploy stage first")
    if problems:
        raise ProvisionError("\n  - ".join(["preflight failed:", *problems]))
    print(
        f"preflight ok: workspace {workspaces[0]['display_name']!r}, keys set, "
        f"model configurations found ({', '.join(sorted(wanted))})"
    )


async def deploy(agent: str, environment: str) -> str:
    """`langgraph deploy` into one agent environment. Returns the deployment URL."""
    tag = f"[{agent} → {environment}]"
    proc = await asyncio.create_subprocess_exec(
        "langgraph", "deploy",
        "--config", AGENTS[agent][0],
        "--agent-id", agent,
        "--agent-environment", environment,
        "--remote", "--no-input", "--json",
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=2**20,
    )
    result, status, tail = {}, None, deque(maxlen=30)
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if not isinstance(event, dict):
            tail.append(line)
            continue
        kind = event.get("event")
        if kind == "result":
            result = event
        elif kind == "log":  # build output: kept for the error message only
            tail.append(event.get("message", ""))
        elif kind == "heartbeat":
            if event.get("status") != status:
                status = event.get("status")
                print(f"{tag} {status}…", flush=True)
        elif kind != "upload_progress" and event.get("message"):
            print(f"{tag} {event['message']}", flush=True)
    await proc.wait()
    if proc.returncode or result.get("status") != "succeeded" or not result.get("url"):
        reason = result.get("message") or f"exit code {proc.returncode}"
        raise ProvisionError(f"{tag} deploy failed: {reason}\n    " + "\n    ".join(tail))
    print(f"{tag} live at {result['url']}", flush=True)
    return result["url"]


def latest_model_error(agent: str, environment: str) -> str:
    """A deployment reports a failed model call as "An internal error occurred". The trace
    has the real error, such as a 402 from a gateway spend policy."""
    try:
        client = Client()
        agent_info = client.request_with_retries("GET", f"/api/v1/agents/{agent}").json()
        env = next(e for e in agent_info["environments"] if e["environment"] == environment.upper())
        run = next(iter(client.list_runs(project_id=env["tracer_session_id"], run_type="llm", error=True, limit=1)), None)
    except Exception:  # best effort: never hide the smoke test's own failure
        return ""
    if not run:
        return ""
    line = re.sub(r"(lsv2_\w\w_|sk-)[\w\-*]+", r"\1***", run.error.strip().splitlines()[-1])  # e.g. "Incorrect API key: sk-…"
    return f"\n  latest model error in the trace: {line[:300]}"


async def deploy_and_smoke(environment: str) -> dict[str, str]:
    urls = await asyncio.gather(*(deploy(agent, environment) for agent in AGENTS))
    for agent, url in zip(AGENTS, urls):
        banner(f"smoke test: {agent} ({environment})")
        try:
            ok, reason = await smoke.main(AGENTS[agent][1], url), "a check failed"
        except RuntimeError as error:
            ok, reason = False, str(error)
        if not ok:
            raise ProvisionError(
                f"smoke test failed for {agent} in {environment}: {reason}"
                + latest_model_error(agent, environment)
            )
    return dict(zip(AGENTS, urls))


async def offline() -> list[str]:
    # Only the guardrail's own model: the classifier forces a tool call, which the main
    # model (Claude Sonnet 5.5) rejects.
    experiments = [await classifier_eval.main(FAST_MODEL)]
    try:
        main_run = await deep_agent_eval.run(MAIN_MODEL)
        fast_run = await deep_agent_eval.run(FAST_MODEL)
    finally:
        cleanup("chatlc-eval-")  # every eval example ran on its own thread sandbox
    await asyncio.to_thread(deep_agent_eval.pairwise, main_run, fast_run)
    return [*experiments, main_run, fast_run]


async def main(skip: set[str]) -> None:
    started = time.monotonic()
    banner("preflight")
    preflight(skip)

    urls: dict[str, dict[str, str]] = {}
    if "deploy" not in skip:
        banner("deploy → development")
        urls["development"] = await deploy_and_smoke("development")
    if "online" not in skip:
        banner("online evals on Production")
        setup_online_evals.main(None)
    if "deploy" not in skip:
        banner("deploy → production")
        urls["production"] = await deploy_and_smoke("production")
    experiments = []
    if "offline" not in skip:
        banner("offline evals")
        experiments = await offline()

    banner(f"done in {(time.monotonic() - started) / 60:.0f} min")
    for environment, by_agent in urls.items():
        for agent, url in by_agent.items():
            print(f"{agent:<26} {environment:<12} {url}")
    if "online" not in skip:
        print(f"online evals: 3 rules on each agent's Production → queue {setup_online_evals.QUEUE_NAME!r}")
    for name in experiments:
        print(f"experiment: {name}")
    if experiments:
        print("compare: dataset chat-langchain-qa → select both deepagent-* experiments → Compare")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[], help="stages to leave out")
    try:
        asyncio.run(main(set(parser.parse_args().skip)))
    except ProvisionError as error:
        raise SystemExit(f"\nprovision stopped: {error}") from None
