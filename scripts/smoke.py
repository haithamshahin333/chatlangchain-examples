"""Drive a running agent (local `langgraph dev` or a LangSmith Deployment) end to end.

    python scripts/smoke.py workflow  <url>
    python scripts/smoke.py deepagent <url>

Checks: an on-topic question gets an answer, an off-topic one gets the refusal,
and (deep agent) the thread keeps the same sandbox across turns.
"""

import argparse
import asyncio
import time
import uuid

from chat_langchain_deepagent.classifier import REFUSAL as DEEPAGENT_REFUSAL
from chat_langchain_deepagent.sandbox import client as sandbox_client
from chat_langchain_deepagent.sandbox import sandbox_name
from chat_langchain_deepagent.settings import SANDBOX_WORKDIR
from chat_langchain_workflow.classifier import REFUSAL as WORKFLOW_REFUSAL
from langgraph_sdk import get_client

GRAPHS = {"workflow": "chat_langchain_workflow", "deepagent": "chat_langchain_deepagent"}
REFUSALS = {"workflow": WORKFLOW_REFUSAL, "deepagent": DEEPAGENT_REFUSAL}


async def ask(client, assistant: str, thread_id: str, text: str) -> str:
    """Stream the run the way Studio does: streaming exercises code paths `runs.wait` skips."""
    start = time.monotonic()
    state = None
    async for event in client.runs.stream(
        thread_id,
        assistant,
        input={"messages": [{"role": "user", "content": text}]},
        config={"recursion_limit": 150},
        stream_mode=["messages-tuple", "values"],
    ):
        if event.event == "error":
            raise RuntimeError(f"run failed while streaming: {event.data}")
        if event.event == "values":
            state = event.data
    answer = state["messages"][-1]["content"]
    if isinstance(answer, list):  # content blocks
        answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
    print(f"\n> {text}\n[{time.monotonic() - start:.0f}s] {answer[:400]}{'…' if len(answer) > 400 else ''}")
    return answer


def check(label: str, ok: bool) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    return ok


async def main(agent: str, url: str) -> bool:
    client = get_client(url=url)
    assistant, refusal = GRAPHS[agent], REFUSALS[agent]
    results = []

    thread = (await client.threads.create(thread_id=str(uuid.uuid4())))["thread_id"]
    answer = await ask(client, assistant, thread, "How do I add memory to a LangGraph agent?")
    results.append(check("on-topic question is answered", refusal not in answer))

    if agent == "deepagent":
        await ask(
            client,
            assistant,
            thread,
            "In the deepagents source, where does create_deep_agent build its middleware stack?",
        )
        sandbox = sandbox_client().get_sandbox(sandbox_name(thread))
        results.append(check(f"thread sandbox {sandbox.name} exists", True))
        await ask(client, assistant, thread, "And where is SkillsMiddleware defined in that repo?")
        listing = sandbox.run(f"ls {SANDBOX_WORKDIR}/repos").stdout
        results.append(check(f"same sandbox kept the clone ({listing.split()})", "deepagents" in listing))

    off_topic = (await client.threads.create())["thread_id"]
    answer = await ask(client, assistant, off_topic, "What's a good recipe for banana bread?")
    results.append(check("off-topic question is refused", refusal in answer))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=GRAPHS)
    parser.add_argument("url")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(args.agent, args.url)) else 1)
