"""Evaluate ONE node of the LangGraph workflow: the `classify` guardrail.

The target calls the node function directly with a state dict. The graph,
retrieval and answer model never run, so the experiment measures the routing
decision and nothing else. The same classifier is the deep agent's middleware
guardrail, so this grades both agents' front door.

    python -m evals.classifier_eval                      # default classifier model
    python -m evals.classifier_eval --model langsmith:custom/claude-sonnet-5-5
"""

import argparse
import asyncio

from langchain_core.messages import convert_to_messages
from langsmith import Client, aevaluate

from chat_langchain.settings import FAST_MODEL
from chat_langchain.workflow import classify
from evals.common import DEEPAGENT_AGENT, WORKFLOW_AGENT, ensure_dataset, short_name, tag_experiment

DATASET = "chat-langchain-relevance"
# The classifier is the workflow's `classify` node AND the deep agent's guardrail.
AGENTS = [WORKFLOW_AGENT, DEEPAGENT_AGENT]


def make_target(model: str):
    async def target(inputs: dict) -> dict:
        state = {"messages": convert_to_messages(inputs["messages"])}
        update = await classify(state, {"configurable": {"classifier_model": model}})
        return update["relevance"]

    return target


def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["relevant"] == reference_outputs["relevant"]


def _counts(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    pairs = [(o["relevant"], r["relevant"]) for o, r in zip(outputs, reference_outputs)]
    return {
        "tp": sum(p and r for p, r in pairs),
        "fp": sum(p and not r for p, r in pairs),
        "fn": sum(not p and r for p, r in pairs),
    }


# "Positive" = relevant, i.e. let through. A false negative refuses a real user.
def precision(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    c = _counts(outputs, reference_outputs)
    return {"key": "precision", "score": c["tp"] / max(c["tp"] + c["fp"], 1)}


def recall(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    c = _counts(outputs, reference_outputs)
    return {"key": "recall", "score": c["tp"] / max(c["tp"] + c["fn"], 1)}


def f1(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    c = _counts(outputs, reference_outputs)
    return {"key": "f1", "score": 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)}


def false_refusal_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    c = _counts(outputs, reference_outputs)
    return {"key": "false_refusal_rate", "score": c["fn"] / max(c["tp"] + c["fn"], 1)}


async def main(model: str) -> str:
    client = Client()
    ensure_dataset(
        client,
        DATASET,
        "relevance.jsonl",
        "Is the latest user message about the LangChain ecosystem? Labels for the classify node.",
        agents=AGENTS,
    )
    results = await aevaluate(
        make_target(model),
        data=DATASET,
        evaluators=[correct],
        summary_evaluators=[precision, recall, f1, false_refusal_rate],
        experiment_prefix=f"classify-node-{short_name(model)}",
        metadata={"models": model, "node": "classify"},
        max_concurrency=8,
        client=client,
    )
    tag_experiment(client, results.experiment_name, AGENTS)
    print(f"experiment: {results.experiment_name} (tagged to {AGENTS})")
    return results.experiment_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=FAST_MODEL)
    asyncio.run(main(parser.parse_args().model))
