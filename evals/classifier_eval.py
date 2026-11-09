"""Evaluate ONE node of the LangGraph workflow: the `classify` guardrail.

The target calls the node function directly with a state dict. The graph,
retrieval and answer model never run, so the experiment measures the routing
decision and nothing else. The deep agent's middleware guardrail is a copy of
this classifier, so while the two copies match, this grades both agents' front door.

    python -m evals.classifier_eval                      # default classifier model
    python -m evals.classifier_eval --model langsmith:custom/claude-sonnet-5-5
"""

import argparse
import asyncio

from chat_langchain_deepagent import classifier as deepagent_classifier
from chat_langchain_workflow import classifier as workflow_classifier
from chat_langchain_workflow.graph import classify
from chat_langchain_workflow.settings import FAST_MODEL
from langchain_core.messages import convert_to_messages
from langsmith import Client, aevaluate

from evals.common import DEEPAGENT_AGENT, WORKFLOW_AGENT, ensure_dataset, short_name, tag_experiment

DATASET = "chat-langchain-relevance"


def graded_agents() -> list[str]:
    """The workflow always; the deep agent too while its guardrail copy matches this one."""
    def substance(module):
        return module.SYSTEM_PROMPT, module.REFUSAL, module.Relevance.model_json_schema()

    if substance(workflow_classifier) == substance(deepagent_classifier):
        return [WORKFLOW_AGENT, DEEPAGENT_AGENT]
    print(
        "note: the deep agent's classifier no longer matches the workflow's "
        "(prompt, refusal or schema), so this experiment is tagged to the workflow only"
    )
    return [WORKFLOW_AGENT]


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
    agents = graded_agents()
    ensure_dataset(
        client,
        DATASET,
        "relevance.jsonl",
        "Is the latest user message about the LangChain ecosystem? Labels for the classify node.",
        agents=agents,
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
    tag_experiment(client, results.experiment_name, agents)
    print(f"experiment: {results.experiment_name} (tagged to {agents})")
    return results.experiment_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=FAST_MODEL)
    asyncio.run(main(parser.parse_args().model))
