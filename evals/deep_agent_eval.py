"""Compare two models on the deep agent, then judge them head to head.

    python -m evals.deep_agent_eval run --model langsmith:custom/claude-sonnet-5-5
    python -m evals.deep_agent_eval run --model langsmith:custom/claude-haiku-5-5
    python -m evals.deep_agent_eval pairwise <experiment-a> <experiment-b>

Each example runs on its own thread, so each gets its own sandbox
(chatlc-eval-<uuid>). scripts/cleanup_sandboxes.py removes them afterwards.
The judge model is fixed so both experiments are graded the same way.
"""

import argparse
import asyncio
import uuid
from functools import cache
from typing import Literal

from chat_langchain_deepagent.agent import make_graph
from chat_langchain_deepagent.classifier import REFUSAL
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langsmith import Client, aevaluate
from langsmith.evaluation import evaluate_comparative
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from pydantic import BaseModel

from evals.common import DEEPAGENT_AGENT, ensure_dataset, short_name, tag_experiment
from evals.settings import JUDGE_MODEL

DATASET = "chat-langchain-qa"


def make_target(model: str):
    async def target(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}", "model": model}}
        graph = await make_graph(config)
        result = await graph.ainvoke(
            {"messages": convert_to_messages(inputs["messages"])},
            {**config, "recursion_limit": 150},
        )
        messages = result["messages"]
        return {
            "answer": messages[-1].text,
            "tools_called": [tc["name"] for m in messages if m.type == "ai" for tc in m.tool_calls],
        }

    return target


ANY_DOCS_TOOL = "search_docs_by_lang_chain|query_docs_filesystem_docs_by_lang_chain|search_api|get_symbol"


def reference(outputs: dict) -> dict:
    """An example's reference fields, however the example was added.

    Rows from evals/data/qa.jsonl carry reference_answer, should_refuse and expected_tools.
    Rows a reviewer adds from the review queue (Add to Dataset) carry the run's `messages`
    instead, final message corrected: that message is the reference answer.
    """
    if "reference_answer" in outputs:
        return outputs
    answer = outputs["messages"][-1]["content"]
    if isinstance(answer, list):  # content blocks
        answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
    refuse = REFUSAL in answer
    return {
        "reference_answer": answer,
        "should_refuse": refuse,
        "expected_tools": [] if refuse else [ANY_DOCS_TOOL],
    }


_judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, feedback_key="correctness", model=JUDGE_MODEL)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return _judge(
        inputs=inputs["messages"][-1]["content"],
        outputs=outputs["answer"],
        reference_outputs=reference(reference_outputs)["reference_answer"],
    )


def refusal_correct(outputs: dict, reference_outputs: dict) -> bool:
    return (REFUSAL in outputs["answer"]) == reference(reference_outputs)["should_refuse"]


def expected_tool_used(outputs: dict, reference_outputs: dict) -> bool:
    """Each expected entry may list alternatives: "search_api|get_symbol"."""
    called = set(outputs["tools_called"])
    ref = reference(reference_outputs)
    if ref["should_refuse"]:
        return not called  # the guardrail should stop the run before any tool
    return all(called & set(entry.split("|")) for entry in ref["expected_tools"])


async def run(model: str) -> str:
    client = Client()
    ensure_dataset(
        client,
        DATASET,
        "qa.jsonl",
        "chat-langchain questions with reference answers, expected tools, and refusals.",
        agents=[DEEPAGENT_AGENT],
    )
    results = await aevaluate(
        make_target(model),
        data=DATASET,
        evaluators=[correctness, refusal_correct, expected_tool_used],
        experiment_prefix=f"deepagent-{short_name(model)}",
        metadata={"models": model, "judge": JUDGE_MODEL},
        max_concurrency=4,
        client=client,
    )
    tag_experiment(client, results.experiment_name, [DEEPAGENT_AGENT])
    print(f"experiment: {results.experiment_name} (tagged to {DEEPAGENT_AGENT})")
    return results.experiment_name


class Preference(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str


PAIRWISE_PROMPT = """\
Two assistants answered the same LangChain question. Using the reference answer as \
ground truth, pick the answer that is more correct, more specific, and better cited. \
Length is not a virtue. Answer "tie" only if they are equally good.

<question>{question}</question>
<reference>{reference}</reference>
<answer_a>{a}</answer_a>
<answer_b>{b}</answer_b>"""

@cache  # built on first use, so importing this module needs no model credentials
def _pairwise_judge():
    return init_chat_model(JUDGE_MODEL).with_structured_output(Preference)


def preference(runs: list, example) -> dict:
    a, b = runs
    verdict = _pairwise_judge().invoke(
        PAIRWISE_PROMPT.format(
            question=example.inputs["messages"][-1]["content"],
            reference=reference(example.outputs)["reference_answer"],
            a=a.outputs["answer"],
            b=b.outputs["answer"],
        )
    )
    scores = {"A": (1, 0), "B": (0, 1), "tie": (0.5, 0.5)}[verdict.winner]
    return {"key": "preference", "scores": {a.id: scores[0], b.id: scores[1]}}


def pairwise(experiment_a: str, experiment_b: str) -> None:
    evaluate_comparative(
        (experiment_a, experiment_b),
        evaluators=[preference],
        randomize_order=True,  # guard against position bias
        experiment_prefix="deepagent-pairwise",
        metadata={"judge": JUDGE_MODEL},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--model", required=True)
    pair_p = sub.add_parser("pairwise")
    pair_p.add_argument("experiment_a")
    pair_p.add_argument("experiment_b")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args.model))
    else:
        pairwise(args.experiment_a, args.experiment_b)
