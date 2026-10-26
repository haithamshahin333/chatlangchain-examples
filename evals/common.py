"""Datasets from JSONL, and tagging them (and their experiments) to LangSmith agents.

In an agent-based workspace, LangSmith keeps a resource-tag key named `Agent` with one
value per agent identifier. Tagging a dataset or an experiment with `Agent=<agent id>`
attaches it to that agent: it shows up in the agent's scope in the UI. A dataset can carry
several agent tags, so one dataset can serve several agents. Experiments are not linked
to an agent automatically, so the eval scripts tag each one after it runs.
"""

import json
from pathlib import Path

from langsmith import Client

DATA = Path(__file__).parent / "data"

WORKFLOW_AGENT = "chat-langchain-workflow"
DEEPAGENT_AGENT = "chat-langchain-deepagent"


def ensure_dataset(
    client: Client, name: str, filename: str, description: str, agents: list[str]
) -> str:
    """Create the dataset from evals/data/<filename> if needed, and tag it to `agents`.

    To change the examples, edit the JSONL, delete the dataset in LangSmith, and rerun.
    """
    tags = [agent_tag(client, agent) for agent in agents]
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        for tag in tags:  # backfill datasets created before they were tagged
            tag_resource(client, tag, "dataset", dataset.id)
        return name
    rows = [json.loads(line) for line in (DATA / filename).read_text().splitlines() if line]
    dataset = client.create_dataset(dataset_name=name, description=description, tag_value_ids=tags)
    client.create_examples(dataset_id=dataset.id, examples=rows)
    print(f"created dataset {name!r} with {len(rows)} examples, tagged to {agents}")
    return name


def tag_experiment(client: Client, experiment_name: str, agents: list[str]) -> None:
    experiment = client.read_project(project_name=experiment_name)
    for agent in agents:
        tag_resource(client, agent_tag(client, agent), "experiment", experiment.id)


def agent_tag(client: Client, agent_id: str) -> str:
    """ID of the `Agent` tag value for an agent identifier.

    LangSmith creates the value when the agent is created (first trace or first deploy).
    """
    keys = client.request_with_retries("GET", "/workspaces/current/tag-keys").json()
    key_id = next(k["id"] for k in keys if k["key"] == "Agent")
    values = client.request_with_retries(
        "GET", f"/workspaces/current/tag-keys/{key_id}/tag-values"
    ).json()
    try:
        return next(v["id"] for v in values if v["value"] == agent_id)
    except StopIteration:
        raise ValueError(
            f"No agent {agent_id!r} in this workspace yet. Run or deploy it once first."
        ) from None


def tag_resource(client: Client, tag_value_id: str, resource_type: str, resource_id) -> None:
    """Apply a tag value to a resource, unless it already has it."""
    current = client.request_with_retries(
        "GET",
        "/workspaces/current/tags/resource",
        request_kwargs={"params": {"resource_type": resource_type, "resource_id": str(resource_id)}},
    ).json()
    if any(v["id"] == tag_value_id for key in current for v in key.get("values", [])):
        return
    client.request_with_retries(
        "POST",
        "/workspaces/current/taggings",
        request_kwargs={
            "json": {
                "tag_value_id": tag_value_id,
                "resource_type": resource_type,
                "resource_id": str(resource_id),
            }
        },
    )


def short_name(model: str) -> str:
    """'langsmith:custom/claude-haiku-5-5' -> 'claude-haiku-5-5', for experiment names."""
    return model.split(":")[-1].split("/")[-1]
