# Every target reads secrets from .env (see .env.example).
#
# `uv run --env-file` does not override a variable your shell already sets, so RUN drops
# the ones that would change where things go:
#   LANGSMITH_API_KEY, LANGSMITH_WORKSPACE_ID  .env alone picks the workspace
#   OPENAI_BASE_URL, OPENAI_API_KEY            model calls take the gateway path from .env
#   LANGSMITH_PROJECT                          traces go to the agent's Local environment
# and LANGSMITH_CONFIG_FILE=/dev/null makes the SDK ignore your `langsmith` CLI login. With
# no workspace in the environment, the SDK would send the login's workspace with the .env key.

WORKFLOW_ID  := chat-langchain-workflow
DEEPAGENT_ID := chat-langchain-deepagent
# Each agent is a standalone LangGraph project; `langgraph deploy` uploads only its folder.
WORKFLOW_DIR  := agents/workflow
DEEPAGENT_DIR := agents/deepagent
MODEL        ?= langsmith:custom/claude-sonnet-5-5

RUN   := env -u LANGSMITH_API_KEY -u LANGSMITH_WORKSPACE_ID -u OPENAI_BASE_URL -u OPENAI_API_KEY \
         -u LANGSMITH_PROJECT LANGSMITH_CONFIG_FILE=/dev/null \
         uv run --project $(CURDIR) --env-file $(CURDIR)/.env
AGENT  = LANGSMITH_AGENT_ENVIRONMENT=local LANGSMITH_AGENT_ID
# Optional key + URL for the online judge. A separate file, because `langgraph deploy`
# uploads everything in .env to the deployments, and they don't need this key.
JUDGE_ENV := $(if $(wildcard .env.judge),--env-file $(CURDIR)/.env.judge)

.PHONY: install provision dev-workflow dev-deepagent eval-classifier eval-deepagent pairwise eval-skill \
        online-evals smoke-workflow smoke-deepagent cleanup \
        deploy-workflow-dev deploy-workflow-prod deploy-deepagent-dev deploy-deepagent-prod

install:
	uv sync

# --- everything at once: deploy dev → online evals → deploy prod → offline evals ---------
provision:  # make provision SKIP=deploy reuses the deployments already there
	$(RUN) $(JUDGE_ENV) python -m scripts.provision $(if $(SKIP),--skip $(SKIP))

# --- local: `langgraph dev` + Studio, traces land in <agent> -> Local -----------------
# `langgraph dev` reads graph paths relative to the current directory, so it runs inside
# the agent's folder. `langgraph deploy` reads them relative to the config file instead.
dev-workflow:
	cd $(WORKFLOW_DIR) && $(RUN) env $(AGENT)=$(WORKFLOW_ID) langgraph dev --port 8124

dev-deepagent:
	cd $(DEEPAGENT_DIR) && $(RUN) env $(AGENT)=$(DEEPAGENT_ID) langgraph dev --port 8125

# --- offline evals (experiments live on the datasets, not in an agent environment) ---
eval-classifier:
	$(RUN) python -m evals.classifier_eval $(if $(CLASSIFIER_MODEL),--model $(CLASSIFIER_MODEL))

eval-deepagent:
	$(RUN) python -m evals.deep_agent_eval run --model $(MODEL)
	$(RUN) python scripts/cleanup_sandboxes.py chatlc-eval-

pairwise:  # make pairwise A=<experiment> B=<experiment>
	$(RUN) python -m evals.deep_agent_eval pairwise $(A) $(B)

eval-skill:  # validate-snippet with vs without; make eval-skill REPS=1 for a quicker pass
	$(RUN) python -m evals.skill_eval $(if $(REPS),--repetitions $(REPS))
	$(RUN) python scripts/cleanup_sandboxes.py chatlc-eval-

# --- smoke tests against a running server: make smoke-deepagent URL=https://... -------
smoke-workflow:
	$(RUN) python scripts/smoke.py workflow $(or $(URL),http://127.0.0.1:8124)

smoke-deepagent:
	$(RUN) python scripts/smoke.py deepagent $(or $(URL),http://127.0.0.1:8125)

# --- online evaluators + review queue on both agents' Production environments ---------
online-evals:  # make online-evals BACKFILL=24 also grades the last 24 hours
	$(RUN) $(JUDGE_ENV) python -m scripts.setup_online_evals $(if $(BACKFILL),--backfill $(BACKFILL))

cleanup:
	$(RUN) python scripts/cleanup_sandboxes.py $(or $(PREFIX),chatlc-eval-)

# --- deploy: one deployment per agent environment ------------------------------------
deploy-workflow-dev:
	$(RUN) langgraph deploy --config $(WORKFLOW_DIR)/langgraph.json --agent-id $(WORKFLOW_ID) --agent-environment development --remote --no-input

deploy-workflow-prod:
	$(RUN) langgraph deploy --config $(WORKFLOW_DIR)/langgraph.json --agent-id $(WORKFLOW_ID) --agent-environment production --remote --no-input

deploy-deepagent-dev:
	$(RUN) langgraph deploy --config $(DEEPAGENT_DIR)/langgraph.json --agent-id $(DEEPAGENT_ID) --agent-environment development --remote --no-input

deploy-deepagent-prod:
	$(RUN) langgraph deploy --config $(DEEPAGENT_DIR)/langgraph.json --agent-id $(DEEPAGENT_ID) --agent-environment production --remote --no-input
