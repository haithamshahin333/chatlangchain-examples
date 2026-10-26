# chat-langchain, twice: a LangSmith tour

Two small assistants that answer questions about LangChain, LangGraph, Deep Agents and
LangSmith. Both search the public docs MCP servers (`docs.langchain.com/mcp` for guides,
`reference.langchain.com/mcp` for the SDK reference), so there is no index to build.

The agents are kept small on purpose. The point is to exercise LangSmith features with them:

| | **LangGraph workflow** | **Deep agent** |
|---|---|---|
| Shape | Fixed graph: `classify → refuse` or `classify → retrieve → respond` | `create_deep_agent` loop: plans, calls tools, delegates |
| Relevance guardrail | The `classify` **node** routes the graph | `RelevanceGuardrail` **middleware** (`before_agent` + `jump_to: end`) |
| Tools | Both MCP searches, called in parallel by `retrieve` | Docs + reference MCP tools, plus `execute` and file tools in a sandbox |
| Sandbox | — | One **LangSmith Sandbox per thread** (`chatlc-<thread_id>`) |
| Subagent | — | `source-diver` clones deepagents / langgraph / langchain and greps them |
| Skill | — | `validate-snippet` checks every import in a code answer against PyPI packages |
| Offline eval | **Node-level**: grades only `classify` against labeled questions | **Model comparison**: Sonnet vs Haiku, then a pairwise judge |
| LangSmith agent | `chat-langchain-workflow` | `chat-langchain-deepagent` |

Both agents share one classifier (`src/chat_langchain/classifier.py`): a node in one agent,
middleware in the other. So the node-level eval grades the deep agent's front door too.

Every model call goes through the **LangSmith LLM Gateway** using the org's saved model
configurations: `langsmith:custom/claude-sonnet-5-5` and `langsmith:custom/claude-haiku-5-5`.
No provider key exists anywhere in this app.

## Layout

```
src/chat_langchain/              the two agents (this is what gets deployed)
  settings.py      every tunable: models, eval judges, MCP URLs, sandbox naming and TTLs
  docs_tools.py    MCP → LangChain tools (MCPAdapter), drops submit_feedback
  classifier.py    shared Relevance schema + prompt + classify()/aclassify()
  workflow.py      LangGraph StateGraph (graph id: chat_langchain_workflow)
  guardrail.py     RelevanceGuardrail(AgentMiddleware)
  sandbox.py       get-or-create the thread's sandbox; seeds /work and the skill
  deep_agent.py    async graph factory make_graph(config) (graph id: chat_langchain_deepagent)
  skills/validate-snippet/   SKILL.md + check_snippet.py (runs inside the sandbox)
evals/
  data/relevance.jsonl      40 labeled questions (26 relevant, 14 off-topic incl. injections)
  data/qa.jsonl             12 questions with reference answers, expected tools, refusals
  common.py                 datasets from JSONL; tags datasets and experiments to agents
  classifier_eval.py        node-only eval + precision/recall/F1/false-refusal summary metrics
  deep_agent_eval.py        `run --model …` and `pairwise A B`
  online/cites_sources.py   code evaluator, uploaded as an online rule
scripts/
  provision.py              everything below in one command (`make provision`)
  smoke.py                  drive a local or deployed agent end to end
  setup_online_evals.py     online evaluators + review queue on Production
  cleanup_sandboxes.py      delete leftover sandboxes
langgraph.workflow.json / langgraph.deepagent.json   one config per agent (deployment)
Makefile                    every command in this README
.env.example                the secrets .env needs (.env itself is never committed)
.dockerignore               keeps .env and local state out of the deploy build
pyproject.toml + uv.lock    dependencies; requirements.txt is the same set for pip
.python-version             3.12, the same Python the deployments build with
```

## Prerequisites

- An **agent-based LangSmith workspace** (LangSmith Cloud) and a **workspace-scoped**
  service key (`lsv2_sk_…`) with `gateway:invoke` + `workspaces:read` (Workspace Admin has
  both) and sandbox access. A personal access token is not enough for the gateway.
- Org model configurations the gateway can use. These are the defaults; other names work
  through the env vars:

  | Role | Default | Override |
  |---|---|---|
  | Main agent | `langsmith:custom/claude-sonnet-5-5` | `CHATLC_MAIN_MODEL` |
  | Guardrail and subagent (must accept a forced tool call) | `langsmith:custom/claude-haiku-5-5` | `CHATLC_FAST_MODEL` |
  | Offline eval judge | `langsmith:custom/claude-sonnet-5-5` | `CHATLC_JUDGE_MODEL` |
  | Online eval judge (a configuration name) | `gpt-5.5` | `CHATLC_ONLINE_JUDGE_CONFIG`, or let the script create one (below) |

  To have the script create the online judge's configuration instead, put an
  OpenAI-compatible endpoint in `.env.judge` (see the end of `.env.example`). It stores the
  key as the workspace secret `CHATLC_ONLINE_JUDGE_API_KEY` and creates or updates the
  workspace configuration `chat-langchain-judge`. The URL must be public, because LangSmith
  blocks private and localhost addresses. Use a separate file because `.env` goes to every
  deployment.
- `uv`, Python 3.12. `requirements.txt` is there if you prefer pip, but the Makefile uses uv.

```bash
uv sync
cp .env.example .env   # fill in the keys; read the comments in it first
```

`.env` holds **deployment secrets only**. `langgraph deploy` uploads every variable in it
(except reserved ones like `LANGSMITH_API_KEY`). Never put `LANGSMITH_AGENT_ID`,
`LANGSMITH_AGENT_ENVIRONMENT` or `LANGSMITH_PROJECT` there: a deployment bound to an agent
environment sets its own project, and those variables next to it turn tracing off.

## One command

```bash
make provision                       # everything, about 20 minutes
make provision SKIP=deploy           # reuse the deployments already there
make provision SKIP="online offline" # deploy and smoke-test only
```

`scripts/provision.py` runs these stages in order and stops at the first failure, so nothing
reaches production after a failed development smoke test:

1. **Preflight.** Checks that the `.env` keys are set and the model configurations exist.
   If `.env.judge` provides an online judge, the online stage creates that configuration
   instead.
2. **Development.** Deploys both agents in parallel, then smoke-tests each. The first deploy
   creates the agents and their environments.
3. **Online evals.** Adds the rules and the review queue to each agent's Production
   environment.
4. **Production.** Deploys and smoke-tests both agents again. The online evaluators grade
   that smoke traffic first.
5. **Offline evals.** Runs the classify node on Haiku, its guardrail model, then the deep
   agent on Sonnet and Haiku. A pairwise judge compares the two deep agent runs, and the eval
   sandboxes are deleted.

What ends up in the workspace:

| Resource | What |
|---|---|
| Agents | `chat-langchain-workflow` and `chat-langchain-deepagent`, each with Local, Development, Staging and Production environments |
| Deployments | 4: development and production for each agent. The `.env` variables become deployment secrets. |
| Datasets | `chat-langchain-relevance` (40 examples, tagged to both agents), `chat-langchain-qa` (12, tagged to the deep agent) |
| Experiments | Per run: 1 `classify-node-*` and 2 `deepagent-*`, tagged to their agents, plus 1 `deepagent-pairwise-*` |
| Online evaluation | 3 rules on each agent's Production environment (`guardrail_audit`, `cites_sources`, `send-to-review`) and the `chat-langchain-review` queue. With `.env.judge`, also the workspace secret `CHATLC_ONLINE_JUDGE_API_KEY` and the configuration `chat-langchain-judge`. |
| Sandboxes | One `chatlc-<thread_id>` per deep agent conversation. Each stops after 10 idle minutes and is deleted a day later. |

It doesn't create the workspace, the keys or the org model configurations (see
Prerequisites). It doesn't set up alerts, dashboards or Insights either.

## Run locally

```bash
make dev-workflow     # Studio on :8124, traces → chat-langchain-workflow → Local
make dev-deepagent    # Studio on :8125, traces → chat-langchain-deepagent → Local
make smoke-workflow   # on-topic answer + off-topic refusal
make smoke-deepagent  # + subagent clone in the thread sandbox, reused on the next turn
```

The Makefile sets `LANGSMITH_AGENT_ID` / `LANGSMITH_AGENT_ENVIRONMENT=local` for local runs.
It also strips `LANGSMITH_API_KEY`, `LANGSMITH_WORKSPACE_ID`, `OPENAI_BASE_URL`,
`OPENAI_API_KEY` and `LANGSMITH_PROJECT` from your shell. `uv run --env-file` doesn't
override a variable the shell already sets, so without this a shell key could point a run at
another workspace. It also sets `LANGSMITH_CONFIG_FILE=/dev/null`, because the SDK otherwise
sends the workspace from your `langsmith` CLI login along with the `.env` key. With a key from
another org, that combination fails with 403. `make provision` also refuses a key that
reaches more than one workspace, such as a personal access token or an org key. It prints
the target workspace's name before it creates anything.

## Evaluate

```bash
make eval-classifier                                             # Haiku (default guardrail model)
make eval-deepagent  MODEL=langsmith:custom/claude-sonnet-5-5
make eval-deepagent  MODEL=langsmith:custom/claude-haiku-5-5
make pairwise A=<sonnet experiment> B=<haiku experiment>
```

- **Node-level eval.** The target calls the `classify` node function with a state dict. The
  graph, retrieval and answer model never run. Summary evaluators report precision, recall,
  F1 and false-refusal rate. A false refusal turns away a real user, so it is the costly error.
  The docs' `example_to_state | app.nodes["x"]` composition no longer works on current
  LangGraph, because a compiled node is not a Runnable any more. Calling the node function is
  the reliable way. `CLASSIFIER_MODEL=…` reruns it on another model, which must accept a
  forced tool call. Claude Sonnet and Opus 5.5 don't (see "Things worth knowing").
- **Model comparison.** Every example runs on its own thread, so it gets its own sandbox.
  The Makefile deletes them afterwards. Evaluators:
  - `correctness`: openevals LLM-as-judge against the reference answer. The judge is fixed
    to Sonnet so both experiments are graded the same way.
  - `refusal_correct`: did the guardrail refuse exactly the off-topic rows?
  - `expected_tool_used`: did the run use the docs search, the reference tools, `task` (the
    subagent) or `execute` (the skill), as each row expects?
  - Cost, tokens and latency come for free. `metadata={"models": …}` fills the Models column.
  - Rows a reviewer adds from the review queue (**Add to Dataset**) hold the run's
    `messages` rather than reference fields. `reference()` uses their final message,
    corrected by the reviewer, as the reference answer.
- **How evals map to agents in an agent-based workspace.**
  - Traces, evaluators, alerts and prebuilt dashboards belong to one *environment* of an agent.
  - Datasets attach to one or more *agents* and to no environment.
  - The attachment is a **resource tag**. LangSmith keeps a tag key named `Agent` with one
    value per agent identifier, created with the agent. Tagging a dataset `Agent=<id>` puts
    it in that agent's scope. The UI does the same when you create a dataset or add runs to
    one while an agent is selected, and the dataset's **Agent** field edits the tags.
  - Experiments are deliberately kept out of agent environments. The API accepts an
    `Agent` tag on an experiment, and we add one so the experiment can be filtered by agent,
    but no view groups experiments by it.
  - `evals/common.py` applies these tags itself. `create_dataset(..., tag_value_ids=[…])`
    tags a new dataset; existing datasets are backfilled through `POST
    /workspaces/current/taggings`. Each experiment is tagged after it runs, because
    `evaluate()` does not link experiments to an agent.
  - What gets tagged:
    - `chat-langchain-relevance` → both agents. The classifier is the workflow's node *and*
      the deep agent's guardrail.
    - `chat-langchain-qa` → `chat-langchain-deepagent`.
  - Experiment traces stay in their own experiment project. They never appear under an
    agent environment.

### Results from the first run (2026-10-20)

| Experiment | Correctness | Refusal | Expected tools | Median latency | Cost |
|---|---|---|---|---|---|
| `classify` node · Haiku 5.5 | 40/40 (precision = recall = F1 = 1.0, 0 false refusals) | — | — | 1.5 s | $0.006 |
| `classify` node · Sonnet 5.5 (before the `function_calling` change; it can't run the classifier now) | 40/40 (same) | — | — | 1.6 s | $0.12 |
| Deep agent · Sonnet 5.5 | 12/12 | 12/12 | 12/12 | 27 s | $1.31 |
| Deep agent · Haiku 5.5 | 10/12 | 12/12 | 11/12 | 23 s | $0.12 |
| Pairwise (Sonnet vs Haiku, judge Sonnet) | 6–6 tie | | | | |

- **Guardrail model:** Haiku matches Sonnet on the guardrail at about a twentieth of the cost.
- **Main agent:** Sonnet is about 11× the cost of Haiku. Haiku's misses:
  - It skipped the snippet check on the `AgentExecutor` question and answered from docs lookups.
  - It got two LangSmith how-tos wrong.
- **Pairwise:** the tie says that when Haiku is right, its answers are as good as Sonnet's.

## Online evaluation and the review queue

```bash
make online-evals              # idempotent: queue + 3 rules on each agent's Production
make online-evals BACKFILL=24  # also grade the last 24 hours of production traffic
```

`scripts/setup_online_evals.py` puts three rules on the **Production** environment of both
agents. In an agent-based workspace an online rule attaches to one environment, which is that
environment's tracing project.

| Rule | Type | Feedback | What it catches |
|---|---|---|---|
| `guardrail_audit` | LLM-as-judge on an org model configuration (`CHATLC_ONLINE_JUDGE_CONFIG`, default `gpt-5.5`), $5/week spend cap | `guardrail_correct` (1/0) | A second opinion on the Haiku relevance guardrail: a wrong refusal of a LangChain question, or an answer to an off-topic one. |
| `cites_sources` | Code evaluator (`evals/online/cites_sources.py`, stdlib only) | `cites_sources` (1/0) | A non-refusal answer with no docs, API-reference or source link, which suggests it answered from memory. |
| `send-to-review` | Automation | — | Root runs with either score at 0 → annotation queue `chat-langchain-review`. |

- **Every rule is filtered to root runs.** Without `eq(is_root, true)`, an online rule
  grades every span in the trace.
- **Sampling is 1.0 for the demo.** Turn it down for real traffic.
- **The automation's filter is on feedback**, so it waits for the scores. Rule order doesn't
  matter, and each run is queued at most once.
- **The queue has a rubric** (`guardrail_correct`, `answer_correct`) and is tagged to both
  agents. Reviewers turn fixes into dataset examples with **Add to Dataset**.
- **The judge uses an OpenAI model on purpose.** A judge from another model family than
  the agents avoids self-preference. The Claude 5.5 Sonnet and Opus presets can't judge here
  anyway: the evaluator runtime forces a tool call to get structured output, and those models
  reject `tool_choice` of type `tool`/`any`. Haiku 5.5 accepts it.
- **The judge rule sends both the configuration's model and its `playground_settings_id`.**
  The model is what runs, and the ID tells LangSmith to resolve the org secret it references.

First run (2026-10-20):
- **`guardrail_correct`:** 1 on all 26 production runs, including the raw-OpenAI-SDK and
  banana-bread refusals.
- **`cites_sources`:** 0 on one deep-agent answer, where the user asked for "no links". That
  run landed in the queue.

## Deploy (development → production)

```bash
make deploy-workflow-dev   && make smoke-workflow  URL=<dev url>
make deploy-deepagent-dev  && make smoke-deepagent URL=<dev url>
make deploy-workflow-prod  && make smoke-workflow  URL=<prod url>
make deploy-deepagent-prod && make smoke-deepagent URL=<prod url>
```

Each target runs `langgraph deploy --config langgraph.<x>.json --agent-id <id>
--agent-environment <env> --remote`. That makes one serverless deployment per agent
environment, four in total. Check the binding under **Agent → Overview → Deployments**.

The first rollout (2026-10-20) passed every smoke check in all four environments: 2/2 for the
workflow and 4/4 for the deep agent, in both development and production. Each deep agent
thread's sandbox was created by the deployment and kept its clone across turns.

`langgraph deploy` prints each deployment's URL, and **Agent → Overview → Deployments** lists
it per environment. Calling a deployment needs a LangSmith API key for the workspace.

## The tour: where to look in LangSmith

| Feature | What to look at |
|---|---|
| **Agent-based workspace** | Two agents. Switch Local / Development / Production in Tracing, Monitoring and Deployments; each environment has its own traces. |
| **Deployments** | Each agent environment is one deployment. Studio shows the 4-node workflow and the deep agent's middleware nodes. |
| **LLM Gateway** | The `gateway` project lists every model call with tokens and cost. The org presets carry the provider credential. Add a small spend policy to see a `402` block. |
| **Tracing** | Workflow: `classify → relevance_classifier → refuse` vs `retrieve → respond`. Deep agent: `RelevanceGuardrail.before_agent` (a short-circuit ends the run), `task → source-diver` nested, `execute` calls, the skill being read. |
| **Threads** | Multi-turn conversations grouped by thread; turn 2 of a deep-agent thread reuses turn 1's clone. |
| **Sandboxes** | `chatlc-<thread_id>`, one per conversation. Idle ones stop after 10 minutes and wake on the next command; stopped ones are deleted after a day. |
| **Datasets & experiments** | `chat-langchain-relevance` (node eval, summary metrics) and `chat-langchain-qa` (Sonnet vs Haiku in Compare, plus the pairwise experiment). |
| **Online evaluation** | Production → Evaluators lists `guardrail_audit` and `cites_sources`. Their scores show as feedback on production traces, and flagged runs wait in the `chat-langchain-review` annotation queue. |
| **Next, no code needed** | Alerts on the Production environment, Insights over Production traffic, per-environment dashboards. |

## Things worth knowing (found while building this)

- **Org OpenAI presets can't be called as `custom/<name>` here.** The presets store
  `openai_api_base=https://gateway.smith.langchain.com/openai/v1` for Playground use. The
  gateway's custom-config route then requests `/openai/v1/v1/responses` and returns `501 path
  not allow-listed`. The Anthropic presets (`…/anthropic`, no `/v1`) work. Plain `openai:`
  models need an `OPENAI_API_KEY` provider secret in the workspace (or Gateway Credits);
  without one the gateway returns `424`.
- **Structured output through `langsmith:` models must use `method="function_calling"`.**
  `init_chat_model("langsmith:…")` is `ChatOpenAI` on the Responses API. Its default
  `json_schema` parsing needs a `parsed` field that only OpenAI's own stream sends. Through
  the gateway to Claude, it fails with `Structured Output response does not have a 'parsed'
  field` whenever the graph streams, which is every Studio chat. `runs.wait` and
  `ainvoke` work, so non-streaming smoke tests and evals miss it. `scripts/smoke.py` now
  streams. `function_calling` forces a tool call, which Haiku 5.5 accepts. Claude
  Sonnet 5.5 and Opus 5.5 reject it, so keep `CHATLC_FAST_MODEL` on Haiku.
- **`langgraph deploy` packs `.env` into the remote build** unless `.dockerignore` excludes
  it. The CLI only skips `.venv`, `.git` and similar by itself. This repo ships a
  `.dockerignore`.
- **A shell `OPENAI_BASE_URL` beats `LANGSMITH_GATEWAY=true`** for `openai:` models. The
  `langsmith:` provider ignores it, and the Makefile strips it anyway.
- **The graph factory runs before the guardrail.** A new thread gets its sandbox on its
  first turn even when that turn is off-topic and refused. That's the trade-off of
  thread-scoped sandboxes created in a factory. Scoping by assistant, or creating the
  sandbox lazily, avoids it.
- **Insights "Generate config" can fail with `RunnableSequence … last … Input should be an
  instance of Runnable`.** A model configuration can be stored in a short "simple" format
  (`modelId`, `envVarName`, `baseUrl`), which is what the LangSmith Terraform provider writes
  for Anthropic. Playground, evaluators and the API convert it to a full LangChain manifest
  first. Insights' config generation reads the stored row as is, so the chain gets a plain
  dict where the model should be. Pick a configuration stored in the full format for Insights.
- **A new workspace can come with a gateway spend policy.** One we hit was named "Quickstart
  monthly workspace limit". Once it's spent, the gateway answers `402 request blocked by
  gateway policies`, and a deployment reports that as `An internal error occurred`.
  `make provision` spends about $3–4 through the gateway, most of it on the deep agent eval
  with Sonnet and the deep agent smoke tests. Raise or remove the cap first. When a smoke
  test fails, the script prints the model error from the trace.
- **The sandbox image** runs as `root`, has `python3` (no `python`), `uv`, `git` and `rg`, and
  can reach github.com and PyPI under the default restricted egress.
- **Deep Agents 0.7 removed backend factories.** Thread scoping is done with an async graph
  factory that reads `config["configurable"]["thread_id"]`. Studio's graph view calls the
  factory without a thread, so the factory builds a sandbox-free graph for it.
