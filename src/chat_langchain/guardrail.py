"""The same relevance classifier, placed as deep-agent middleware.

`before_agent` runs once per user turn, before the agent loop starts. On an
off-topic question it writes a refusal and jumps to the end, so the model,
tools, subagent and sandbox never run. In a trace it shows up as the node
`RelevanceGuardrail.before_agent` with `relevance_classifier` as its child.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from chat_langchain.classifier import REFUSAL, Relevance, aclassify, classify
from chat_langchain.settings import FAST_MODEL


class RelevanceGuardrail(AgentMiddleware):
    def __init__(self, model: str = FAST_MODEL) -> None:
        super().__init__()
        self.model = model

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._decide(classify(state["messages"], model=self.model))

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._decide(await aclassify(state["messages"], model=self.model))

    @staticmethod
    def _decide(relevance: Relevance) -> dict[str, Any] | None:
        if relevance.relevant:
            return None
        return {"messages": [AIMessage(REFUSAL)], "jump_to": "end"}
