"""The deep agent's search tools, straight from the public LangChain MCP servers.

docs.langchain.com/mcp       -> search_docs_by_lang_chain, query_docs_filesystem_docs_by_lang_chain
reference.langchain.com/mcp  -> search_api, get_symbol
"""

import asyncio
import warnings

from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool

with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from langchain.mcp import MCPAdapter

from chat_langchain_deepagent.settings import DOCS_MCP_URL, REFERENCE_MCP_URL

# Not something a Q&A bot should call on a user's behalf.
_EXCLUDED = {"submit_feedback"}

_tools: list[BaseTool] | None = None


async def load_docs_tools() -> list[BaseTool]:
    """List the tools once per process; the adapted tools stay callable afterwards."""
    global _tools
    if _tools is None:
        docs, reference = await asyncio.gather(
            MCPAdapter(DOCS_MCP_URL).list_tools(),
            MCPAdapter(REFERENCE_MCP_URL).list_tools(),
        )
        _tools = [t for t in docs + reference if t.name not in _EXCLUDED]
    return _tools
