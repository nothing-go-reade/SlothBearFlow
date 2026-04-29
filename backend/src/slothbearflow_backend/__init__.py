"""LangChain 生产级 Agent 模板包。
"""

from slothbearflow_backend.config import Settings, get_settings
from slothbearflow_backend.llm import get_chat_llm, llm_supports_tools
from slothbearflow_backend.prompt import get_agent_prompt, get_basic_chat_prompt
from slothbearflow_backend.tools.registry import build_tools
from slothbearflow_backend.agent.agent_executor import build_agent_executor
from slothbearflow_backend.rag.milvus_store import get_vector_store
from slothbearflow_backend.output_parser import structured_chat_output_from_text

__all__ = [
    "get_settings",
    "Settings",
    "get_chat_llm",
    "llm_supports_tools",
    "get_agent_prompt",
    "get_basic_chat_prompt",
    "build_tools",
    "build_agent_executor",
    "get_vector_store",
    "structured_chat_output_from_text",
]

__version__ = "0.1.0"
