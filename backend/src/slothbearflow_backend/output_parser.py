from __future__ import annotations

from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from slothbearflow_backend import Settings, get_chat_llm, get_settings
from slothbearflow_backend.output_schema import ChatOutput, Citation


def get_chat_output_parser() -> PydanticOutputParser[ChatOutput]:
    """经典 PydanticOutputParser（可用于提示词内嵌 format_instructions）。"""
    return PydanticOutputParser(pydantic_object=ChatOutput)


def format_instructions() -> str:
    return get_chat_output_parser().get_format_instructions()


def structured_chat_output_from_text(
    agent_text: str,
    *,
    rag_hint: str = "",
    citations: Optional[List[Citation]] = None,
    tools_used: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
) -> ChatOutput:
    """在 Agent 自由文本输出后，再用一次 LLM 约束为 ChatOutput（生产常用二段式）。"""
    settings = settings or get_settings()
    llm = get_chat_llm(settings, temperature=0.0).with_structured_output(ChatOutput)
    hint = f"\n\n【检索来源线索】\n{rag_hint}" if rag_hint else ""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "将助手草稿整理为结构化 JSON 字段，不要编造事实。"
                + hint,
            ),
            ("human", "{draft}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"draft": agent_text})
    if citations:
        result.citations = citations
        if not result.source:
            result.source = rag_hint
    if tools_used:
        result.tools_used = tools_used
    return result
