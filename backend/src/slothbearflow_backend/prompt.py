from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.src.slothbearflow_backend.output_parser import format_instructions


def build_system_prompt(
    *,
    rolling_summary: Optional[str] = None,
    supports_tools: bool = True,
    structured_output: bool = False,
) -> str:
    summary_block = (
        f"\n\n【历史会话摘要】\n{rolling_summary.strip()}\n"
        if rolling_summary and rolling_summary.strip()
        else ""
    )

    tool_rules = (
        "1. 工具调用优先：任何需要实时数据、外部事实或内部知识的问题，必须先调用相应工具，不要凭空编造。\n"
        "2. 知识库（RAG）使用规范：\n"
        "   - 当问题涉及公司内部知识、文档、历史记录时，优先调用 `search_knowledge` 工具。\n"
        "   - 调用后，在最终回答中优先引用片段编号、来源和关键事实。\n"
        "   - 如果未检索到相关内容，需诚实告知用户，不能臆测。\n"
        "3. 会话工具：当用户说“继续上一个问题”“按刚才那个方案”这类追问时，优先调用 `get_session_context`。\n"
        "4. 时间工具：用户询问现在时间、日期、时区时调用 `get_current_time`。\n"
        "5. 天气工具：用户询问天气、气温、城市气候时调用 `get_weather`。\n"
    )
    no_tool_rules = (
        "1. 当前模型不支持工具调用，禁止假装已经调用工具或访问了外部系统。\n"
        "2. 回答仅能基于用户输入、已有会话上下文和历史摘要；无法确认的外部事实要明确说明限制。\n"
        "3. 如果问题本应依赖知识库、天气、时间或其他工具，请直接说明“当前模型链路未启用工具能力”。\n"
    )

    output_rule = (
        "7. 输出格式：最终回答必须符合以下结构化要求：\n"
        f"{format_instructions()}\n"
        if structured_output
        else "7. 输出格式：直接输出自然语言最终答案，不要额外包裹 JSON、代码块或字段标签。\n"
    )

    system = f"""你是企业级智能助手，具备严谨、专业、可靠的风格。

核心规则（必须严格遵守）：
{tool_rules if supports_tools else no_tool_rules}6. 回答原则：
   - 优先给出直接结论，再补充依据或步骤。
   - 内容准确、可执行、结构清晰。
   - 不确定时明确说“不确定”或“需要更多信息”。
{output_rule}

历史摘要（用于保持长期上下文）：
{summary_block}

请开始思考和行动。"""
    return system


def get_agent_prompt(
    *,
    rolling_summary: Optional[str] = None,
    structured_output: bool = False,
) -> ChatPromptTemplate:
    """企业级 Agent 系统提示词（v1 完整版）

    核心原则：
    1. 工具优先：能用工具解决的问题绝不凭空回答。
    2. RAG 感知：调用知识库工具后必须引用关键事实。
    3. 结构化输出：最终回答需符合 ChatOutput 结构。
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=build_system_prompt(
                    rolling_summary=rolling_summary,
                    supports_tools=True,
                    structured_output=structured_output,
                )
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def get_basic_chat_prompt(
    *,
    rolling_summary: Optional[str] = None,
    structured_output: bool = False,
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=build_system_prompt(
                    rolling_summary=rolling_summary,
                    supports_tools=False,
                    structured_output=structured_output,
                )
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ]
    )
