from __future__ import annotations

from typing import Optional

from langchain_ollama import ChatOllama

from app import Settings, get_settings


def get_chat_llm(
    settings: Optional[Settings] = None, *, temperature: float = 0.2
) -> ChatOllama:
    settings = settings or get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
