from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app import Settings, get_settings


def _normalize_provider(settings: Settings) -> str:
    return str(settings.llm_provider or "ollama").strip().lower()


def get_llm_model_name(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    if str(settings.llm_model or "").strip():
        return str(settings.llm_model).strip()
    provider = _normalize_provider(settings)
    if provider == "openai":
        return settings.openai_model
    return settings.ollama_model


def llm_supports_tools(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    if settings.llm_supports_tools is not None:
        return bool(settings.llm_supports_tools)
    provider = _normalize_provider(settings)
    if provider == "openai":
        return bool(settings.openai_model_supports_tools)
    return bool(settings.ollama_model_supports_tools)


def build_openai_chat_model(**kwargs: object) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


def build_ollama_chat_model(**kwargs: object) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(**kwargs)


def get_chat_llm(
    settings: Optional[Settings] = None, *, temperature: float = 0.2
) -> BaseChatModel:
    settings = settings or get_settings()
    provider = _normalize_provider(settings)
    model_name = get_llm_model_name(settings)

    if provider == "openai":
        kwargs: dict[str, object] = {
            "model": model_name,
            "temperature": temperature,
        }
        if str(settings.openai_api_key or "").strip():
            kwargs["api_key"] = settings.openai_api_key
        if str(settings.openai_base_url or "").strip():
            kwargs["base_url"] = settings.openai_base_url
        return build_openai_chat_model(**kwargs)

    if provider == "ollama":
        return build_ollama_chat_model(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported llm provider: {settings.llm_provider}")
