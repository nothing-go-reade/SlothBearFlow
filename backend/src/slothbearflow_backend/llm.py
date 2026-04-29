from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from backend.src.slothbearflow_backend import Settings, get_settings


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


def _drop_none_values(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if v is not None}


def _merge_dict(
    base: Optional[dict[str, Any]], override: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    if not merged:
        return None
    return merged


def _resolve_temperature(
    settings: Settings, provider: str, explicit_temperature: Optional[float]
) -> float:
    if explicit_temperature is not None:
        return explicit_temperature
    if provider == "openai" and settings.openai_temperature is not None:
        return settings.openai_temperature
    if settings.llm_temperature is not None:
        return settings.llm_temperature
    return 0.2


def _resolve_openai_reasoning_effort(settings: Settings) -> Optional[str]:
    if settings.openai_reasoning_effort is not None:
        return settings.openai_reasoning_effort
    if settings.llm_reasoning_effort is not None:
        return settings.llm_reasoning_effort

    deep_think = settings.openai_deep_think
    if deep_think is None:
        deep_think = settings.llm_deep_think
    if deep_think is True:
        return "high"
    return None


def get_chat_llm(
    settings: Optional[Settings] = None, *, temperature: Optional[float] = None
) -> BaseChatModel:
    settings = settings or get_settings()
    provider = _normalize_provider(settings)
    model_name = get_llm_model_name(settings)
    resolved_temperature = _resolve_temperature(settings, provider, temperature)

    if provider == "openai":
        model_kwargs = _merge_dict(
            settings.llm_model_kwargs_json, settings.openai_model_kwargs_json
        )
        extra_body = _merge_dict(
            settings.llm_extra_body_json, settings.openai_extra_body_json
        )
        kwargs: dict[str, object] = _drop_none_values(
            {
                "model": model_name,
                "temperature": resolved_temperature,
                "top_p": (
                    settings.openai_top_p
                    if settings.openai_top_p is not None
                    else settings.llm_top_p
                ),
                "max_tokens": (
                    settings.openai_max_tokens
                    if settings.openai_max_tokens is not None
                    else settings.llm_max_tokens
                ),
                "reasoning_effort": _resolve_openai_reasoning_effort(settings),
                "api_key": (
                    settings.openai_api_key
                    if str(settings.openai_api_key or "").strip()
                    else None
                ),
                "base_url": (
                    settings.openai_base_url
                    if str(settings.openai_base_url or "").strip()
                    else None
                ),
                "model_kwargs": model_kwargs,
                "extra_body": extra_body,
            }
        )
        return build_openai_chat_model(**kwargs)

    if provider == "ollama":
        return build_ollama_chat_model(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=resolved_temperature,
        )

    raise ValueError(f"Unsupported llm provider: {settings.llm_provider}")
