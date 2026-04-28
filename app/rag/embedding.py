from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings

from app import Settings, get_settings


def get_embedding_provider(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    provider = str(settings.embedding_provider or "").strip().lower()
    if provider:
        return provider
    return str(settings.llm_provider or "ollama").strip().lower()


def get_embedding_model_name(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    if str(settings.embedding_model or "").strip():
        return str(settings.embedding_model).strip()
    provider = get_embedding_provider(settings)
    if provider == "openai":
        return settings.openai_embed_model
    return settings.ollama_embed_model


def build_openai_embeddings(**kwargs: object) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(**kwargs)


def build_ollama_embeddings(**kwargs: object) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(**kwargs)


def get_embedding_function(settings: Optional[Settings] = None) -> Embeddings:
    settings = settings or get_settings()
    provider = get_embedding_provider(settings)
    model_name = get_embedding_model_name(settings)

    if provider == "openai":
        kwargs: dict[str, object] = {"model": model_name}
        if str(settings.openai_api_key or "").strip():
            kwargs["api_key"] = settings.openai_api_key
        if str(settings.openai_base_url or "").strip():
            kwargs["base_url"] = settings.openai_base_url
        return build_openai_embeddings(**kwargs)

    if provider == "ollama":
        return build_ollama_embeddings(
            model=model_name,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
