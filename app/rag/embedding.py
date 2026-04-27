from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app import Settings, get_settings


def get_embedding_function(settings: Optional[Settings] = None) -> Embeddings:
    settings = settings or get_settings()
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
