from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_sections import (
    AppConfigMixin,
    EmbeddingConfigMixin,
    LegalConfigMixin,
    LLMConfigMixin,
    PdfPipelineConfigMixin,
    RetrievalConfigMixin,
    StorageConfigMixin,
    TabularConfigMixin,
)


class Settings(
    AppConfigMixin,
    LLMConfigMixin,
    EmbeddingConfigMixin,
    StorageConfigMixin,
    RetrievalConfigMixin,
    LegalConfigMixin,
    TabularConfigMixin,
    PdfPipelineConfigMixin,
    BaseSettings,
):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
