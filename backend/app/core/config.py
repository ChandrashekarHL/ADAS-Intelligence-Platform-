"""Application settings loaded from environment / .env.

Secret values (e.g. OPENAI_API_KEY) must only ever enter the process via the
environment. They are never logged, serialized into reports, or committed.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM access. "fake" runs the whole pipeline offline with scripted responses.
    llm_provider: Literal["openai", "fake"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_timeout_s: float = 60.0
    llm_max_attempts: int = 3

    # SQLite for MVP; swap to a PostgreSQL URL in production without code changes
    # (see docs/postgres-migration.md).
    database_url: str = "sqlite:///./aip.sqlite"

    # Large time-series data lives on the filesystem (CSV/Parquet), not in the DB.
    data_dir: Path = Path("../data")
    # Built RAG index (app.rag.cli build). Retrieval is skipped when it does not exist.
    index_dir: Path = Path("../data/index")
    # Uploaded files and generated reports are stored under here, per project.
    workspace_dir: Path = Path("../data/workspace")
    # Browser origins allowed to call the API (comma-separated). The Next.js dev server.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


def get_settings() -> Settings:
    return Settings()
