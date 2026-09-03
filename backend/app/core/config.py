"""Application settings loaded from environment / .env.

Secret values (e.g. OPENAI_API_KEY) must only ever enter the process via the
environment. They are never logged, serialized into reports, or committed.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # SQLite for MVP; swap to a PostgreSQL URL in production without code changes
    # (see docs/postgres-migration.md).
    database_url: str = "sqlite:///./aip.sqlite"

    # Large time-series data lives on the filesystem (CSV/Parquet), not in the DB.
    data_dir: Path = Path("../data")


def get_settings() -> Settings:
    return Settings()
