"""Configuration management."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 4096

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "research_assistant"
    postgres_user: str = "postgres"
    postgres_password: str = "research_pass"
    database_url: Optional[str] = None

    # LangGraph
    max_iterations: int = 25
    recursion_limit: int = 50

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent)

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def logs_dir(self) -> Path:
        path = self.project_root / "logs"
        path.mkdir(exist_ok=True)
        return path


settings = Settings()
