"""
Environment-based configuration for dev (SQLite) and prod (PostgreSQL).

Usage:
    # Dev mode (default) - uses SQLite
    python -m app.main --mode dev

    # Prod mode - uses DATABASE_URL from environment
    python -m app.main --mode prod
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Base settings shared across environments."""

    # Application
    APP_NAME: str = "GraphQL API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database pool settings (production)
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    @property
    def is_dev(self) -> bool:
        return getattr(self, 'ENV_MODE', 'dev') == 'dev'

    @property
    def db_url(self) -> str:
        """Sync database URL."""
        raise NotImplementedError("Use subclass")

    @property
    def async_db_url(self) -> str:
        """Async database URL for SQLAlchemy."""
        raise NotImplementedError("Use subclass")


class DevSettings(Settings):
    """Development settings - uses local SQLite."""

    ENV_MODE: str = "dev"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def db_url(self) -> str:
        return "sqlite:///./dev.db"

    @property
    def async_db_url(self) -> str:
        return "sqlite+aiosqlite:///./dev.db"


class ProdSettings(Settings):
    """Production settings - uses PostgreSQL via DATABASE_URL."""

    ENV_MODE: str = "prod"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def db_url(self) -> str:
        """Return DATABASE_URL for sync operations."""
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is required in prod mode")
        return self.DATABASE_URL

    @property
    def async_db_url(self) -> str:
        """Convert DATABASE_URL to async format (asyncpg).

        Railway format: postgresql://postgres:PASSWORD@postgres.railway.internal:5432/railway
        Heroku format:  postgres://user:pass@host:5432/db
        """
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is required in prod mode")

        url = self.DATABASE_URL

        # Heroku uses postgres:// (deprecated), convert to postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)

        # Add asyncpg driver for async SQLAlchemy
        # postgresql://... -> postgresql+asyncpg://...
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)

        return url


def get_settings(env_mode: str = "dev") -> Settings:
    """Get settings based on environment mode."""
    if env_mode == "prod":
        return ProdSettings()
    return DevSettings()
