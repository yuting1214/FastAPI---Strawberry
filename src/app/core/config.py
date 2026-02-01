from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    debug: bool = False

    # Database
    database_url: str = ""

    # Pool settings (production)
    db_pool_size: int = 5
    db_max_overflow: int = 10

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    @property
    def db_url(self) -> str:
        """Return SQLite for dev if DATABASE_URL not set, else use DATABASE_URL."""
        if self.is_dev and not self.database_url:
            return "sqlite+aiosqlite:///./dev.db"
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
