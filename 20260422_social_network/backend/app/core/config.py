from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Kizuna"
    api_prefix: str = "/api"
    database_url: str = "postgresql+psycopg://kizuna:kizuna@localhost:5432/kizuna"
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", env_prefix="KIZUNA_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
