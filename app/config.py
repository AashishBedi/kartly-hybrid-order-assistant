from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_PATH: Path = Path("data/kartly.db")
    GROQ_API_KEY: str = ""
    # Check the Groq console for current model names before setting these values.
    ROUTER_MODEL: str = "replace-with-current-groq-router-model"
    ANSWER_MODEL: str = "replace-with-current-groq-answer-model"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
