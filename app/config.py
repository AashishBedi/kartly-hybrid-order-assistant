from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_PATH: Path = Path("data/kartly.db")
    GROQ_API_KEY: str = ""
    # Check the Groq console for current model names before setting these values.
    ROUTER_MODEL: str = "replace-with-current-groq-router-model"
    ANSWER_MODEL: str = "replace-with-current-groq-answer-model"
    LOG_LEVEL: str = "INFO"
    CHROMA_PATH: Path = Path("data/chroma")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    RETRIEVAL_TOP_K: int = 4
    JOBS_DB_PATH: Path = Path("data/jobs.db")
    MAX_UPLOAD_BYTES: int = 200000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
