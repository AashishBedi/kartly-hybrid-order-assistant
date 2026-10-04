from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_PATH: Path = Path("data/kartly.db")
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_TIMEOUT_SECONDS: float = 20
    LLM_MAX_RETRIES: int = 2
    EVAL_CACHE: str = "0"
    # Estimated USD per 1M tokens; edit to match your model. The free tier
    # costs nothing, but we log an estimate as if paid.
    PRICE_INPUT_PER_MTOK: float = 0.10
    PRICE_OUTPUT_PER_MTOK: float = 0.30
    # Check the Groq console for current model names before setting these values.
    ROUTER_MODEL: str = "replace-with-current-groq-router-model"
    ANSWER_MODEL: str = "replace-with-current-groq-answer-model"
    LOG_LEVEL: str = "INFO"
    CHROMA_PATH: Path = Path("data/chroma")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    WARMUP_ON_STARTUP: bool = True
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    RETRIEVAL_TOP_K: int = 4
    # Chunks with cosine distance above this are treated as not relevant.
    RELEVANCE_MAX_DISTANCE: float = 0.62
    RETURN_WINDOW_DAYS: int = 30
    JOBS_DB_PATH: Path = Path("data/jobs.db")
    MAX_UPLOAD_BYTES: int = 200000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
