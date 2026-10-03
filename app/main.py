import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.config import settings
from app.deps import get_embedder, get_store


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.WARMUP_ON_STARTUP:
        embedder_provider = app.dependency_overrides.get(
            get_embedder,
            get_embedder,
        )
        try:
            embedder_provider().warm_up()
        except Exception:
            logger.warning("Embedder warm-up failed; startup will continue.", exc_info=True)

        store_provider = app.dependency_overrides.get(get_store, get_store)
        try:
            store_provider()
        except Exception:
            logger.warning("Chroma warm-up failed; startup will continue.", exc_info=True)

    yield


app = FastAPI(
    title="Kartly Hybrid Order Support Assistant",
    lifespan=lifespan,
)
app.include_router(ask_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
