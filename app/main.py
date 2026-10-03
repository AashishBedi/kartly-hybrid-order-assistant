from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router


app = FastAPI(title="Kartly Hybrid Order Support Assistant")
app.include_router(ask_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
