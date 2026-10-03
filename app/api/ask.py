import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.data.access import CUSTOMER_EXISTS_SQL
from app.deps import (
    get_embedder,
    get_llm_client,
    get_readonly_connection,
    get_store,
)
from app.llm.client import LLMClient
from app.pipeline.ask import handle_ask
from app.rag.embeddings import Embedder
from app.rag.store import PolicyStore


router = APIRouter()


class AskRequest(BaseModel):
    customer_id: Annotated[int, Field(strict=True, gt=0)]
    question: Annotated[str, Field(min_length=1, max_length=500)]

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


@router.post("/ask")
def ask(
    body: AskRequest,
    store: PolicyStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
    llm_client: LLMClient = Depends(get_llm_client),
    conn: sqlite3.Connection = Depends(get_readonly_connection),
) -> dict:
    customer = conn.execute(CUSTOMER_EXISTS_SQL, (body.customer_id,)).fetchone()
    if customer is None:
        raise HTTPException(status_code=404, detail="customer not found")

    return handle_ask(
        customer_id=body.customer_id,
        question=body.question,
        store=store,
        embedder=embedder,
        llm_client=llm_client,
        conn=conn,
        router_model=settings.ROUTER_MODEL,
        answer_model=settings.ANSWER_MODEL,
        top_k=settings.RETRIEVAL_TOP_K,
        max_distance=settings.RELEVANCE_MAX_DISTANCE,
    )
