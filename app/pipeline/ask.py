import sqlite3
from datetime import date
from typing import Any
from uuid import uuid4

from app.config import settings
from app.llm.client import LLMClient, LLMError
from app.observability import finish_request, start_request
from app.pipeline.answer import generate_answer
from app.pipeline.data_evidence import DataEvidence, gather_data
from app.pipeline.grounding import check_grounding
from app.pipeline.policy_evidence import PolicyEvidence, gather_policy
from app.rag.embeddings import Embedder
from app.rag.store import PolicyStore
from app.routing.router import Route, route_question
from app.routing.rules import check_blocked


_POLICY_FALLBACK_PREFIX = (
    "I couldn't generate a full answer right now, but here is the most "
    "relevant policy:"
)


def handle_ask(
    customer_id: int,
    question: str,
    store: PolicyStore,
    embedder: Embedder,
    llm_client: LLMClient,
    conn: sqlite3.Connection,
    today: date | None = None,
    router_model: str = settings.ROUTER_MODEL,
    answer_model: str = settings.ANSWER_MODEL,
    top_k: int = settings.RETRIEVAL_TOP_K,
    max_distance: float = settings.RELEVANCE_MAX_DISTANCE,
) -> dict[str, Any]:
    """Run the routed, evidence-grounded ask pipeline."""
    request_id = str(uuid4())
    start_request(request_id)
    try:
        decision = route_question(question, llm_client, router_model)
        if decision.blocked:
            block = check_blocked(question)
            answer = block.message if block is not None else decision.reason
            return _response(
                answer=answer,
                route=decision.route,
                blocked=True,
                grounded=False,
                degraded=False,
                request_id=request_id,
            )

        data_evidence = _empty_data_evidence()
        policy_evidence = _empty_policy_evidence()
        resolved_today = today if today is not None else date.today()

        if decision.route in {"data", "combined"}:
            data_evidence = gather_data(
                question,
                decision.order_id,
                customer_id,
                resolved_today,
                conn=conn,
            )
        if decision.route in {"policy", "combined"}:
            policy_evidence = gather_policy(
                question,
                store,
                embedder,
                top_k,
                max_distance,
            )

        grounding = check_grounding(
            decision.route,
            data_evidence,
            policy_evidence,
        )
        if not grounding.ok:
            return _response(
                answer=grounding.message or "",
                route=decision.route,
                blocked=False,
                grounded=False,
                degraded=False,
                request_id=request_id,
            )

        degraded = False
        try:
            answer = generate_answer(
                question,
                decision.route,
                data_evidence,
                policy_evidence,
                llm_client,
                answer_model,
            )
        except LLMError:
            answer = _fallback_answer(
                decision.route,
                data_evidence,
                policy_evidence,
            )
            degraded = True

        return _response(
            answer=answer,
            route=decision.route,
            blocked=False,
            grounded=True,
            degraded=degraded,
            request_id=request_id,
            data_evidence=data_evidence,
            policy_evidence=policy_evidence,
        )
    finally:
        finish_request()


def _empty_data_evidence() -> DataEvidence:
    return DataEvidence(
        tool="",
        query_results=[],
        rows=[],
        found=False,
        facts={},
    )


def _empty_policy_evidence() -> PolicyEvidence:
    return PolicyEvidence(chunks=[], found=False, best_distance=None)


def _response(
    *,
    answer: str,
    route: Route,
    blocked: bool,
    grounded: bool,
    degraded: bool,
    request_id: str,
    data_evidence: DataEvidence | None = None,
    policy_evidence: PolicyEvidence | None = None,
) -> dict[str, Any]:
    sql_sources = []
    chunk_sources = []
    if grounded and data_evidence is not None:
        sql_sources = [
            {
                "tool": result.tool,
                "sql": result.sql,
                "params": list(result.params),
            }
            for result in data_evidence.query_results
        ]
    if grounded and policy_evidence is not None:
        chunk_sources = [
            chunk["chunk_id"] for chunk in policy_evidence.chunks
        ]

    return {
        "answer": answer,
        "route": route,
        "blocked": blocked,
        "grounded": grounded,
        "degraded": degraded,
        "sources": {"sql": sql_sources, "chunks": chunk_sources},
        "request_id": request_id,
    }


def _fallback_answer(
    route: Route,
    data_evidence: DataEvidence,
    policy_evidence: PolicyEvidence,
) -> str:
    sections = []
    if route in {"data", "combined"}:
        sections.append(_data_fallback(data_evidence))
    if route in {"policy", "combined"} and policy_evidence.chunks:
        sections.append(
            f"{_POLICY_FALLBACK_PREFIX} {policy_evidence.chunks[0]['text']}"
        )
    return "\n\n".join(sections)


def _data_fallback(evidence: DataEvidence) -> str:
    facts = evidence.facts
    if facts:
        order_id = evidence.rows[0].get("order_id") if evidence.rows else None
        lead = f"Order {order_id}" if order_id is not None else "Your order"
        items = list(dict.fromkeys(row["product_name"] for row in evidence.rows))
        delivered = facts["delivered_date"] or "not delivered"
        return (
            f"{lead} has status {facts['status']}. It was ordered on "
            f"{facts['order_date']} and delivered: {delivered}. "
            f"Items: {', '.join(items)}."
        )

    if evidence.tool == "count_orders_by_status":
        counts = ", ".join(
            f"{row['status']}: {row['count']}" for row in evidence.rows
        )
        return f"Your order counts are {counts}."

    orders = ", ".join(
        f"order {row['id']} is {row['status']} (ordered {row['order_date']})"
        for row in evidence.rows
    )
    return f"I found these orders: {orders}."
