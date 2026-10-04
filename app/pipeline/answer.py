import json

from app.llm.client import LLMClient
from app.pipeline.data_evidence import DataEvidence
from app.pipeline.policy_evidence import PolicyEvidence
from app.routing.router import Route


_SYSTEM_PROMPT = """You are Kartly's order support assistant.
Answer ONLY from the evidence provided. If the evidence does not contain the
answer, say so. Never invent order details, dates, or policy rules. Use computed
facts exactly as given and do not recompute dates. Be concise and friendly.
The order total comes only from the order-level total field; never add up item values or repeated values.
Mention the order number when it is relevant."""


def generate_answer(
    question: str,
    route: Route,
    data_evidence: DataEvidence,
    policy_evidence: PolicyEvidence,
    llm_client: LLMClient,
    model: str,
) -> str:
    """Generate an answer from only the evidence allowed by the route."""
    sections = [f"Question:\n{question}"]

    if route in {"data", "combined"}:
        data_payload = _answer_data_payload(data_evidence)
        sections.append(
            "Order evidence (JSON):\n"
            + json.dumps(data_payload, ensure_ascii=False, sort_keys=True)
        )

    if route in {"policy", "combined"}:
        policy_payload = [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
            for chunk in policy_evidence.chunks
        ]
        sections.append(
            "Policy chunks (JSON):\n"
            + json.dumps(policy_payload, ensure_ascii=False, sort_keys=True)
        )

    response = llm_client.chat(
        model,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(sections)},
        ],
        stage="answer",
        temperature=0,
        max_tokens=400,
    )
    return response.text


def _answer_data_payload(data_evidence: DataEvidence) -> dict[str, object]:
    if data_evidence.tool != "get_order" or not data_evidence.rows:
        return {
            "rows": data_evidence.rows,
            "computed_facts": data_evidence.facts,
        }

    first_row = data_evidence.rows[0]
    order = {
        field: first_row.get(field)
        for field in (
            "order_id",
            "status",
            "order_date",
            "shipped_date",
            "delivered_date",
            "shipping_method",
            "total",
        )
    }
    items = [
        {
            "name": row["product_name"],
            "quantity": row["quantity"],
            "unit_price": row["unit_price"],
            "line_total": round(row["quantity"] * row["unit_price"], 2),
        }
        for row in data_evidence.rows
    ]
    return {
        "order": order,
        "items": items,
        "computed_facts": data_evidence.facts,
    }
