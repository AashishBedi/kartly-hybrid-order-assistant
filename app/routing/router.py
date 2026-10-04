"""Route support questions with deterministic guards and an LLM classifier."""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm.client import LLMClient, LLMError
from app.routing.rules import check_blocked, extract_order_id


Route = Literal["data", "policy", "combined", "out_of_scope"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    order_id: int | None
    blocked: bool
    reason: str
    fallback_used: bool


class _RouterResponse(BaseModel):
    route: Route
    reason: str

    model_config = ConfigDict(extra="forbid")


_SYSTEM_PROMPT = """Classify the user's Kartly support question into one route:
- data: facts about the customer's own orders, such as status, items, dates,
  totals, or listing or counting their orders. Example: "Where is my order?"
- policy: Kartly rules about returns, refunds, shipping, warranty, exchanges, or
  payments, with no reference to a specific order. Example: "What is the return
  policy?"
- combined: needs both the customer's order facts and a policy rule. Examples:
  "Can I still return order 1042?" and "Is my headphones order covered by
  warranty?"
- out_of_scope: anything else, including chit-chat, other companies, general
  knowledge, and requests to perform actions such as cancelling, refunding, or
  changing something now. This assistant is read-only.

Reply ONLY with JSON in this form:
{"route": "data|policy|combined|out_of_scope", "reason": "<short>"}
"""

_POLICY_PATTERNS = (
    re.compile(r"\breturn(?:s|ed|ing)?\b"),
    re.compile(r"\brefund(?:s|ed|ing)?\b"),
    re.compile(r"\bexchange(?:s|d|ing)?\b"),
    re.compile(r"\bwarrant(?:y|ies)\b"),
    re.compile(r"\bshipping\b"),
    re.compile(r"\bdelivery\s+time\b"),
    re.compile(r"\bcancel(?:s|led|ing|lation)?\b"),
    re.compile(r"\bpolic(?:y|ies)\b"),
    re.compile(r"\bdo\s+you\s+offer\b"),
    re.compile(r"\bdo\s+you\s+accept\b"),
    re.compile(r"\bcan\s+i\s+pay\b"),
    re.compile(r"\bcan\s+i\s+use\b"),
)
_DATA_PATTERNS = (
    re.compile(r"\bhow\s+many\s+orders?\b"),
    re.compile(r"\bmy\s+orders?\b"),
    re.compile(r"\bi\s+ordered\b"),
    re.compile(r"\bi\s+placed\b"),
    re.compile(r"\bhave\s+i\b"),
    re.compile(r"\blast\s+order\b"),
    re.compile(r"\brecent\s+orders\b"),
    re.compile(r"\border\s+status\b"),
    re.compile(r"\bwhere\s+is\b"),
    re.compile(r"\btracking\b"),
    re.compile(r"\bitems\b"),
    re.compile(r"\btotal\b"),
    re.compile(r"\bordered\b"),
)


def route_question(
    question: str,
    llm_client: LLMClient,
    model: str,
) -> RouteDecision:
    """Classify a question after applying the deterministic request guard."""
    order_id = extract_order_id(question)
    block = check_blocked(question)
    if block is not None:
        return RouteDecision(
            route="out_of_scope",
            order_id=order_id,
            blocked=True,
            reason=block.category,
            fallback_used=False,
        )

    try:
        response = llm_client.chat(
            model,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            json_mode=True,
            stage="router",
            max_tokens=100,
        )
    except LLMError:
        return _fallback_decision(question, order_id, "llm_error")

    try:
        parsed = _RouterResponse.model_validate_json(response.text)
    except ValidationError:
        return _fallback_decision(
            question,
            order_id,
            "invalid_llm_response",
        )

    return RouteDecision(
        route=parsed.route,
        order_id=order_id,
        blocked=False,
        reason=parsed.reason,
        fallback_used=False,
    )


def fallback_route(question: str, order_id: int | None) -> Route:
    """Classify a question using a small, deterministic keyword heuristic."""
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    has_policy_word = any(
        pattern.search(normalized) for pattern in _POLICY_PATTERNS
    )
    has_data_word = any(pattern.search(normalized) for pattern in _DATA_PATTERNS)

    if has_policy_word and (order_id is not None or has_data_word):
        return "combined"
    if has_data_word or order_id is not None:
        return "data"
    if has_policy_word:
        return "policy"
    return "policy"


def _fallback_decision(
    question: str,
    order_id: int | None,
    reason: str,
) -> RouteDecision:
    return RouteDecision(
        route=fallback_route(question, order_id),
        order_id=order_id,
        blocked=False,
        reason=reason,
        fallback_used=True,
    )
