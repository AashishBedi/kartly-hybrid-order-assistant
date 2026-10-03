"""Resolve labelled-case checks against live seeded data at evaluation time."""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from decimal import Decimal
from typing import Any

from app.data.access import run_tool
from app.pipeline.data_evidence import compute_order_facts


YES_PHRASES = ("eligible", "can return", "within")
NO_PHRASES = (
    "not eligible",
    "ineligible",
    "cannot",
    "can't",
    "not returnable",
    "outside",
    "ended",
    "expired",
)
REFUSAL_PHRASES = (
    "can't",
    "cannot",
    "couldn't",
    "unable",
    "don't have",
    "do not have",
    "not available",
    "no information",
    "only help with",
    "not something i can",
)


def resolve_check(
    check: dict[str, Any],
    customer_id: int,
    conn: sqlite3.Connection,
    today: date,
) -> dict[str, Any]:
    """Return the concrete expectation for one abstract case check."""
    kind = check["kind"]
    if kind in {"policy_number", "must_not_contain"}:
        return dict(check)
    if kind == "refusal":
        return {"kind": kind, "phrases": list(REFUSAL_PHRASES)}

    order_id = check["order_id"]
    rows = run_tool(
        "get_order",
        {"order_id": order_id},
        customer_id,
        conn=conn,
    ).rows
    if not rows:
        raise ValueError(
            f"Order {order_id} does not belong to customer {customer_id}"
        )

    if kind == "order_status":
        return {"kind": kind, "value": rows[0]["status"]}
    if kind == "order_total":
        total = Decimal(str(rows[0]["total"]))
        return {"kind": kind, "values": _total_formats(total)}
    if kind == "item_name":
        names = list(dict.fromkeys(row["product_name"] for row in rows))
        return {"kind": kind, "values": names}
    if kind == "return_decision":
        facts = compute_order_facts(rows, today)
        if facts["final_sale_items"]:
            return {
                "kind": kind,
                "decision": "no",
                "reason": "final sale",
            }
        if facts["within_return_window"] and facts["returnable_items"]:
            return {"kind": kind, "decision": "yes", "reason": None}

        reason = facts["not_returnable_reason"] or "not eligible"
        if "not delivered" in reason or facts["delivered_date"] is None:
            reason = "not delivered"
        elif "window ended" in reason:
            reason = "window ended"
        return {"kind": kind, "decision": "no", "reason": reason}

    raise ValueError(f"Unknown check kind: {kind}")


def resolve_case(
    case: dict[str, Any],
    conn: sqlite3.Connection,
    today: date,
) -> list[dict[str, Any]]:
    """Resolve every check for a case using one explicit evaluation date."""
    return [
        resolve_check(check, case["customer_id"], conn, today)
        for check in case["expect"]
    ]


def check_answer(answer: str, resolved: dict[str, Any]) -> bool:
    """Evaluate one resolved check without making any model or API calls."""
    normalized = re.sub(r"\s+", " ", answer).strip().casefold()
    kind = resolved["kind"]

    if kind == "order_status":
        return _contains_word(normalized, resolved["value"])
    if kind == "order_total":
        return any(_contains_number(answer, value) for value in resolved["values"])
    if kind == "item_name":
        return any(value.casefold() in normalized for value in resolved["values"])
    if kind == "policy_number":
        return _contains_number(answer, str(resolved["value"]))
    if kind == "refusal":
        return any(phrase in normalized for phrase in resolved["phrases"])
    if kind == "must_not_contain":
        return str(resolved["value"]).casefold() not in normalized
    if kind == "return_decision":
        return _check_return_decision(normalized, resolved)
    raise ValueError(f"Unknown resolved check kind: {kind}")


def _total_formats(total: Decimal) -> list[str]:
    plain = format(total, "f")
    fixed = f"{total:.2f}"
    grouped = f"{total:,.2f}"
    values = {plain, fixed, grouped}
    if total == total.to_integral():
        integer = str(int(total))
        values.update({integer, f"{int(total):,}", f"{int(total):,}.00"})
    return sorted(values)


def _contains_word(answer: str, value: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", answer) is not None


def _contains_number(answer: str, value: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", answer) is not None


def _check_return_decision(answer: str, resolved: dict[str, Any]) -> bool:
    has_no = any(phrase in answer for phrase in NO_PHRASES)
    if resolved["decision"] == "yes":
        return not has_no and any(phrase in answer for phrase in YES_PHRASES)
    if not has_no:
        return False

    reason = resolved["reason"]
    reason_phrases = {
        "final sale": ("final sale", "final-sale"),
        "not delivered": (
            "not delivered",
            "not yet delivered",
            "hasn't been delivered",
            "wait until",
        ),
        "window ended": (
            "window ended",
            "window has ended",
            "outside",
            "expired",
            "past the",
            "more than 30",
        ),
    }
    phrases = reason_phrases.get(reason, (str(reason).casefold(),))
    return any(phrase in answer for phrase in phrases)
