import calendar
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.config import settings
from app.data.access import run_tool
from app.data.queries import ORDER_STATUSES, QueryResult


_COUNT_PATTERN = re.compile(r"\b(?:how\s+many|count|number\s+of)\b", re.IGNORECASE)
_STATUS_PATTERN = re.compile(
    rf"\b({'|'.join(re.escape(status) for status in ORDER_STATUSES)})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DataEvidence:
    tool: str
    query_results: list[QueryResult]
    rows: list[dict[str, Any]]
    found: bool
    facts: dict[str, Any]


def plan_tool(question: str, order_id: int | None) -> tuple[str, dict[str, Any]]:
    """Choose a data tool using deterministic question rules."""
    if order_id is not None:
        return "get_order", {"order_id": order_id}
    if _COUNT_PATTERN.search(question):
        return "count_orders_by_status", {}

    status_match = _STATUS_PATTERN.search(question)
    if status_match is not None:
        return "list_orders", {"status": status_match.group(1).casefold()}
    return "list_orders", {}


def gather_data(
    question: str,
    order_id: int | None,
    customer_id: int,
    today: date,
    conn: sqlite3.Connection | None = None,
) -> DataEvidence:
    """Run the planned customer-scoped tool and derive any order facts."""
    tool, args = plan_tool(question, order_id)
    result = run_tool(tool, args, customer_id, conn=conn)
    rows = result.rows
    facts = compute_order_facts(rows, today) if tool == "get_order" and rows else {}
    return DataEvidence(
        tool=tool,
        query_results=[result],
        rows=rows,
        found=bool(rows),
        facts=facts,
    )


def compute_order_facts(
    rows: list[dict[str, Any]],
    today: date,
) -> dict[str, Any]:
    """Compute return and warranty facts for rows from ``get_order``."""
    if not rows:
        return {}

    order = rows[0]
    status = order["status"]
    delivered_date = _parse_optional_date(order.get("delivered_date"))
    return_window_end = (
        delivered_date + timedelta(days=settings.RETURN_WINDOW_DAYS)
        if delivered_date is not None
        else None
    )
    within_return_window = (
        status == "delivered"
        and return_window_end is not None
        and today <= return_window_end
    )

    final_sale_items = [
        row["product_name"] for row in rows if row["is_final_sale"] == 1
    ]
    returnable_items = [
        row["product_name"]
        for row in rows
        if within_return_window and row["is_final_sale"] != 1
    ]
    warranty_per_item = [
        {
            "product": row["product_name"],
            "warranty_months": row["warranty_months"],
            "warranty_end_date": (
                _add_months(delivered_date, row["warranty_months"]).isoformat()
                if delivered_date is not None
                and status == "delivered"
                and row["warranty_months"] > 0
                else None
            ),
        }
        for row in rows
    ]

    return {
        "status": status,
        "order_date": order["order_date"],
        "delivered_date": (
            delivered_date.isoformat() if delivered_date is not None else None
        ),
        "days_since_delivery": (
            (today - delivered_date).days if delivered_date is not None else None
        ),
        "return_window_end": (
            return_window_end.isoformat() if return_window_end is not None else None
        ),
        "within_return_window": within_return_window,
        "final_sale_items": final_sale_items,
        "returnable_items": returnable_items,
        "warranty_per_item": warranty_per_item,
        "not_returnable_reason": _not_returnable_reason(
            status,
            delivered_date,
            return_window_end,
            within_return_window,
        ),
    }


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _not_returnable_reason(
    status: str,
    delivered_date: date | None,
    return_window_end: date | None,
    within_return_window: bool,
) -> str | None:
    if status == "cancelled":
        return "status is cancelled"
    if status == "returned":
        return "status is returned"
    if status != "delivered" or delivered_date is None:
        return "not delivered yet"
    if not within_return_window and return_window_end is not None:
        return f"return window ended on {return_window_end.isoformat()}"
    return None
