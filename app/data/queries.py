from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


ORDER_STATUSES = ("placed", "shipped", "delivered", "cancelled", "returned")


@dataclass(frozen=True)
class QueryResult:
    tool: str
    sql: str
    params: tuple[Any, ...]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ArgumentSchema:
    name: str
    type: type
    required: bool
    allowed_values: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class ToolSpec:
    function: Callable[..., QueryResult]
    arguments: tuple[ArgumentSchema, ...]


GET_ORDER_SQL = """\
SELECT
    o.id AS order_id,
    o.customer_id,
    o.status,
    o.order_date,
    o.shipped_date,
    o.delivered_date,
    o.total,
    o.shipping_method,
    oi.id AS order_item_id,
    oi.product_id,
    p.name AS product_name,
    p.category,
    oi.quantity,
    oi.unit_price,
    p.warranty_months,
    p.is_final_sale
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.id
JOIN products AS p ON p.id = oi.product_id
WHERE o.id = ? AND o.customer_id = ?
ORDER BY oi.id"""

LIST_ORDERS_SQL = """\
SELECT
    o.id,
    o.customer_id,
    o.status,
    o.order_date,
    o.shipped_date,
    o.delivered_date,
    o.total,
    o.shipping_method
FROM orders AS o
WHERE o.customer_id = ?
ORDER BY o.order_date DESC, o.id DESC
LIMIT ?"""

LIST_ORDERS_BY_STATUS_SQL = """\
SELECT
    o.id,
    o.customer_id,
    o.status,
    o.order_date,
    o.shipped_date,
    o.delivered_date,
    o.total,
    o.shipping_method
FROM orders AS o
WHERE o.customer_id = ? AND o.status = ?
ORDER BY o.order_date DESC, o.id DESC
LIMIT ?"""

COUNT_ORDERS_BY_STATUS_SQL = """\
SELECT
    o.status,
    COUNT(*) AS count
FROM orders AS o
WHERE o.customer_id = ?
GROUP BY o.status
ORDER BY o.status"""


def get_order(customer_id: int, order_id: int) -> QueryResult:
    return QueryResult(
        tool="get_order",
        sql=GET_ORDER_SQL,
        params=(order_id, customer_id),
        rows=[],
    )


def list_orders(
    customer_id: int,
    status: str | None = None,
    limit: int = 10,
) -> QueryResult:
    if status is None:
        sql = LIST_ORDERS_SQL
        params = (customer_id, limit)
    else:
        sql = LIST_ORDERS_BY_STATUS_SQL
        params = (customer_id, status, limit)

    return QueryResult(tool="list_orders", sql=sql, params=params, rows=[])


def count_orders_by_status(customer_id: int) -> QueryResult:
    return QueryResult(
        tool="count_orders_by_status",
        sql=COUNT_ORDERS_BY_STATUS_SQL,
        params=(customer_id,),
        rows=[],
    )


TOOLS: dict[str, ToolSpec] = {
    "get_order": ToolSpec(
        function=get_order,
        arguments=(
            ArgumentSchema(name="order_id", type=int, required=True),
        ),
    ),
    "list_orders": ToolSpec(
        function=list_orders,
        arguments=(
            ArgumentSchema(
                name="status",
                type=str,
                required=False,
                allowed_values=ORDER_STATUSES,
            ),
            ArgumentSchema(name="limit", type=int, required=False),
        ),
    ),
    "count_orders_by_status": ToolSpec(
        function=count_orders_by_status,
        arguments=(),
    ),
}
