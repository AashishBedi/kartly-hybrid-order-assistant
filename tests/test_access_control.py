import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from app.config import settings
from app.data.access import assert_safe_sql, run_tool
from app.data.errors import AccessError
from app.data.queries import TOOLS
from app.db.connection import get_readonly_connection, get_write_connection


SCHEMA_PATH = Path(__file__).parents[1] / "app" / "db" / "schema.sql"


@pytest.fixture
def db_connection() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    with connection:
        connection.executemany(
            "INSERT INTO customers (id, name, email, created_at) VALUES (?, ?, ?, ?)",
            [
                (1, "Customer One", "one@example.com", "2026-01-01"),
                (2, "Customer Two", "two@example.com", "2026-01-02"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO products
                (id, name, category, price, warranty_months, is_final_sale)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "Everyday Laptop", "electronics", 800.0, 24, 0),
                (2, "Final Sale Camera", "electronics", 300.0, 12, 1),
                (3, "Cotton Shirt", "clothing", 40.0, 0, 0),
            ],
        )
        connection.executemany(
            """
            INSERT INTO orders
                (id, customer_id, status, order_date, shipped_date,
                 delivered_date, total, shipping_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "delivered", "2026-01-10", "2026-01-11", "2026-01-13", 1100.0, "express"),
                (2, 1, "placed", "2026-02-10", None, None, 40.0, "standard"),
                (3, 2, "shipped", "2026-03-10", "2026-03-11", None, 300.0, "express"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO order_items
                (id, order_id, product_id, quantity, unit_price)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, 1, 1, 1, 800.0),
                (2, 1, 2, 1, 300.0),
                (3, 2, 3, 1, 40.0),
                (4, 3, 2, 1, 300.0),
            ],
        )

    try:
        yield connection
    finally:
        connection.close()


def test_get_order_returns_order_and_items_for_owner(
    db_connection: sqlite3.Connection,
) -> None:
    result = run_tool("get_order", {"order_id": 1}, 1, conn=db_connection)

    assert result.tool == "get_order"
    assert result.params == (1, 1)
    assert len(result.rows) == 2
    assert {row["product_name"] for row in result.rows} == {
        "Everyday Laptop",
        "Final Sale Camera",
    }
    final_sale_item = next(row for row in result.rows if row["is_final_sale"] == 1)
    assert final_sale_item["warranty_months"] == 12
    assert all(row["customer_id"] == 1 for row in result.rows)


def test_get_order_hides_another_customers_order_like_a_missing_order(
    db_connection: sqlite3.Connection,
) -> None:
    another_customers_order = run_tool(
        "get_order", {"order_id": 3}, 1, conn=db_connection
    )
    nonexistent_order = run_tool(
        "get_order", {"order_id": 999}, 1, conn=db_connection
    )

    assert another_customers_order.rows == nonexistent_order.rows == []
    assert another_customers_order.tool == nonexistent_order.tool == "get_order"
    assert another_customers_order.sql == nonexistent_order.sql


def test_list_orders_is_scoped_and_supports_status_filter(
    db_connection: sqlite3.Connection,
) -> None:
    all_orders = run_tool("list_orders", {}, 1, conn=db_connection)
    delivered_orders = run_tool(
        "list_orders", {"status": "delivered"}, 1, conn=db_connection
    )

    assert [row["id"] for row in all_orders.rows] == [2, 1]
    assert [row["id"] for row in delivered_orders.rows] == [1]
    assert all(row["customer_id"] == 1 for row in all_orders.rows)


@pytest.mark.parametrize("limit", [0, 21])
def test_list_orders_rejects_limit_outside_allowed_range(
    db_connection: sqlite3.Connection,
    limit: int,
) -> None:
    with pytest.raises(AccessError):
        run_tool("list_orders", {"limit": limit}, 1, conn=db_connection)


def test_count_orders_by_status_is_scoped_to_customer(
    db_connection: sqlite3.Connection,
) -> None:
    result = run_tool("count_orders_by_status", {}, 1, conn=db_connection)

    assert result.rows == [
        {"status": "delivered", "count": 1},
        {"status": "placed", "count": 1},
    ]


@pytest.mark.parametrize("forbidden_arg", ["customer_id", "sql", "query"])
def test_forbidden_argument_names_raise_access_error(
    db_connection: sqlite3.Connection,
    forbidden_arg: str,
) -> None:
    with pytest.raises(AccessError):
        run_tool(
            "count_orders_by_status",
            {forbidden_arg: "untrusted"},
            1,
            conn=db_connection,
        )


@pytest.mark.parametrize("order_id", ["1 OR 1=1", "1; DROP TABLE orders"])
def test_get_order_rejects_string_order_ids(
    db_connection: sqlite3.Connection,
    order_id: str,
) -> None:
    with pytest.raises(AccessError):
        run_tool("get_order", {"order_id": order_id}, 1, conn=db_connection)


def test_unknown_tool_name_raises_access_error(
    db_connection: sqlite3.Connection,
) -> None:
    with pytest.raises(AccessError):
        run_tool("run_sql", {}, 1, conn=db_connection)


def test_nonexistent_customer_raises_access_error(
    db_connection: sqlite3.Connection,
) -> None:
    with pytest.raises(AccessError):
        run_tool("list_orders", {}, 999, conn=db_connection)


@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "UPDATE customers SET email='x'",
        "PRAGMA table_info(orders)",
    ],
)
def test_assert_safe_sql_rejects_unsafe_statements(unsafe_sql: str) -> None:
    with pytest.raises(AccessError):
        assert_safe_sql(unsafe_sql)


def test_assert_safe_sql_accepts_normal_select() -> None:
    assert_safe_sql("SELECT id FROM orders WHERE customer_id = ?")


def test_get_readonly_connection_rejects_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "readonly.db"
    monkeypatch.setattr(settings, "DATABASE_PATH", database_path)

    write_connection = get_write_connection()
    try:
        write_connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        write_connection.close()

    readonly_connection = get_readonly_connection()
    try:
        with pytest.raises(sqlite3.OperationalError):
            readonly_connection.execute(
                """
                INSERT INTO customers (name, email, created_at)
                VALUES (?, ?, ?)
                """,
                ("Blocked", "blocked@example.com", "2026-01-01"),
            )
    finally:
        readonly_connection.close()


def test_all_registered_tools_keep_values_in_separate_params() -> None:
    customer_id = 987654321
    order_id = 123456789
    plans = [
        TOOLS["get_order"].function(customer_id, order_id=order_id),
        TOOLS["list_orders"].function(customer_id, limit=7),
        TOOLS["list_orders"].function(
            customer_id, status="delivered", limit=7
        ),
        TOOLS["count_orders_by_status"].function(customer_id),
    ]

    assert {plan.tool for plan in plans} == set(TOOLS)
    for plan in plans:
        assert "customer_id = ?" in plan.sql
        assert customer_id in plan.params
        assert str(customer_id) not in plan.sql
        assert ";" not in plan.sql

    assert order_id in plans[0].params
    assert str(order_id) not in plans[0].sql
    assert "delivered" in plans[2].params
    assert "'delivered'" not in plans[2].sql
