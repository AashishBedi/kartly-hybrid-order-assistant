import sqlite3
from collections.abc import Generator
from datetime import date
from pathlib import Path

import pytest

from app.pipeline.data_evidence import gather_data, plan_tool


SCHEMA_PATH = Path(__file__).parents[1] / "app" / "db" / "schema.sql"
TODAY = date(2026, 10, 3)


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
                (1, "Everyday Laptop", "electronics", 800.0, 12, 0),
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
                (1, 1, "delivered", "2026-09-01", "2026-09-02", "2026-09-03", 1100.0, "express"),
                (2, 1, "delivered", "2026-08-31", "2026-09-01", "2026-09-02", 40.0, "standard"),
                (3, 1, "delivered", "2026-10-01", "2026-10-02", "2026-10-03", 40.0, "express"),
                (4, 1, "placed", "2026-10-02", None, None, 40.0, "standard"),
                (5, 1, "cancelled", "2026-09-25", None, None, 40.0, "standard"),
                (6, 1, "returned", "2026-08-01", "2026-08-02", "2026-08-04", 40.0, "standard"),
                (7, 2, "delivered", "2026-09-30", "2026-10-01", "2026-10-02", 300.0, "express"),
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
                (4, 3, 3, 1, 40.0),
                (5, 4, 3, 1, 40.0),
                (6, 5, 3, 1, 40.0),
                (7, 6, 3, 1, 40.0),
                (8, 7, 2, 1, 300.0),
            ],
        )

    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("question", "order_id", "expected"),
    [
        ("Where is it?", 42, ("get_order", {"order_id": 42})),
        ("How many orders do I have?", None, ("count_orders_by_status", {})),
        ("COUNT my orders", None, ("count_orders_by_status", {})),
        ("What is the number of orders?", None, ("count_orders_by_status", {})),
        ("Show my placed orders", None, ("list_orders", {"status": "placed"})),
        ("Which orders SHIPPED?", None, ("list_orders", {"status": "shipped"})),
        ("List delivered purchases", None, ("list_orders", {"status": "delivered"})),
        ("Show cancelled orders", None, ("list_orders", {"status": "cancelled"})),
        ("Have any been returned?", None, ("list_orders", {"status": "returned"})),
        ("List my orders", None, ("list_orders", {})),
        ("Show unshipped orders", None, ("list_orders", {})),
        ("How many delivered?", 9, ("get_order", {"order_id": 9})),
    ],
)
def test_plan_tool(
    question: str,
    order_id: int | None,
    expected: tuple[str, dict[str, object]],
) -> None:
    assert plan_tool(question, order_id) == expected


def test_gather_data_hides_another_customers_order(
    db_connection: sqlite3.Connection,
) -> None:
    evidence = gather_data(
        "Where is order 7?", 7, customer_id=1, today=TODAY, conn=db_connection
    )

    assert evidence.tool == "get_order"
    assert len(evidence.query_results) == 1
    assert evidence.query_results[0].params == (7, 1)
    assert evidence.rows == []
    assert evidence.found is False
    assert evidence.facts == {}


@pytest.mark.parametrize(
    ("order_id", "days_since_delivery", "window_end", "within"),
    [
        (1, 30, "2026-10-03", True),
        (2, 31, "2026-10-02", False),
        (3, 0, "2026-11-02", True),
    ],
)
def test_return_window_boundaries(
    db_connection: sqlite3.Connection,
    order_id: int,
    days_since_delivery: int,
    window_end: str,
    within: bool,
) -> None:
    evidence = gather_data(
        "Can I return this order?",
        order_id,
        customer_id=1,
        today=TODAY,
        conn=db_connection,
    )

    assert evidence.facts["days_since_delivery"] == days_since_delivery
    assert evidence.facts["return_window_end"] == window_end
    assert evidence.facts["within_return_window"] is within


@pytest.mark.parametrize(
    ("order_id", "reason"),
    [
        (4, "not delivered yet"),
        (5, "status is cancelled"),
        (6, "status is returned"),
    ],
)
def test_non_delivered_statuses_are_not_returnable(
    db_connection: sqlite3.Connection,
    order_id: int,
    reason: str,
) -> None:
    evidence = gather_data(
        "Can I return this order?",
        order_id,
        customer_id=1,
        today=TODAY,
        conn=db_connection,
    )

    assert evidence.facts["within_return_window"] is False
    assert evidence.facts["returnable_items"] == []
    assert evidence.facts["not_returnable_reason"] == reason


def test_final_sale_item_is_not_returnable(
    db_connection: sqlite3.Connection,
) -> None:
    evidence = gather_data(
        "Can I return order 1?", 1, customer_id=1, today=TODAY, conn=db_connection
    )

    assert evidence.facts["final_sale_items"] == ["Final Sale Camera"]
    assert evidence.facts["returnable_items"] == ["Everyday Laptop"]


def test_warranty_end_date_for_twelve_month_item(
    db_connection: sqlite3.Connection,
) -> None:
    evidence = gather_data(
        "Is order 1 under warranty?",
        1,
        customer_id=1,
        today=TODAY,
        conn=db_connection,
    )

    laptop_warranty = next(
        item
        for item in evidence.facts["warranty_per_item"]
        if item["product"] == "Everyday Laptop"
    )
    assert laptop_warranty == {
        "product": "Everyday Laptop",
        "warranty_months": 12,
        "warranty_end_date": "2027-09-03",
    }
