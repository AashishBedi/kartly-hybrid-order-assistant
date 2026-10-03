import json
import re
import sqlite3
from pathlib import Path

import pytest

from app.config import settings
from app.routing.rules import extract_order_id


CASES_PATH = Path(__file__).parents[1] / "eval" / "cases.json"
CASE_KEYS = {
    "id",
    "question",
    "customer_id",
    "category",
    "expected_route",
    "expected_blocked",
    "answerable",
    "expect",
}
ROUTES = {"data", "policy", "combined", "out_of_scope"}
CATEGORIES = {
    "data",
    "policy",
    "combined",
    "out_of_scope",
    "unanswerable",
    "adversarial",
}
ORDER_CHECKS = {
    "order_status",
    "order_total",
    "item_name",
    "return_decision",
}
VALUE_CHECKS = {"policy_number", "must_not_contain"}
CROSS_CUSTOMER_CASE_IDS = {"A04"}


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db() -> sqlite3.Connection:
    uri = f"file:{settings.DATABASE_PATH.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    yield conn
    conn.close()


def test_every_case_matches_schema(cases: list[dict]) -> None:
    for case in cases:
        assert set(case) == CASE_KEYS, case.get("id")
        assert re.fullmatch(r"[DPCOUA]\d{2}", case["id"])
        assert isinstance(case["question"], str) and case["question"].strip()
        assert type(case["customer_id"]) is int and case["customer_id"] > 0
        assert case["category"] in CATEGORIES
        assert case["expected_route"] in ROUTES
        assert type(case["expected_blocked"]) is bool
        assert type(case["answerable"]) is bool
        assert isinstance(case["expect"], list)
        for check in case["expect"]:
            _assert_check_schema(check)


def test_case_ids_are_unique(cases: list[dict]) -> None:
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_minimum_size_routes_and_adversarial_coverage(
    cases: list[dict],
) -> None:
    assert len(cases) >= 30
    assert {case["expected_route"] for case in cases} == ROUTES
    assert any(case["category"] == "adversarial" for case in cases)


def test_referenced_orders_exist_and_have_expected_ownership(
    cases: list[dict],
    db: sqlite3.Connection,
) -> None:
    for case in cases:
        order_ids = {
            check["order_id"]
            for check in case["expect"]
            if check["kind"] in ORDER_CHECKS
        }
        question_order_id = extract_order_id(case["question"])
        if question_order_id is not None:
            order_ids.add(question_order_id)

        assert order_ids or case["id"] not in CROSS_CUSTOMER_CASE_IDS
        for order_id in order_ids:
            row = db.execute(
                "SELECT customer_id FROM orders WHERE id = ?",
                (order_id,),
            ).fetchone()
            assert row is not None, (case["id"], order_id)
            if case["id"] in CROSS_CUSTOMER_CASE_IDS:
                assert row["customer_id"] != case["customer_id"]
            else:
                assert row["customer_id"] == case["customer_id"]


def test_cross_customer_cases_include_non_disclosure_checks(
    cases: list[dict],
) -> None:
    by_id = {case["id"]: case for case in cases}
    for case_id in CROSS_CUSTOMER_CASE_IDS:
        case = by_id[case_id]
        kinds = [check["kind"] for check in case["expect"]]
        assert "refusal" in kinds
        assert "must_not_contain" in kinds


def _assert_check_schema(check: dict) -> None:
    assert isinstance(check, dict)
    kind = check.get("kind")
    if kind in ORDER_CHECKS:
        assert set(check) == {"kind", "order_id"}
        assert type(check["order_id"]) is int and check["order_id"] > 0
        return
    if kind in VALUE_CHECKS:
        assert set(check) == {"kind", "value"}
        assert isinstance(check["value"], str) and check["value"]
        return
    assert kind == "refusal"
    assert set(check) == {"kind"}
