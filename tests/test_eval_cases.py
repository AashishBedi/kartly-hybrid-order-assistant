import json
import re
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.config import settings
from app.pipeline.grounding import (
    ORDER_NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    POLICY_NOT_FOUND_MESSAGE,
)
from app.routing.rules import _BLOCK_MESSAGES, check_blocked, extract_order_id
from eval.resolve import REFUSAL_PHRASES, check_answer, resolve_check


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
VALUES_CHECKS = {"must_contain_any"}
CROSS_CUSTOMER_CASE_IDS = {"A04"}
APP_REFUSAL_MESSAGES = (
    ORDER_NOT_FOUND_MESSAGE,
    POLICY_NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    *_BLOCK_MESSAGES.values(),
)


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
    assert {case["category"] for case in cases} == CATEGORIES


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


def test_strengthened_cases_use_the_new_check_kinds(cases: list[dict]) -> None:
    by_id = {case["id"]: case for case in cases}

    assert by_id["D04"]["expect"] == [{"kind": "order_count"}]
    assert by_id["D05"]["expect"] == [
        {"kind": "order_ids_listed", "n": 3}
    ]
    assert by_id["P04"]["expect"][0] == {
        "kind": "must_contain_any",
        "values": [
            "manufacturing",
            "not cover",
            "does not cover",
            "not covered",
            "excluded",
        ],
    }
    assert by_id["P05"]["expect"][0] == {
        "kind": "must_contain_any",
        "values": ["placed"],
    }
    for case_id in ("P04", "P05"):
        assert any(
            check["kind"] == "must_not_contain"
            for check in by_id[case_id]["expect"]
        )


def test_order_count_resolves_from_db_and_matches_a_whole_word(
    cases: list[dict],
    db: sqlite3.Connection,
) -> None:
    case = next(case for case in cases if case["id"] == "D04")
    resolved = resolve_check(
        case["expect"][0],
        case["customer_id"],
        db,
        date.today(),
    )
    expected = db.execute(
        "SELECT COUNT(*) FROM orders WHERE customer_id = ?",
        (case["customer_id"],),
    ).fetchone()[0]

    assert resolved == {"kind": "order_count", "value": expected}
    assert check_answer(f"You have {expected} orders.", resolved)
    assert not check_answer(f"You have 1{expected} orders.", resolved)


def test_order_ids_listed_resolves_latest_delivered_orders(
    cases: list[dict],
    db: sqlite3.Connection,
) -> None:
    case = next(case for case in cases if case["id"] == "D05")
    resolved = resolve_check(
        case["expect"][0],
        case["customer_id"],
        db,
        date.today(),
    )
    expected = [
        row["id"]
        for row in db.execute(
            """
            SELECT id
            FROM orders
            WHERE customer_id = ? AND status = 'delivered'
            ORDER BY order_date DESC, id DESC
            LIMIT 3
            """,
            (case["customer_id"],),
        ).fetchall()
    ]

    assert resolved == {"kind": "order_ids_listed", "values": expected}
    assert check_answer(
        "Delivered orders: " + ", ".join(f"#{value}" for value in expected),
        resolved,
    )
    assert not check_answer(
        "Delivered orders: " + ", ".join(f"#{value}" for value in expected[:-1]),
        resolved,
    )


def test_must_contain_any_is_case_insensitive(cases: list[dict]) -> None:
    by_id = {case["id"]: case for case in cases}

    assert check_answer(
        "Water damage is NOT COVERED by the warranty.",
        by_id["P04"]["expect"][0],
    )
    assert not check_answer(
        "Please contact support for warranty details.",
        by_id["P04"]["expect"][0],
    )
    assert check_answer(
        "Only PLACED orders may be cancelled.",
        by_id["P05"]["expect"][0],
    )


@pytest.mark.parametrize("message", APP_REFUSAL_MESSAGES)
def test_refusal_check_accepts_application_messages(
    message: str,
    db: sqlite3.Connection,
) -> None:
    resolved = resolve_check({"kind": "refusal"}, 1, db, date.today())

    assert message.casefold() in REFUSAL_PHRASES
    assert check_answer(message, resolved)


@pytest.mark.parametrize(
    "message",
    [
        "I can't answer that.",
        "I cannot answer that.",
        "I am unable to answer that.",
        "I don't have that information.",
        "No information is available.",
        "That is not mentioned in the policy.",
        "I couldn't find that order.",
    ],
)
def test_refusal_check_accepts_generic_phrases(
    message: str,
    db: sqlite3.Connection,
) -> None:
    resolved = resolve_check({"kind": "refusal"}, 1, db, date.today())

    assert check_answer(message, resolved)


def test_a05_sql_tautology_is_blocked(cases: list[dict]) -> None:
    case = next(case for case in cases if case["id"] == "A05")

    result = check_blocked(case["question"])

    assert case["expected_blocked"] is True
    assert result is not None
    assert result.category == "sql_or_write"


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
    if kind in VALUES_CHECKS:
        assert set(check) == {"kind", "values"}
        assert isinstance(check["values"], list) and check["values"]
        assert all(
            isinstance(value, str) and value for value in check["values"]
        )
        return
    if kind == "order_count":
        assert set(check) == {"kind"}
        return
    if kind == "order_ids_listed":
        assert set(check) == {"kind", "n"}
        assert type(check["n"]) is int and check["n"] > 0
        return
    assert kind == "refusal"
    assert set(check) == {"kind"}
