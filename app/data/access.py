import re
import sqlite3
from typing import Any

from app.data.errors import AccessError
from app.data.queries import QueryResult, TOOLS
from app.db.connection import get_readonly_connection


CUSTOMER_EXISTS_SQL = "SELECT 1 FROM customers WHERE id = ?"
FORBIDDEN_SQL_WORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "pragma",
    "replace",
)
FORBIDDEN_SQL_PATTERN = re.compile(
    rf"\b(?:{'|'.join(FORBIDDEN_SQL_WORDS)})\b",
    flags=re.IGNORECASE,
)


def assert_safe_sql(sql: str) -> None:
    if not isinstance(sql, str) or not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise AccessError("Only SELECT statements are allowed")
    if ";" in sql:
        raise AccessError("SQL must not contain a semicolon")
    if FORBIDDEN_SQL_PATTERN.search(sql):
        raise AccessError("SQL contains a forbidden keyword")


def _validate_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = TOOLS[tool_name]
    schemas = {argument.name: argument for argument in spec.arguments}
    unknown_args = set(args) - set(schemas)
    if unknown_args:
        names = ", ".join(sorted(str(name) for name in unknown_args))
        raise AccessError(f"Unexpected argument(s): {names}")

    validated: dict[str, Any] = {}
    for name, schema in schemas.items():
        if name not in args:
            if schema.required:
                raise AccessError(f"Missing required argument: {name}")
            continue

        value = args[name]
        if value is None and not schema.required:
            validated[name] = value
            continue
        if type(value) is not schema.type:
            raise AccessError(f"Invalid type for argument: {name}")
        if schema.allowed_values is not None and value not in schema.allowed_values:
            raise AccessError(f"Invalid value for argument: {name}")
        validated[name] = value

    if tool_name == "get_order" and validated["order_id"] <= 0:
        raise AccessError("order_id must be a positive integer")
    if tool_name == "list_orders":
        limit = validated.get("limit", 10)
        if not 1 <= limit <= 20:
            raise AccessError("limit must be between 1 and 20")

    return validated


def run_tool(
    tool_name: str,
    args: dict[str, Any],
    customer_id: int,
    conn: sqlite3.Connection | None = None,
) -> QueryResult:
    if tool_name not in TOOLS:
        raise AccessError(f"Unknown tool: {tool_name}")
    if not isinstance(args, dict):
        raise AccessError("Tool arguments must be a dictionary")
    if type(customer_id) is not int or customer_id <= 0:
        raise AccessError("Invalid customer_id")

    validated_args = _validate_args(tool_name, args)
    owns_connection = conn is None
    connection = conn if conn is not None else get_readonly_connection()

    try:
        assert_safe_sql(CUSTOMER_EXISTS_SQL)
        customer = connection.execute(CUSTOMER_EXISTS_SQL, (customer_id,)).fetchone()
        if customer is None:
            raise AccessError("Customer does not exist")

        query = TOOLS[tool_name].function(customer_id, **validated_args)
        assert_safe_sql(query.sql)
        cursor = connection.execute(query.sql, query.params)
        column_names = [column[0] for column in cursor.description]
        rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        return QueryResult(
            tool=query.tool,
            sql=query.sql,
            params=query.params,
            rows=rows,
        )
    finally:
        if owns_connection:
            connection.close()
