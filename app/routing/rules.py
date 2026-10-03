"""Deterministic request checks used before any model-based routing."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockResult:
    category: str
    message: str


_ORDER_ID_PATTERN = re.compile(
    r"(?<!\w)#\s*(?P<hash_id>\d+)\b"
    r"|\border\s*(?:(?:number|no\.?|id)\s*)?#?\s*(?P<order_id>\d+)\b",
    re.IGNORECASE,
)

_SQL_PATTERNS = (
    re.compile(r"\bdrop\s+table\b"),
    re.compile(r"\bdelete\s+from\b"),
    re.compile(r"\binsert\s+into\b"),
    re.compile(r"\bupdate\b.{0,200}\bset\b"),
    re.compile(r"\balter\s+table\b"),
    re.compile(r"\btruncate\s+(?:table\s+)?[a-z_][\w.]*\b"),
    re.compile(r"^select\b.+\bfrom\b"),
    re.compile(
        r";\s*(?:select|insert|update|delete|drop|alter|truncate|create|"
        r"replace|attach|pragma)\b"
    ),
)

_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:previous|all)\s+instructions\b"),
    re.compile(r"\bignore\s+your\s+rules\b"),
    re.compile(r"\bdisregard\s+the\s+above\b"),
    re.compile(r"\breveal\s+your\s+system\s+prompt\b"),
    re.compile(r"\bshow\s+(?:me\s+)?your\s+prompt\b"),
    re.compile(r"\byou\s+are\s+now\b"),
    re.compile(r"\bdeveloper\s+mode\b"),
    re.compile(r"\bjailbreak\b"),
)

_REQUEST_PATTERN = re.compile(
    r"\b(?:show|list|give|send|provide|display|reveal|find|fetch|see|view|"
    r"tell|lookup|look\s+up)\b"
)
_QUESTION_PATTERN = re.compile(r"\b(?:what|where)\b")
_PII_PATTERN = re.compile(
    r"\b(?:emails?|email\s+addresses?|phone(?:\s+numbers?)?|"
    r"personal\s+(?:details|information|data))\b"
)
_THIRD_PARTY_SCOPE_PATTERN = re.compile(
    r"\b(?:all|every|other|another)\s+customers?(?:['\u2019]s|['\u2019])?\b"
    r"|\bsomeone\s+else(?:['\u2019]s)?\b"
    r"|\beveryone(?:['\u2019]s)?\b"
)
_CUSTOMER_DATA_PATTERN = re.compile(
    r"\b(?:orders?|purchases?|account|data|details|emails?|email\s+addresses?|"
    r"phone(?:\s+numbers?)?|addresses?|refunds?|status)\b"
)
_ALL_PII_PATTERN = re.compile(
    r"\ball\s+(?:customer\s+|customers['\u2019]?\s+)?"
    r"(?:emails?|email\s+addresses?|phone(?:\s+numbers?)?)\b"
)
_ALL_ORDERS_PATTERN = re.compile(
    r"\b(?:list|show|give|send|provide|display|get|fetch)\s+(?:me\s+)?"
    r"all\s+(?:customer\s+|customers['\u2019]?\s+)?orders\b"
)
_CUSTOMER_ID_PATTERN = re.compile(r"\bcustomer\s+id\s*#?\s*\d+\b")
_CUSTOMER_ID_REQUEST_PATTERN = re.compile(
    r"\b(?:show|list|give|get|display|find|fetch|see|view|lookup|look\s+up)\b"
    r".{0,100}\bcustomer\s+id\s*#?\s*\d+\b"
)

_BLOCK_MESSAGES = {
    "sql_or_write": "I can't help with database commands or data changes.",
    "prompt_injection": "I can only help with Kartly orders and policies.",
    "other_customers": (
        "I can only help with your own orders and Kartly policies, so I can't "
        "do that."
    ),
    "pii_bulk": "I can't provide other customers' personal information.",
}


def extract_order_id(question: str) -> int | None:
    """Return the first explicitly referenced order number, if present."""
    match = _ORDER_ID_PATTERN.search(question)
    if match is None:
        return None
    return int(match.group("hash_id") or match.group("order_id"))


def check_blocked(question: str) -> BlockResult | None:
    """Return a deterministic block reason for unsafe requests."""
    normalized = re.sub(r"\s+", " ", question).strip().casefold()

    if any(pattern.search(normalized) for pattern in _SQL_PATTERNS):
        return _blocked("sql_or_write")

    if any(pattern.search(normalized) for pattern in _PROMPT_INJECTION_PATTERNS):
        return _blocked("prompt_injection")

    has_request = bool(_REQUEST_PATTERN.search(normalized))
    has_question = bool(_QUESTION_PATTERN.search(normalized))
    has_third_party_scope = bool(_THIRD_PARTY_SCOPE_PATTERN.search(normalized))
    has_pii = bool(_PII_PATTERN.search(normalized))

    if _ALL_PII_PATTERN.search(normalized) or (
        (has_request or has_question) and has_third_party_scope and has_pii
    ):
        return _blocked("pii_bulk")

    if _ALL_ORDERS_PATTERN.search(normalized):
        return _blocked("other_customers")

    if has_third_party_scope and (
        has_request or _CUSTOMER_DATA_PATTERN.search(normalized)
    ):
        return _blocked("other_customers")

    if _CUSTOMER_ID_PATTERN.search(normalized) and (
        _CUSTOMER_ID_REQUEST_PATTERN.search(normalized)
        or _CUSTOMER_DATA_PATTERN.search(normalized)
    ):
        return _blocked("other_customers")

    return None


def _blocked(category: str) -> BlockResult:
    return BlockResult(category=category, message=_BLOCK_MESSAGES[category])
