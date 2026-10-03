import sqlite3

from app.config import settings


def get_write_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_readonly_connection() -> sqlite3.Connection:
    """Open the only database connection used by request-serving code."""
    database_uri = f"file:{settings.DATABASE_PATH}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection
