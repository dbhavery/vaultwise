"""Database initialization and connection management for Vaultwise.

Uses SQLite with WAL mode for concurrent reads. All table schemas are
defined here and created on first startup via init_db().
"""

import os
import sqlite3

DB_PATH = "vaultwise.db"


def _get_db_path() -> str:
    """Return the database file path from env or default."""
    return os.environ.get("VAULTWISE_DB", DB_PATH)


_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    doc_type TEXT DEFAULT 'text',
    word_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT,
    confidence REAL DEFAULT 0.0,
    helpful INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_doc_ids TEXT,
    status TEXT DEFAULT 'draft',
    auto_generated INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    article_id TEXT REFERENCES articles(id),
    title TEXT NOT NULL,
    questions_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_log (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    query TEXT,
    response_time_ms INTEGER,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection with row factory enabled."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Initialize the database, creating all tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript(_TABLES_SQL)
        conn.commit()
    finally:
        conn.close()
