"""Shared test fixtures for the Vaultwise test suite."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path):
    """Use a temporary database for each test, ensuring isolation."""
    db_path = str(tmp_path / "test_vaultwise.db")
    os.environ["VAULTWISE_DB"] = db_path
    os.environ["LLM_PROVIDER"] = "mock"
    yield
    os.environ.pop("VAULTWISE_DB", None)
    os.environ.pop("LLM_PROVIDER", None)


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI TestClient with initialized (empty) database."""
    from src.database import init_db
    from src.main import app, rebuild_search_index, _search_index
    from src.routers import search as search_router, qa as qa_router

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def test_lifespan(a):
        init_db()
        rebuild_search_index()
        search_router.set_search_index(_search_index)
        qa_router.set_search_index(_search_index)
        yield

    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan


@pytest.fixture
def seeded_client(tmp_path):
    """Create a FastAPI TestClient with pre-populated seed data.

    Seeds the database with documents, articles, quizzes, questions,
    and usage log entries so tests can verify list/read endpoints
    against realistic data.
    """
    from src.database import init_db, get_connection
    from src.main import app, rebuild_search_index, _search_index
    from src.routers import search as search_router, qa as qa_router

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def test_lifespan(a):
        init_db()

        conn = get_connection()
        try:
            now = datetime.now(timezone.utc)
            doc_ids = []

            # --- Seed 5 documents with chunks ---
            seed_docs = [
                ("Employee Handbook 2026", "Employee handbook covering PTO, benefits, and company policies in detail."),
                ("REST API Documentation v3.0", "API documentation with authentication, rate limiting, and endpoint specs."),
                ("New Employee Onboarding Guide", "Onboarding guide covering first week orientation and required training."),
                ("Information Security Policy", "Security policy for data classification, access control, and incident response."),
                ("System Architecture Overview", "Architecture overview of microservices, infrastructure, and monitoring."),
            ]
            for i, (title, content) in enumerate(seed_docs):
                # Expand content to be substantial enough for chunking
                expanded = " ".join([content] * 20)
                doc_id = uuid4().hex
                doc_ids.append(doc_id)
                created = (now - timedelta(days=30 - i * 5)).isoformat()
                word_count = len(expanded.split())
                conn.execute(
                    "INSERT INTO documents (id, title, source, content, doc_type, word_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, title, "upload", expanded, "text", word_count, created, created),
                )
                # Create a chunk for each doc
                chunk_id = uuid4().hex
                conn.execute(
                    "INSERT INTO chunks (id, doc_id, content, chunk_index, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, expanded, 0, created),
                )

            # --- Seed 10 questions ---
            for i in range(10):
                q_id = uuid4().hex
                created = (now - timedelta(days=i % 7, hours=i)).isoformat()
                conn.execute(
                    "INSERT INTO questions (id, query, answer, sources, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (q_id, f"Sample question {i}?", f"Sample answer {i}.",
                     json.dumps([doc_ids[i % len(doc_ids)]]), 0.8 + i * 0.01, created),
                )

            # --- Seed 3 articles ---
            article_ids = []
            for i in range(3):
                art_id = uuid4().hex
                article_ids.append(art_id)
                created = (now - timedelta(days=20 - i * 7)).isoformat()
                conn.execute(
                    "INSERT INTO articles (id, title, content, source_doc_ids, status, auto_generated, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (art_id, f"Knowledge Article {i}", f"Article content about topic {i}. " * 30,
                     json.dumps([doc_ids[i % len(doc_ids)]]), "published", 1, created),
                )

            # --- Seed 2 quizzes ---
            for i in range(2):
                quiz_id = uuid4().hex
                questions_data = [
                    {"question": f"Quiz question {j}?", "options": ["A", "B", "C", "D"],
                     "correct_index": 0, "explanation": f"Explanation {j}"}
                    for j in range(4)
                ]
                created = (now - timedelta(days=15 - i * 5)).isoformat()
                conn.execute(
                    "INSERT INTO quizzes (id, article_id, title, questions_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (quiz_id, article_ids[i], f"Quiz: Article {i}",
                     json.dumps(questions_data), created),
                )

            # --- Seed 20 usage log entries ---
            actions = ["search", "ask", "generate_article", "generate_quiz"]
            for i in range(20):
                log_id = uuid4().hex
                created = (now - timedelta(days=i % 7, hours=i, minutes=i * 3)).isoformat()
                conn.execute(
                    "INSERT INTO usage_log (id, action, query, response_time_ms, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (log_id, actions[i % len(actions)], f"query {i}", 100 + i * 10, created),
                )

            conn.commit()
        finally:
            conn.close()

        rebuild_search_index()
        search_router.set_search_index(_search_index)
        qa_router.set_search_index(_search_index)
        yield

    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan


@pytest.fixture
def sample_doc() -> dict:
    """Return a sample document dict for testing."""
    return {
        "title": "Test Document: Python Best Practices",
        "content": (
            "Python best practices include using type hints for all function signatures. "
            "Virtual environments should be used for dependency isolation. "
            "Code should follow PEP 8 style guidelines consistently. "
            "Testing is essential and pytest is the recommended framework. "
            "Documentation should be maintained alongside the code. "
            "Error handling should be explicit and never silently swallow exceptions. "
            "Logging should provide context about what operation was being performed. "
            "Configuration values should be externalized, not hardcoded in the source. "
            "Dependencies should be pinned to exact versions for reproducibility. "
            "Code reviews help catch issues early and share knowledge across the team."
        ),
        "doc_type": "text",
        "source": "upload",
    }


@pytest.fixture
def long_doc() -> dict:
    """Return a document long enough to produce multiple chunks."""
    paragraphs = []
    topics = [
        "database management", "API design", "authentication security",
        "deployment automation", "monitoring observability", "testing strategies",
        "documentation practices", "code review processes",
    ]
    for topic in topics:
        para = " ".join(
            [f"The {topic} domain covers many important concepts that professionals must understand."] * 20
        )
        paragraphs.append(para)

    return {
        "title": "Comprehensive Engineering Guide",
        "content": "\n\n".join(paragraphs),
        "doc_type": "text",
        "source": "upload",
    }


@pytest.fixture
def sample_markdown_doc() -> dict:
    """Return a sample markdown document."""
    return {
        "title": "guide.md",
        "content": (
            "# Python Guide\n\n"
            "## Getting Started\n\n"
            "Python is a versatile programming language used in web development, "
            "data science, automation, and artificial intelligence.\n\n"
            "## Best Practices\n\n"
            "- Use **type hints** on all function signatures\n"
            "- Write **docstrings** for all public functions\n"
            "- Follow **PEP 8** style guidelines\n\n"
            "## Testing\n\n"
            "Use `pytest` as the primary testing framework. "
            "Write unit tests for all business logic.\n"
        ),
        "doc_type": "markdown",
        "source": "upload",
    }
