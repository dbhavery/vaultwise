"""Shared test fixtures for Vaultwise tests."""

import os
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test_vaultwise.db")
    os.environ["VAULTWISE_DB"] = db_path
    # Reset search module global state
    from vaultwise import search
    search._vocabulary = {}
    search._idf = None
    search._chunk_vectors = {}
    search._chunk_metadata = {}
    yield
    os.environ.pop("VAULTWISE_DB", None)


def _make_test_app(run_seed: bool = False):
    """Create a FastAPI app instance for testing.

    Args:
        run_seed: If True, run the seed data on startup.

    Returns:
        FastAPI app with test-appropriate lifespan.
    """
    from vaultwise.database import init_db, is_db_empty
    from vaultwise.search import build_index
    from vaultwise.seed import run_seed as do_seed
    from vaultwise.main import app

    # Replace the lifespan with a test-specific one
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def test_lifespan(a):
        init_db()
        if run_seed and is_db_empty():
            do_seed()
        build_index()
        yield

    app.router.lifespan_context = test_lifespan
    return app, original_lifespan


@pytest.fixture
def client():
    """Create a FastAPI TestClient with initialized (empty) database."""
    app, original_lifespan = _make_test_app(run_seed=False)
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan


@pytest.fixture
def seeded_client():
    """Create a TestClient with seed data loaded."""
    app, original_lifespan = _make_test_app(run_seed=True)
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan


@pytest.fixture
def sample_doc():
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
