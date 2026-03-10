"""Vaultwise main application — FastAPI server with all routes.

Initializes the database, builds the TF-IDF search index from all
stored document chunks, and mounts all API routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.database import get_connection, init_db
from src.routers import documents, qa, quizzes, search, training
from src.services.tfidf_engine import TFIDFIndex

logger = logging.getLogger("vaultwise")

# Global search index shared across the application
_search_index = TFIDFIndex()


def rebuild_search_index() -> None:
    """Rebuild the TF-IDF search index from all chunks in the database."""
    global _search_index
    _search_index.clear()

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT c.id, c.content, c.doc_id, d.title as doc_title "
            "FROM chunks c JOIN documents d ON c.doc_id = d.id "
            "ORDER BY c.doc_id, c.chunk_index"
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        _search_index.add_document(
            doc_id=row["id"],
            text=row["content"],
            metadata={
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "content": row["content"],
            },
        )

    _search_index.build()
    logger.info(
        "Search index built: %d documents, %d vocabulary terms",
        _search_index.document_count,
        _search_index.vocabulary_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database and build search index."""
    logger.info("Starting Vaultwise v%s", __version__)
    init_db()
    rebuild_search_index()

    # Inject search index into routers that need it
    search.set_search_index(_search_index)
    qa.set_search_index(_search_index)

    yield
    logger.info("Vaultwise shutting down.")


app = FastAPI(
    title="Vaultwise",
    version=__version__,
    description="Knowledge Management Platform — document ingestion, TF-IDF search, AI Q&A, training articles and quizzes",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(qa.router)
app.include_router(training.router)
app.include_router(quizzes.router)


# ---------------------------------------------------------------------------
# Health and utility endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Health check with system stats."""
    conn = get_connection()
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    return {
        "status": "ok",
        "version": __version__,
        "documents": doc_count,
        "chunks": chunk_count,
    }


@app.post("/api/reindex")
def api_reindex() -> dict:
    """Force rebuild of the TF-IDF search index.

    Useful after bulk ingestion or manual database changes.
    """
    rebuild_search_index()
    search.set_search_index(_search_index)
    qa.set_search_index(_search_index)
    return {
        "reindexed": True,
        "document_count": _search_index.document_count,
        "vocabulary_size": _search_index.vocabulary_size,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Vaultwise server."""
    import os
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    host = os.environ.get("VAULTWISE_HOST", "0.0.0.0")
    port = int(os.environ.get("VAULTWISE_PORT", "8090"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
