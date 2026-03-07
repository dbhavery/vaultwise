"""Vaultwise main application: FastAPI server with all routes."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vaultwise import __version__
from vaultwise.analytics import get_knowledge_gaps, get_overview, get_usage_stats, update_gap_status
from vaultwise.database import get_connection, init_db, is_db_empty
from vaultwise.ingest import delete_document, get_document, ingest_document, list_documents
from vaultwise.models import (
    ArticleGenerateRequest,
    ArticleUpdateRequest,
    AskRequest,
    DocumentCreate,
    QuizGenerateRequest,
    SearchRequest,
)
from vaultwise.qa import ask_question, list_questions
from vaultwise.search import build_index
from vaultwise.seed import run_seed
from vaultwise.training import (
    generate_article,
    generate_quiz,
    get_article,
    get_quiz,
    list_articles,
    list_quizzes,
    update_article,
)

logger = logging.getLogger("vaultwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database, seed data, build search index."""
    logger.info("Starting Vaultwise v%s", __version__)
    init_db()
    if is_db_empty():
        logger.info("Database is empty, running seed data...")
        run_seed()
        logger.info("Seed data loaded.")
    build_index()
    logger.info("Search index built.")
    yield
    logger.info("Vaultwise shutting down.")


app = FastAPI(
    title="Vaultwise",
    version=__version__,
    description="Knowledge Management Platform",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for dashboard
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint with system stats."""
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


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.get("/api/documents")
def api_list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List documents with pagination."""
    docs, total = list_documents(limit=limit, offset=offset)
    return {"documents": docs, "total": total}


@app.post("/api/documents")
async def api_create_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
):
    """Upload a document via multipart form (file or title+content)."""
    if file is not None:
        file_content = (await file.read()).decode("utf-8", errors="replace")
        file_title = title or file.filename or "Untitled"
        result = ingest_document(
            title=file_title,
            content=file_content,
            source=source or "upload",
            doc_type=doc_type,
        )
    elif title and content:
        result = ingest_document(
            title=title,
            content=content,
            source=source or "upload",
            doc_type=doc_type,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or title+content")

    # Rebuild search index after ingestion
    build_index()
    return result


@app.post("/api/documents/json")
def api_create_document_json(doc: DocumentCreate):
    """Upload a document via JSON body."""
    result = ingest_document(
        title=doc.title,
        content=doc.content,
        source=doc.source,
        doc_type=doc.doc_type,
    )
    build_index()
    return result


@app.get("/api/documents/{doc_id}")
def api_get_document(doc_id: str):
    """Get a document with its chunks."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/api/documents/{doc_id}")
def api_delete_document(doc_id: str):
    """Delete a document and its chunks."""
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    build_index()
    return {"deleted": True, "id": doc_id}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/api/search")
def api_search(req: SearchRequest):
    """Search for relevant chunks using TF-IDF similarity."""
    from vaultwise.search import search_chunks
    results = search_chunks(req.query, limit=req.limit)
    return {"results": results, "query": req.query}


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------

@app.post("/api/ask")
def api_ask(req: AskRequest):
    """Ask a question and get an AI-generated answer with sources."""
    result = ask_question(req.query)
    return result


@app.get("/api/questions")
def api_list_questions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List recent questions."""
    return list_questions(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Training: Articles
# ---------------------------------------------------------------------------

@app.get("/api/articles")
def api_list_articles(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all articles."""
    return list_articles(limit=limit, offset=offset)


@app.post("/api/articles/generate")
def api_generate_article(req: ArticleGenerateRequest):
    """Generate a knowledge article from specified documents."""
    try:
        result = generate_article(req.doc_ids)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/articles/{article_id}")
def api_get_article(article_id: str):
    """Get an article by ID."""
    article = get_article(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.patch("/api/articles/{article_id}")
def api_update_article(article_id: str, req: ArticleUpdateRequest):
    """Update an article's status, title, or content."""
    result = update_article(article_id, status=req.status, title=req.title, content=req.content)
    if result is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return result


# ---------------------------------------------------------------------------
# Training: Quizzes
# ---------------------------------------------------------------------------

@app.get("/api/quizzes")
def api_list_quizzes(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all quizzes."""
    return list_quizzes(limit=limit, offset=offset)


@app.post("/api/quizzes/generate")
def api_generate_quiz(req: QuizGenerateRequest):
    """Generate a quiz from an article."""
    try:
        result = generate_quiz(req.article_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/quizzes/{quiz_id}")
def api_get_quiz(quiz_id: str):
    """Get a quiz by ID."""
    quiz = get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return quiz


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/api/analytics/overview")
def api_analytics_overview():
    """Get overview statistics."""
    return get_overview()


@app.get("/api/analytics/gaps")
def api_analytics_gaps(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List knowledge gaps sorted by frequency."""
    return get_knowledge_gaps(limit=limit, offset=offset)


@app.patch("/api/analytics/gaps/{gap_id}")
def api_update_gap(gap_id: str, status: str = Query(...)):
    """Update a knowledge gap's status."""
    try:
        updated = update_gap_status(gap_id, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not updated:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    return {"updated": True, "id": gap_id, "status": status}


@app.get("/api/analytics/usage")
def api_analytics_usage(days: int = Query(default=7, ge=1, le=90)):
    """Get usage statistics."""
    return get_usage_stats(days=days)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Vaultwise server."""
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")


if __name__ == "__main__":
    main()
