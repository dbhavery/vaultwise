"""Question-answering router with source citations."""

from fastapi import APIRouter, HTTPException, Query

from src.models import AskRequest
from src.services.qa_engine import ask_question, list_questions

router = APIRouter(prefix="/api", tags=["qa"])

# The search index is injected from main.py
_search_index = None


def set_search_index(index: object) -> None:
    """Set the search index reference for Q&A context retrieval.

    Args:
        index: A TFIDFIndex instance.
    """
    global _search_index
    _search_index = index


@router.post("/ask")
def api_ask(req: AskRequest) -> dict:
    """Ask a question and get an AI-generated answer with source citations.

    Retrieves relevant chunks via TF-IDF search, passes them as context
    to the configured LLM provider, and returns the answer with sources.
    """
    search_results = []
    if _search_index is not None:
        search_results = _search_index.search(req.query, limit=5)

    result = ask_question(req.query, search_results)
    return result


@router.get("/questions")
def api_list_questions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """List recent questions with answers."""
    return list_questions(limit=limit, offset=offset)
