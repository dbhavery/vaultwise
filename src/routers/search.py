"""TF-IDF search router."""

from fastapi import APIRouter

from src.models import SearchRequest

router = APIRouter(prefix="/api/search", tags=["search"])

# The search index is injected via app state at startup.
# Each endpoint accesses it through the module-level reference.
_search_index = None


def set_search_index(index: object) -> None:
    """Set the global search index reference (called from main at startup).

    Args:
        index: A TFIDFIndex instance.
    """
    global _search_index
    _search_index = index


def get_search_index() -> object:
    """Get the current search index."""
    return _search_index


@router.post("")
def api_search(req: SearchRequest) -> dict:
    """Search for relevant chunks using TF-IDF cosine similarity.

    Tokenizes the query, builds a TF-IDF vector, and scores all indexed
    chunks using cosine similarity. Returns results sorted by relevance.
    """
    if _search_index is None:
        return {"results": [], "query": req.query}

    raw_results = _search_index.search(req.query, limit=req.limit)

    results = []
    for r in raw_results:
        meta = r.get("metadata", {})
        results.append({
            "chunk_id": r["id"],
            "content": meta.get("content", ""),
            "score": r["score"],
            "doc_title": meta.get("doc_title", ""),
            "doc_id": meta.get("doc_id", ""),
        })

    return {"results": results, "query": req.query}


@router.get("/stats")
def api_search_stats() -> dict:
    """Get statistics about the search index."""
    if _search_index is None:
        return {"document_count": 0, "vocabulary_size": 0, "built": False}

    return {
        "document_count": _search_index.document_count,
        "vocabulary_size": _search_index.vocabulary_size,
        "built": _search_index._built,
    }
