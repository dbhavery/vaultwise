"""Training article generation router."""

from fastapi import APIRouter, HTTPException, Query

from src.models import ArticleGenerateRequest, ArticleUpdateRequest
from src.services.article_generator import (
    generate_article,
    get_article,
    list_articles,
    update_article,
)

router = APIRouter(prefix="/api/articles", tags=["training"])


@router.get("")
def api_list_articles(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """List all training articles."""
    return list_articles(limit=limit, offset=offset)


@router.post("/generate")
def api_generate_article(req: ArticleGenerateRequest) -> dict:
    """Generate a training article from specified documents.

    Takes a list of document IDs and synthesizes a comprehensive
    knowledge article covering the key points from each source.
    """
    try:
        return generate_article(req.doc_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{article_id}")
def api_get_article(article_id: str) -> dict:
    """Get an article by ID."""
    article = get_article(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.patch("/{article_id}")
def api_update_article(article_id: str, req: ArticleUpdateRequest) -> dict:
    """Update an article's status, title, or content."""
    result = update_article(article_id, status=req.status, title=req.title, content=req.content)
    if result is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return result
