"""Training article generator — synthesizes knowledge articles from documents.

Fetches source documents, builds a prompt for the LLM, and falls back to
an extractive article when no LLM is available.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from src.database import get_connection
from src.services.qa_engine import call_llm, _log_usage

logger = logging.getLogger("vaultwise.article_generator")


def _get_documents_content(doc_ids: list[str]) -> list[dict]:
    """Fetch documents by their IDs.

    Args:
        doc_ids: List of document IDs.

    Returns:
        List of dicts with id, title, content.
    """
    if not doc_ids:
        return []

    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in doc_ids)
        rows = conn.execute(
            f"SELECT id, title, content FROM documents WHERE id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _build_article_prompt(docs: list[dict]) -> str:
    """Build a prompt for LLM article generation.

    Args:
        docs: Source documents.

    Returns:
        Prompt string.
    """
    content_parts = []
    for doc in docs:
        content_parts.append(f"## {doc['title']}\n\n{doc['content']}\n")
    combined = "\n---\n".join(content_parts)

    return (
        "You are a technical writer. Based on the following source documents, "
        "create a well-structured knowledge article that synthesizes the key information. "
        "Use clear headings, bullet points where appropriate, and a summary at the end. "
        "The article should be educational and easy to follow.\n\n"
        f"Source Documents:\n\n{combined}\n\n"
        "Write the knowledge article now:"
    )


def _extractive_article(docs: list[dict]) -> tuple[str, str]:
    """Generate a fallback article from document excerpts.

    Args:
        docs: Source documents.

    Returns:
        Tuple of (title, content).
    """
    titles = [d["title"] for d in docs]
    combined_title = f"Knowledge Article: {', '.join(titles[:3])}"

    sections: list[str] = [f"# {combined_title}\n"]
    sections.append(
        "This article summarizes key information from the following source documents: "
        + ", ".join(f"**{t}**" for t in titles) + ".\n"
    )

    for doc in docs:
        sections.append(f"\n## Key Points from: {doc['title']}\n")
        paragraphs = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]
        for para in paragraphs[:3]:
            if len(para) > 500:
                para = para[:500] + "..."
            sections.append(f"{para}\n")

    sections.append("\n## Summary\n")
    sections.append(
        "This article covers essential knowledge drawn from organizational documents. "
        "Review the source materials for complete details.\n"
    )

    return combined_title, "\n".join(sections)


def generate_article(doc_ids: list[str]) -> dict:
    """Generate a knowledge article from specified documents.

    Args:
        doc_ids: List of document IDs to base the article on.

    Returns:
        Dict with article id, title, content, status, source_doc_ids, created_at.

    Raises:
        ValueError: If no documents found for the given IDs.
    """
    docs = _get_documents_content(doc_ids)
    if not docs:
        raise ValueError(f"No documents found for IDs: {doc_ids}")

    prompt = _build_article_prompt(docs)
    llm_response = call_llm(prompt)

    if llm_response:
        lines = llm_response.strip().split("\n")
        title = lines[0].lstrip("#").strip() if lines else "Knowledge Article"
        if len(title) > 200:
            title = title[:200]
        content = llm_response
    else:
        title, content = _extractive_article(docs)

    article_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO articles (id, title, content, source_doc_ids, status, auto_generated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (article_id, title, content, json.dumps(doc_ids), "draft", 1, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _log_usage("generate_article")

    return {
        "id": article_id,
        "title": title,
        "content": content,
        "status": "draft",
        "source_doc_ids": json.dumps(doc_ids),
        "auto_generated": 1,
        "created_at": now,
    }


def get_article(article_id: str) -> dict | None:
    """Get an article by ID.

    Args:
        article_id: The article ID.

    Returns:
        Article dict or None.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_articles(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all articles with pagination.

    Args:
        limit: Max results.
        offset: Pagination offset.

    Returns:
        List of article dicts.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM articles ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_article(
    article_id: str,
    status: str | None = None,
    title: str | None = None,
    content: str | None = None,
) -> dict | None:
    """Update an article's fields.

    Args:
        article_id: The article ID.
        status: New status (draft, published, archived).
        title: New title.
        content: New content.

    Returns:
        Updated article dict, or None if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if row is None:
            return None

        updates: list[str] = []
        params: list = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if content is not None:
            updates.append("content = ?")
            params.append(content)

        if updates:
            params.append(article_id)
            conn.execute(
                f"UPDATE articles SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
