"""Document ingestion — parsing, chunking, and storage.

Supports plain text and markdown documents. Content is split into
overlapping chunks for granular search indexing.
"""

import json
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from src.database import get_connection


def _get_chunk_size() -> int:
    """Return chunk size from env or default."""
    return int(os.environ.get("CHUNK_SIZE", "500"))


def _get_chunk_overlap() -> int:
    """Return chunk overlap from env or default."""
    return int(os.environ.get("CHUNK_OVERLAP", "50"))


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Input text.

    Returns:
        Word count.
    """
    return len(text.split())


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks by word count.

    Args:
        text: Full document text.
        chunk_size: Target words per chunk (default from env/500).
        overlap: Overlap words between chunks (default from env/50).

    Returns:
        List of chunk strings.
    """
    if chunk_size is None:
        chunk_size = _get_chunk_size()
    if overlap is None:
        overlap = _get_chunk_overlap()

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()
        if chunk_text_str:
            chunks.append(chunk_text_str)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def detect_doc_type(title: str, content: str) -> str:
    """Detect document type from title extension or content heuristics.

    Args:
        title: Document title or filename.
        content: Document body text.

    Returns:
        Detected type string: 'markdown', 'python', or 'text'.
    """
    title_lower = title.lower()
    if title_lower.endswith(".md"):
        return "markdown"
    if title_lower.endswith(".py"):
        return "python"
    # Check content for markdown heading patterns
    if re.search(r"^#{1,6}\s", content, re.MULTILINE):
        return "markdown"
    if re.search(r"^(def |class |import |from )", content, re.MULTILINE):
        return "python"
    return "text"


def strip_markdown(text: str) -> str:
    """Strip markdown formatting to produce plain text for indexing.

    Removes headings markers, bold/italic markers, links, and code fences
    while preserving the actual text content.

    Args:
        text: Markdown-formatted text.

    Returns:
        Plain text version.
    """
    # Remove code fences
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def ingest_document(
    title: str,
    content: str,
    source: str = "upload",
    doc_type: str | None = None,
) -> dict:
    """Ingest a document: store it, chunk it, save chunks.

    Args:
        title: Document title.
        content: Full document text.
        source: Origin of the document (upload, url, api).
        doc_type: Document type. Auto-detected if None.

    Returns:
        Dict with id, title, chunk_count, word_count.
    """
    if doc_type is None:
        doc_type = detect_doc_type(title, content)

    doc_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    word_count = count_words(content)

    # For markdown, strip formatting before chunking for better search
    indexable_content = strip_markdown(content) if doc_type == "markdown" else content

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO documents (id, title, source, content, doc_type, word_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, title, source, content, doc_type, word_count, now, now),
        )

        chunks = chunk_text(indexable_content)
        for i, chunk_content in enumerate(chunks):
            chunk_id = uuid4().hex
            conn.execute(
                "INSERT INTO chunks (id, doc_id, content, chunk_index, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk_content, i, now),
            )

        conn.commit()
        return {
            "id": doc_id,
            "title": title,
            "chunk_count": len(chunks),
            "word_count": word_count,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_document(doc_id: str) -> dict | None:
    """Get a document by ID, including its chunks.

    Args:
        doc_id: The document ID.

    Returns:
        Dict with document fields and chunks list, or None.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None

        doc = dict(row)
        chunk_rows = conn.execute(
            "SELECT id, content, chunk_index FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        doc["chunks"] = [dict(c) for c in chunk_rows]
        return doc
    finally:
        conn.close()


def list_documents(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """List documents with pagination.

    Args:
        limit: Max documents to return.
        offset: Pagination offset.

    Returns:
        Tuple of (document list, total count).
    """
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks.

    Args:
        doc_id: The document ID.

    Returns:
        True if document existed and was deleted.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
