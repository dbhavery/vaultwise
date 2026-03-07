"""Document ingestion: upload, text extraction, chunking, and embedding storage."""

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from vaultwise.database import get_connection
from vaultwise.search import compute_embedding


CHUNK_SIZE = 500  # target words per chunk
CHUNK_OVERLAP = 50  # overlap words between chunks


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count.

    Args:
        text: The full document text.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def _detect_doc_type(title: str, content: str) -> str:
    """Detect document type from title extension or content heuristics."""
    title_lower = title.lower()
    if title_lower.endswith(".md"):
        return "markdown"
    if title_lower.endswith(".py"):
        return "python"
    if title_lower.endswith(".pdf"):
        return "pdf"
    # Check content for markdown patterns
    if re.search(r"^#{1,6}\s", content, re.MULTILINE):
        return "markdown"
    if re.search(r"^(def |class |import |from )", content, re.MULTILINE):
        return "python"
    return "text"


def ingest_document(
    title: str,
    content: str,
    source: str = "upload",
    doc_type: str | None = None,
) -> dict:
    """Ingest a document: store it, chunk it, compute embeddings.

    Args:
        title: Document title.
        content: Full document text content.
        source: Source of the document (upload, url, api).
        doc_type: Document type. Auto-detected if None.

    Returns:
        Dict with document id, title, chunk_count, and word_count.
    """
    if doc_type is None:
        doc_type = _detect_doc_type(title, content)

    doc_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    word_count = _count_words(content)

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO documents (id, title, source, content, doc_type, word_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, title, source, content, doc_type, word_count, now, now),
        )

        chunks = _chunk_text(content)
        for i, chunk_content in enumerate(chunks):
            chunk_id = uuid4().hex
            embedding = compute_embedding(chunk_content)
            embedding_json = json.dumps(embedding) if embedding is not None else None
            conn.execute(
                "INSERT INTO chunks (id, doc_id, content, chunk_index, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk_content, i, embedding_json, now),
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
        Dict with document fields and a chunks list, or None if not found.
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
        offset: Offset for pagination.

    Returns:
        Tuple of (list of document dicts, total count).
    """
    conn = get_connection()
    try:
        total_row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        total = total_row[0]

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
        True if the document existed and was deleted.
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
