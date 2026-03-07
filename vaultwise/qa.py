"""Question-answering module: retrieve context, generate answers, track gaps."""

import json
import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger("vaultwise.qa")

import httpx

from vaultwise.database import get_connection
from vaultwise.search import search_chunks, _log_usage

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 60.0  # seconds


def _build_context_prompt(query: str, chunks: list[dict]) -> str:
    """Build a prompt with retrieved context for the LLM.

    Args:
        query: User's question.
        chunks: List of relevant chunk dicts from search.

    Returns:
        Formatted prompt string.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['doc_title']}]\n{chunk['content']}\n"
        )

    context_block = "\n".join(context_parts)

    return (
        "You are a helpful knowledge assistant for an organization. "
        "Answer the question based ONLY on the provided context documents. "
        "If the context doesn't contain enough information to answer fully, say so honestly. "
        "Cite sources by referring to the document titles.\n\n"
        f"## Context Documents\n\n{context_block}\n\n"
        f"## Question\n\n{query}\n\n"
        "## Answer\n\n"
    )


def _call_ollama(prompt: str) -> str | None:
    """Call the Ollama API to generate a response.

    Args:
        prompt: The full prompt to send.

    Returns:
        Generated text, or None if Ollama is unavailable.
    """
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return None


def _mock_answer(query: str, chunks: list[dict]) -> str:
    """Generate a mock answer when Ollama is unavailable.

    Summarizes the top retrieved chunks as the answer.

    Args:
        query: User's question.
        chunks: Retrieved chunks.

    Returns:
        Mock answer string.
    """
    if not chunks:
        return (
            "I couldn't find any relevant documents to answer your question. "
            "This topic may not be covered in the current knowledge base."
        )

    parts: list[str] = []
    parts.append("Based on the retrieved documents:\n\n")
    for chunk in chunks[:3]:
        excerpt = chunk["content"][:300]
        if len(chunk["content"]) > 300:
            excerpt += "..."
        parts.append(f"From **{chunk['doc_title']}**: {excerpt}\n\n")

    return "".join(parts).strip()


def _compute_confidence(chunks: list[dict]) -> float:
    """Compute a confidence score based on search result quality.

    Args:
        chunks: Search results with scores.

    Returns:
        Confidence between 0.0 and 1.0.
    """
    if not chunks:
        return 0.0

    top_score = chunks[0]["score"]
    avg_score = sum(c["score"] for c in chunks) / len(chunks)
    # Weighted combination: 60% top score, 40% average
    confidence = 0.6 * top_score + 0.4 * avg_score
    return round(min(max(confidence, 0.0), 1.0), 3)


def _detect_knowledge_gap(query: str, confidence: float) -> None:
    """If confidence is low, record or update a knowledge gap.

    Args:
        query: The user's query.
        confidence: The computed confidence score.
    """
    if confidence >= 0.3:
        return

    # Extract a topic from the query (first 5 significant words)
    words = query.strip().split()
    topic = " ".join(words[:6]).lower().strip("?.,!")

    conn = get_connection()
    try:
        # Check if a similar gap already exists
        existing = conn.execute(
            "SELECT id, frequency, sample_queries FROM knowledge_gaps WHERE topic = ?",
            (topic,),
        ).fetchone()

        now = datetime.now(timezone.utc).isoformat()

        if existing:
            freq = existing["frequency"] + 1
            samples = json.loads(existing["sample_queries"] or "[]")
            if query not in samples:
                samples.append(query)
                if len(samples) > 10:
                    samples = samples[-10:]
            conn.execute(
                "UPDATE knowledge_gaps SET frequency = ?, sample_queries = ?, last_asked = ? WHERE id = ?",
                (freq, json.dumps(samples), now, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO knowledge_gaps (id, topic, frequency, sample_queries, status, last_asked, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid4().hex, topic, 1, json.dumps([query]), "open", now, now),
            )
        conn.commit()
    except Exception:
        logger.warning("Failed to record knowledge gap for topic=%r", topic, exc_info=True)
        conn.rollback()
    finally:
        conn.close()


def ask_question(query: str) -> dict:
    """Answer a question using retrieved context and LLM.

    Args:
        query: The user's question.

    Returns:
        Dict with answer, sources, confidence, and question_id.
    """
    start = time.time()

    # Retrieve relevant chunks
    chunks = search_chunks(query, limit=5)

    # Build prompt and get answer
    prompt = _build_context_prompt(query, chunks)
    llm_answer = _call_ollama(prompt)

    if llm_answer is None:
        answer = _mock_answer(query, chunks)
    else:
        answer = llm_answer

    confidence = _compute_confidence(chunks)
    elapsed_ms = int((time.time() - start) * 1000)

    # Build source references
    sources: list[dict] = []
    seen_doc_ids: set[str] = set()
    for chunk in chunks[:3]:
        if chunk["doc_id"] not in seen_doc_ids:
            seen_doc_ids.add(chunk["doc_id"])
            excerpt = chunk["content"][:200]
            if len(chunk["content"]) > 200:
                excerpt += "..."
            sources.append({
                "doc_id": chunk["doc_id"],
                "title": chunk["doc_title"],
                "excerpt": excerpt,
            })

    # Store the question in the database
    question_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    source_ids = [s["doc_id"] for s in sources]

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO questions (id, query, answer, sources, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question_id, query, answer, json.dumps(source_ids), confidence, now),
        )
        conn.commit()
    finally:
        conn.close()

    # Log usage
    _log_usage("ask", query, elapsed_ms)

    # Detect knowledge gaps
    _detect_knowledge_gap(query, confidence)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "question_id": question_id,
    }


def rate_question(question_id: str, helpful: bool) -> bool:
    """Rate a question as helpful or not.

    Args:
        question_id: The question ID.
        helpful: True if the answer was helpful.

    Returns:
        True if the question existed and was updated.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM questions WHERE id = ?", (question_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE questions SET helpful = ? WHERE id = ?",
            (1 if helpful else 0, question_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def list_questions(limit: int = 50, offset: int = 0) -> list[dict]:
    """List recent questions.

    Args:
        limit: Max results.
        offset: Pagination offset.

    Returns:
        List of question dicts.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM questions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
