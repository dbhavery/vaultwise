"""Question-answering engine with source citations.

Uses the TF-IDF search index to retrieve relevant chunks, builds a
context prompt, and calls a pluggable LLM backend to generate answers.
Falls back to extractive answers when no LLM is available.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import httpx

from src.database import get_connection

logger = logging.getLogger("vaultwise.qa")


# ---------------------------------------------------------------------------
# LLM backend interface
# ---------------------------------------------------------------------------

def _get_llm_provider() -> str:
    """Get the configured LLM provider."""
    return os.environ.get("LLM_PROVIDER", "mock")


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama for text generation.

    Args:
        prompt: The full prompt.

    Returns:
        Generated text, or None if unavailable.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "60"))

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return None


def _call_openai(prompt: str) -> Optional[str]:
    """Call OpenAI-compatible API for text generation.

    Args:
        prompt: The full prompt.

    Returns:
        Generated text, or None if unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, KeyError):
        return None


def call_llm(prompt: str) -> Optional[str]:
    """Call the configured LLM provider.

    Dispatches to the appropriate backend based on LLM_PROVIDER env var.

    Args:
        prompt: The prompt to send.

    Returns:
        Generated text, or None if the provider is 'mock' or unavailable.
    """
    provider = _get_llm_provider()
    if provider == "ollama":
        return _call_ollama(prompt)
    elif provider == "openai":
        return _call_openai(prompt)
    # Default: mock provider returns None (triggers fallback)
    return None


# ---------------------------------------------------------------------------
# Prompt building and answer generation
# ---------------------------------------------------------------------------

def build_context_prompt(query: str, chunks: list[dict]) -> str:
    """Build a RAG prompt with retrieved context for the LLM.

    Args:
        query: User's question.
        chunks: List of relevant chunk dicts from search.

    Returns:
        Formatted prompt string.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        doc_title = chunk.get("doc_title", chunk.get("metadata", {}).get("doc_title", "Unknown"))
        content = chunk.get("content", chunk.get("metadata", {}).get("content", ""))
        context_parts.append(f"[Source {i}: {doc_title}]\n{content}\n")

    context_block = "\n".join(context_parts)

    return (
        "You are a helpful knowledge assistant. "
        "Answer the question based ONLY on the provided context documents. "
        "If the context doesn't contain enough information, say so honestly. "
        "Cite sources by referring to the document titles.\n\n"
        f"## Context Documents\n\n{context_block}\n\n"
        f"## Question\n\n{query}\n\n"
        "## Answer\n\n"
    )


def extractive_answer(query: str, chunks: list[dict]) -> str:
    """Generate a fallback extractive answer from retrieved chunks.

    Used when no LLM backend is available. Summarizes the top
    retrieved chunks as the answer.

    Args:
        query: User's question.
        chunks: Retrieved chunks from search.

    Returns:
        Extractive answer string.
    """
    if not chunks:
        return (
            "I couldn't find any relevant documents to answer your question. "
            "This topic may not be covered in the current knowledge base."
        )

    parts: list[str] = ["Based on the available documents:\n\n"]
    for chunk in chunks[:3]:
        doc_title = chunk.get("doc_title", chunk.get("metadata", {}).get("doc_title", "Unknown"))
        content = chunk.get("content", chunk.get("metadata", {}).get("content", ""))
        excerpt = content[:300]
        if len(content) > 300:
            excerpt += "..."
        parts.append(f"From **{doc_title}**: {excerpt}\n\n")

    return "".join(parts).strip()


def compute_confidence(chunks: list[dict]) -> float:
    """Compute answer confidence from search result quality.

    Uses a weighted combination of top score and average score.

    Args:
        chunks: Search results with scores.

    Returns:
        Confidence between 0.0 and 1.0.
    """
    if not chunks:
        return 0.0

    scores = [c.get("score", 0.0) for c in chunks]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    confidence = 0.6 * top_score + 0.4 * avg_score
    return round(min(max(confidence, 0.0), 1.0), 3)


def ask_question(query: str, search_results: list[dict]) -> dict:
    """Answer a question using search results and LLM.

    Args:
        query: The user's question.
        search_results: Pre-fetched search results from the TF-IDF index.

    Returns:
        Dict with answer, sources, confidence, question_id.
    """
    start = time.time()

    # Prepare chunks with consistent field names
    chunks = []
    for r in search_results:
        meta = r.get("metadata", {})
        chunks.append({
            "doc_title": meta.get("doc_title", "Unknown"),
            "content": meta.get("content", ""),
            "doc_id": meta.get("doc_id", ""),
            "score": r.get("score", 0.0),
        })

    # Build prompt and attempt LLM call
    prompt = build_context_prompt(query, chunks)
    llm_answer = call_llm(prompt)

    if llm_answer is None:
        answer = extractive_answer(query, chunks)
    else:
        answer = llm_answer

    confidence = compute_confidence(chunks)
    elapsed_ms = int((time.time() - start) * 1000)

    # Build source citations
    sources: list[dict] = []
    seen_doc_ids: set[str] = set()
    for chunk in chunks[:3]:
        doc_id = chunk["doc_id"]
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            content = chunk["content"]
            excerpt = content[:200]
            if len(content) > 200:
                excerpt += "..."
            sources.append({
                "doc_id": doc_id,
                "title": chunk["doc_title"],
                "excerpt": excerpt,
            })

    # Persist the question to the database
    question_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO questions (id, query, answer, sources, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question_id, query, answer, json.dumps([s["doc_id"] for s in sources]), confidence, now),
        )
        conn.commit()
    except Exception:
        logger.warning("Failed to persist question: %s", query, exc_info=True)
        conn.rollback()
    finally:
        conn.close()

    # Log usage
    _log_usage("ask", query, elapsed_ms)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "question_id": question_id,
    }


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


def _log_usage(action: str, query: str | None = None, response_time_ms: int | None = None) -> None:
    """Log a usage event to the database."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO usage_log (id, action, query, response_time_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, action, query, response_time_ms, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        logger.debug("Failed to log usage event", exc_info=True)
        conn.rollback()
    finally:
        conn.close()
