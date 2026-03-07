"""Training material generation: articles and quizzes from documents."""

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from vaultwise.database import get_connection
from vaultwise.search import _log_usage

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 90.0


def _call_ollama(prompt: str) -> str | None:
    """Call Ollama for text generation. Returns None if unavailable."""
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


def _get_documents_content(doc_ids: list[str]) -> list[dict]:
    """Fetch documents by their IDs.

    Args:
        doc_ids: List of document IDs.

    Returns:
        List of dicts with id, title, content for found documents.
    """
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


def _mock_article(docs: list[dict]) -> tuple[str, str]:
    """Generate a mock article from document excerpts when LLM is unavailable.

    Args:
        docs: List of document dicts with title and content.

    Returns:
        Tuple of (title, content).
    """
    titles = [d["title"] for d in docs]
    combined_title = f"Knowledge Article: {', '.join(titles[:3])}"

    sections: list[str] = []
    sections.append(f"# {combined_title}\n")
    sections.append(
        "This article summarizes key information from the following source documents: "
        + ", ".join(f"**{t}**" for t in titles)
        + ".\n"
    )

    for doc in docs:
        sections.append(f"\n## Key Points from: {doc['title']}\n")
        # Extract first few meaningful paragraphs
        paragraphs = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]
        for para in paragraphs[:3]:
            # Trim to reasonable length
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
    """Generate a knowledge article from the specified documents.

    Args:
        doc_ids: List of document IDs to base the article on.

    Returns:
        Dict with article id, title, content, status.

    Raises:
        ValueError: If no documents found for the given IDs.
    """
    docs = _get_documents_content(doc_ids)
    if not docs:
        raise ValueError(f"No documents found for IDs: {doc_ids}")

    # Build LLM prompt
    content_parts = []
    for doc in docs:
        content_parts.append(f"## {doc['title']}\n\n{doc['content']}\n")
    combined_content = "\n---\n".join(content_parts)

    prompt = (
        "You are a technical writer. Based on the following source documents, "
        "create a well-structured knowledge article that synthesizes the key information. "
        "Use clear headings, bullet points where appropriate, and a summary at the end. "
        "The article should be educational and easy to follow.\n\n"
        f"Source Documents:\n\n{combined_content}\n\n"
        "Write the knowledge article now:"
    )

    llm_response = _call_ollama(prompt)
    if llm_response:
        # Try to extract a title from the LLM response
        lines = llm_response.strip().split("\n")
        title = lines[0].lstrip("#").strip() if lines else "Knowledge Article"
        if len(title) > 200:
            title = title[:200]
        content = llm_response
    else:
        title, content = _mock_article(docs)

    # Store the article
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


def _mock_quiz(article_title: str, article_content: str) -> list[dict]:
    """Generate mock quiz questions from article content.

    Extracts key sentences and creates comprehension questions.

    Args:
        article_title: Title of the source article.
        article_content: Content of the article.

    Returns:
        List of quiz question dicts.
    """
    # Extract sentences that contain key information
    sentences = re.split(r"[.!?]+", article_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    questions: list[dict] = []

    if len(sentences) >= 1:
        s = sentences[0]
        words = s.split()
        key_phrase = " ".join(words[:6])
        questions.append({
            "question": f"According to the article, what is described about {key_phrase.lower().rstrip(',')}?",
            "options": [
                s[:100] + ("..." if len(s) > 100 else ""),
                "This topic is not covered in the article",
                "The article provides no specific details on this",
                "None of the above apply",
            ],
            "correct_index": 0,
            "explanation": f"The article states: '{s[:150]}...'",
        })

    if len(sentences) >= 3:
        s = sentences[2]
        questions.append({
            "question": f"What key point does the article '{article_title}' make?",
            "options": [
                "The article focuses only on introductory concepts",
                s[:100] + ("..." if len(s) > 100 else ""),
                "No specific points are made in the article",
                "The content is purely theoretical",
            ],
            "correct_index": 1,
            "explanation": f"The article specifically mentions: '{s[:150]}...'",
        })

    if len(sentences) >= 5:
        s = sentences[4]
        questions.append({
            "question": "Which of the following best reflects the article's content?",
            "options": [
                "Only background information is provided",
                "The article lacks specific guidance",
                "All mentioned topics are out of scope",
                s[:100] + ("..." if len(s) > 100 else ""),
            ],
            "correct_index": 3,
            "explanation": f"This is directly stated in the article: '{s[:150]}...'",
        })

    if len(sentences) >= 7:
        s = sentences[6]
        questions.append({
            "question": "Based on the article, which statement is accurate?",
            "options": [
                s[:100] + ("..." if len(s) > 100 else ""),
                "The article does not address this topic",
                "This information contradicts the main article",
                "No evidence supports this claim",
            ],
            "correct_index": 0,
            "explanation": f"The article confirms: '{s[:150]}...'",
        })

    # Ensure we have at least 4 questions by padding with generic ones
    while len(questions) < 4:
        questions.append({
            "question": f"What is the main purpose of the article '{article_title}'?",
            "options": [
                "To provide entertainment",
                "To share organizational knowledge and best practices",
                "To replace existing documentation",
                "To criticize current processes",
            ],
            "correct_index": 1,
            "explanation": "Knowledge articles are designed to share organizational knowledge and best practices.",
        })

    return questions[:4]


def generate_quiz(article_id: str) -> dict:
    """Generate a quiz from an article.

    Args:
        article_id: The article to base the quiz on.

    Returns:
        Dict with quiz id, title, questions.

    Raises:
        ValueError: If the article is not found.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, title, content FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(f"Article not found: {article_id}")

    article_title = row["title"]
    article_content = row["content"]

    prompt = (
        "Based on the following article, create exactly 4 multiple-choice quiz questions. "
        "Each question should test comprehension of the article content. "
        "Format your response as a JSON array where each element has: "
        '"question" (string), "options" (array of 4 strings), '
        '"correct_index" (0-3), "explanation" (string).\n\n'
        f"Article: {article_title}\n\n{article_content}\n\n"
        "Respond ONLY with the JSON array, no other text:"
    )

    llm_response = _call_ollama(prompt)
    quiz_questions: list[dict] | None = None

    if llm_response:
        # Try to parse JSON from response
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", llm_response, re.DOTALL)
            if json_match:
                quiz_questions = json.loads(json_match.group())
                # Validate structure
                for q in quiz_questions:
                    if not all(k in q for k in ("question", "options", "correct_index", "explanation")):
                        quiz_questions = None
                        break
        except (json.JSONDecodeError, TypeError, KeyError):
            quiz_questions = None

    if quiz_questions is None:
        quiz_questions = _mock_quiz(article_title, article_content)

    quiz_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    quiz_title = f"Quiz: {article_title}"

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO quizzes (id, article_id, title, questions_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (quiz_id, article_id, quiz_title, json.dumps(quiz_questions), now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _log_usage("generate_quiz")

    return {
        "id": quiz_id,
        "article_id": article_id,
        "title": quiz_title,
        "questions_json": json.dumps(quiz_questions),
        "created_at": now,
    }


def list_articles(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all articles.

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


def update_article(article_id: str, status: str | None = None, title: str | None = None, content: str | None = None) -> dict | None:
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


def list_quizzes(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all quizzes.

    Args:
        limit: Max results.
        offset: Pagination offset.

    Returns:
        List of quiz dicts.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM quizzes ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_quiz(quiz_id: str) -> dict | None:
    """Get a quiz by ID.

    Args:
        quiz_id: The quiz ID.

    Returns:
        Quiz dict or None.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
