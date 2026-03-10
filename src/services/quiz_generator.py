"""Quiz generator — creates multiple-choice quizzes from article content.

Attempts LLM-powered quiz generation first, falling back to an extractive
approach that pulls key sentences from the article to form questions.
"""

import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from src.database import get_connection
from src.services.qa_engine import call_llm, _log_usage

logger = logging.getLogger("vaultwise.quiz_generator")


def _extractive_quiz(article_title: str, article_content: str) -> list[dict]:
    """Generate quiz questions from article content using extraction.

    Pulls key sentences and creates comprehension questions around them.

    Args:
        article_title: Title of the source article.
        article_content: Full article text.

    Returns:
        List of quiz question dicts (4 questions).
    """
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

    # Pad to at least 4 questions
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

    Tries LLM generation first; falls back to extractive quiz if the
    LLM is unavailable or returns invalid JSON.

    Args:
        article_id: The source article ID.

    Returns:
        Dict with quiz id, article_id, title, questions_json, created_at.

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

    llm_response = call_llm(prompt)
    quiz_questions: list[dict] | None = None

    if llm_response:
        try:
            json_match = re.search(r"\[.*\]", llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate structure
                valid = True
                for q in parsed:
                    if not all(k in q for k in ("question", "options", "correct_index", "explanation")):
                        valid = False
                        break
                if valid:
                    quiz_questions = parsed
        except (json.JSONDecodeError, TypeError, KeyError):
            quiz_questions = None

    if quiz_questions is None:
        quiz_questions = _extractive_quiz(article_title, article_content)

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


def list_quizzes(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all quizzes with pagination.

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
