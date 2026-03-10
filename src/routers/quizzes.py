"""Quiz generation router."""

from fastapi import APIRouter, HTTPException, Query

from src.models import QuizGenerateRequest
from src.services.quiz_generator import generate_quiz, get_quiz, list_quizzes

router = APIRouter(prefix="/api/quizzes", tags=["quizzes"])


@router.get("")
def api_list_quizzes(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """List all quizzes."""
    return list_quizzes(limit=limit, offset=offset)


@router.post("/generate")
def api_generate_quiz(req: QuizGenerateRequest) -> dict:
    """Generate a quiz from an article.

    Creates 4 multiple-choice comprehension questions based on the
    article content. Attempts LLM generation first, falls back to
    extractive question generation.
    """
    try:
        return generate_quiz(req.article_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{quiz_id}")
def api_get_quiz(quiz_id: str) -> dict:
    """Get a quiz by ID."""
    quiz = get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return quiz
