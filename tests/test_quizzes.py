"""Tests for quiz generation."""

import json

import pytest

from src.database import init_db
from src.services.article_generator import generate_article
from src.services.ingestion import ingest_document
from src.services.quiz_generator import generate_quiz, get_quiz, list_quizzes


class TestQuizGenerator:
    def _create_article(self, sample_doc: dict) -> str:
        """Helper: ingest doc and generate article, return article ID."""
        doc = ingest_document(**sample_doc)
        article = generate_article([doc["id"]])
        return article["id"]

    def test_generate_quiz(self, sample_doc) -> None:
        init_db()
        article_id = self._create_article(sample_doc)
        quiz = generate_quiz(article_id)

        assert "id" in quiz
        assert quiz["article_id"] == article_id
        assert "Quiz:" in quiz["title"]

        questions = json.loads(quiz["questions_json"])
        assert len(questions) == 4
        for q in questions:
            assert "question" in q
            assert "options" in q
            assert len(q["options"]) == 4
            assert "correct_index" in q
            assert 0 <= q["correct_index"] <= 3
            assert "explanation" in q

    def test_generate_quiz_article_not_found(self) -> None:
        init_db()
        with pytest.raises(ValueError, match="Article not found"):
            generate_quiz("nonexistent")

    def test_get_quiz(self, sample_doc) -> None:
        init_db()
        article_id = self._create_article(sample_doc)
        quiz = generate_quiz(article_id)
        retrieved = get_quiz(quiz["id"])
        assert retrieved is not None
        assert retrieved["id"] == quiz["id"]

    def test_get_quiz_not_found(self) -> None:
        init_db()
        assert get_quiz("nonexistent") is None

    def test_list_quizzes(self, sample_doc) -> None:
        init_db()
        article_id = self._create_article(sample_doc)
        generate_quiz(article_id)
        quizzes = list_quizzes()
        assert len(quizzes) == 1


class TestQuizzesAPI:
    def _setup_article(self, client, sample_doc) -> str:
        """Helper: create doc + article via API, return article ID."""
        doc_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = doc_resp.json()["id"]
        art_resp = client.post("/api/articles/generate", json={"doc_ids": [doc_id]})
        return art_resp.json()["id"]

    def test_generate_quiz_endpoint(self, client, sample_doc) -> None:
        article_id = self._setup_article(client, sample_doc)
        resp = client.post("/api/quizzes/generate", json={"article_id": article_id})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        questions = json.loads(data["questions_json"])
        assert len(questions) == 4

    def test_generate_quiz_bad_article(self, client) -> None:
        resp = client.post("/api/quizzes/generate", json={"article_id": "bad_id"})
        assert resp.status_code == 400

    def test_list_quizzes_endpoint(self, client) -> None:
        resp = client.get("/api/quizzes")
        assert resp.status_code == 200

    def test_get_quiz_endpoint(self, client, sample_doc) -> None:
        article_id = self._setup_article(client, sample_doc)
        quiz_resp = client.post("/api/quizzes/generate", json={"article_id": article_id})
        quiz_id = quiz_resp.json()["id"]

        resp = client.get(f"/api/quizzes/{quiz_id}")
        assert resp.status_code == 200

    def test_get_quiz_not_found(self, client) -> None:
        resp = client.get("/api/quizzes/nonexistent")
        assert resp.status_code == 404
