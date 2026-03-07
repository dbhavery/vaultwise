"""Tests for the training module (articles and quizzes)."""

import json

from vaultwise.database import get_connection, init_db


class TestArticles:
    """Tests for article generation and management."""

    def test_generate_article_from_docs(self, client, sample_doc):
        """Generating an article should create one from the given docs."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        art_resp = client.post("/api/articles/generate", json={"doc_ids": [doc_id]})
        assert art_resp.status_code == 200
        data = art_resp.json()
        assert "id" in data
        assert "title" in data
        assert "content" in data
        assert len(data["content"]) > 0
        assert data["status"] == "draft"

    def test_generate_article_invalid_docs(self, client):
        """Generating an article with invalid doc IDs should fail."""
        resp = client.post("/api/articles/generate", json={"doc_ids": ["nonexistent"]})
        assert resp.status_code == 400

    def test_list_articles(self, seeded_client):
        """Listing articles should return seed data."""
        resp = seeded_client.get("/api/articles")
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) >= 1

    def test_update_article_status(self, client, sample_doc):
        """Updating an article's status should persist."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        art = client.post("/api/articles/generate", json={"doc_ids": [doc_id]}).json()
        # Update to published
        update_resp = client.patch(f"/api/articles/{art['id']}", json={"status": "published"})
        assert update_resp.status_code == 200
        assert update_resp.json()["status"] == "published"

    def test_get_article_detail(self, client, sample_doc):
        """Getting an article by ID should return full content."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        art = client.post("/api/articles/generate", json={"doc_ids": [doc_id]}).json()
        detail = client.get(f"/api/articles/{art['id']}").json()
        assert detail["id"] == art["id"]
        assert len(detail["content"]) > 0


class TestQuizzes:
    """Tests for quiz generation."""

    def test_generate_quiz_from_article(self, client, sample_doc):
        """Generating a quiz should create one with questions."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        art = client.post("/api/articles/generate", json={"doc_ids": [doc_id]}).json()
        quiz_resp = client.post("/api/quizzes/generate", json={"article_id": art["id"]})
        assert quiz_resp.status_code == 200
        data = quiz_resp.json()
        assert "id" in data
        assert "questions_json" in data
        questions = json.loads(data["questions_json"])
        assert len(questions) >= 1

    def test_quiz_format_valid(self, client, sample_doc):
        """Quiz questions should have the expected structure."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        art = client.post("/api/articles/generate", json={"doc_ids": [doc_id]}).json()
        quiz = client.post("/api/quizzes/generate", json={"article_id": art["id"]}).json()
        questions = json.loads(quiz["questions_json"])
        for q in questions:
            assert "question" in q
            assert "options" in q
            assert isinstance(q["options"], list)
            assert len(q["options"]) >= 2
            assert "correct_index" in q
            assert "explanation" in q
            assert 0 <= q["correct_index"] < len(q["options"])

    def test_generate_quiz_invalid_article(self, client):
        """Generating a quiz for a nonexistent article should fail."""
        resp = client.post("/api/quizzes/generate", json={"article_id": "nonexistent"})
        assert resp.status_code == 400

    def test_list_quizzes(self, seeded_client):
        """Listing quizzes should return seed data."""
        resp = seeded_client.get("/api/quizzes")
        assert resp.status_code == 200
        quizzes = resp.json()
        assert len(quizzes) >= 1

    def test_get_quiz_detail(self, seeded_client):
        """Getting a quiz by ID should return full quiz data."""
        quizzes = seeded_client.get("/api/quizzes").json()
        assert len(quizzes) > 0
        quiz = seeded_client.get(f"/api/quizzes/{quizzes[0]['id']}").json()
        assert quiz["id"] == quizzes[0]["id"]
        assert "questions_json" in quiz
