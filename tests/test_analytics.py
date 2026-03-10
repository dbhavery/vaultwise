"""Tests for analytics-related endpoints and application health."""

import json
from uuid import uuid4
from datetime import datetime, timezone


class TestOverview:
    """Tests for the health/overview endpoint."""

    def test_overview_stats_correct(self, seeded_client):
        """Health endpoint should return correct counts after seeding."""
        resp = seeded_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["documents"] >= 5  # seed has 5 docs
        assert data["chunks"] >= 5  # at least one chunk per doc

    def test_overview_empty_db(self, client):
        """Health endpoint on empty DB should return zero counts."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == 0
        assert data["chunks"] == 0


class TestSeededDocuments:
    """Tests verifying seeded data is accessible via the API."""

    def test_seeded_documents_listed(self, seeded_client):
        """List documents should return the seeded data."""
        resp = seeded_client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 5

    def test_seeded_articles_listed(self, seeded_client):
        """List articles should return seeded articles."""
        resp = seeded_client.get("/api/articles")
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) >= 1

    def test_seeded_quizzes_listed(self, seeded_client):
        """List quizzes should return seeded quizzes."""
        resp = seeded_client.get("/api/quizzes")
        assert resp.status_code == 200
        quizzes = resp.json()
        assert len(quizzes) >= 1

    def test_seeded_questions_listed(self, seeded_client):
        """List questions should return seeded questions."""
        resp = seeded_client.get("/api/questions")
        assert resp.status_code == 200
        questions = resp.json()
        assert len(questions) >= 10


class TestSearchIndex:
    """Tests verifying search works on seeded data."""

    def test_search_on_seeded_data(self, seeded_client):
        """Search should return results from seeded documents."""
        resp = seeded_client.post("/api/search", json={"query": "security policy"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_search_stats(self, seeded_client):
        """Search stats endpoint should reflect indexed documents."""
        resp = seeded_client.get("/api/search/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] >= 5
        assert data["vocabulary_size"] > 0
        assert data["built"] is True


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return status and counts."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "documents" in data
        assert "chunks" in data
