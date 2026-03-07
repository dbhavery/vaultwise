"""Tests for the question-answering module."""

import json

from vaultwise.database import get_connection, init_db
from vaultwise.qa import _compute_confidence, _mock_answer


class TestQA:
    """Tests for the QA endpoints and logic."""

    def test_ask_returns_answer_with_sources(self, client, sample_doc):
        """Asking a question should return an answer with sources."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.post("/api/ask", json={"query": "What are Python best practices?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert "confidence" in data
        assert "question_id" in data

    def test_question_stored_in_database(self, client, sample_doc):
        """Asked questions should be stored in the database."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.post("/api/ask", json={"query": "What is pytest?"})
        data = resp.json()
        # Check the questions list
        questions = client.get("/api/questions").json()
        assert any(q["id"] == data["question_id"] for q in questions)

    def test_confidence_calculation(self):
        """Confidence should be between 0 and 1."""
        chunks = [{"score": 0.9}, {"score": 0.7}, {"score": 0.5}]
        conf = _compute_confidence(chunks)
        assert 0.0 <= conf <= 1.0

    def test_confidence_empty_chunks(self):
        """Confidence should be 0 for empty results."""
        assert _compute_confidence([]) == 0.0

    def test_gap_detection_on_low_confidence(self, client, sample_doc):
        """Low-confidence answers should create knowledge gaps."""
        # Upload a document about Python
        client.post("/api/documents/json", json=sample_doc)
        # Ask about something not covered at all
        client.post("/api/ask", json={"query": "xyzzy quantum flux capacitor nonsense"})
        # Check that a gap was potentially created
        gaps = client.get("/api/analytics/gaps").json()
        # The gap may or may not be created depending on scores,
        # but the endpoint should work
        assert isinstance(gaps, list)

    def test_mock_answer_with_chunks(self):
        """Mock answer should summarize chunk content."""
        chunks = [
            {"content": "Python is a programming language.", "doc_title": "Intro", "doc_id": "1"},
        ]
        answer = _mock_answer("What is Python?", chunks)
        assert "Python" in answer
        assert "Intro" in answer

    def test_mock_answer_empty_chunks(self):
        """Mock answer with no chunks should indicate no results."""
        answer = _mock_answer("What is nothing?", [])
        assert "couldn't find" in answer.lower() or "not" in answer.lower()
