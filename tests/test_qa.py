"""Tests for the Q&A engine and API endpoint."""

import pytest

from src.database import init_db
from src.services.qa_engine import (
    ask_question,
    build_context_prompt,
    compute_confidence,
    extractive_answer,
    list_questions,
)


class TestQAEngine:
    def test_compute_confidence_no_chunks(self) -> None:
        assert compute_confidence([]) == 0.0

    def test_compute_confidence_high_scores(self) -> None:
        chunks = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]
        confidence = compute_confidence(chunks)
        assert 0.7 < confidence <= 1.0

    def test_compute_confidence_low_scores(self) -> None:
        chunks = [{"score": 0.1}, {"score": 0.05}]
        confidence = compute_confidence(chunks)
        assert 0.0 < confidence < 0.3

    def test_extractive_answer_no_chunks(self) -> None:
        answer = extractive_answer("what is python?", [])
        assert "couldn't find" in answer.lower()

    def test_extractive_answer_with_chunks(self) -> None:
        chunks = [{
            "doc_title": "Python Guide",
            "content": "Python is a versatile programming language.",
            "doc_id": "doc1",
            "score": 0.8,
        }]
        answer = extractive_answer("what is python?", chunks)
        assert "Python Guide" in answer
        assert "Python" in answer

    def test_build_context_prompt(self) -> None:
        chunks = [{
            "doc_title": "Guide",
            "content": "Important information here",
            "doc_id": "d1",
            "score": 0.5,
        }]
        prompt = build_context_prompt("test question?", chunks)
        assert "Source 1: Guide" in prompt
        assert "Important information" in prompt
        assert "test question?" in prompt

    def test_ask_question_mock_provider(self) -> None:
        init_db()
        # With mock provider, should get extractive answer
        search_results = [{
            "id": "chunk1",
            "score": 0.7,
            "metadata": {
                "doc_title": "Test Doc",
                "content": "Python is great for data science.",
                "doc_id": "doc1",
            },
        }]
        result = ask_question("what is python?", search_results)
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "question_id" in result

    def test_ask_question_no_results(self) -> None:
        init_db()
        result = ask_question("obscure topic?", [])
        assert result["confidence"] == 0.0
        assert "couldn't find" in result["answer"].lower()

    def test_list_questions_empty(self) -> None:
        init_db()
        questions = list_questions()
        assert questions == []


class TestQAAPI:
    def test_ask_endpoint(self, client, sample_doc) -> None:
        client.post("/api/documents/json", json=sample_doc)
        client.post("/api/reindex")

        resp = client.post("/api/ask", json={"query": "What are Python best practices?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data

    def test_ask_empty_knowledge_base(self, client) -> None:
        resp = client.post("/api/ask", json={"query": "What is quantum computing?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == 0.0

    def test_list_questions_endpoint(self, client) -> None:
        resp = client.get("/api/questions")
        assert resp.status_code == 200
