"""Tests for the TF-IDF search API endpoint."""

import pytest


class TestSearchAPI:
    def test_search_empty_index(self, client) -> None:
        resp = client.post("/api/search", json={"query": "python"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_search_after_ingestion(self, client, sample_doc) -> None:
        # Ingest a document
        client.post("/api/documents/json", json=sample_doc)
        # Rebuild index
        client.post("/api/reindex")

        resp = client.post("/api/search", json={"query": "python testing pytest"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        assert data["results"][0]["score"] > 0

    def test_search_with_limit(self, client, sample_doc) -> None:
        client.post("/api/documents/json", json=sample_doc)
        client.post("/api/reindex")

        resp = client.post("/api/search", json={"query": "python", "limit": 1})
        data = resp.json()
        assert len(data["results"]) <= 1

    def test_search_returns_metadata(self, client, sample_doc) -> None:
        client.post("/api/documents/json", json=sample_doc)
        client.post("/api/reindex")

        resp = client.post("/api/search", json={"query": "python best practices"})
        data = resp.json()
        if data["results"]:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "content" in result
            assert "score" in result
            assert "doc_title" in result
            assert "doc_id" in result

    def test_search_relevance_ordering(self, client) -> None:
        # Create two documents with different relevance
        client.post("/api/documents/json", json={
            "title": "Python Deep Dive",
            "content": "Python programming Python language Python ecosystem Python frameworks Python tools",
        })
        client.post("/api/documents/json", json={
            "title": "Java Intro",
            "content": "Java programming language. Brief mention of Python interoperability.",
        })
        client.post("/api/reindex")

        resp = client.post("/api/search", json={"query": "Python programming"})
        data = resp.json()
        results = data["results"]
        if len(results) >= 2:
            # Higher score = more relevant
            assert results[0]["score"] >= results[1]["score"]

    def test_search_stats(self, client, sample_doc) -> None:
        client.post("/api/documents/json", json=sample_doc)
        client.post("/api/reindex")

        resp = client.get("/api/search/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] >= 1
        assert data["vocabulary_size"] >= 1
        assert data["built"] is True

    def test_search_validation(self, client) -> None:
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422
