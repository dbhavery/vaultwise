"""Tests for training article generation."""

import pytest

from src.database import init_db
from src.services.article_generator import (
    generate_article,
    get_article,
    list_articles,
    update_article,
)
from src.services.ingestion import ingest_document


class TestArticleGenerator:
    def test_generate_article_from_document(self, sample_doc) -> None:
        init_db()
        doc = ingest_document(**sample_doc)
        result = generate_article([doc["id"]])

        assert "id" in result
        assert "title" in result
        assert "content" in result
        assert result["status"] == "draft"
        assert len(result["content"]) > 0

    def test_generate_article_no_docs(self) -> None:
        init_db()
        with pytest.raises(ValueError, match="No documents found"):
            generate_article(["nonexistent_id"])

    def test_get_article(self, sample_doc) -> None:
        init_db()
        doc = ingest_document(**sample_doc)
        article = generate_article([doc["id"]])

        retrieved = get_article(article["id"])
        assert retrieved is not None
        assert retrieved["title"] == article["title"]

    def test_get_article_not_found(self) -> None:
        init_db()
        assert get_article("nonexistent") is None

    def test_list_articles_empty(self) -> None:
        init_db()
        articles = list_articles()
        assert articles == []

    def test_list_articles(self, sample_doc) -> None:
        init_db()
        doc = ingest_document(**sample_doc)
        generate_article([doc["id"]])

        articles = list_articles()
        assert len(articles) == 1

    def test_update_article_status(self, sample_doc) -> None:
        init_db()
        doc = ingest_document(**sample_doc)
        article = generate_article([doc["id"]])

        updated = update_article(article["id"], status="published")
        assert updated is not None
        assert updated["status"] == "published"

    def test_update_article_not_found(self) -> None:
        init_db()
        assert update_article("nonexistent", status="published") is None


class TestTrainingAPI:
    def test_generate_article_endpoint(self, client, sample_doc) -> None:
        create_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = create_resp.json()["id"]

        resp = client.post("/api/articles/generate", json={"doc_ids": [doc_id]})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "title" in data

    def test_generate_article_bad_ids(self, client) -> None:
        resp = client.post("/api/articles/generate", json={"doc_ids": ["bad_id"]})
        assert resp.status_code == 400

    def test_list_articles_endpoint(self, client) -> None:
        resp = client.get("/api/articles")
        assert resp.status_code == 200

    def test_get_article_endpoint(self, client, sample_doc) -> None:
        create_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = create_resp.json()["id"]
        art_resp = client.post("/api/articles/generate", json={"doc_ids": [doc_id]})
        article_id = art_resp.json()["id"]

        resp = client.get(f"/api/articles/{article_id}")
        assert resp.status_code == 200

    def test_get_article_not_found(self, client) -> None:
        resp = client.get("/api/articles/nonexistent")
        assert resp.status_code == 404

    def test_update_article_endpoint(self, client, sample_doc) -> None:
        create_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = create_resp.json()["id"]
        art_resp = client.post("/api/articles/generate", json={"doc_ids": [doc_id]})
        article_id = art_resp.json()["id"]

        resp = client.patch(f"/api/articles/{article_id}", json={"status": "published"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"
