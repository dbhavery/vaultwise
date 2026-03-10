"""Tests for the documents API router."""

import pytest


class TestDocumentsAPI:
    def test_list_documents_empty(self, client) -> None:
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total"] == 0

    def test_create_document_json(self, client, sample_doc) -> None:
        resp = client.post("/api/documents/json", json=sample_doc)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["title"] == sample_doc["title"]
        assert data["word_count"] > 0

    def test_get_document(self, client, sample_doc) -> None:
        create_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = create_resp.json()["id"]

        resp = client.get(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == doc_id
        assert "chunks" in data

    def test_get_document_not_found(self, client) -> None:
        resp = client.get("/api/documents/nonexistent")
        assert resp.status_code == 404

    def test_delete_document(self, client, sample_doc) -> None:
        create_resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = create_resp.json()["id"]

        resp = client.delete(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify it's gone
        resp = client.get(f"/api/documents/{doc_id}")
        assert resp.status_code == 404

    def test_delete_document_not_found(self, client) -> None:
        resp = client.delete("/api/documents/nonexistent")
        assert resp.status_code == 404

    def test_create_document_validation_error(self, client) -> None:
        resp = client.post("/api/documents/json", json={"title": "", "content": "x"})
        assert resp.status_code == 422

    def test_list_documents_pagination(self, client) -> None:
        # Create 3 documents
        for i in range(3):
            client.post("/api/documents/json", json={
                "title": f"Doc {i}",
                "content": f"Content for document number {i} with enough words",
            })

        resp = client.get("/api/documents?limit=2&offset=0")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["documents"]) == 2

        resp = client.get("/api/documents?limit=2&offset=2")
        data = resp.json()
        assert len(data["documents"]) == 1
