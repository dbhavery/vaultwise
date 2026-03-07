"""Tests for the document ingestion module."""

import json

from vaultwise.database import get_connection, init_db
from vaultwise.ingest import (
    _chunk_text,
    _count_words,
    _detect_doc_type,
    delete_document,
    get_document,
    ingest_document,
    list_documents,
)
from vaultwise.search import build_index


class TestChunking:
    """Tests for text chunking logic."""

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        text = "This is a short document with only a few words."
        chunks = _chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        words = ["word"] * 1200
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # Each chunk should be roughly chunk_size words
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 500

    def test_overlap_between_chunks(self):
        """Chunks should overlap by the specified number of words."""
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) >= 2
        # Check that overlap is present
        chunk0_words = set(chunks[0].split()[-10:])
        chunk1_words = set(chunks[1].split()[:10])
        assert chunk0_words == chunk1_words

    def test_empty_text_no_chunks(self):
        """Empty text should produce no chunks."""
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []


class TestWordCount:
    """Tests for word counting."""

    def test_word_count(self):
        assert _count_words("one two three") == 3

    def test_word_count_empty(self):
        assert _count_words("") == 0


class TestDocTypeDetection:
    """Tests for document type detection."""

    def test_markdown_extension(self):
        assert _detect_doc_type("readme.md", "anything") == "markdown"

    def test_python_extension(self):
        assert _detect_doc_type("main.py", "anything") == "python"

    def test_markdown_content(self):
        assert _detect_doc_type("notes", "# Heading\nsome content") == "markdown"

    def test_default_text(self):
        assert _detect_doc_type("notes", "just plain text here") == "text"


class TestDocumentIngest:
    """Tests for the full ingestion pipeline."""

    def test_upload_document(self, client, sample_doc):
        """Uploading a document should store it and create chunks."""
        resp = client.post("/api/documents/json", json=sample_doc)
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == sample_doc["title"]
        assert data["word_count"] > 0
        assert data["chunk_count"] >= 1

    def test_upload_creates_chunks(self, client, sample_doc):
        """Uploaded document should be retrievable with its chunks."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        detail = client.get(f"/api/documents/{doc_id}").json()
        assert detail["id"] == doc_id
        assert len(detail["chunks"]) >= 1

    def test_word_count_calculated(self, client, sample_doc):
        """Word count should be calculated on ingestion."""
        resp = client.post("/api/documents/json", json=sample_doc)
        assert resp.json()["word_count"] == len(sample_doc["content"].split())

    def test_list_documents(self, client, sample_doc):
        """Documents should appear in the list after upload."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["documents"]) == 1

    def test_delete_document(self, client, sample_doc):
        """Deleting a document should remove it and its chunks."""
        resp = client.post("/api/documents/json", json=sample_doc)
        doc_id = resp.json()["id"]
        del_resp = client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 200
        get_resp = client.get(f"/api/documents/{doc_id}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent(self, client):
        """Deleting a nonexistent document should return 404."""
        resp = client.delete("/api/documents/nonexistent123")
        assert resp.status_code == 404
