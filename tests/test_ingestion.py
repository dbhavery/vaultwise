"""Tests for document ingestion — parsing, chunking, and storage."""

import pytest

from src.database import init_db
from src.services.ingestion import (
    chunk_text,
    count_words,
    detect_doc_type,
    delete_document,
    get_document,
    ingest_document,
    list_documents,
    strip_markdown,
)


class TestCountWords:
    def test_basic_count(self) -> None:
        assert count_words("hello world foo bar") == 4

    def test_empty_string(self) -> None:
        assert count_words("") == 0

    def test_single_word(self) -> None:
        assert count_words("hello") == 1


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_empty_text(self) -> None:
        assert chunk_text("", chunk_size=500, overlap=50) == []

    def test_long_text_multiple_chunks(self) -> None:
        words = ["word"] * 1200
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_overlap_produces_shared_content(self) -> None:
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        # The end of chunk 0 should overlap with the start of chunk 1
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        overlap_words = set(chunk0_words[-20:]) & set(chunk1_words[:20])
        assert len(overlap_words) > 0


class TestDetectDocType:
    def test_markdown_by_extension(self) -> None:
        assert detect_doc_type("notes.md", "plain content") == "markdown"

    def test_python_by_extension(self) -> None:
        assert detect_doc_type("script.py", "x = 1") == "python"

    def test_markdown_by_content(self) -> None:
        assert detect_doc_type("notes", "# Heading\nsome content") == "markdown"

    def test_python_by_content(self) -> None:
        assert detect_doc_type("code", "def hello():\n    pass") == "python"

    def test_default_text(self) -> None:
        assert detect_doc_type("file", "plain text content") == "text"


class TestStripMarkdown:
    def test_removes_headings(self) -> None:
        result = strip_markdown("# Title\n## Subtitle")
        assert "#" not in result
        assert "Title" in result

    def test_removes_bold(self) -> None:
        result = strip_markdown("This is **bold** text")
        assert "**" not in result
        assert "bold" in result

    def test_removes_code_fences(self) -> None:
        result = strip_markdown("Before\n```python\ncode\n```\nAfter")
        assert "```" not in result
        assert "Before" in result
        assert "After" in result

    def test_preserves_link_text(self) -> None:
        result = strip_markdown("Visit [Google](https://google.com)")
        assert "Google" in result
        assert "https" not in result


class TestIngestion:
    def test_ingest_and_retrieve(self, sample_doc: dict) -> None:
        init_db()
        result = ingest_document(**sample_doc)
        assert "id" in result
        assert result["title"] == sample_doc["title"]
        assert result["word_count"] > 0

        doc = get_document(result["id"])
        assert doc is not None
        assert doc["title"] == sample_doc["title"]
        assert len(doc["chunks"]) >= 1

    def test_ingest_auto_detects_type(self) -> None:
        init_db()
        result = ingest_document(
            title="readme.md",
            content="# Hello\nWorld",
        )
        doc = get_document(result["id"])
        assert doc is not None
        assert doc["doc_type"] == "markdown"

    def test_list_documents(self, sample_doc: dict) -> None:
        init_db()
        ingest_document(**sample_doc)
        ingest_document(title="Second Doc", content="More content here")

        docs, total = list_documents(limit=10, offset=0)
        assert total == 2
        assert len(docs) == 2

    def test_delete_document(self, sample_doc: dict) -> None:
        init_db()
        result = ingest_document(**sample_doc)
        assert delete_document(result["id"]) is True
        assert get_document(result["id"]) is None

    def test_delete_nonexistent(self) -> None:
        init_db()
        assert delete_document("nonexistent_id") is False
