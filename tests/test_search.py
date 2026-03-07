"""Tests for the search module."""

from vaultwise.database import init_db
from vaultwise.ingest import ingest_document
from vaultwise.search import (
    _tokenize,
    build_index,
    compute_embedding,
    search_chunks,
)


class TestTokenizer:
    """Tests for the tokenizer."""

    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World Python Programming")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize("the quick brown fox and the lazy dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world! how's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how's" in tokens


class TestSearchIndex:
    """Tests for the search index and similarity search."""

    def test_search_returns_relevant_results(self, client, sample_doc):
        """Search should return results relevant to the query."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.post("/api/search", json={"query": "python type hints"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0
        # The top result should be from our document
        assert data["results"][0]["doc_title"] == sample_doc["title"]

    def test_search_score_ordering(self, client, sample_doc):
        """Results should be ordered by score descending."""
        client.post("/api/documents/json", json=sample_doc)
        client.post("/api/documents/json", json={
            "title": "Cooking Guide",
            "content": "This is a guide about cooking pasta and baking bread. Italian cuisine is delicious.",
            "doc_type": "text",
            "source": "upload",
        })
        resp = client.post("/api/search", json={"query": "python testing pytest", "limit": 10})
        data = resp.json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_handled(self, client, sample_doc):
        """An empty query should return an error."""
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422  # Validation error

    def test_no_results_for_unrelated_query(self, client, sample_doc):
        """A completely unrelated query should return no or low-scoring results."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.post("/api/search", json={"query": "quantum physics dark matter"})
        data = resp.json()
        # Either no results or very low scores
        if data["results"]:
            assert all(r["score"] < 0.5 for r in data["results"])

    def test_search_with_limit(self, client, sample_doc):
        """Limit parameter should control result count."""
        client.post("/api/documents/json", json=sample_doc)
        resp = client.post("/api/search", json={"query": "python", "limit": 2})
        data = resp.json()
        assert len(data["results"]) <= 2

    def test_compute_embedding_returns_list(self, client, sample_doc):
        """compute_embedding should return a list of floats after index is built."""
        client.post("/api/documents/json", json=sample_doc)
        emb = compute_embedding("python testing")
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) > 0
