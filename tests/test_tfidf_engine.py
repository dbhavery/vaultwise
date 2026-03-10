"""Tests for the pure Python TF-IDF engine — the core differentiator.

Verifies tokenization, TF/IDF math, vector operations, cosine similarity,
and the full TFIDFIndex search lifecycle.
"""

import math

from src.services.tfidf_engine import (
    STOP_WORDS,
    TFIDFIndex,
    build_tfidf_vector,
    compute_idf,
    compute_tf,
    cosine_similarity,
    l2_norm,
    normalize_vector,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = tokenize("Hello World from Python")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_stop_word_removal(self) -> None:
        tokens = tokenize("the quick brown fox is a very fast animal")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "very" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_single_char_removal(self) -> None:
        tokens = tokenize("I am a big fan of x and y")
        # Single characters should be removed
        for t in tokens:
            assert len(t) > 1

    def test_lowercase(self) -> None:
        tokens = tokenize("UPPERCASE Words MiXeD")
        for t in tokens:
            assert t == t.lower()

    def test_punctuation_handling(self) -> None:
        tokens = tokenize("Hello, world! How's it going?")
        assert "hello" in tokens
        assert "how's" in tokens  # contraction preserved
        assert "going" in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_only_stop_words(self) -> None:
        assert tokenize("the a an is are was") == []

    def test_numbers_preserved(self) -> None:
        tokens = tokenize("Python 3.10 has 42 features")
        assert "python" in tokens
        assert "42" in tokens


# ---------------------------------------------------------------------------
# Term Frequency tests
# ---------------------------------------------------------------------------

class TestComputeTF:
    def test_basic_tf(self) -> None:
        tokens = ["python", "testing", "python", "code"]
        tf = compute_tf(tokens)
        assert tf["python"] == 2 / 4
        assert tf["testing"] == 1 / 4
        assert tf["code"] == 1 / 4

    def test_single_token(self) -> None:
        tf = compute_tf(["hello"])
        assert tf["hello"] == 1.0

    def test_empty_tokens(self) -> None:
        tf = compute_tf([])
        assert tf == {}

    def test_all_same_token(self) -> None:
        tf = compute_tf(["python", "python", "python"])
        assert tf["python"] == 1.0


# ---------------------------------------------------------------------------
# Inverse Document Frequency tests
# ---------------------------------------------------------------------------

class TestComputeIDF:
    def test_basic_idf(self) -> None:
        docs = [
            ["python", "code"],
            ["python", "testing"],
            ["java", "code"],
        ]
        idf = compute_idf(docs)
        # "python" appears in 2/3 docs
        # "java" appears in 1/3 docs
        # java should have higher IDF than python
        assert idf["java"] > idf["python"]

    def test_universal_term_lower_idf(self) -> None:
        docs = [
            ["common", "unique1"],
            ["common", "unique2"],
            ["common", "unique3"],
        ]
        idf = compute_idf(docs)
        # "common" appears in all docs - should have lowest IDF
        assert idf["common"] < idf["unique1"]

    def test_smoothing_prevents_zero(self) -> None:
        docs = [["hello", "world"]]
        idf = compute_idf(docs)
        # With smoothing, IDF should always be > 0
        assert all(v > 0 for v in idf.values())

    def test_empty_corpus(self) -> None:
        idf = compute_idf([])
        assert idf == {}

    def test_idf_formula_correctness(self) -> None:
        docs = [["alpha"], ["alpha", "beta"], ["gamma"]]
        idf = compute_idf(docs)
        n = 3
        # alpha appears in 2 docs
        expected_alpha = math.log((n + 1) / (2 + 1)) + 1.0
        assert abs(idf["alpha"] - expected_alpha) < 1e-9


# ---------------------------------------------------------------------------
# Vector operation tests
# ---------------------------------------------------------------------------

class TestVectorOps:
    def test_l2_norm(self) -> None:
        vec = {"a": 3.0, "b": 4.0}
        assert abs(l2_norm(vec) - 5.0) < 1e-9

    def test_l2_norm_empty(self) -> None:
        assert l2_norm({}) == 0.0

    def test_normalize_vector(self) -> None:
        vec = {"a": 3.0, "b": 4.0}
        normed = normalize_vector(vec)
        norm = l2_norm(normed)
        assert abs(norm - 1.0) < 1e-9

    def test_normalize_zero_vector(self) -> None:
        assert normalize_vector({}) == {}

    def test_build_tfidf_vector(self) -> None:
        tf = {"python": 0.5, "testing": 0.25, "unknown": 0.25}
        idf = {"python": 2.0, "testing": 3.0}
        vec = build_tfidf_vector(tf, idf)
        assert abs(vec["python"] - 1.0) < 1e-9  # 0.5 * 2.0
        assert abs(vec["testing"] - 0.75) < 1e-9  # 0.25 * 3.0
        assert "unknown" not in vec  # not in IDF

    def test_cosine_similarity_identical(self) -> None:
        vec = {"python": 0.5, "testing": 0.5}
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self) -> None:
        vec_a = {"python": 1.0}
        vec_b = {"java": 1.0}
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_similarity_partial_overlap(self) -> None:
        vec_a = {"python": 1.0, "testing": 1.0}
        vec_b = {"python": 1.0, "java": 1.0}
        sim = cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_cosine_similarity_empty(self) -> None:
        assert cosine_similarity({}, {"a": 1.0}) == 0.0
        assert cosine_similarity({"a": 1.0}, {}) == 0.0


# ---------------------------------------------------------------------------
# TFIDFIndex integration tests
# ---------------------------------------------------------------------------

class TestTFIDFIndex:
    def test_empty_index_search(self) -> None:
        index = TFIDFIndex()
        index.build()
        results = index.search("hello world")
        assert results == []

    def test_single_document_search(self) -> None:
        index = TFIDFIndex()
        index.add_document("doc1", "Python testing best practices guide", {"title": "Test Doc"})
        index.build()

        results = index.search("python testing")
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] > 0

    def test_multi_document_ranking(self) -> None:
        index = TFIDFIndex()
        index.add_document("doc1", "Python is a programming language", {"title": "Python"})
        index.add_document("doc2", "Java is a programming language", {"title": "Java"})
        index.add_document("doc3", "Python testing with pytest framework", {"title": "Testing"})
        index.build()

        results = index.search("python", limit=3)
        # Docs mentioning "python" should rank higher
        ids = [r["id"] for r in results]
        assert "doc1" in ids
        assert "doc3" in ids

    def test_limit_parameter(self) -> None:
        index = TFIDFIndex()
        for i in range(10):
            index.add_document(f"doc{i}", f"document number {i} about testing", {})
        index.build()

        results = index.search("testing", limit=3)
        assert len(results) <= 3

    def test_metadata_preserved(self) -> None:
        meta = {"title": "My Document", "author": "Test"}
        index = TFIDFIndex()
        index.add_document("doc1", "some text content here", meta)
        index.build()

        results = index.search("text content")
        assert len(results) == 1
        assert results[0]["metadata"] == meta

    def test_get_vector(self) -> None:
        index = TFIDFIndex()
        index.add_document("doc1", "Python programming guide", {})
        index.build()

        vec = index.get_vector("doc1")
        assert vec is not None
        assert isinstance(vec, dict)
        assert len(vec) > 0

    def test_get_vector_nonexistent(self) -> None:
        index = TFIDFIndex()
        index.build()
        assert index.get_vector("nonexistent") is None

    def test_clear(self) -> None:
        index = TFIDFIndex()
        index.add_document("doc1", "hello world", {})
        index.build()
        assert index.document_count == 1

        index.clear()
        assert index.document_count == 0
        assert index.vocabulary_size == 0

    def test_document_count(self) -> None:
        index = TFIDFIndex()
        index.add_document("d1", "one", {})
        index.add_document("d2", "two", {})
        assert index.document_count == 2

    def test_no_results_for_stop_words_only(self) -> None:
        index = TFIDFIndex()
        index.add_document("doc1", "the document is about things", {})
        index.build()

        results = index.search("the is a")
        assert results == []
