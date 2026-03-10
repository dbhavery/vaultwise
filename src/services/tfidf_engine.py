"""Pure Python TF-IDF search engine — zero ML dependencies.

This module implements the full TF-IDF pipeline from scratch:
  1. Tokenization with stop word removal
  2. Term Frequency (TF) calculation
  3. Inverse Document Frequency (IDF) calculation
  4. TF-IDF vector construction
  5. Cosine similarity scoring

No sklearn, numpy, or any search/ML library is used. All math is
implemented with Python builtins and the standard library.

Mathematical foundations:
  TF(t, d) = count(t in d) / len(d)
  IDF(t) = log((N + 1) / (df(t) + 1)) + 1   (smoothed)
  TF-IDF(t, d) = TF(t, d) * IDF(t)
  cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
"""

import math
import re
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Stop words — common English words that carry little discriminative value
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "about", "up", "down", "also", "these", "those", "any", "many",
})

# Regex for extracting alphanumeric tokens (including contractions)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, removing stop words and single chars.

    Args:
        text: Raw input text.

    Returns:
        List of cleaned token strings.
    """
    words = _TOKEN_RE.findall(text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


# ---------------------------------------------------------------------------
# TF-IDF sparse vector (dict-based for memory efficiency)
# ---------------------------------------------------------------------------

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term frequency for a token list.

    TF(t, d) = count(t in d) / total_tokens_in_d

    Args:
        tokens: List of tokens from a single document/chunk.

    Returns:
        Dict mapping each token to its normalized frequency.
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across a corpus.

    IDF(t) = log((N + 1) / (df(t) + 1)) + 1

    The +1 smoothing prevents division by zero and reduces the impact
    of very rare terms.

    Args:
        documents: List of token lists (one per document/chunk).

    Returns:
        Dict mapping each term to its IDF weight.
    """
    n = len(documents)
    if n == 0:
        return {}

    # Count how many documents each term appears in
    df: Counter[str] = Counter()
    for doc_tokens in documents:
        unique_terms = set(doc_tokens)
        df.update(unique_terms)

    idf: dict[str, float] = {}
    for term, doc_freq in df.items():
        idf[term] = math.log((n + 1) / (doc_freq + 1)) + 1.0

    return idf


def build_tfidf_vector(
    tf: dict[str, float],
    idf: dict[str, float],
) -> dict[str, float]:
    """Build a TF-IDF vector (sparse dict) from TF and IDF components.

    TF-IDF(t, d) = TF(t, d) * IDF(t)

    Args:
        tf: Term frequency dict for one document.
        idf: Global IDF dict.

    Returns:
        Sparse vector as {term: tfidf_weight}.
    """
    vector: dict[str, float] = {}
    for term, tf_val in tf.items():
        if term in idf:
            weight = tf_val * idf[term]
            if weight > 0:
                vector[term] = weight
    return vector


def l2_norm(vector: dict[str, float]) -> float:
    """Compute the L2 (Euclidean) norm of a sparse vector.

    ||v|| = sqrt(sum(v_i^2))

    Args:
        vector: Sparse vector as dict.

    Returns:
        L2 norm as float.
    """
    if not vector:
        return 0.0
    return math.sqrt(sum(v * v for v in vector.values()))


def normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    """L2-normalize a sparse vector to unit length.

    Args:
        vector: Sparse vector as dict.

    Returns:
        Normalized sparse vector (or empty dict if norm is zero).
    """
    norm = l2_norm(vector)
    if norm == 0:
        return {}
    return {term: val / norm for term, val in vector.items()}


def cosine_similarity(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    """Compute cosine similarity between two sparse vectors.

    cos(a, b) = dot(a, b) / (||a|| * ||b||)

    For pre-normalized vectors (unit length), this reduces to just
    the dot product.

    Args:
        vec_a: First sparse vector.
        vec_b: Second sparse vector.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not vec_a or not vec_b:
        return 0.0

    # Iterate over the smaller vector for efficiency
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot_product = sum(
        val * vec_b[term]
        for term, val in vec_a.items()
        if term in vec_b
    )

    norm_a = l2_norm(vec_a)
    norm_b = l2_norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# TF-IDF Index — the main search structure
# ---------------------------------------------------------------------------

class TFIDFIndex:
    """In-memory TF-IDF search index over a corpus of text chunks.

    Usage:
        index = TFIDFIndex()
        index.add_document("id1", "some text content", {"title": "Doc 1"})
        index.add_document("id2", "other text content", {"title": "Doc 2"})
        index.build()
        results = index.search("query text", limit=5)

    The index stores sparse TF-IDF vectors keyed by document/chunk ID.
    After all documents are added, call build() to compute IDF weights
    and finalize the vectors.
    """

    def __init__(self) -> None:
        self._raw_tokens: dict[str, list[str]] = {}  # id -> tokens
        self._metadata: dict[str, dict] = {}  # id -> arbitrary metadata
        self._idf: dict[str, float] = {}
        self._vectors: dict[str, dict[str, float]] = {}  # id -> normalized tfidf vec
        self._built: bool = False

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return len(self._raw_tokens)

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the IDF dictionary."""
        return len(self._idf)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._raw_tokens.clear()
        self._metadata.clear()
        self._idf.clear()
        self._vectors.clear()
        self._built = False

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the index (must call build() after all adds).

        Args:
            doc_id: Unique identifier for this document/chunk.
            text: Raw text content.
            metadata: Optional dict of metadata (e.g., title, source).
        """
        tokens = tokenize(text)
        self._raw_tokens[doc_id] = tokens
        self._metadata[doc_id] = metadata or {}
        self._built = False

    def build(self) -> None:
        """Compute IDF weights and build TF-IDF vectors for all documents.

        Must be called after adding documents and before searching.
        """
        if not self._raw_tokens:
            self._idf = {}
            self._vectors = {}
            self._built = True
            return

        # Collect all token lists for IDF computation
        all_token_lists = list(self._raw_tokens.values())
        self._idf = compute_idf(all_token_lists)

        # Build and normalize TF-IDF vector for each document
        self._vectors = {}
        for doc_id, tokens in self._raw_tokens.items():
            tf = compute_tf(tokens)
            tfidf_vec = build_tfidf_vector(tf, self._idf)
            self._vectors[doc_id] = normalize_vector(tfidf_vec)

        self._built = True

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search the index for documents matching the query.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.

        Returns:
            List of result dicts with keys: id, score, metadata.
            Sorted by descending similarity score.
        """
        if not self._built or not self._vectors:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Build query TF-IDF vector using the corpus IDF
        query_tf = compute_tf(query_tokens)
        query_tfidf = build_tfidf_vector(query_tf, self._idf)
        query_vec = normalize_vector(query_tfidf)

        if not query_vec:
            return []

        # Score all documents using cosine similarity
        scored: list[tuple[str, float]] = []
        for doc_id, doc_vec in self._vectors.items():
            # Both vectors are already normalized, so cosine = dot product
            score = sum(
                val * doc_vec[term]
                for term, val in query_vec.items()
                if term in doc_vec
            )
            if score > 0.0:
                scored.append((doc_id, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for doc_id, score in scored[:limit]:
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "metadata": self._metadata.get(doc_id, {}),
            })

        return results

    def get_vector(self, doc_id: str) -> Optional[dict[str, float]]:
        """Get the normalized TF-IDF vector for a document.

        Args:
            doc_id: The document ID.

        Returns:
            Sparse vector dict, or None if not found.
        """
        return self._vectors.get(doc_id)

    def get_metadata(self, doc_id: str) -> Optional[dict]:
        """Get metadata for a document.

        Args:
            doc_id: The document ID.

        Returns:
            Metadata dict, or None if not found.
        """
        return self._metadata.get(doc_id)
