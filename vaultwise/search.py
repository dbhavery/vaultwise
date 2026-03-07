"""Vector search using TF-IDF embeddings and cosine similarity."""

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from vaultwise.database import get_connection

# Global TF-IDF state -- rebuilt on startup and after ingestion
_vocabulary: dict[str, int] = {}  # word -> index
_idf: np.ndarray | None = None   # IDF weights
_chunk_vectors: dict[str, np.ndarray] = {}  # chunk_id -> TF-IDF vector
_chunk_metadata: dict[str, dict] = {}  # chunk_id -> {doc_id, content, doc_title}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping punctuation."""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    # Remove very common stop words for better discrimination
    stop_words = {
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
    }
    return [w for w in words if w not in stop_words and len(w) > 1]


def _compute_tf(tokens: list[str]) -> Counter:
    """Compute term frequency (raw count) for a token list."""
    return Counter(tokens)


def build_index() -> None:
    """Build the TF-IDF index from all chunks in the database.

    This loads all chunks, builds a vocabulary, computes IDF weights,
    and stores TF-IDF vectors in memory for fast similarity search.
    """
    global _vocabulary, _idf, _chunk_vectors, _chunk_metadata

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT c.id, c.content, c.doc_id, d.title as doc_title "
            "FROM chunks c JOIN documents d ON c.doc_id = d.id "
            "ORDER BY c.doc_id, c.chunk_index"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        _vocabulary = {}
        _idf = None
        _chunk_vectors = {}
        _chunk_metadata = {}
        return

    # Tokenize all chunks
    chunk_tokens: list[tuple[str, list[str], dict]] = []
    for row in rows:
        tokens = _tokenize(row["content"])
        meta = {
            "doc_id": row["doc_id"],
            "content": row["content"],
            "doc_title": row["doc_title"],
        }
        chunk_tokens.append((row["id"], tokens, meta))

    # Build vocabulary from all tokens
    vocab_counter: Counter = Counter()
    for _, tokens, _ in chunk_tokens:
        vocab_counter.update(set(tokens))  # count document frequency

    # Keep top N terms by document frequency (or all if small)
    max_vocab = 10000
    most_common = vocab_counter.most_common(max_vocab)
    _vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

    vocab_size = len(_vocabulary)
    num_docs = len(chunk_tokens)

    # Compute IDF: log(N / df) with smoothing
    _idf = np.zeros(vocab_size, dtype=np.float64)
    for word, idx in _vocabulary.items():
        df = vocab_counter[word]
        _idf[idx] = math.log((num_docs + 1) / (df + 1)) + 1  # smoothed IDF

    # Compute TF-IDF vector for each chunk
    _chunk_vectors = {}
    _chunk_metadata = {}
    for chunk_id, tokens, meta in chunk_tokens:
        tf = _compute_tf(tokens)
        vec = np.zeros(vocab_size, dtype=np.float64)
        for word, count in tf.items():
            if word in _vocabulary:
                idx = _vocabulary[word]
                vec[idx] = count * _idf[idx]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        _chunk_vectors[chunk_id] = vec
        _chunk_metadata[chunk_id] = meta


def compute_embedding(text: str) -> list[float] | None:
    """Compute a TF-IDF embedding vector for the given text.

    If the vocabulary hasn't been built yet, returns None.

    Args:
        text: Input text to embed.

    Returns:
        List of floats representing the TF-IDF vector, or None.
    """
    if not _vocabulary or _idf is None:
        return None

    tokens = _tokenize(text)
    tf = _compute_tf(tokens)
    vec = np.zeros(len(_vocabulary), dtype=np.float64)
    for word, count in tf.items():
        if word in _vocabulary:
            idx = _vocabulary[word]
            vec[idx] = count * _idf[idx]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def search_chunks(query: str, limit: int = 5) -> list[dict]:
    """Search for chunks matching the query using cosine similarity.

    Args:
        query: Search query text.
        limit: Maximum number of results.

    Returns:
        List of dicts with chunk_id, content, score, doc_title, doc_id.
    """
    if not _chunk_vectors or _idf is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    tf = _compute_tf(tokens)
    query_vec = np.zeros(len(_vocabulary), dtype=np.float64)
    for word, count in tf.items():
        if word in _vocabulary:
            idx = _vocabulary[word]
            query_vec[idx] = count * _idf[idx]

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    query_vec = query_vec / query_norm

    # Compute cosine similarity with all chunk vectors
    scores: list[tuple[str, float]] = []
    for chunk_id, chunk_vec in _chunk_vectors.items():
        score = float(np.dot(query_vec, chunk_vec))
        if score > 0.0:
            scores.append((chunk_id, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Log the search action
    _log_usage("search", query)

    results: list[dict] = []
    for chunk_id, score in scores[:limit]:
        meta = _chunk_metadata[chunk_id]
        results.append({
            "chunk_id": chunk_id,
            "content": meta["content"],
            "score": round(score, 4),
            "doc_title": meta["doc_title"],
            "doc_id": meta["doc_id"],
        })

    return results


def _log_usage(action: str, query: str | None = None, response_time_ms: int | None = None) -> None:
    """Log a usage event to the database."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO usage_log (id, action, query, response_time_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, action, query, response_time_ms, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        # Usage logging should not break the main flow
        conn.rollback()
    finally:
        conn.close()
