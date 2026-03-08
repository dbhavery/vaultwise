# Vaultwise
[![CI](https://github.com/dbhavery/vaultwise/actions/workflows/ci.yml/badge.svg)](https://github.com/dbhavery/vaultwise/actions/workflows/ci.yml)

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Tests Passing](https://img.shields.io/badge/tests-51%20passing-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)

A RAG-powered knowledge management platform that ingests organizational documents, enables semantic search through a **from-scratch TF-IDF implementation**, provides AI-driven Q&A with source citations, automatically generates training materials and quizzes, and surfaces knowledge gaps where documentation is lacking. Built for teams that need to turn scattered documents into searchable, actionable knowledge.

---

## Features

- **Document Ingestion** -- Upload text, Markdown, or Python files. Documents are automatically chunked with configurable overlap for optimal retrieval.
- **Semantic Search** -- From-scratch TF-IDF vectorization with cosine similarity ranking. No third-party search libraries; the entire pipeline is implemented with NumPy.
- **AI-Powered Q&A** -- Retrieval-augmented generation using Ollama (local LLM). Questions are answered with source citations and confidence scores.
- **Training Material Generation** -- Automatically synthesize knowledge articles from multiple source documents, then generate multiple-choice quizzes from those articles.
- **Knowledge Gap Detection** -- Low-confidence answers are tracked as knowledge gaps, surfacing topics where documentation is missing or insufficient.
- **Analytics Dashboard** -- Usage statistics, confidence trends, and gap frequency analysis via a built-in REST API and static dashboard.
- **Seed Data** -- Ships with realistic demo content (employee handbook, API docs, security policies) so the platform is usable out of the box.

## How Search Works

Vaultwise implements TF-IDF (Term Frequency-Inverse Document Frequency) **from scratch** using only NumPy -- no scikit-learn, no Elasticsearch, no vector database.

The pipeline:

1. **Tokenization** -- Text is lowercased, split into alphanumeric tokens, and filtered through a stop-word list for better discrimination.
2. **Vocabulary Construction** -- Document frequency is counted across all chunks. The top 10,000 terms by document frequency form the vocabulary.
3. **IDF Computation** -- Each term receives a smoothed IDF weight: `log((N + 1) / (df + 1)) + 1`, where `N` is the total chunk count and `df` is the document frequency.
4. **TF-IDF Vectors** -- Each chunk is represented as a sparse vector where each dimension is `term_count * idf_weight`. Vectors are L2-normalized for cosine similarity.
5. **Query Matching** -- The query is vectorized through the same pipeline, then scored against all chunk vectors via `np.dot()`. Results are ranked by cosine similarity.

This approach demonstrates understanding of information retrieval fundamentals without leaning on opaque library calls.

## Architecture

```
Client Request
      |
      v
 [FastAPI Server]  ──── /api/documents   ──> Ingest (chunk + embed)
      |                  /api/search      ──> TF-IDF cosine similarity
      |                  /api/ask         ──> RAG: retrieve + LLM generate
      |                  /api/articles    ──> Training article generation
      |                  /api/quizzes     ──> Quiz generation from articles
      |                  /api/analytics   ──> Usage stats + knowledge gaps
      |
      v
 [SQLite + WAL]         [Ollama (local LLM)]
  - documents            - qwen3:8b
  - chunks               - context-grounded answers
  - questions             - article synthesis
  - articles              - quiz generation
  - quizzes
  - knowledge_gaps
  - usage_log
```

## Tech Stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Framework     | FastAPI + Uvicorn                 |
| Search Engine | TF-IDF + NumPy (from scratch)    |
| LLM Backend   | Ollama (qwen3:8b, local)         |
| Database      | SQLite with WAL mode              |
| HTTP Client   | httpx (async-capable)             |
| Validation    | Pydantic v2                       |
| Testing       | pytest + pytest-asyncio           |
| Language      | Python 3.10+                      |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dbhavery/vaultwise.git
cd vaultwise

# Install with dev dependencies
pip install -e ".[dev]"

# Start the server (seeds demo data on first run)
python -m vaultwise.main
```

The server starts at `http://localhost:8090`. The dashboard is served at `http://localhost:8090/dashboard/`.

**Optional:** For AI-powered Q&A and training generation, install and run [Ollama](https://ollama.com) with a model:

```bash
ollama pull qwen3:8b
```

Without Ollama, the platform falls back to extractive summaries from retrieved chunks.

## API Endpoints

### Health

| Method | Endpoint   | Description                       |
|--------|------------|-----------------------------------|
| GET    | `/health`  | Status check with document counts |

### Documents

| Method | Endpoint                  | Description                          |
|--------|---------------------------|--------------------------------------|
| GET    | `/api/documents`          | List documents (paginated)           |
| POST   | `/api/documents`          | Upload via multipart form            |
| POST   | `/api/documents/json`     | Upload via JSON body                 |
| GET    | `/api/documents/{doc_id}` | Get document with chunks             |
| DELETE | `/api/documents/{doc_id}` | Delete document and its chunks       |

### Search

| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| POST   | `/api/search`  | Semantic search over document chunks     |

### Q&A

| Method | Endpoint          | Description                                |
|--------|-------------------|--------------------------------------------|
| POST   | `/api/ask`        | Ask a question, get answer with sources    |
| GET    | `/api/questions`  | List recent questions (paginated)          |

### Training Materials

| Method | Endpoint                      | Description                          |
|--------|-------------------------------|--------------------------------------|
| GET    | `/api/articles`               | List articles                        |
| POST   | `/api/articles/generate`      | Generate article from documents      |
| GET    | `/api/articles/{article_id}`  | Get article by ID                    |
| PATCH  | `/api/articles/{article_id}`  | Update article status/content        |
| GET    | `/api/quizzes`                | List quizzes                         |
| POST   | `/api/quizzes/generate`       | Generate quiz from article           |
| GET    | `/api/quizzes/{quiz_id}`      | Get quiz by ID                       |

### Analytics

| Method | Endpoint                       | Description                        |
|--------|--------------------------------|------------------------------------|
| GET    | `/api/analytics/overview`      | Dashboard statistics               |
| GET    | `/api/analytics/gaps`          | Knowledge gaps by frequency        |
| PATCH  | `/api/analytics/gaps/{gap_id}` | Update gap status                  |
| GET    | `/api/analytics/usage`         | Usage stats over N days            |

## Testing

```bash
# Run the full suite
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_search.py -v
```

All 51 tests use isolated temporary databases via the `_use_temp_db` fixture -- no test pollution, no cleanup required.

## Project Structure

```
vaultwise/
├── vaultwise/
│   ├── __init__.py         # Package version
│   ├── main.py             # FastAPI app, routes, entry point
│   ├── models.py           # Pydantic request/response schemas
│   ├── database.py         # SQLite connection, schema, migrations
│   ├── ingest.py           # Document upload, chunking, embedding
│   ├── search.py           # TF-IDF index, cosine similarity search
│   ├── qa.py               # RAG Q&A, confidence scoring, gap detection
│   ├── training.py         # Article and quiz generation
│   ├── analytics.py        # Usage stats, knowledge gap tracking
│   └── seed.py             # Demo data seeder
├── tests/
│   ├── conftest.py         # Shared fixtures (temp DB, test clients)
│   ├── test_ingest.py      # Ingestion and chunking tests
│   ├── test_search.py      # TF-IDF index and search tests
│   ├── test_qa.py          # Q&A pipeline tests
│   ├── test_training.py    # Article/quiz generation tests
│   └── test_analytics.py   # Analytics and gap tracking tests
├── dashboard/
│   └── index.html          # Static analytics dashboard
├── pyproject.toml          # Project metadata and dependencies
├── LICENSE                 # MIT License
└── README.md
```

## License

[MIT](LICENSE) -- Copyright 2026 Don Havery
