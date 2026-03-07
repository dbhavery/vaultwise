# Vaultwise -- Knowledge Management Platform

Ingest company documents, search with AI, auto-generate training materials, and analyze knowledge gaps.

## Quick Start

```bash
pip install -e ".[dev]"
python -m vaultwise.main
```

Dashboard: http://localhost:8090/dashboard/

## API

- `GET /health` -- status and counts
- `POST /api/documents` -- upload documents
- `POST /api/search` -- semantic search
- `POST /api/ask` -- question answering
- `POST /api/articles/generate` -- generate training articles
- `POST /api/quizzes/generate` -- generate quizzes
- `GET /api/analytics/overview` -- analytics dashboard

## Tests

```bash
python -m pytest tests/ -v
```
