# Vaultwise

Knowledge management platform that ingests documents, searches them with a TF-IDF engine built from scratch using only NumPy, and generates training materials and quizzes -- turning scattered documentation into searchable, testable knowledge.

## Why I Built This

Organizations accumulate documentation across wikis, drives, and repos, but nobody can find anything when they need it. Search is either keyword-exact (useless) or requires deploying heavyweight infrastructure (Elasticsearch, vector databases). Vaultwise sits in the middle: a from-scratch TF-IDF implementation that delivers relevance-ranked results with zero ML dependencies, plus an AI layer that generates training content from the knowledge base so teams can actually learn from their own docs.

## What It Does

- **TF-IDF search engine from scratch** -- implemented using only NumPy with no scikit-learn, no Elasticsearch, no vector database. Full pipeline: tokenization, stop-word filtering, IDF weighting, L2-normalized vectors, cosine similarity ranking. Instant indexing, explainable relevance scores.
- **Document ingestion with chunking** -- handles PDF, DOCX, and plain text with configurable chunk sizes and overlap for optimal retrieval granularity
- **RAG-powered Q&A with citations** -- retrieves relevant chunks, generates answers grounded in source material via Ollama, and includes confidence scores and source references
- **Training material generation** -- synthesizes structured learning modules from multiple source documents, converting documentation into scannable training content
- **Quiz generation with difficulty scaling** -- creates multiple-choice assessments from source material with answer validation, supporting active recall over passive reading

## Architecture

```
Client Request
      |
      v
 [FastAPI Server]  ---- /api/documents   --> Ingest (chunk + index)
      |                  /api/search      --> TF-IDF cosine similarity
      |                  /api/ask         --> RAG: retrieve + LLM generate
      |                  /api/articles    --> Training article generation
      |                  /api/quizzes     --> Quiz generation from articles
      |                  /api/analytics   --> Usage stats + knowledge gaps
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

The search engine operates independently of any LLM. AI features (Q&A, training generation, quiz generation) are additive -- the platform is fully functional for document management and search without Ollama running.

## Key Technical Decisions

- **TF-IDF from scratch over scikit-learn/Whoosh** -- complete control over tokenization, stop-word filtering, and scoring. Every step is inspectable and debuggable. Also demonstrates understanding of information retrieval fundamentals without opaque library calls. Trade-off: no built-in optimizations like inverted index compression, but the 10,000-term vocabulary cap keeps memory bounded.
- **Quiz generation over simple Q&A** -- active recall (answering questions) produces stronger retention than passive reading. Generating assessments from source material closes the loop between "we documented it" and "the team actually knows it." Trade-off: quiz quality depends on prompt engineering to avoid trivial or ambiguous questions.
- **Flat categories over hierarchical taxonomy** -- simpler mental model for users, avoids the over-classification problem where documents end up in deeply nested categories nobody navigates. Trade-off: less granularity for large knowledge bases, but tagging handles that need without imposing structure.
- **Chunk overlap for retrieval** -- overlapping chunks prevent relevant content from being split across chunk boundaries. Trade-off: slightly more storage and index size, but retrieval quality improves significantly on multi-paragraph answers.

## Results & Metrics

- 20 API endpoints across documents, search, Q&A, training, quizzes, and analytics
- 7 database tables with WAL mode for concurrent read access
- 51 tests covering ingestion, TF-IDF indexing, search ranking, Q&A pipeline, training generation, and analytics
- TF-IDF vocabulary capped at 10,000 terms with smoothed IDF weighting
- Ships with realistic seed data (employee handbook, API docs, security policies) for immediate usability

## Live Demo

[HuggingFace Space](https://huggingface.co/spaces/dbhavery/vaultwise-knowledge)

## Quick Start

```bash
git clone https://github.com/dbhavery/vaultwise.git
cd vaultwise
pip install -e ".[dev]"

# Start the server (seeds demo data on first run)
python -m vaultwise.main
```

Server starts at `http://localhost:8090`. Dashboard at `/dashboard/`.

For AI-powered Q&A and training generation, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull qwen3:8b
```

Without Ollama, the platform falls back to extractive summaries from retrieved chunks.

## Lessons Learned

- **Tokenization matters more than the algorithm.** Stemming, stop-word removal, and n-gram handling have bigger impact on search quality than TF-IDF formula variants. Spent more time tuning the tokenizer than the scoring math.
- **Training content generation needs strong structural constraints.** Without explicit format requirements in the prompt, LLMs produce wall-of-text content instead of scannable learning modules. Had to specify section headers, bullet limits, and summary requirements to get usable output.
- **Knowledge gaps are a feature, not a failure.** Low-confidence answers reveal where documentation is missing. Tracking these systematically turns a search miss into an actionable improvement signal for the documentation team.

## Tests

51 tests across 5 modules. All tests use isolated temporary databases via fixtures -- no test pollution, no cleanup required.

```bash
python -m pytest tests/ -v
```
