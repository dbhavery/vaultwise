"""Pydantic models for Vaultwise request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# --- Document models ---

class DocumentCreate(BaseModel):
    """Request body for creating a document via JSON."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    doc_type: str = Field(default="text")
    source: str = Field(default="upload")


class ChunkOut(BaseModel):
    """A single chunk returned in document detail."""
    id: str
    content: str
    chunk_index: int


class DocumentOut(BaseModel):
    """Response model for a document."""
    id: str
    title: str
    source: str
    content: str
    doc_type: str
    word_count: int
    created_at: str
    updated_at: str


class DocumentDetail(DocumentOut):
    """Document with its chunks."""
    chunks: list[ChunkOut] = []


class DocumentList(BaseModel):
    """Paginated document list."""
    documents: list[DocumentOut]
    total: int


# --- Search models ---

class SearchRequest(BaseModel):
    """Request body for search."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    """A single search result."""
    chunk_id: str
    content: str
    score: float
    doc_title: str
    doc_id: str


class SearchResponse(BaseModel):
    """Response for search endpoint."""
    results: list[SearchResult]
    query: str


# --- QA models ---

class AskRequest(BaseModel):
    """Request body for asking a question."""
    query: str = Field(..., min_length=1)


class SourceRef(BaseModel):
    """Reference to a source document in an answer."""
    doc_id: str
    title: str
    excerpt: str


class AskResponse(BaseModel):
    """Response for ask endpoint."""
    answer: str
    sources: list[SourceRef]
    confidence: float
    question_id: str


# --- Training models ---

class ArticleGenerateRequest(BaseModel):
    """Request to generate an article from documents."""
    doc_ids: list[str] = Field(..., min_length=1)


class ArticleOut(BaseModel):
    """Response model for an article."""
    id: str
    title: str
    content: str
    source_doc_ids: Optional[str] = None
    status: str
    auto_generated: int
    created_at: str


class ArticleUpdateRequest(BaseModel):
    """Request to update an article."""
    status: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None


class QuizGenerateRequest(BaseModel):
    """Request to generate a quiz from an article."""
    article_id: str


class QuizQuestion(BaseModel):
    """A single quiz question."""
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class QuizOut(BaseModel):
    """Response model for a quiz."""
    id: str
    article_id: Optional[str] = None
    title: str
    questions_json: str
    created_at: str


# --- Analytics models ---

class OverviewStats(BaseModel):
    """Overview analytics."""
    total_docs: int
    total_questions: int
    avg_confidence: float
    gaps_count: int
    questions_today: int


class KnowledgeGapOut(BaseModel):
    """A knowledge gap entry."""
    id: str
    topic: str
    frequency: int
    sample_queries: Optional[str] = None
    status: str
    last_asked: str
    created_at: str


class UsageEntry(BaseModel):
    """A single usage log entry."""
    date: str
    count: int


class UsageStats(BaseModel):
    """Usage statistics."""
    queries_per_day: list[UsageEntry]
    total_actions: int
