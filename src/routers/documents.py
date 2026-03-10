"""Document CRUD and ingestion router."""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from src.models import DocumentCreate
from src.services.ingestion import (
    delete_document,
    get_document,
    ingest_document,
    list_documents,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def api_list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List documents with pagination."""
    docs, total = list_documents(limit=limit, offset=offset)
    return {"documents": docs, "total": total}


@router.post("")
async def api_create_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
) -> dict:
    """Upload a document via multipart form (file or title+content)."""
    if file is not None:
        file_content = (await file.read()).decode("utf-8", errors="replace")
        file_title = title or file.filename or "Untitled"
        result = ingest_document(
            title=file_title,
            content=file_content,
            source=source or "upload",
            doc_type=doc_type,
        )
    elif title and content:
        result = ingest_document(
            title=title,
            content=content,
            source=source or "upload",
            doc_type=doc_type,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or title+content")

    return result


@router.post("/json")
def api_create_document_json(doc: DocumentCreate) -> dict:
    """Upload a document via JSON body."""
    return ingest_document(
        title=doc.title,
        content=doc.content,
        source=doc.source,
        doc_type=doc.doc_type,
    )


@router.get("/{doc_id}")
def api_get_document(doc_id: str) -> dict:
    """Get a document with its chunks."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
def api_delete_document(doc_id: str) -> dict:
    """Delete a document and its chunks."""
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "id": doc_id}
