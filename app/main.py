from __future__ import annotations


import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .embeddings import embed_text
from .generative_llm import generate_answer as generate_rag_answer
from .reranker import rerank_texts
from .splitter import recursive_split
from .vectorstore import cag_check, get_qdrant_client, upsert_chunks
from .vectorstore import ensure_collection as ensure_qdrant_collection

load_dotenv()


# Pydantic models for API
class IngestTextRequest(BaseModel):
    text: str
    source: Optional[str] = "upload"


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    answerable: bool
    cag_max_score: float
    retrieved_contexts: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Settings (simplified, assume from config or inline)
# -----------------------------------------------------------------------------


class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")
    cag_threshold: float = float(os.getenv("CAG_THRESHOLD", "0.5"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    reranked_top_k: int = int(os.getenv("RERANKED_TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    static_response_message: str = os.getenv(
        "STATIC_RESPONSE_MESSAGE",
        "На основе имеющейся у меня базы знаний, я не могу предоставить ответ на ваш вопрос.",
    )


settings = Settings()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF."""
    if not content:
        raise ValueError("Empty PDF content")
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    texts = [page.extract_text() or "" for page in reader.pages]
    full = "\n\n".join(t.strip() for t in texts if t.strip())
    if not full:
        raise ValueError("No extractable text in PDF")
    return full


def build_points_for_qdrant(
    doc_id: str, chunks: List[str], vectors: List[List[float]], source: str
) -> List[Dict[str, Any]]:
    """Build points for Qdrant upsert."""
    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors count mismatch")
    import uuid

    points = []
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        if not text.strip():
            continue
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{idx}"))
        points.append(
            {
                "id": point_id,
                "vector": vec,
                "payload": {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "text": text.strip(),
                    "source": source,
                },
            }
        )
    if not points:
        raise ValueError("No valid chunks to upsert")
    return points


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------


def ingest_document_text(text: str, source: str = "upload") -> Dict[str, Any]:
    """Ingest text document."""
    from .embeddings import embed_texts

    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")
    doc_id = str(uuid4())
    chunks = recursive_split(
        text,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise ValueError("No chunks produced")
    vectors = embed_texts(chunks)
    points = build_points_for_qdrant(doc_id, chunks, vectors, source)
    upsert_chunks(points)
    return {"doc_id": doc_id, "num_chunks": len(points)}


def ingest_document_pdf(pdf_bytes: bytes, source: str) -> Dict[str, Any]:
    """Ingest PDF document."""
    text = extract_text_from_pdf(pdf_bytes)
    return ingest_document_text(text, source or "upload")


# -----------------------------------------------------------------------------
# Query pipeline
# -----------------------------------------------------------------------------


def perform_cag_check(query: str) -> float:
    """Perform CAG check: Get max similarity score."""
    query_vec = embed_text(query)
    return cag_check(query_vec)


def retrieve_and_rerank(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve top_k, rerank, build context."""
    query_vec = embed_text(query)
    client = get_qdrant_client()
    # Retrieve more initially for reranking
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vec,
        limit=settings.retrieval_top_k,
        with_payload=True,
        with_vectors=False,
    )
    retrieved_texts = [
        r.payload.get("text", "").strip()
        for r in results
        if r.payload and r.payload.get("text")
    ]
    retrieved_metadata = [
        {
            "text": r.payload.get("text", ""),
            "score": float(getattr(r, "score", 0.0)),
            "doc_id": r.payload.get("doc_id"),
            "chunk_index": r.payload.get("chunk_index"),
            "source": r.payload.get("source"),
        }
        for r in results
        if r.payload and r.payload.get("text")
    ]
    # Rerank
    if len(retrieved_texts) > settings.reranked_top_k:
        reranked_texts = rerank_texts(query, retrieved_texts)
        reranked_texts = reranked_texts[: settings.reranked_top_k]
        # Filter metadata to match reranked
        reranked_metadata = [
            m for m in retrieved_metadata if m["text"].strip() in reranked_texts
        ]
    else:
        reranked_texts = retrieved_texts[: settings.reranked_top_k]
        reranked_metadata = retrieved_metadata[: settings.reranked_top_k]

    context = "\n\n".join(reranked_texts)
    if len(context) > settings.max_context_chars:
        context = context[: settings.max_context_chars]
    return context, reranked_metadata


def handle_rag_query(query: str) -> Dict[str, Any]:
    """Handle full RAG query."""
    context, retrieved_contexts = retrieve_and_rerank(query)
    if not context.strip():
        return {
            "answer": settings.static_response_message,
            "answerable": False,
            "retrieved_contexts": [],
        }
    answer = generate_rag_answer(query, context)
    # For simplicity, assume answerable if answer is generated (LLM follows instructions)
    answerable = not (answer.startswith("Я не могу") or "не могу" in answer.lower())
    return {
        "answer": answer,
        "answerable": answerable,
        "retrieved_contexts": retrieved_contexts,
    }


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(title="RAG Backend with CAG and Reranking", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    ensure_qdrant_collection()


# Ingest endpoints
@app.post("/ingest/text")
async def ingest_text(req: IngestTextRequest) -> IngestResponse:
    try:
        result = ingest_document_text(req.text, req.source or "upload")
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    content = await file.read()
    try:
        result = ingest_document_pdf(content, filename or "upload")
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Query endpoint
@app.post("/query")
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query required")

    max_similarity = perform_cag_check(q)

    if max_similarity <= settings.cag_threshold:
        # No RAG needed
        return QueryResponse(
            answer=settings.static_response_message,
            answerable=False,
            cag_max_score=max_similarity,
            retrieved_contexts=[],
        )
    else:
        # Perform RAG
        result = handle_rag_query(q)
        return QueryResponse(
            answer=result["answer"],
            answerable=result["answerable"],
            cag_max_score=max_similarity,
            retrieved_contexts=result["retrieved_contexts"],
        )
