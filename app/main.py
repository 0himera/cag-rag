from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import logging

from .embeddings import embed_text
from .generative_llm import generate_answer as generate_rag_answer
from .reranker import rerank_texts
from .splitter import recursive_split
from .vectorstore import cag_check, get_qdrant_client, upsert_chunks
from .vectorstore import ensure_collection as ensure_qdrant_collection
from .config import settings

load_dotenv()

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


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
# Using centralized config from app.config
# -----------------------------------------------------------------------------

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


def perform_cag_check(query: str) -> Dict[str, Any]:
    """Perform CAG check: Get detailed similarity analysis."""
    query_vec = embed_text(query)
    return cag_check(query_vec)


def retrieve_and_rerank(query: str, query_vector: List[float] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve top_k, rerank, build context."""
    # Use provided query_vector or create new one
    if query_vector is None:
        query_vec = embed_text(query)
    else:
        query_vec = query_vector
        
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
        # Filter metadata to match reranked texts using proper mapping
        reranked_metadata = []
        for reranked_text in reranked_texts:
            # Find matching metadata by text content
            for metadata in retrieved_metadata:
                if metadata["text"].strip() == reranked_text.strip():
                    reranked_metadata.append(metadata)
                    break
    else:
        reranked_texts = retrieved_texts[: settings.reranked_top_k]
        reranked_metadata = retrieved_metadata[: settings.reranked_top_k]

    context = "\n\n".join(reranked_texts)
    if len(context) > settings.max_context_chars:
        context = context[: settings.max_context_chars]
    return context, reranked_metadata


def handle_rag_query(query: str, query_vector: List[float] = None) -> Dict[str, Any]:
    """Handle full RAG query."""
    start_time = time.time()
    logger.info(f"Starting RAG for query: '{query[:50]}...'")
    retrieval_start = time.time()
    context, retrieved_contexts = retrieve_and_rerank(query, query_vector)
    retrieval_time = time.time() - retrieval_start
    logger.info(f"Retrieval completed in {retrieval_time:.2f}s")
    if not context.strip():
        logger.warning(f"No context retrieved for query: '{query[:50]}...'")
        return {
            "answer": settings.static_response_message,
            "answerable": False,
            "retrieved_contexts": [],
        }
    generation_start = time.time()
    answer = generate_rag_answer(query, context)
    generation_time = time.time() - generation_start
    logger.info(f"Generation completed in {generation_time:.2f}s")
    # Use answerability classifier instead of keyword matching
    try:
        from .answerability import classify_answerability
        classification = classify_answerability(
            question=query,
            candidate_answer=answer,
            qa_score=0.8,  # Default confidence score
            contexts=retrieved_contexts
        )
        answerable = classification["answerable"]
        answer = classification["final_answer"]
    except Exception as e:
        # Fallback to simple keyword matching if classifier fails
        logger.warning(f"Answerability classifier failed: {e}, using fallback")
        answerable = not (answer.startswith("Я не могу") or "не могу" in answer.lower())
    total_time = time.time() - start_time
    logger.info(
        f"RAG completed for query: '{query[:50]}...', total time: {total_time:.2f}s"
    )
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
    start_time = time.time()
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query required")

    logger.info(f"Processing query: '{q[:50]}...'")
    cag_start = time.time()
    cag_result = perform_cag_check(q)
    cag_time = time.time() - cag_start
    logger.info(f"CAG check completed in {cag_time:.2f}s")
    logger.info(f"CAG result: {cag_result}")
    
    if not cag_result["should_rag"]:
        logger.info(
            f"CAG triggered static response for query: '{q[:50]}...' (reason: {cag_result['reason']}, score: {cag_result['max_score']:.2f})"
        )
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f}s")
        # No RAG needed
        return QueryResponse(
            answer=settings.static_response_message,
            answerable=False,
            cag_max_score=cag_result["max_score"],
            retrieved_contexts=[],
        )
    else:
        logger.info(
            f"CAG passed, proceeding to RAG for query: '{q[:50]}...' (max_score: {cag_result['max_score']:.2f}, avg_score: {cag_result['avg_score']:.2f})"
        )
        # Perform RAG with cached query vector
        result = handle_rag_query(q, query_vector=embed_text(q))
        total_time = time.time() - start_time
        logger.info(
            f"Query processing completed: '{q[:50]}...', total time: {total_time:.2f}s"
        )
        return QueryResponse(
            answer=result["answer"],
            answerable=result["answerable"],
            cag_max_score=cag_result["max_score"],
            retrieved_contexts=result["retrieved_contexts"],
        )
