from __future__ import annotations

import io
import uuid
from typing import Any, Dict, List, Optional

from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file represented as bytes.

    This function performs a straightforward extraction of textual content from each page.
    Any pages without extractable text are skipped.

    :param pdf_bytes: Raw bytes of the uploaded PDF file.
    :return: Concatenated text content of the PDF.
    """
    if not pdf_bytes:
        raise ValueError("No PDF bytes provided for extraction.")

    try:
        # `PdfReader` accepts a file-like object; using BytesIO for in-memory bytes.
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to read PDF: {exc}") from exc

    texts: List[str] = []

    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            # Skip problematic pages but continue extracting remaining content.
            # For production use, consider structured logging instead of print.
            print(f"Warning: failed to extract text from page {page_idx}: {exc}")
            page_text = ""

        if page_text.strip():
            texts.append(page_text.strip())

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        raise ValueError("No extractable text found in the PDF document.")

    return full_text


def generate_document_id() -> str:
    """
    Generate a unique document identifier.

    This is intentionally simple and stable; caller can override if needed.
    """
    return str(uuid.uuid4())


def build_chunk_payload(
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    source: str = "upload",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct the payload metadata for a single chunk.

    :param doc_id: Unique document identifier.
    :param chunk_index: Index of this chunk within the document.
    :param chunk_text: The text content of this chunk.
    :param source: Source label (e.g. filename, user-defined tag).
    :param extra_metadata: Optional additional metadata to attach.
    :return: Payload dictionary suitable for storage in a vector DB.
    """
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "text": chunk_text,
        "source": source,
    }

    if extra_metadata:
        # User-defined metadata should not overwrite required keys unless explicit.
        for k, v in extra_metadata.items():
            if k not in payload:
                payload[k] = v

    return payload


def prepare_chunks_for_upsert(
    doc_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    source: str = "upload",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Combine text chunks and their embeddings into a list of upsertable point dicts.

    This function only prepares the data structure:
    it does not perform any actual database operations.

    Expected output schema for each point:
      {
        "id": "<doc_id>-<chunk_index>",
        "vector": [...],
        "payload": {
            "doc_id": str,
            "chunk_index": int,
            "text": str,
            "source": str,
            ... extra_metadata
        }
      }

    :param doc_id: Unique document identifier.
    :param chunks: List of chunk texts.
    :param embeddings: Corresponding list of embedding vectors, one per chunk.
    :param source: Source label of the document/chunks.
    :param extra_metadata: Optional extra metadata to include in each payload.
    :return: List of points ready for upsert into Qdrant (or similar).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
        )

    points: List[Dict[str, Any]] = []
    for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
        if not chunk_text or not str(chunk_text).strip():
            continue
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"Invalid embedding vector for chunk index {idx}.")

        point_id = f"{doc_id}-{idx}"
        payload = build_chunk_payload(
            doc_id=doc_id,
            chunk_index=idx,
            chunk_text=str(chunk_text).strip(),
            source=source,
            extra_metadata=extra_metadata,
        )

        points.append(
            {
                "id": point_id,
                "vector": list(vector),
                "payload": payload,
            }
        )

    if not points:
        raise ValueError("No valid chunks to upsert after preprocessing.")

    return points
