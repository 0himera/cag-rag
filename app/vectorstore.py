import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant vector store.

    Values are intentionally simple and environment-driven so this module
    can be reused independently from the rest of the application.
    """

    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")

    # jina-embeddings-v4 (via current Jina API) returns 2048-dim vectors.
    # Keep configurable so it can be overridden if the model changes.

    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "2048"))

    distance: str = os.getenv("QDRANT_DISTANCE", "Cosine")


_config = QdrantConfig()
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Get a singleton Qdrant client instance.

    This is a lightweight helper to avoid recreating the client for each call.
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            url=_config.url,
            api_key=_config.api_key,
        )
    return _client


def _to_distance_enum(distance: str) -> qmodels.Distance:
    """
    Map a string distance name to the Qdrant Distance enum.
    Defaults to COSINE for unknown values.
    """
    normalized = (distance or "").strip().lower()
    if normalized in ("cosine", "cos"):
        return qmodels.Distance.COSINE
    if normalized in ("euclid", "euclidean", "l2"):
        return qmodels.Distance.EUCLID
    if normalized in ("dot", "dotproduct", "inner"):
        return qmodels.Distance.DOT
    return qmodels.Distance.COSINE


def ensure_collection() -> None:
    """
    Ensure that the target collection exists with the expected configuration.

    - If the collection does not exist, it will be created.
    - If it exists, no destructive action is taken.
    """
    client = get_qdrant_client()
    distance_enum = _to_distance_enum(_config.distance)

    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if _config.collection_name not in existing:
        client.create_collection(
            collection_name=_config.collection_name,
            vectors_config=qmodels.VectorParams(
                size=_config.vector_size,
                distance=distance_enum,
            ),
        )


def upsert_chunks(
    chunks: Iterable[Dict[str, Any]],
    *,
    collection_name: Optional[str] = None,
) -> None:
    """
    Upsert multiple chunks into Qdrant.

    Each chunk dict must contain:
      - id: int | str
      - vector: list[float]
      - payload: dict (should at least contain "text")

    Example chunk:
      {
        "id": "doc-uuid-0",
        "vector": [...],
        "payload": {
            "text": "chunk text",
            "source": "upload",
            "doc_id": "doc-uuid",
            "chunk_index": 0
        }
      }
    """
    cn = collection_name or _config.collection_name
    client = get_qdrant_client()

    points: Sequence[qmodels.PointStruct] = []
    for c in chunks:
        if "id" not in c or "vector" not in c:
            raise ValueError("Each chunk must have 'id' and 'vector' fields")
        payload = c.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("Chunk 'payload' must be a dict if provided")

        points.append(
            qmodels.PointStruct(
                id=c["id"],
                vector=c["vector"],
                payload=payload,
            )
        )

    if not points:
        return

    client.upsert(
        collection_name=cn,
        points=points,
    )


def search_vectors(
    query_vector: Sequence[float],
    top_k: int = 5,
    *,
    collection_name: Optional[str] = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    score_threshold: Optional[float] = None,
) -> List[qmodels.ScoredPoint]:
    """
    Perform a similarity search in Qdrant.

    Args:
        query_vector: The embedding vector for the query.
        top_k: Number of top results to retrieve.
        collection_name: Optional override for the collection name.
        with_payload: Whether to include payloads in results.
        with_vectors: Whether to include stored vectors.
        score_threshold: Optional minimum similarity score.

    Returns:
        List of ScoredPoint objects.
    """
    if not query_vector:
        raise ValueError("query_vector must not be empty")

    cn = collection_name or _config.collection_name
    client = get_qdrant_client()

    results = client.search(
        collection_name=cn,
        query_vector=list(query_vector),
        limit=top_k,
        with_payload=with_payload,
        with_vectors=with_vectors,
        score_threshold=score_threshold,
    )
    return results


def delete_collection(collection_name: Optional[str] = None) -> None:
    """
    Delete a collection, primarily for testing or maintenance.

    This is not used in the main RAG flow but can be handy in tests.
    """
    cn = collection_name or _config.collection_name
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=cn)
    except Exception:
        # Intentionally swallow errors to avoid breaking callers if collection does not exist.
        # For stricter behavior, handle specific exceptions or re-raise.
        pass


def cag_check(query_vector: Sequence[float]) -> Dict[str, Any]:
    """
    Perform improved CAG (Context-Aware Gate) check with multiple metrics.
    
    Returns detailed information about the check:
    - should_rag: bool - whether to proceed with RAG
    - max_score: float - highest similarity score
    - avg_score: float - average similarity of top results  
    - num_results: int - number of results found
    - reason: str - explanation of decision
    """
    if not query_vector:
        return {
            "should_rag": False,
            "max_score": 0.0,
            "avg_score": 0.0,
            "num_results": 0,
            "reason": "no_query_vector"
        }

    cn = _config.collection_name
    client = get_qdrant_client()
    
    # Get top 3 results instead of just 1 for better analysis
    try:
        results = client.search(
            collection_name=cn,
            query_vector=list(query_vector),
            limit=3,  # Get multiple results for analysis
            with_payload=False,
            with_vectors=False,
            score_threshold=None,
        )
        
        if not results:
            return {
                "should_rag": False,
                "max_score": 0.0,
                "avg_score": 0.0,
                "num_results": 0,
                "reason": "no_results"
            }
        
        scores = [float(r.score) for r in results if r.score is not None]
        if not scores:
            return {
                "should_rag": False,
                "max_score": 0.0,
                "avg_score": 0.0,
                "num_results": 0,
                "reason": "no_valid_scores"
            }
        
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        num_results = len(scores)
        
        # Multi-criteria decision making
        # Default threshold from config, but can be overridden
        threshold = float(os.getenv("CAG_THRESHOLD", "0.5"))
        
        # More sophisticated logic:
        # 1. Max score must exceed threshold
        # 2. Average score should be reasonable (70% of threshold)
        # 3. Need at least 2 results for confidence
        should_rag = (
            max_score > threshold and
            avg_score > threshold * 0.7 and
            num_results >= 2
        )
        
        if should_rag:
            reason = "passed_all_criteria"
        elif max_score <= threshold:
            reason = "max_score_below_threshold"
        elif avg_score <= threshold * 0.7:
            reason = "avg_score_too_low"
        else:
            reason = "insufficient_results"
            
        return {
            "should_rag": should_rag,
            "max_score": max_score,
            "avg_score": avg_score,
            "num_results": num_results,
            "reason": reason
        }
        
    except Exception as e:
        # If collection doesn't exist or search fails, return safe defaults
        return {
            "should_rag": False,
            "max_score": 0.0,
            "avg_score": 0.0,
            "num_results": 0,
            "reason": f"search_error: {str(e)}"
        }
