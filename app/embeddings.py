from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class JinaEmbeddingConfig:
    """
    Configuration for calling the Jina embeddings API.

    This module is responsible for:
      - Creating embeddings for documents and queries.
      - Using the `jina-embeddings-v4` model via Jina's HTTP API.
      - Returning embeddings as plain Python lists of floats.

    Environment variables:

      JINA_API_KEY
          Your Jina AI API key (required).
          Example: "jina_xxx..."

      JINA_EMBEDDING_MODEL
          Model name to use. Default: "jina-embeddings-v4"

      JINA_EMBEDDING_TASK
          Task type for embeddings. Common values:
            - "text-matching" (default, good for RAG / similarity)
            - "text-classification", ...
          See Jina docs for supported tasks.

      JINA_EMBEDDING_ENDPOINT
          Override base URL for embeddings if needed.
          Default: "https://api.jina.ai/v1/embeddings"

      JINA_EMBEDDING_TIMEOUT
          Request timeout in seconds. Default: 30

      JINA_EMBEDDING_EXPECTED_DIM
          Expected embedding dimension (for validation).
          For jina-embeddings-v4 it's typically 1024.
          Default: 1024
    """

    api_key: str = os.getenv("JINA_API_KEY", "")
    model: str = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
    task: str = os.getenv("JINA_EMBEDDING_TASK", "text-matching")
    endpoint: str = os.getenv(
        "JINA_EMBEDDING_ENDPOINT", "https://api.jina.ai/v1/embeddings"
    )
    timeout: int = int(os.getenv("JINA_EMBEDDING_TIMEOUT", "30"))
    expected_dim: int = int(os.getenv("JINA_EMBEDDING_EXPECTED_DIM", "2048"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "JINA_API_KEY is not set. Configure your Jina AI API key for embeddings."
            )
        if not self.model:
            raise RuntimeError(
                "JINA_EMBEDDING_MODEL is not configured. Set a valid Jina embeddings model."
            )
        if not self.endpoint:
            raise RuntimeError(
                "JINA_EMBEDDING_ENDPOINT is not configured. Set a valid Jina embeddings endpoint."
            )


_config = JinaEmbeddingConfig()


def _build_headers() -> Dict[str, str]:
    """
    Build HTTP headers for Jina embeddings API.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_config.api_key}",
    }


def _build_payload(texts: List[str]) -> Dict[str, Any]:
    """
    Build request payload for Jina embeddings API.

    The API expects:
      {
        "model": "jina-embeddings-v4",
        "task": "text-matching",
        "input": [
          { "text": "..." },
          ...
        ]
      }
    """
    input_items = [{"text": t} for t in texts]
    return {
        "model": _config.model,
        "task": _config.task,
        "input": input_items,
    }


def _parse_embeddings_response(
    response_json: Dict[str, Any],
    expected_count: int,
) -> List[List[float]]:
    """
    Parse embeddings from Jina API response.

    Expected response structure (simplified):

      {
        "data": [
          {
            "embedding": [...],
            "index": 0,
            ...
          },
          ...
        ]
      }

    Returns:
      List[List[float]]: embedding vectors.

    Raises:
      RuntimeError on unexpected structures or mismatched counts.
    """
    if "data" not in response_json or not isinstance(response_json["data"], list):
        raise RuntimeError(f"Invalid Jina embeddings response format: {response_json}")

    data = response_json["data"]
    if len(data) != expected_count:
        # Not fatal, but suspicious. Log and continue with min(len(data), expected_count).
        logger.warning(
            "Jina embeddings response count (%d) != requested count (%d)",
            len(data),
            expected_count,
        )

    embeddings: List[List[float]] = []
    for item in data:
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Missing or invalid 'embedding' field in item: {item}")
        vec = [float(x) for x in emb]
        if _config.expected_dim and len(vec) != _config.expected_dim:
            logger.warning(
                "Embedding dim mismatch: expected %d, got %d",
                _config.expected_dim,
                len(vec),
            )
        embeddings.append(vec)

    if not embeddings:
        raise RuntimeError("No embeddings returned by Jina API.")

    return embeddings


def _encode_batch(texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """
    Internal helper: send batch of texts to Jina embeddings API and return vectors.
    Includes retry logic for transient failures.
    """
    if not texts:
        return []

    _config.validate()

    payload = _build_payload(texts)
    headers = _build_headers()

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                _config.endpoint,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=_config.timeout,
            )
            
            if resp.status_code == 200:
                break
            elif resp.status_code in (429, 503):  # Rate limit or service unavailable
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"API rate limited, retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    body_preview = resp.text[:500] if resp.text else ""
                    raise RuntimeError(
                        f"Jina embeddings API returned {resp.status_code} after retries: {body_preview}"
                    )
            else:
                body_preview = resp.text[:500] if resp.text else ""
                raise RuntimeError(
                    f"Jina embeddings API returned {resp.status_code}: {body_preview}"
                )
        except requests.exceptions.Timeout as exc:
            if attempt < max_retries - 1:
                logger.warning(f"API timeout, retrying (attempt {attempt + 1}/{max_retries})...")
                continue
            else:
                raise RuntimeError(f"Failed to call Jina embeddings API after {max_retries} retries: {exc}") from exc
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning(f"API call failed, retrying (attempt {attempt + 1}/{max_retries})...")
                continue
            else:
                raise RuntimeError(f"Failed to call Jina embeddings API: {exc}") from exc

    try:
        resp_json = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse Jina embeddings API response as JSON: {exc}"
        ) from exc

    return _parse_embeddings_response(resp_json, expected_count=len(texts))


def embed_text(text: str) -> List[float]:
    """
    Compute an embedding vector for a single text via Jina embeddings API.

    Args:
        text: Input text (non-empty).

    Returns:
        List[float]: Embedding vector.

    Raises:
        ValueError: If text is None or empty.
        RuntimeError: If remote call fails or response is invalid.
    """
    if text is None:
        raise ValueError("Text for embedding must not be None.")
    text = text.strip()
    if not text:
        raise ValueError("Text for embedding must not be empty.")

    return _encode_batch([text])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for multiple texts via Jina embeddings API.

    Args:
        texts: List of input texts (none of them should be None/empty).

    Returns:
        List[List[float]]: List of embedding vectors.

    Raises:
        ValueError: If list is None or any element is invalid.
        RuntimeError: If remote call fails or response is invalid.
    """
    if texts is None:
        raise ValueError("Texts list must not be None.")

    cleaned: List[str] = []
    for idx, t in enumerate(texts):
        if t is None:
            raise ValueError(f"Text at index {idx} is None.")
        s = str(t).strip()
        if not s:
            raise ValueError(f"Text at index {idx} is empty or whitespace.")
        cleaned.append(s)

    if not cleaned:
        return []

    return _encode_batch(cleaned)
