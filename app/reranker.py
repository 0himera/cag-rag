from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class JinaRerankerConfig:
    """
    Configuration for Jina reranker API.

    Environment variables:
      JINA_RERANKER_API_KEY
          API key for reranker (defaults to JINA_API_KEY if empty).
      JINA_RERANKER_MODEL
          Model name, default: jina-reranker-v3-base-en
      JINA_RERANKER_ENDPOINT
          API endpoint, default: https://api.jina.ai/v1/rerank
      JINA_RERANKER_TOP_K
          Top k to return, default: 5
    """

    api_key: str = os.getenv("JINA_RERANKER_API_KEY") or os.getenv("JINA_API_KEY", "")
    model: str = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3-base-en")
    endpoint: str = os.getenv("JINA_RERANKER_ENDPOINT", "https://api.jina.ai/v1/rerank")
    top_k: int = int(os.getenv("JINA_RERANKER_TOP_K", "5"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "JINA_RERANKER_API_KEY is not set. Configure Jina reranker API key."
            )
        if not self.model:
            raise RuntimeError("JINA_RERANKER_MODEL is not configured.")


_config = JinaRerankerConfig()


def _build_rerank_payload(query: str, documents: List[str]) -> Dict[str, Any]:
    """
    Build payload for Jina reranker API.
    """
    return {
        "query": query,
        "documents": documents,
        "model": _config.model,
        "top_k": _config.top_k,
    }


def _build_rerank_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_config.api_key}",
    }


def rerank_texts(query: str, texts: List[str], max_retries: int = 3) -> List[str]:
    """
    Rerank a list of texts based on relevance to the query using Jina reranker.
    Includes retry logic for transient failures.

    Args:
        query: The query string.
        texts: List of text documents to rerank.
        max_retries: Maximum number of retry attempts.

    Returns:
        List of top-k reranked texts, ordered by relevance.
    """
    if not texts:
        return []
    if not query:
        return texts[: _config.top_k]

    _config.validate()

    payload = _build_rerank_payload(query, texts)
    headers = _build_rerank_headers()

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                _config.endpoint,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=30,
            )
            
            if resp.status_code == 200:
                break
            elif resp.status_code in (429, 503):  # Rate limit or service unavailable
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    body = resp.text[:500]
                    raise RuntimeError(f"Jina reranker API error {resp.status_code} after retries: {body}")
            else:
                body = resp.text[:500]
                raise RuntimeError(f"Jina reranker API error {resp.status_code}: {body}")
        except requests.exceptions.Timeout as exc:
            if attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Failed to call Jina reranker API after {max_retries} retries: {exc}") from exc
        except Exception as exc:
            if attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Failed to call Jina reranker API: {exc}") from exc

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Jina reranker response: {exc}") from exc

    # Validate response structure
    if "results" not in data:
        raise RuntimeError("Invalid Jina reranker response structure: missing 'results' field")
    
    if not isinstance(data["results"], list):
        raise RuntimeError("Invalid Jina reranker response: 'results' must be a list")

    results = data["results"][: _config.top_k]

    # Extract the text from each result with proper validation
    reranked_texts = []
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid result at index {idx}: expected dict, got {type(result)}")
        
        doc = result.get("document", "")
        if isinstance(doc, dict):
            text = doc.get("text", "")
        elif isinstance(doc, str):
            text = doc
        else:
            text = str(doc) if doc else ""
        
        text = text.strip()
        if text:  # Only add non-empty texts
            reranked_texts.append(text)

    if not reranked_texts:
        raise RuntimeError("No valid reranked texts extracted from API response")
    
    return reranked_texts
