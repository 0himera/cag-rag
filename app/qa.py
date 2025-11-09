from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from huggingface_hub import InferenceClient


@dataclass
class QAConfig:
    """
    Configuration for remote QA inference via Hugging Face Inference API.

    Environment variables:

      HF_API_KEY            - required; Hugging Face API token with inference access
      HF_QA_MODEL           - QA model id, default: "timpal0l/mdeberta-v3-base-squad2"
      HF_QA_PROVIDER        - provider identifier, default: "hf-inference"
      MAX_CONTEXT_CHARS     - hard cap on context length passed into QA model
    """

    api_key: str = os.getenv("HF_API_KEY", "")
    model: str = os.getenv("HF_QA_MODEL", "timpal0l/mdeberta-v3-base-squad2")
    provider: str = os.getenv("HF_QA_PROVIDER", "hf-inference")
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "HF_API_KEY is not set. Please provide a valid Hugging Face API token."
            )
        if not self.model:
            raise RuntimeError(
                "HF_QA_MODEL is not configured. Please specify a QA model id."
            )


_config = QAConfig()


def _get_client() -> InferenceClient:
    """
    Lazily create a Hugging Face InferenceClient configured for QA.
    """
    _config.validate()
    return InferenceClient(
        provider=_config.provider,
        api_key=_config.api_key,
    )


def _truncate_context(context: str) -> str:
    """
    Truncate context string to MAX_CONTEXT_CHARS to avoid overly long payloads.
    """
    if not context:
        return ""
    if len(context) <= _config.max_context_chars:
        return context
    return context[: _config.max_context_chars]


def generate_answer(question: str, context: str) -> Dict[str, Any]:
    """
    Call remote QA model via Hugging Face Inference API.

    Inputs:
      - question: user query
      - context: concatenated retrieved chunks

    Returns:
      {
        "answer": str,   # predicted answer text (may be empty)
        "score": float,  # model confidence score if returned, else 0.0
      }

    Notes:
      - If context is empty/whitespace, returns empty answer with score 0.0.
      - This function is intentionally minimal; upstream caller is responsible
        for retrieval and answerability classification.
    """
    question = (question or "").strip()
    context = (context or "").strip()

    if not question or not context:
        return {"answer": "", "score": 0.0}

    context = _truncate_context(context)

    client = _get_client()

    # The huggingface_hub InferenceClient.question_answering returns
    # a dict like: {"answer": "...", "score": 0.98, "start": ..., "end": ...}
    result = client.question_answering(
        question=question,
        context=context,
        model=_config.model,
    )

    # Be defensive: normalize result shape.
    if isinstance(result, dict):
        answer = (result.get("answer") or "").strip()
        score = float(result.get("score") or 0.0)
    else:
        # Unexpected format: loggable via caller; return conservative output.
        answer = ""
        score = 0.0

    return {
        "answer": answer,
        "score": score,
    }


def build_context_from_hits(
    hits: List[Dict[str, Any]],
    max_context_chars: int | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build concatenated context and normalized hits from retrieval results.

    Expected hit schema (flexible but recommended):
      {
        "text": str,
        "score": float | None,
        "doc_id": str | None,
        "chunk_index": int | None,
        "source": str | None,
      }

    Returns:
      (context_str, normalized_hits)
    """
    if max_context_chars is None:
        max_context_chars = _config.max_context_chars

    context_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []

    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        item = {
            "text": text,
            "score": float(h.get("score")) if h.get("score") is not None else None,
            "doc_id": h.get("doc_id"),
            "chunk_index": h.get("chunk_index"),
            "source": h.get("source"),
        }
        normalized.append(item)
        context_parts.append(text)

    context = "\n\n".join(context_parts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    return context, normalized


def run_rag_qa(
    query: str,
    retrieved_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    End-to-end helper:

      - Build context from retrieved hits
      - Call remote QA model via Hugging Face Inference API

    This does NOT do:
      - retrieval (caller provides hits)
      - answerability classification (external OpenRouter step in another module)

    Returns:
      {
        "question": str,
        "context": str,
        "answer": str,
        "qa_score": float,
        "hits": List[Dict[str, Any]],
      }
    """
    query = (query or "").strip()
    if not query:
        return {
            "question": "",
            "context": "",
            "answer": "",
            "qa_score": 0.0,
            "hits": [],
        }

    context, normalized_hits = build_context_from_hits(retrieved_hits)
    qa_result = generate_answer(query, context)

    return {
        "question": query,
        "context": context,
        "answer": qa_result.get("answer", ""),
        "qa_score": float(qa_result.get("score", 0.0)),
        "hits": normalized_hits,
    }


def rag_result_to_json(result: Dict[str, Any]) -> str:
    """
    Serialize RAG QA result to JSON (useful for logging/debugging).
    """

    def _default(o: Any) -> Any:
        if isinstance(o, (set,)):
            return list(o)
        return str(o)

    return json.dumps(result, ensure_ascii=False, default=_default, indent=2)
