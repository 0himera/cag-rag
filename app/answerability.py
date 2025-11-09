from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from .config import settings
from dotenv import load_dotenv
load_dotenv()

def _get_client() -> OpenAI:
    """
    Initialize an OpenAI-style client pointed at OpenRouter.
    Requires OPENROUTER_API_KEY to be set in the environment (via settings).
    """
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. "
            "Set it in the environment to enable answerability classification."
        )

    # Note:
    # - `base_url` points to OpenRouter's OpenAI-compatible endpoint.
    # - `api_key` is your OpenRouter API key (never hardcode it).
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


def classify_answerability(
    question: str,
    candidate_answer: str,
    qa_score: float,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Use an external LLM (via OpenRouter) to decide if the RAG answer is grounded.

    Inputs:
      - question: original user query
      - candidate_answer: answer produced by the local QA model
      - qa_score: confidence score from the QA model (float in [0,1] or model-specific)
      - contexts: list of retrieved chunks with optional fields:
          {
            "text": str,
            "score": float | None,
            "source": str | None,
            "doc_id": str | None,
            "chunk_index": int | None
          }

    Returns a dict with shape:
      {
        "answerable": bool,
        "reason": str,
        "final_answer": str,
      }

    Behavior:
      - If answerable is True:
          `final_answer` should be a faithful and concise variant of candidate_answer,
          grounded only in provided contexts.
      - If answerable is False:
          `final_answer` should politely state that the system cannot answer based
          on available documents (no hallucinated details).
    """
    client = _get_client()

    # Trim and sanitize contexts for prompt compactness.
    # Do not leak excessive data to the classifier, but keep enough for grounding.
    compact_contexts: List[Dict[str, Any]] = []
    for c in contexts:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        compact_contexts.append(
            {
                "text": text[
                    :400
                ],  # ensure bounded size per chunk in classifier prompt
                "score": c.get("score"),
                "source": c.get("source"),
                "doc_id": c.get("doc_id"),
                "chunk_index": c.get("chunk_index"),
            }
        )

    # System message enforces strict JSON output and grounding rules.
    system_prompt = (
        "You are an answerability and hallucination classifier for a Retrieval-Augmented Generation (RAG) backend.\n"
        "You will receive:\n"
        "1) The user's question.\n"
        "2) Retrieved context chunks from a document store.\n"
        "3) A candidate answer proposed by a QA model.\n"
        "4) The QA model's confidence score.\n\n"
        "Your tasks:\n"
        "- Decide if the candidate answer is well-supported ONLY by the provided contexts.\n"
        "- Mark `answerable` as true if the answer is directly grounded and non-contradictory.\n"
        "- Mark `answerable` as false if:\n"
        "  * The information is missing, weakly supported, or contradicted by the contexts, OR\n"
        "  * The QA score is very low and you cannot verify grounding, OR\n"
        "  * The candidate answer is empty, vague, or clearly fabricated relative to context.\n"
        "- When `answerable` is true:\n"
        "  * `final_answer` should either repeat the candidate answer or slightly refine it for clarity,\n"
        "    but MUST NOT introduce new facts that are not clearly supported by the contexts.\n"
        "- When `answerable` is false:\n"
        "  * `final_answer` must be a brief, honest message like:\n"
        "    'I cannot answer this reliably based on the available documents.' or similar.\n"
        "  * Do NOT fabricate or guess missing information.\n\n"
        "Output format:\n"
        "- Respond with a single valid JSON object, no extra text.\n"
        "- Required keys:\n"
        "  * answerable: boolean\n"
        "  * reason: string (short rationale referencing support/lack of support in context)\n"
        "  * final_answer: string\n"
        "Do not wrap JSON in markdown. Do not include comments. Only raw JSON."
    )

    request_payload = {
        "question": question,
        "candidate_answer": candidate_answer,
        "qa_score": qa_score,
        "contexts": compact_contexts,
    }

    completion = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(request_payload, ensure_ascii=False),
            },
        ],
        # Enforce JSON-only output when supported by the target model.
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=300,
    )

    raw_content = (completion.choices[0].message.content or "").strip()

    # Conservative parsing: if anything goes wrong, abstain instead of hallucinating.
    try:
        parsed = json.loads(raw_content)

        answerable = bool(parsed.get("answerable"))
        reason = str(parsed.get("reason", "")).strip()
        final_answer = str(parsed.get("final_answer", "")).strip()

        # Ensure minimal sensible fallbacks if fields are missing.
        if not reason:
            if answerable:
                reason = "The answer is supported by the provided context."
            else:
                reason = (
                    "The answer is not sufficiently supported by the provided context."
                )

        if not final_answer:
            if answerable and candidate_answer:
                final_answer = candidate_answer.strip()
            else:
                final_answer = (
                    "I cannot answer this reliably based on the available documents."
                )

        return {
            "answerable": answerable,
            "reason": reason,
            "final_answer": final_answer,
        }

    except Exception:
        # If the classifier response is invalid/unparsable, fail closed (not answerable).
        return {
            "answerable": False,
            "reason": "Failed to parse answerability classifier response.",
            "final_answer": (
                "I cannot reliably answer this based on the available documents."
            ),
        }
