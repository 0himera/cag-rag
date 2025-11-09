from __future__ import annotations

import os
from dataclasses import dataclass
# from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class OpenRouterConfig:
    """
    Configuration for OpenRouter generative LLM.
    """

    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model: str = os.getenv("OPENROUTER_MODEL", "openrouter/polaris-alpha")

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")


_config = OpenRouterConfig()


def _get_client() -> OpenAI:
    _config.validate()
    return OpenAI(base_url=_config.base_url, api_key=_config.api_key)


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using the generative LLM on OpenRouter.

    This function creates a RAG-style prompt with context and question,
    instructing the LLM to answer only based on context or abstain if unable.

    Args:
        question: User query.
        context: Retrieved and reranked context strings, concatenated.

    Returns:
        Generated answer from the LLM.
    """
    if not question.strip():
        return "Вопрос не задан."

    if not context.strip():
        return "Контекст отсутствует, не могу ответить."

    client = _get_client()

    system_prompt = (
        "You are a helpful assistant. Answer the user's question based strictly on the provided context. "
        "If the answer cannot be found in the context, say: 'Я не могу ответить на этот вопрос на основе предоставленной информации.' "
        "Do not use external knowledge. Keep answers concise and relevant."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        completion = client.chat.completions.create(
            model=_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""
    except Exception as exc:
        return f"Ошибка при генерации ответа: {exc}"
