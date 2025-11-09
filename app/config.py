import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Central configuration for the RAG backend with CAG and reranking.

    All values are primarily sourced from environment variables so that
    the service can be configured without code changes.
    """

    # Qdrant configuration
    qdrant_url: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"),
        description="Base URL for Qdrant instance.",
    )
    qdrant_api_key: str | None = Field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY"),
        description="Optional API key for secured Qdrant instances.",
    )
    qdrant_collection: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "documents"),
        description="Name of the Qdrant collection for document chunks.",
    )
    qdrant_vector_size: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_VECTOR_SIZE", "2048")),
        description="Dimensionality of embedding vectors in Qdrant.",
    )

    # Jina embeddings configuration
    jina_api_key: str = Field(
        default_factory=lambda: os.getenv("JINA_API_KEY", ""),
        description="API key for Jina embeddings.",
    )
    jina_embedding_model: str = Field(
        default_factory=lambda: os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4"),
        description="Model name for Jina embeddings.",
    )
    jina_embedding_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "JINA_EMBEDDING_ENDPOINT", "https://api.jina.ai/v1/embeddings"
        ),
        description="Endpoint URL for Jina embeddings API.",
    )
    jina_embedding_task: str = Field(
        default_factory=lambda: os.getenv("JINA_EMBEDDING_TASK", "text-matching"),
        description="Task type for Jina embeddings.",
    )
    jina_embedding_expected_dim: int = Field(
        default_factory=lambda: int(os.getenv("JINA_EMBEDDING_EXPECTED_DIM", "2048")),
        description="Expected embedding dimension.",
    )

    # Jina reranker configuration
    jina_reranker_api_key: str = Field(
        default_factory=lambda: os.getenv("JINA_RERANKER_API_KEY", ""),
        description="API key for Jina reranker (defaults to JINA_API_KEY if empty).",
    )
    jina_reranker_model: str = Field(
        default_factory=lambda: os.getenv(
            "JINA_RERANKER_MODEL", "jina-reranker-v3-base-en"
        ),
        description="Model name for Jina reranker.",
    )
    jina_reranker_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "JINA_RERANKER_ENDPOINT", "https://api.jina.ai/v1/rerank"
        ),
        description="Endpoint URL for Jina reranker API.",
    )
    jina_reranker_top_k: int = Field(
        default_factory=lambda: int(os.getenv("JINA_RERANKER_TOP_K", "5")),
        description="Number of contexts to rerank.",
    )

    # OpenRouter configuration (for generative LLM)
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        description="API key for OpenRouter.",
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        description="Base URL for OpenRouter.",
    )
    openrouter_model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL", "openrouter/polaris-alpha"
        ),
        description="Model for OpenRouter generative LLM.",
    )

    # CAG configuration
    cag_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CAG_THRESHOLD", "0.5")),
        description="Threshold for CAG check (max similarity to decide if RAG is needed).",
    )

    # Retrieval configuration
    retrieval_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10")),
        description="Number of chunks to retrieve initially for reranking.",
    )
    reranked_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANKED_TOP_K", "5")),
        description="Number of contexts to keep after reranking for RAG prompt.",
    )
    max_context_chars: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "4000")),
        description="Max context length in RAG prompt.",
    )

    # Chunking parameters
    chunk_size_tokens: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE_TOKENS", "500")),
        description="Chunk size for document splitting.",
    )
    chunk_overlap_tokens: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
        description="Overlap between chunks.",
    )

    # Static response for CAG failure
    static_response_message: str = Field(
        default_factory=lambda: os.getenv(
            "STATIC_RESPONSE_MESSAGE",
            "На основе имеющейся у меня базы знаний, я не могу предоставить ответ на ваш вопрос.",
        ),
        description="Static response when CAG decides no RAG is needed.",
    )

    class Config:
        validate_assignment = True


settings = Settings()
