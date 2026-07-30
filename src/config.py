"""Centralized configuration via Pydantic BaseSettings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # OpenAI
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: int = 1024

    # Fallback LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector Store
    VECTOR_STORE_TYPE: Literal["faiss", "chroma"] = "chroma"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "hybridrag_docs"
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # Retrieval
    RETRIEVAL_TOP_K: int = 20
    RERANKER_TOP_K: int = 5
    RRF_K: int = 60
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    SEMANTIC_THRESHOLD: float = 0.75

    # Query Expansion
    QUERY_EXPANSION_MODE: Literal["hyde", "multiquery", "none"] = "multiquery"
    QUERY_EXPANSION_COUNT: int = 3

    # Agent
    DEFAULT_MODE: Literal["simple_rag", "hybrid_rag", "agentic"] = "hybrid_rag"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_RELOAD: bool = True
    API_LOG_LEVEL: str = "info"

    # Evaluation
    RAGAS_LLM_MODEL: str = "gpt-4o-mini"
    EVAL_DATASET_PATH: str = "./src/evaluation/test_dataset.json"
    EVAL_RESULTS_PATH: str = "./eval_results/"

    # MLflow
    MLFLOW_TRACKING_URI: str = "./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "hybridrag-pro"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/hybridrag.log"


settings = Settings()
