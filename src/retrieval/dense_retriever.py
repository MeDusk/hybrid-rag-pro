"""Dense retriever wrapping FAISS search from HybridIndexer."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.ingestion.indexer import HybridIndexer


class DenseRetriever:
    """Retrieves documents using FAISS dense vector similarity search.

    Args:
        indexer: HybridIndexer instance with a built or loaded FAISS index.
        top_k: Number of top documents to retrieve.
    """

    def __init__(
        self,
        indexer: HybridIndexer,
        top_k: int = settings.RETRIEVAL_TOP_K,
    ) -> None:
        self.indexer = indexer
        self.top_k = top_k

    def retrieve(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve top-k documents by dense similarity.

        Args:
            query: Natural language query string.

        Returns:
            List of (Document, score) tuples, descending by score.
        """
        logger.debug(f"DenseRetriever | query='{query[:60]}'")
        return self.indexer.search_dense(query, top_k=self.top_k)
