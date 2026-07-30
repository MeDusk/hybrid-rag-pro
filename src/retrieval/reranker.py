"""Cross-Encoder reranker for precision boosting after hybrid fusion."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import settings


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs and produce
    a relevance score, achieving higher precision than bi-encoder retrieval
    at the cost of higher latency.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
        top_k: Number of top documents to keep after reranking.
    """

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        top_k: int = settings.RERANKER_TOP_K,
    ) -> None:
        self.top_k = top_k
        logger.info(f"Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)
        logger.info("Cross-encoder reranker ready.")

    def rerank(
        self,
        query: str,
        documents: List[tuple[Document, float]],
    ) -> List[tuple[Document, float]]:
        """Rerank (document, score) pairs using the cross-encoder.

        Args:
            query: Original query string.
            documents: Candidate (Document, initial_score) pairs from retrieval.

        Returns:
            Top-k (Document, rerank_score) pairs sorted by descending score.
        """
        if not documents:
            logger.warning("Reranker received empty document list.")
            return []

        docs_only = [doc for doc, _ in documents]
        pairs = [[query, doc.page_content] for doc in docs_only]

        logger.info(f"Reranking {len(pairs)} candidates -> top-{self.top_k}")
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(docs_only, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        top = ranked[: self.top_k]
        logger.info(
            f"Reranker top scores: {[round(s, 4) for _, s in top]}"
        )
        return top
