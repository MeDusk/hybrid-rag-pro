"""Hybrid Fusion retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridFusion:
    """Combines dense and sparse retrieval results via Reciprocal Rank Fusion.

    RRF formula:
        score_RRF(d) = sum(1 / (k + rank_i(d))) for each retrieval list i

    where k=60 is a smoothing constant that reduces the impact of high ranks.

    Args:
        dense_retriever: DenseRetriever for FAISS-based results.
        sparse_retriever: SparseRetriever for BM25-based results.
        rrf_k: RRF smoothing constant (default 60 per original paper).
        top_k: Final number of fused results to return.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = settings.RRF_K,
        top_k: int = settings.RETRIEVAL_TOP_K,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

    def retrieve(self, query: str) -> List[tuple[Document, float]]:
        """Execute hybrid retrieval and fuse results with RRF.

        Args:
            query: Natural language query string.

        Returns:
            List of (Document, rrf_score) tuples sorted by descending RRF score.
        """
        logger.info(f"HybridFusion | query='{query[:60]}'")

        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)

        fused = self._rrf_fusion(
            result_lists=[dense_results, sparse_results]
        )

        logger.info(
            f"HybridFusion | dense={len(dense_results)} "
            f"sparse={len(sparse_results)} fused={len(fused)}"
        )
        return fused[: self.top_k]

    def retrieve_multi(
        self, queries: List[str]
    ) -> List[tuple[Document, float]]:
        """Execute hybrid retrieval for multiple query variants and fuse all results.

        Used in combination with QueryExpander for multi-query retrieval.

        Args:
            queries: List of query strings (original + expanded variants).

        Returns:
            Deduplicated list of (Document, rrf_score) tuples.
        """
        all_result_lists: list[list[tuple[Document, float]]] = []

        for q in queries:
            dense = self.dense_retriever.retrieve(q)
            sparse = self.sparse_retriever.retrieve(q)
            all_result_lists.extend([dense, sparse])

        fused = self._rrf_fusion(result_lists=all_result_lists)
        logger.info(
            f"HybridFusion multi-query | {len(queries)} queries "
            f"-> {len(fused)} fused results"
        )
        return fused[: self.top_k]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        result_lists: List[List[tuple[Document, float]]],
    ) -> List[tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion across multiple ranked result lists.

        Args:
            result_lists: Each element is a ranked list of (Document, score).

        Returns:
            Merged list sorted by descending RRF score.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for result_list in result_lists:
            for rank, (doc, _) in enumerate(result_list):
                doc_id = self._doc_id(doc)
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
                doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids]

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Generate a stable unique identifier for a document chunk."""
        chunk_id = doc.metadata.get("chunk_id", "")
        source = doc.metadata.get("source", "")
        return f"{source}::{chunk_id}::{hash(doc.page_content)}"
