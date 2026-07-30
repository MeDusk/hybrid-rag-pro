"""Hybrid indexer combining FAISS (dense) and BM25 (sparse) with ChromaDB persistence."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.embedder import EmbeddingModel


class HybridIndexer:
    """Builds and persists FAISS dense index + BM25 sparse index from documents.

    Args:
        embedder: EmbeddingModel instance for dense vectorization.
        faiss_index_path: Directory to persist FAISS index and metadata.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        faiss_index_path: str = settings.FAISS_INDEX_PATH,
    ) -> None:
        self.embedder = embedder or EmbeddingModel()
        self.faiss_index_path = Path(faiss_index_path)
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)

        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._texts: List[str] = []

        logger.info("HybridIndexer initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, documents: List[Document]) -> None:
        """Build FAISS and BM25 indices from a list of documents.

        Args:
            documents: Chunked LangChain Document objects to index.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        self._documents = documents
        self._texts = [doc.page_content for doc in documents]

        logger.info(f"Building indices from {len(documents)} chunks...")
        self._build_faiss()
        self._build_bm25()
        logger.info("Both indices built successfully.")

    def save(self) -> None:
        """Persist FAISS index, BM25 model, and document metadata to disk."""
        faiss.write_index(
            self._faiss_index,
            str(self.faiss_index_path / "index.faiss"),
        )
        with open(self.faiss_index_path / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        with open(self.faiss_index_path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        metadata = {"num_docs": len(self._documents), "dim": self.embedder.dimension}
        with open(self.faiss_index_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Indices saved to {self.faiss_index_path}")

    def load(self) -> None:
        """Load persisted FAISS index, BM25 model, and documents from disk.

        Raises:
            FileNotFoundError: If index files are not found.
        """
        index_file = self.faiss_index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {index_file}. Run build() first."
            )

        self._faiss_index = faiss.read_index(str(index_file))
        with open(self.faiss_index_path / "bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        with open(self.faiss_index_path / "documents.pkl", "rb") as f:
            self._documents = pickle.load(f)
            self._texts = [doc.page_content for doc in self._documents]

        logger.info(
            f"Indices loaded | {len(self._documents)} docs | "
            f"FAISS dim={self._faiss_index.d}"
        )

    def search_dense(
        self, query: str, top_k: int = settings.RETRIEVAL_TOP_K
    ) -> List[tuple[Document, float]]:
        """Search the FAISS index using dense vector similarity.

        Args:
            query: Query string to search.
            top_k: Number of top results to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        self._check_built()
        query_vec = self.embedder.encode_single(query).reshape(1, -1)
        scores, indices = self._faiss_index.search(query_vec, top_k)
        results = [
            (self._documents[idx], float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx != -1
        ]
        logger.debug(f"Dense search returned {len(results)} results for query: '{query[:50]}'")
        return results

    def search_sparse(
        self, query: str, top_k: int = settings.RETRIEVAL_TOP_K
    ) -> List[tuple[Document, float]]:
        """Search using BM25 sparse retrieval.

        Args:
            query: Query string to search.
            top_k: Number of top results to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        self._check_built()
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = [
            (self._documents[idx], float(bm25_scores[idx]))
            for idx in top_indices
        ]
        logger.debug(f"Sparse search returned {len(results)} results for query: '{query[:50]}'")
        return results

    @property
    def documents(self) -> List[Document]:
        """Return all indexed documents."""
        return self._documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_faiss(self) -> None:
        """Build the FAISS IndexFlatIP from document embeddings."""
        vectors = self.embedder.encode(self._texts)
        dim = vectors.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(vectors)
        logger.info(f"FAISS index built | {self._faiss_index.ntotal} vectors | dim={dim}")

    def _build_bm25(self) -> None:
        """Build BM25Okapi index from tokenized document texts."""
        tokenized_corpus = [text.lower().split() for text in self._texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built | {len(tokenized_corpus)} documents")

    def _check_built(self) -> None:
        """Raise RuntimeError if indices are not yet built or loaded."""
        if self._faiss_index is None or self._bm25 is None:
            raise RuntimeError(
                "Indices not built. Call build() or load() first."
            )
