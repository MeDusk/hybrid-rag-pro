"""Unit tests for the retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_fusion import HybridFusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_expander import QueryExpander


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "doc1.txt", "chunk_id": 0}),
        Document(page_content="FAISS is a library for dense vector search.", metadata={"source": "doc2.txt", "chunk_id": 1}),
        Document(page_content="BM25 is a sparse retrieval algorithm.", metadata={"source": "doc3.txt", "chunk_id": 2}),
        Document(page_content="Cross-encoders rerank candidate documents.", metadata={"source": "doc4.txt", "chunk_id": 3}),
        Document(page_content="LangGraph enables agentic workflows.", metadata={"source": "doc5.txt", "chunk_id": 4}),
    ]


@pytest.fixture
def mock_dense_retriever(sample_documents):
    mock = MagicMock()
    mock.retrieve.return_value = [
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.82),
        (sample_documents[2], 0.71),
    ]
    return mock


@pytest.fixture
def mock_sparse_retriever(sample_documents):
    mock = MagicMock()
    mock.retrieve.return_value = [
        (sample_documents[2], 4.5),
        (sample_documents[0], 3.1),
        (sample_documents[3], 2.7),
    ]
    return mock


# ---------------------------------------------------------------------------
# HybridFusion tests
# ---------------------------------------------------------------------------

class TestHybridFusion:

    def test_rrf_returns_fused_results(
        self, mock_dense_retriever, mock_sparse_retriever, sample_documents
    ):
        fusion = HybridFusion(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            top_k=5,
        )
        results = fusion.retrieve("test query")
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, _ in results)

    def test_rrf_scores_are_positive(self, mock_dense_retriever, mock_sparse_retriever):
        fusion = HybridFusion(mock_dense_retriever, mock_sparse_retriever, top_k=5)
        results = fusion.retrieve("query")
        assert all(score > 0 for _, score in results)

    def test_rrf_deduplicates_documents(
        self, mock_dense_retriever, mock_sparse_retriever, sample_documents
    ):
        """Document appearing in both lists should appear only once in fused results."""
        fusion = HybridFusion(mock_dense_retriever, mock_sparse_retriever, top_k=10)
        results = fusion.retrieve("query")
        contents = [doc.page_content for doc, _ in results]
        assert len(contents) == len(set(contents)), "Duplicate documents found in fused results"

    def test_rrf_formula_correctness(self, mock_dense_retriever, mock_sparse_retriever):
        """doc0 appears rank=0 in dense, rank=1 in sparse -> RRF score = 1/61 + 1/62."""
        fusion = HybridFusion(mock_dense_retriever, mock_sparse_retriever, rrf_k=60, top_k=10)
        results = fusion.retrieve("query")
        scores = {doc.page_content: score for doc, score in results}
        rag_doc_score = scores.get("RAG combines retrieval with generation.", 0)
        expected = 1 / (60 + 1) + 1 / (60 + 2)  # rank 0 in dense, rank 1 in sparse
        assert abs(rag_doc_score - expected) < 1e-9

    def test_retrieve_multi_expands_queries(
        self, mock_dense_retriever, mock_sparse_retriever
    ):
        fusion = HybridFusion(mock_dense_retriever, mock_sparse_retriever, top_k=5)
        results = fusion.retrieve_multi(["query 1", "query 2", "query 3"])
        assert len(results) > 0
        assert mock_dense_retriever.retrieve.call_count == 3

    def test_top_k_respected(self, mock_dense_retriever, mock_sparse_retriever):
        fusion = HybridFusion(mock_dense_retriever, mock_sparse_retriever, top_k=2)
        results = fusion.retrieve("query")
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# CrossEncoderReranker tests
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_ce_cls, sample_documents):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.4, 0.7, 0.2, 0.85])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(top_k=3)
        docs_with_scores = [(doc, 0.5) for doc in sample_documents]
        results = reranker.rerank("test query", docs_with_scores)

        assert len(results) == 3

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_sorted_descending(self, mock_ce_cls, sample_documents):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6, 0.1, 0.8])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(top_k=5)
        results = reranker.rerank("query", [(doc, 0.5) for doc in sample_documents])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_input(self, mock_ce_cls):
        mock_ce_cls.return_value = MagicMock()
        reranker = CrossEncoderReranker(top_k=3)
        assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# QueryExpander tests
# ---------------------------------------------------------------------------

class TestQueryExpander:

    @patch("src.retrieval.query_expander.ChatOpenAI")
    def test_multiquery_returns_original_plus_variants(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_chain_result = "Variant one\nVariant two\nVariant three"
        mock_llm_cls.return_value = mock_llm

        expander = QueryExpander(mode="multiquery", n_variants=3)
        with patch.object(expander, "_multiquery", return_value=["q", "v1", "v2", "v3"]):
            results = expander.expand("original query")
            assert len(results) == 4
            assert results[0] == "q"

    def test_none_mode_returns_original_only(self):
        expander = QueryExpander.__new__(QueryExpander)
        expander.mode = "none"
        expander.n_variants = 3
        results = expander.expand("my query")
        assert results == ["my query"]

    @patch("src.retrieval.query_expander.ChatOpenAI")
    def test_expansion_fallback_on_error(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        expander = QueryExpander(mode="multiquery", n_variants=3)
        with patch.object(expander, "_multiquery", side_effect=Exception("LLM error")):
            results = expander.expand("fallback query")
            assert results == ["fallback query"]
