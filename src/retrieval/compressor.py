"""Contextual compression using LLMChainExtractor."""

from __future__ import annotations

from typing import List

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import settings


class ContextualCompressor:
    """Compresses retrieved chunks by extracting only query-relevant content.

    Uses LangChain's LLMChainExtractor to ask an LLM to extract only the
    parts of each chunk that are relevant to the query, reducing noise
    passed to the generation step.

    Args:
        model_name: LLM model identifier for compression.
    """

    def __init__(self, model_name: str = settings.OPENAI_MODEL) -> None:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
        )
        self._compressor = LLMChainExtractor.from_llm(llm)
        logger.info(f"ContextualCompressor initialized | model={model_name}")

    def compress(
        self,
        query: str,
        documents: List[tuple[Document, float]],
    ) -> List[tuple[Document, float]]:
        """Extract only query-relevant content from each document chunk.

        Args:
            query: The user query for relevance filtering.
            documents: List of (Document, score) pairs from retrieval/reranking.

        Returns:
            Filtered list of (Document, score) pairs with compressed content.
            Chunks deemed fully irrelevant are dropped.
        """
        if not documents:
            return []

        docs_only = [doc for doc, _ in documents]
        scores = [score for _, score in documents]

        logger.info(f"Compressing {len(docs_only)} chunks for query: '{query[:60]}'")

        compressed: List[tuple[Document, float]] = []
        for doc, score in zip(docs_only, scores):
            try:
                result = self._compressor.compress_documents(
                    documents=[doc], query=query
                )
                if result:
                    compressed.append((result[0], score))
                else:
                    logger.debug(f"Chunk dropped by compressor (not relevant): chunk_id={doc.metadata.get('chunk_id')}")
            except Exception as e:
                logger.warning(f"Compression failed for chunk, keeping original: {e}")
                compressed.append((doc, score))

        logger.info(f"Compression: {len(docs_only)} -> {len(compressed)} chunks kept")
        return compressed
