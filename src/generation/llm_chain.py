"""Conversational RAG chain with streaming support and memory."""

from __future__ import annotations

from typing import AsyncIterator, Iterator, List, Optional

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import settings
from src.generation.prompt_templates import RAGPromptTemplates
from src.ingestion.indexer import HybridIndexer
from src.retrieval.hybrid_fusion import HybridFusion
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_expander import QueryExpander
from src.retrieval.compressor import ContextualCompressor


class HybridRAGChain:
    """End-to-end Hybrid RAG chain with conversational memory and streaming.

    Pipeline:
        1. Condense follow-up questions using chat history.
        2. Expand query (MultiQuery / HyDE).
        3. Hybrid retrieval (Dense + Sparse + RRF fusion).
        4. Cross-encoder reranking.
        5. Contextual compression.
        6. LLM generation with anti-hallucination prompt.

    Args:
        indexer: Loaded HybridIndexer instance.
        use_reranker: Whether to apply cross-encoder reranking.
        use_compressor: Whether to apply contextual compression.
        use_query_expansion: Whether to expand query before retrieval.
        memory_k: Number of past conversation turns to keep in memory.
    """

    def __init__(
        self,
        indexer: HybridIndexer,
        use_reranker: bool = True,
        use_compressor: bool = False,
        use_query_expansion: bool = True,
        memory_k: int = 5,
    ) -> None:
        self.use_reranker = use_reranker
        self.use_compressor = use_compressor
        self.use_query_expansion = use_query_expansion

        # LLM
        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
            streaming=True,
        )

        # Memory
        self._memory = ConversationBufferWindowMemory(
            k=memory_k,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
        )

        # Retrieval pipeline
        dense = DenseRetriever(indexer)
        sparse = SparseRetriever(indexer)
        self._fusion = HybridFusion(dense, sparse)

        if use_reranker:
            self._reranker = CrossEncoderReranker()

        if use_compressor:
            self._compressor = ContextualCompressor()

        if use_query_expansion:
            self._expander = QueryExpander()

        # Chains
        self._condense_chain = (
            RAGPromptTemplates.CONDENSE_PROMPT
            | self._llm
            | StrOutputParser()
        )
        self._rag_chain = (
            RAGPromptTemplates.RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

        logger.info(
            f"HybridRAGChain ready | reranker={use_reranker} "
            f"compressor={use_compressor} expansion={use_query_expansion}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """Run a full RAG query and return answer with sources.

        Args:
            question: User question string.
            top_k: Override default retrieval top_k if provided.

        Returns:
            Dict with keys: answer (str), sources (list[dict]), chat_history.
        """
        standalone_question = self._condense_question(question)
        retrieved = self._retrieve(standalone_question, top_k)
        context = RAGPromptTemplates.format_context(retrieved)
        chat_history = self._memory.load_memory_variables({})["chat_history"]

        logger.info(f"Generating answer | context_chunks={len(retrieved)}")
        answer = self._rag_chain.invoke({
            "context": context,
            "question": standalone_question,
            "chat_history": chat_history,
        })

        self._memory.save_context(
            {"input": question},
            {"answer": answer},
        )

        sources = self._format_sources(retrieved)
        return {"answer": answer, "sources": sources, "chat_history": chat_history}

    def stream(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream the RAG answer token by token.

        Args:
            question: User question string.
            top_k: Override default retrieval top_k if provided.

        Yields:
            String tokens as they are generated by the LLM.
        """
        standalone_question = self._condense_question(question)
        retrieved = self._retrieve(standalone_question, top_k)
        context = RAGPromptTemplates.format_context(retrieved)
        chat_history = self._memory.load_memory_variables({})["chat_history"]

        full_answer = ""
        for token in self._rag_chain.stream({
            "context": context,
            "question": standalone_question,
            "chat_history": chat_history,
        }):
            full_answer += token
            yield token

        self._memory.save_context(
            {"input": question},
            {"answer": full_answer},
        )

    async def astream(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Async stream the RAG answer token by token (for FastAPI SSE).

        Args:
            question: User question string.
            top_k: Override default retrieval top_k if provided.

        Yields:
            String tokens as they are generated by the LLM.
        """
        standalone_question = self._condense_question(question)
        retrieved = self._retrieve(standalone_question, top_k)
        context = RAGPromptTemplates.format_context(retrieved)
        chat_history = self._memory.load_memory_variables({})["chat_history"]

        full_answer = ""
        async for token in self._rag_chain.astream({
            "context": context,
            "question": standalone_question,
            "chat_history": chat_history,
        }):
            full_answer += token
            yield token

        self._memory.save_context(
            {"input": question},
            {"answer": full_answer},
        )

    def clear_memory(self) -> None:
        """Reset conversation memory."""
        self._memory.clear()
        logger.info("Conversation memory cleared.")

    @property
    def history(self) -> List[BaseMessage]:
        """Return the current conversation history as LangChain messages."""
        return self._memory.load_memory_variables({})["chat_history"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _condense_question(self, question: str) -> str:
        """Rewrite question as standalone if chat history exists."""
        history = self._memory.load_memory_variables({})["chat_history"]
        if not history:
            return question
        try:
            return self._condense_chain.invoke({
                "question": question,
                "chat_history": history,
            })
        except Exception as e:
            logger.warning(f"Question condensing failed: {e}")
            return question

    def _retrieve(
        self,
        question: str,
        top_k: Optional[int],
    ) -> List[tuple[Document, float]]:
        """Run the full retrieval pipeline: expand -> fuse -> rerank -> compress."""
        if self.use_query_expansion:
            queries = self._expander.expand(question)
            results = self._fusion.retrieve_multi(queries)
        else:
            results = self._fusion.retrieve(question)

        if top_k:
            results = results[:top_k]

        if self.use_reranker:
            results = self._reranker.rerank(question, results)

        if self.use_compressor:
            results = self._compressor.compress(question, results)

        return results

    @staticmethod
    def _format_sources(
        documents: List[tuple[Document, float]],
    ) -> List[dict]:
        """Convert (Document, score) pairs into serializable source dicts."""
        return [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "chunk_id": doc.metadata.get("chunk_id", None),
                "score": round(score, 6),
            }
            for doc, score in documents
        ]
