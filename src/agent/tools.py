"""LangChain tools available to the LangGraph ReAct agent."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from src.retrieval.hybrid_fusion import HybridFusion


def build_agent_tools(fusion: HybridFusion) -> list:
    """Build the list of LangChain tools available to the ReAct agent.

    Args:
        fusion: HybridFusion instance for document retrieval.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def search_hybrid(query: str) -> str:
        """Search the document knowledge base using hybrid retrieval (dense + sparse + RRF).
        Use this tool to find relevant information before answering any factual question.

        Args:
            query: The search query string.

        Returns:
            Formatted string of top retrieved document chunks with sources.
        """
        logger.info(f"Tool:search_hybrid | query='{query[:60]}'")
        results = fusion.retrieve(query)
        if not results:
            return "No relevant documents found for this query."

        parts: list[str] = []
        for i, (doc, score) in enumerate(results[:5], start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f" p.{page}" if page else ""
            parts.append(
                f"[{i}] {source}{page_str} (score={score:.4f})\n{doc.page_content[:400]}"
            )
        return "\n\n---\n\n".join(parts)

    @tool
    def get_document_metadata(source_name: str) -> str:
        """Retrieve metadata about a specific indexed document by its source name.
        Use this when you need information about when a document was indexed,
        its size, or number of chunks.

        Args:
            source_name: The filename or source identifier of the document.

        Returns:
            Formatted metadata string for the requested document.
        """
        logger.info(f"Tool:get_document_metadata | source='{source_name}'")
        docs = fusion.dense_retriever.indexer.documents
        matches = [
            doc for doc in docs
            if source_name.lower() in doc.metadata.get("source", "").lower()
        ]
        if not matches:
            return f"No document found with source matching '{source_name}'."

        sources = set(doc.metadata.get("source", "unknown") for doc in matches)
        return (
            f"Found {len(matches)} chunks from {len(sources)} file(s): "
            f"{', '.join(sources)}"
        )

    @tool
    def summarize_chunk(text: str) -> str:
        """Summarize a long text chunk to its key points.
        Use this when a retrieved chunk is too long to include in full context.

        Args:
            text: The text content to summarize.

        Returns:
            A concise bullet-point summary of the key information.
        """
        from langchain_openai import ChatOpenAI
        from src.config import settings

        logger.info("Tool:summarize_chunk | summarizing chunk")
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
        )
        prompt = (
            f"Summarize the following text into concise bullet points "
            f"(max 5 bullets):\n\n{text[:2000]}"
        )
        result = llm.invoke(prompt)
        return result.content

    return [search_hybrid, get_document_metadata, summarize_chunk]
