"""Synthetic test dataset generator for RAGAS evaluation (20 Q&A pairs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from loguru import logger


TEST_DATASET: List[dict] = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": "RAG is a technique that combines a retrieval system with a language model. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate, grounded answers.",
        "contexts": []
    },
    {
        "question": "What is the difference between dense and sparse retrieval?",
        "ground_truth": "Dense retrieval uses neural embeddings (e.g., FAISS) to find semantically similar documents. Sparse retrieval uses keyword matching algorithms like BM25 based on term frequency. Hybrid search combines both for better coverage.",
        "contexts": []
    },
    {
        "question": "How does Reciprocal Rank Fusion (RRF) work?",
        "ground_truth": "RRF merges ranked lists from multiple retrievers by assigning each document a score of 1/(k + rank), where k=60. Documents appearing high in multiple lists get boosted scores, making it robust to individual retriever weaknesses.",
        "contexts": []
    },
    {
        "question": "What is the role of a cross-encoder reranker?",
        "ground_truth": "A cross-encoder jointly encodes the query and each candidate document together, producing a relevance score more accurate than bi-encoder models. It is applied after initial retrieval to rerank the top-k candidates.",
        "contexts": []
    },
    {
        "question": "What is HyDE in the context of query expansion?",
        "ground_truth": "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer passage from the query using an LLM, then uses that passage as the retrieval query instead of the original question, improving recall for complex queries.",
        "contexts": []
    },
    {
        "question": "What is MultiQuery retrieval?",
        "ground_truth": "MultiQuery generates several paraphrased versions of the original query using an LLM. Each variant is used to retrieve documents independently, and all results are merged and deduplicated to increase recall.",
        "contexts": []
    },
    {
        "question": "What is contextual compression in RAG?",
        "ground_truth": "Contextual compression extracts only the parts of retrieved chunks that are relevant to the query, using an LLM. This reduces noise in the context window and improves answer quality by filtering out irrelevant content.",
        "contexts": []
    },
    {
        "question": "What is FAISS and how is it used in this pipeline?",
        "ground_truth": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. In this pipeline, document chunks are embedded and indexed in FAISS for fast nearest-neighbor retrieval using cosine similarity.",
        "contexts": []
    },
    {
        "question": "What is BM25 and why is it used alongside dense retrieval?",
        "ground_truth": "BM25 is a probabilistic sparse retrieval algorithm based on term frequency and inverse document frequency. It complements dense retrieval by capturing exact keyword matches that semantic search may miss.",
        "contexts": []
    },
    {
        "question": "What is the purpose of ConversationBufferWindowMemory in the RAG chain?",
        "ground_truth": "ConversationBufferWindowMemory stores the last k conversation turns, allowing the RAG chain to rewrite follow-up questions as standalone queries and provide contextually relevant answers in multi-turn conversations.",
        "contexts": []
    },
    {
        "question": "What is LangGraph and how is it used in this project?",
        "ground_truth": "LangGraph is a library for building stateful multi-step LLM applications as graphs. In this project it powers the agentic router, which classifies queries and routes them to simple RAG, ReAct agent, or rejection nodes.",
        "contexts": []
    },
    {
        "question": "What chunking strategies are implemented in HybridRAG-Pro?",
        "ground_truth": "Two strategies are implemented: RecursiveCharacterTextSplitter for structural splits with configurable size and overlap, and SemanticChunker using sentence embeddings to split at semantic boundaries.",
        "contexts": []
    },
    {
        "question": "What file formats does the ingestion pipeline support?",
        "ground_truth": "The ingestion pipeline supports PDF, TXT, DOCX, HTML, and Markdown files using dedicated LangChain document loaders for each format.",
        "contexts": []
    },
    {
        "question": "What are the RAGAS evaluation metrics used in this project?",
        "ground_truth": "RAGAS evaluates four metrics: faithfulness (are answers grounded in context?), answer_relevancy (is the answer relevant to the question?), context_precision (are retrieved chunks precise?), and context_recall (are all relevant chunks retrieved?).",
        "contexts": []
    },
    {
        "question": "What is the anti-hallucination strategy in the generation prompt?",
        "ground_truth": "The system prompt instructs the LLM to answer ONLY from the provided context, never fabricate facts, always cite sources, and respond with 'I don't have enough information' when the answer is not in the context.",
        "contexts": []
    },
    {
        "question": "How does streaming work in the RAG chain?",
        "ground_truth": "The RAG chain supports both synchronous streaming via stream() yielding tokens in a for-loop, and asynchronous streaming via astream() using async generators, enabling real-time token delivery through FastAPI SSE endpoints.",
        "contexts": []
    },
    {
        "question": "How are documents persisted in this pipeline?",
        "ground_truth": "FAISS index, BM25 model, and document objects are serialized to disk using faiss.write_index and pickle. A metadata JSON file records the number of documents and embedding dimension for validation on reload.",
        "contexts": []
    },
    {
        "question": "What embedding model is used and what is its dimension?",
        "ground_truth": "The pipeline uses sentence-transformers/all-MiniLM-L6-v2 which produces 384-dimensional normalized vectors optimized for semantic similarity tasks with low latency.",
        "contexts": []
    },
    {
        "question": "What is the ReAct agent routing condition?",
        "ground_truth": "Queries classified as requiring comparison or multi-step analysis are routed to the ReAct agent. The agent loops through tool calls (search_hybrid, get_document_metadata, summarize_chunk) until no more tool calls are pending.",
        "contexts": []
    },
    {
        "question": "How is configuration managed in HybridRAG-Pro?",
        "ground_truth": "All configuration is centralized in a Pydantic BaseSettings class that reads from a .env file. This ensures no secrets are hardcoded, all parameters are typed, and configuration can be overridden via environment variables.",
        "contexts": []
    },
]


def save_test_dataset(output_path: str = "src/evaluation/test_dataset.json") -> None:
    """Serialize the test dataset to a JSON file.

    Args:
        output_path: Path where the JSON file will be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(TEST_DATASET, f, indent=2, ensure_ascii=False)
    logger.info(f"Test dataset saved: {len(TEST_DATASET)} Q&A pairs -> {path}")


def load_test_dataset(path: str = "src/evaluation/test_dataset.json") -> List[dict]:
    """Load the test dataset from a JSON file.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of Q&A dicts with keys: question, ground_truth, contexts.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"Dataset file not found at {p}, using in-memory dataset.")
        return TEST_DATASET
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} Q&A pairs from {p}")
    return data
