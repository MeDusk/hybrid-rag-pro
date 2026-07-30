"""HybridRAG-Pro FastAPI application.

Endpoints:
    POST /query   -> Hybrid RAG query with optional streaming (SSE)
    POST /ingest  -> Upload and index a document asynchronously
    GET  /health  -> System health status
    GET  /metrics -> Latest RAGAS evaluation scores
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.ingestion.chunker import HybridChunker
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.indexer import HybridIndexer
from src.generation.llm_chain import HybridRAGChain
from src.agent.router import AgenticRouter
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_fusion import HybridFusion


# ---------------------------------------------------------------------------
# Global singletons (initialized at startup)
# ---------------------------------------------------------------------------

_embedder: Optional[EmbeddingModel] = None
_indexer: Optional[HybridIndexer] = None
_rag_chain: Optional[HybridRAGChain] = None
_agent_router: Optional[AgenticRouter] = None
_upload_dir = Path("./data/uploads")
_upload_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lifespan: load indices at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize heavy singletons once at startup."""
    global _embedder, _indexer, _rag_chain, _agent_router

    logger.info("Starting HybridRAG-Pro API...")
    _embedder = EmbeddingModel()
    _indexer = HybridIndexer(embedder=_embedder)

    index_path = Path(settings.FAISS_INDEX_PATH) / "index.faiss"
    if index_path.exists():
        logger.info("Loading existing indices from disk...")
        _indexer.load()
        dense = DenseRetriever(_indexer)
        sparse = SparseRetriever(_indexer)
        fusion = HybridFusion(dense, sparse)
        _rag_chain = HybridRAGChain(_indexer)
        _agent_router = AgenticRouter(fusion)
        logger.info("Indices loaded. API ready.")
    else:
        logger.warning(
            "No existing index found. POST /ingest to add documents before querying."
        )

    yield
    logger.info("Shutting down HybridRAG-Pro API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HybridRAG-Pro",
    description=(
        "Production-ready Hybrid RAG API — "
        "semantic + keyword search, reranking, query expansion, agentic routing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., min_length=3, max_length=2000, description="User query")
    top_k: int = Field(default=settings.RERANKER_TOP_K, ge=1, le=20)
    use_reranker: bool = Field(default=True)
    mode: str = Field(
        default=settings.DEFAULT_MODE,
        description="Retrieval mode: simple_rag | hybrid_rag | agentic",
    )
    stream: bool = Field(default=False, description="Enable SSE token streaming")


class QueryResponse(BaseModel):
    """Response body for POST /query (non-streaming)."""
    answer: str
    sources: List[dict]
    route: Optional[str] = None
    latency_ms: float


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    filename: str
    num_chunks: int
    message: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    index_loaded: bool
    num_indexed_docs: int
    version: str


# ---------------------------------------------------------------------------
# Helper: ensure chain is ready
# ---------------------------------------------------------------------------

def _require_chain() -> HybridRAGChain:
    """Raise 503 if the RAG chain is not initialized."""
    if _rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST /ingest to add documents first.",
        )
    return _rag_chain


def _require_router() -> AgenticRouter:
    """Raise 503 if the agent router is not initialized."""
    if _agent_router is None:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST /ingest to add documents first.",
        )
    return _agent_router


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, summary="Run a RAG query")
async def query_endpoint(request: QueryRequest) -> JSONResponse | StreamingResponse:
    """Execute a hybrid RAG query.

    - **hybrid_rag** mode: Dense + Sparse + RRF + CrossEncoder reranker.
    - **agentic** mode: LangGraph router with ReAct agent.
    - **simple_rag** mode: Direct retrieval without reranking.
    - Set **stream=true** to receive a Server-Sent Events stream.
    """
    logger.info(f"POST /query | mode={request.mode} stream={request.stream} q='{request.query[:60]}'")
    start = time.perf_counter()

    if request.stream:
        chain = _require_chain()

        async def event_generator() -> AsyncIterator[dict]:
            try:
                async for token in chain.astream(request.query, top_k=request.top_k):
                    yield {"event": "token", "data": token}
                yield {"event": "done", "data": "[DONE]"}
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield {"event": "error", "data": str(e)}

        return EventSourceResponse(event_generator())

    if request.mode == "agentic":
        router = _require_router()
        result = await asyncio.get_event_loop().run_in_executor(
            None, router.run, request.query
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return JSONResponse(content={
            "answer": result["answer"],
            "sources": [],
            "route": result["route"],
            "latency_ms": latency_ms,
        })

    chain = _require_chain()
    result = await asyncio.get_event_loop().run_in_executor(
        None, chain.query, request.query, request.top_k
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return JSONResponse(content={
        "answer": result["answer"],
        "sources": result["sources"],
        "route": request.mode,
        "latency_ms": latency_ms,
    })


@app.post("/ingest", response_model=IngestResponse, summary="Upload and index a document")
async def ingest_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a document (PDF, TXT, DOCX, HTML, MD) and add it to the index.

    - Saves the file to disk.
    - Chunks, embeds, and indexes it with FAISS + BM25.
    - Persists the updated index.
    """
    global _indexer, _rag_chain, _agent_router

    allowed_extensions = {".pdf", ".txt", ".docx", ".html", ".htm", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    save_path = _upload_dir / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    logger.info(f"File saved: {save_path} ({len(content)} bytes)")

    chunker = HybridChunker()
    try:
        new_chunks = chunker.load_and_chunk(save_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Chunking failed: {e}")

    existing_docs = _indexer.documents if _indexer else []
    all_docs = existing_docs + new_chunks

    if _embedder is None:
        _embedder = EmbeddingModel()
    _indexer = HybridIndexer(embedder=_embedder)
    _indexer.build(all_docs)
    _indexer.save()

    dense = DenseRetriever(_indexer)
    sparse = SparseRetriever(_indexer)
    fusion = HybridFusion(dense, sparse)
    _rag_chain = HybridRAGChain(_indexer)
    _agent_router = AgenticRouter(fusion)

    logger.info(f"Ingestion complete | {len(new_chunks)} new chunks | total={len(all_docs)}")
    return IngestResponse(
        filename=file.filename,
        num_chunks=len(new_chunks),
        message=f"Successfully indexed {len(new_chunks)} chunks. Total docs: {len(all_docs)}.",
    )


@app.get("/health", response_model=HealthResponse, summary="System health check")
async def health_endpoint() -> HealthResponse:
    """Return API health status and index statistics."""
    index_loaded = _indexer is not None and len(_indexer.documents) > 0
    return HealthResponse(
        status="ok" if index_loaded else "degraded",
        index_loaded=index_loaded,
        num_indexed_docs=len(_indexer.documents) if _indexer else 0,
        version=app.version,
    )


@app.get("/metrics", summary="Latest RAGAS evaluation scores")
async def metrics_endpoint() -> JSONResponse:
    """Return the most recent RAGAS evaluation scores from disk.

    Returns 404 if no evaluation has been run yet.
    """
    results_dir = Path(settings.EVAL_RESULTS_PATH)
    score_files = sorted(results_dir.glob("ragas_scores_*.json"), reverse=True)

    if not score_files:
        raise HTTPException(
            status_code=404,
            detail="No RAGAS evaluation results found. Run eval/evaluate.py first.",
        )

    latest = score_files[0]
    with open(latest, "r", encoding="utf-8") as f:
        scores = json.load(f)

    logger.info(f"Serving RAGAS metrics from: {latest.name}")
    return JSONResponse(content={"source_file": latest.name, "scores": scores})


@app.delete("/memory", summary="Clear conversation memory")
async def clear_memory_endpoint() -> JSONResponse:
    """Reset the RAG chain's conversation memory."""
    chain = _require_chain()
    chain.clear_memory()
    return JSONResponse(content={"message": "Conversation memory cleared."})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.API_LOG_LEVEL,
    )
