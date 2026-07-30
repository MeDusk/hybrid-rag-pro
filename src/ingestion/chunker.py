"""Hybrid chunker combining RecursiveCharacterTextSplitter and SemanticChunker."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from src.config import settings


LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
}


class HybridChunker:
    """Loads documents and splits them using a hybrid chunking strategy.

    Strategy:
        1. RecursiveCharacterTextSplitter for hard structural splits.
        2. SemanticChunker (optional) for semantically coherent chunks.

    Args:
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        use_semantic: Whether to apply semantic chunking on top.
        semantic_threshold: Cosine similarity threshold for semantic splits.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        use_semantic: bool = False,
        semantic_threshold: float = settings.SEMANTIC_THRESHOLD,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold

        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        if use_semantic:
            self._semantic_splitter = SemanticChunker(
                embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=semantic_threshold,
            )

        logger.info(
            f"HybridChunker initialized | chunk_size={chunk_size} "
            f"overlap={chunk_overlap} semantic={use_semantic}"
        )

    def load_file(self, file_path: str | Path) -> List[Document]:
        """Load a single file based on its extension.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of raw LangChain Document objects.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {list(LOADER_MAP.keys())}"
            )

        logger.info(f"Loading file: {path.name} [{ext}]")
        loader = loader_cls(str(path))
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} raw page(s) from {path.name}")
        return documents

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using the configured strategy.

        Args:
            documents: Raw LangChain documents to chunk.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        if self.use_semantic:
            logger.info("Applying SemanticChunker...")
            chunks = self._semantic_splitter.split_documents(documents)
        else:
            chunks = self._recursive_splitter.split_documents(documents)

        # Enrich metadata with chunk index
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} document(s)")
        return chunks

    def load_and_chunk(self, file_path: str | Path) -> List[Document]:
        """Convenience method: load a file and chunk it in one call.

        Args:
            file_path: Path to the file to process.

        Returns:
            List of chunked Document objects.
        """
        documents = self.load_file(file_path)
        return self.chunk(documents)
