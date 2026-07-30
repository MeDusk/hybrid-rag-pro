"""Embedding model wrapper using SentenceTransformers."""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingModel:
    """Wrapper around SentenceTransformer for dense vector generation.

    Provides batch encoding with normalization for cosine similarity.

    Args:
        model_name: HuggingFace model identifier.
        device: Inference device ('cpu', 'cuda', 'mps').
        normalize: Whether to L2-normalize output vectors.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready | dim={self.dimension}")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts per inference batch.

        Returns:
            Float32 numpy array of shape (len(texts), dimension).
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")

        logger.debug(f"Encoding {len(texts)} texts (batch_size={batch_size})")
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string into a dense vector.

        Args:
            text: Input string.

        Returns:
            Float32 numpy array of shape (dimension,).
        """
        return self.encode([text])[0]
