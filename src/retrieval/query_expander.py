"""Query expansion using HyDE or MultiQuery via LLM."""

from __future__ import annotations

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import settings


MULTIQUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant helping to improve document retrieval.

Given the following question, generate {n} different reformulations of it 
that cover different angles, phrasings, and aspects of the same intent.
Return ONLY the reformulations, one per line, no numbering, no explanation.

Original question: {query}

Reformulations:"""
)

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant. Given the following question, write a short 
hypothetical passage (2-4 sentences) that would directly answer it. 
This passage will be used to find similar real documents.

Question: {query}

Hypothetical answer passage:"""
)


class QueryExpander:
    """Expands a user query into multiple variants using an LLM.

    Supports two modes:
        - multiquery: Generates N paraphrased versions of the query.
        - hyde: Generates a hypothetical document (HyDE technique).

    Args:
        mode: Expansion mode ('multiquery' or 'hyde').
        n_variants: Number of query variants to generate (multiquery only).
    """

    def __init__(
        self,
        mode: str = settings.QUERY_EXPANSION_MODE,
        n_variants: int = settings.QUERY_EXPANSION_COUNT,
    ) -> None:
        if mode not in ("multiquery", "hyde", "none"):
            raise ValueError(f"Invalid expansion mode: {mode}")

        self.mode = mode
        self.n_variants = n_variants

        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.4,
            api_key=settings.OPENAI_API_KEY,
        )
        self._parser = StrOutputParser()
        logger.info(f"QueryExpander initialized | mode={mode} n={n_variants}")

    def expand(self, query: str) -> List[str]:
        """Expand a query into multiple variants.

        Args:
            query: Original user query string.

        Returns:
            List containing the original query + generated variants.
            Returns [query] if mode is 'none'.
        """
        if self.mode == "none":
            return [query]

        try:
            if self.mode == "multiquery":
                return self._multiquery(query)
            else:
                return self._hyde(query)
        except Exception as e:
            logger.warning(f"QueryExpander failed ({e}), falling back to original query.")
            return [query]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _multiquery(self, query: str) -> List[str]:
        """Generate N paraphrased query variants."""
        chain = MULTIQUERY_PROMPT | self._llm | self._parser
        result = chain.invoke({"query": query, "n": self.n_variants})
        variants = [line.strip() for line in result.strip().split("\n") if line.strip()]
        all_queries = [query] + variants[: self.n_variants]
        logger.info(f"MultiQuery expanded to {len(all_queries)} queries")
        return all_queries

    def _hyde(self, query: str) -> List[str]:
        """Generate a hypothetical answer document (HyDE)."""
        chain = HYDE_PROMPT | self._llm | self._parser
        hypothetical_doc = chain.invoke({"query": query})
        logger.info("HyDE generated hypothetical document.")
        return [query, hypothetical_doc.strip()]
