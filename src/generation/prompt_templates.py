"""Custom prompt templates for anti-hallucination RAG generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class RAGPromptTemplates:
    """Collection of prompt templates for different RAG modes.

    All templates enforce strict grounding:
    the LLM must answer ONLY from the provided context.
    """

    # ------------------------------------------------------------------
    # Standard RAG — strict grounding
    # ------------------------------------------------------------------
    SYSTEM_RAG = """You are a precise and helpful AI assistant.
Your answers must be grounded EXCLUSIVELY in the context provided below.

Strict rules:
- If the answer is not found in the context, respond ONLY with: "I don't have enough information in the provided documents to answer this question."
- Never fabricate facts, statistics, names, or dates.
- Always cite the source document when possible (use metadata: source, page).
- Be concise and structured. Use bullet points for lists.
- Do not repeat the question in your answer.

Context:
{context}"""

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # ------------------------------------------------------------------
    # Condense question — for conversational follow-up
    # ------------------------------------------------------------------
    CONDENSE_SYSTEM = """Given a chat history and the latest user question, 
rewrite the question to be standalone and self-contained, 
without referring to the chat history. 
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, no explanation."""

    CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # ------------------------------------------------------------------
    # Agentic ReAct — for multi-step reasoning
    # ------------------------------------------------------------------
    SYSTEM_REACT = """You are an advanced AI research assistant with access to tools.
Use the tools to retrieve relevant information before answering.

Rules:
- Always use search_hybrid before answering factual questions.
- If you need document metadata, use get_document_metadata.
- If the context is too long, use summarize_chunk to condense it.
- Never answer from memory alone — always ground in retrieved documents.
- If after 3 tool calls you still cannot answer, say so honestly."""

    # ------------------------------------------------------------------
    # Rejection — out-of-scope queries
    # ------------------------------------------------------------------
    REJECTION_TEMPLATE = (
        "I'm sorry, but your question appears to be outside the scope "
        "of the documents I have access to. "
        "Please ask a question related to the indexed content."
    )

    # ------------------------------------------------------------------
    # Context formatter
    # ------------------------------------------------------------------
    @staticmethod
    def format_context(
        documents: list,
        include_scores: bool = True,
    ) -> str:
        """Format retrieved (Document, score) pairs into a context string.

        Args:
            documents: List of (Document, score) tuples.
            include_scores: Whether to include relevance scores.

        Returns:
            Formatted context string ready to inject into a prompt.
        """
        parts: list[str] = []
        for i, (doc, score) in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            score_str = f" | score={score:.4f}" if include_scores else ""
            header = f"[{i}] Source: {source}" + (f" p.{page}" if page else "") + score_str
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)
