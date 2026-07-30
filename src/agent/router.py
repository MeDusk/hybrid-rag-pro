"""LangGraph agentic router with conditional routing logic."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.agent.tools import build_agent_tools
from src.config import settings
from src.generation.prompt_templates import RAGPromptTemplates
from src.retrieval.hybrid_fusion import HybridFusion


# ------------------------------------------------------------------
# Graph State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """State passed between LangGraph nodes."""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str
    final_answer: str


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

class AgenticRouter:
    """LangGraph-based agentic router with three routing paths.

    Routing logic:
        - SIMPLE_RAG   : Factual, direct questions -> single RAG retrieval
        - REACT_AGENT  : Comparative / analytical questions -> multi-step ReAct
        - REJECT       : Off-topic queries -> polite rejection

    Args:
        fusion: HybridFusion instance used by agent tools.
    """

    def __init__(self, fusion: HybridFusion) -> None:
        self._fusion = fusion
        self._tools = build_agent_tools(fusion)

        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
        )
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._tool_node = ToolNode(self._tools)
        self._graph = self._build_graph()
        logger.info("AgenticRouter initialized with LangGraph")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """Execute the agentic routing pipeline for a given query.

        Args:
            query: User query string.

        Returns:
            Dict with keys: answer (str), route (str), messages (list).
        """
        logger.info(f"AgenticRouter.run | query='{query[:60]}'")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "route": "",
            "final_answer": "",
        }
        result = self._graph.invoke(initial_state)
        return {
            "answer": result["final_answer"],
            "route": result["route"],
            "messages": [m.content for m in result["messages"]],
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("simple_rag", self._simple_rag_node)
        graph.add_node("react_agent", self._react_agent_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("reject", self._reject_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_decision,
            {
                "simple_rag": "simple_rag",
                "react_agent": "react_agent",
                "reject": "reject",
            },
        )

        graph.add_edge("simple_rag", "finalize")
        graph.add_edge("reject", "finalize")

        graph.add_conditional_edges(
            "react_agent",
            self._should_continue_react,
            {
                "continue": "tools",
                "end": "finalize",
            },
        )
        graph.add_edge("tools", "react_agent")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _classify_node(self, state: AgentState) -> AgentState:
        """Classify the query type to determine routing path."""
        classify_prompt = f"""Classify the following user query into exactly one category:

- SIMPLE_RAG: A direct factual question that can be answered with a single retrieval.
- REACT_AGENT: A complex question requiring comparison, analysis, or multiple steps.
- REJECT: A question that is completely unrelated to any document-based knowledge base.

Query: \"{state['query']}\"

Respond with ONLY one word: SIMPLE_RAG, REACT_AGENT, or REJECT."""

        response = self._llm.invoke([HumanMessage(content=classify_prompt)])
        route_raw = response.content.strip().upper()

        if "REACT" in route_raw:
            route = "react_agent"
        elif "REJECT" in route_raw:
            route = "reject"
        else:
            route = "simple_rag"

        logger.info(f"Classify node -> route='{route}'")
        return {**state, "route": route}

    def _simple_rag_node(self, state: AgentState) -> AgentState:
        """Handle simple factual queries with direct hybrid retrieval."""
        from src.retrieval.reranker import CrossEncoderReranker
        from src.generation.prompt_templates import RAGPromptTemplates

        query = state["query"]
        results = self._fusion.retrieve(query)
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, results)
        context = RAGPromptTemplates.format_context(results)

        prompt = RAGPromptTemplates.RAG_PROMPT.format_messages(
            context=context,
            question=query,
            chat_history=[],
        )
        response = self._llm.invoke(prompt)
        logger.info("SimpleRAG node completed.")
        return {
            **state,
            "messages": state["messages"] + [response],
            "final_answer": response.content,
        }

    def _react_agent_node(self, state: AgentState) -> AgentState:
        """Handle complex queries via ReAct agent with tool calls."""
        system_msg = SystemMessage(content=RAGPromptTemplates.SYSTEM_REACT)
        messages = [system_msg] + state["messages"]
        response = self._llm_with_tools.invoke(messages)
        logger.info(
            f"ReAct node | tool_calls={len(getattr(response, 'tool_calls', []))}"
        )
        return {
            **state,
            "messages": state["messages"] + [response],
            "final_answer": response.content if response.content else state["final_answer"],
        }

    def _reject_node(self, state: AgentState) -> AgentState:
        """Return a polite rejection for off-topic queries."""
        logger.info("Reject node triggered.")
        return {
            **state,
            "final_answer": RAGPromptTemplates.REJECTION_TEMPLATE,
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize: ensure final_answer is set from last AI message if empty."""
        if not state["final_answer"]:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    return {**state, "final_answer": msg.content}
        return state

    # ------------------------------------------------------------------
    # Conditional edge functions
    # ------------------------------------------------------------------

    @staticmethod
    def _route_decision(state: AgentState) -> Literal["simple_rag", "react_agent", "reject"]:
        """Return routing decision from state."""
        return state["route"]

    @staticmethod
    def _should_continue_react(
        state: AgentState,
    ) -> Literal["continue", "end"]:
        """Continue ReAct loop if there are pending tool calls, else end."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if tool_calls:
            return "continue"
        return "end"
