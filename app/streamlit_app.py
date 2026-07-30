"""HybridRAG-Pro — Streamlit Chat Interface.

Features:
    - Multi-turn chat with conversation history
    - Sidebar: mode selection, top_k, reranker toggle, document upload
    - Sources panel with relevance scores per answer
    - Evaluation tab with RAGAS metrics visualization
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8080"

st.set_page_config(
    page_title="HybridRAG-Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources: list[dict] = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency: float = 0.0
if "last_route" not in st.session_state:
    st.session_state.last_route: str = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/HybridRAG--Pro-v1.0-2196F3?style=for-the-badge&logo=python",
        use_container_width=True,
    )
    st.markdown("### ⚙️ Configuration")

    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid_rag", "simple_rag", "agentic"],
        index=0,
        help="hybrid_rag: Dense+Sparse+RRF+Reranker | agentic: LangGraph router | simple_rag: direct",
    )

    top_k = st.slider(
        "Top-K results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of chunks passed to the LLM after reranking.",
    )

    use_reranker = st.toggle("Enable Cross-Encoder Reranker", value=True)
    use_streaming = st.toggle("Enable Token Streaming", value=False)

    st.divider()
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop a file to index",
        type=["pdf", "txt", "docx", "html", "md"],
        help="Supported: PDF, TXT, DOCX, HTML, Markdown",
    )

    if uploaded_file is not None:
        if st.button("🚀 Index Document", use_container_width=True):
            with st.spinner("Indexing..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(
                        f"✅ Indexed **{data['num_chunks']}** chunks from `{data['filename']}`"
                    )
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        try:
            httpx.delete(f"{API_BASE}/memory", timeout=10.0)
        except Exception:
            pass
        st.rerun()

    st.markdown("### 🟢 System Status")
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5.0).json()
        status_color = "🟢" if health["status"] == "ok" else "🟡"
        st.markdown(
            f"{status_color} **{health['status'].upper()}** — "
            f"`{health['num_indexed_docs']}` docs indexed"
        )
    except Exception:
        st.markdown("🔴 **API unreachable** — start with `uvicorn api.main:app`")


# ---------------------------------------------------------------------------
# Main layout: Chat tab + Evaluation tab
# ---------------------------------------------------------------------------

tab_chat, tab_eval = st.tabs(["  💬 Chat", "  📊 Evaluation"])


# ===========================================================================
# TAB 1: CHAT
# ===========================================================================

with tab_chat:
    st.markdown("## 🧠 HybridRAG-Pro")
    st.caption(
        "Production-ready Hybrid RAG — Dense + Sparse + RRF + CrossEncoder + LangGraph Agent"
    )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if use_streaming and mode != "agentic":
                # SSE streaming mode
                placeholder = st.empty()
                full_response = ""
                try:
                    with httpx.stream(
                        "POST",
                        f"{API_BASE}/query",
                        json={
                            "query": prompt,
                            "top_k": top_k,
                            "use_reranker": use_reranker,
                            "mode": mode,
                            "stream": True,
                        },
                        timeout=60.0,
                    ) as stream_resp:
                        for line in stream_resp.iter_lines():
                            if line.startswith("data:"):
                                token = line[5:].strip()
                                if token == "[DONE]":
                                    break
                                full_response += token
                                placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"❌ Streaming error: {e}"
                    placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            else:
                # Standard (non-streaming) mode
                with st.spinner("🔍 Retrieving and generating..."):
                    try:
                        start = time.perf_counter()
                        resp = httpx.post(
                            f"{API_BASE}/query",
                            json={
                                "query": prompt,
                                "top_k": top_k,
                                "use_reranker": use_reranker,
                                "mode": mode,
                                "stream": False,
                            },
                            timeout=60.0,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        answer = data["answer"]
                        st.session_state.last_sources = data.get("sources", [])
                        st.session_state.last_latency = data.get("latency_ms", 0.0)
                        st.session_state.last_route = data.get("route", mode)
                    except Exception as e:
                        answer = f"❌ API error: {e}"
                        st.session_state.last_sources = []

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    # Sources panel (shown after last answer)
    if st.session_state.last_sources:
        with st.expander(
            f"📎 **Sources** ({len(st.session_state.last_sources)} chunks) — "
            f"latency: `{st.session_state.last_latency}ms` — "
            f"route: `{st.session_state.last_route}`",
            expanded=False,
        ):
            for i, src in enumerate(st.session_state.last_sources, 1):
                score = src.get("score", 0)
                source_name = Path(src.get("source", "unknown")).name
                page = src.get("page", "")
                page_str = f" — p.{page}" if page else ""
                bar_pct = min(int(score * 1000), 100)

                st.markdown(
                    f"**[{i}]** `{source_name}`{page_str} — "
                    f"score: **{score:.4f}**"
                )
                st.progress(bar_pct, text=None)
                st.caption(src.get("content", "")[:300] + "...")
                if i < len(st.session_state.last_sources):
                    st.divider()


# ===========================================================================
# TAB 2: EVALUATION
# ===========================================================================

with tab_eval:
    st.markdown("## 📊 RAGAS Evaluation Dashboard")
    st.caption(
        "Faithfulness • Answer Relevancy • Context Precision • Context Recall"
    )

    col_run, col_refresh = st.columns([2, 1])

    with col_refresh:
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.rerun()

    # Fetch latest RAGAS scores from API
    metrics_data: dict | None = None
    try:
        metrics_resp = httpx.get(f"{API_BASE}/metrics", timeout=10.0)
        if metrics_resp.status_code == 200:
            metrics_data = metrics_resp.json()
    except Exception:
        pass

    if metrics_data:
        scores = metrics_data["scores"]
        source_file = metrics_data.get("source_file", "")
        st.success(f"✅ Results from: `{source_file}`")

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        metric_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
        metric_values = [scores.get(m, 0.0) for m in metric_names]

        # KPI cards
        cols = st.columns(4)
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
        thresholds = [0.85, 0.80, 0.80, 0.75]
        for col, label, value, color, threshold in zip(
            cols, metric_labels, metric_values, colors, thresholds
        ):
            delta = round(value - threshold, 3)
            col.metric(
                label=label.replace("\n", " "),
                value=f"{value:.3f}",
                delta=f"{delta:+.3f} vs target",
                delta_color="normal",
            )

        st.divider()

        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=metric_values + [metric_values[0]],
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.2)",
            line=dict(color="#2196F3", width=2),
            name="HybridRAG-Pro",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.85, 0.80, 0.80, 0.75, 0.85],
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            fillcolor="rgba(255, 152, 0, 0.1)",
            line=dict(color="#FF9800", width=1, dash="dash"),
            name="Target threshold",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="RAGAS Metrics Radar",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"Evaluated on **{scores.get('num_samples', '?')}** samples — "
            f"`{scores.get('evaluated_at', 'unknown')}`"
        )

    else:
        st.info(
            "💡 No evaluation results yet. "
            "Run `python -m src.evaluation.ragas_eval` to generate metrics."
        )
        st.code(
            "# Run evaluation from project root\n"
            "python -c \"from src.evaluation.ragas_eval import RAGASEvaluator; "
            "from src.generation.llm_chain import HybridRAGChain; "
            "from src.ingestion.indexer import HybridIndexer; "
            "i = HybridIndexer(); i.load(); "
            "chain = HybridRAGChain(i); "
            "RAGASEvaluator(chain).run()\"",
            language="bash",
        )
