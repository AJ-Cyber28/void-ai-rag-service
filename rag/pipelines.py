"""
Step 4: Haystack RAG Retrieval Pipelines

Two modes:
  1. Focused — ticker-specific, hybrid retrieval (stock profiles + SEC filings)
  2. Global — cross-universe, searches all chunks with diversity

Hybrid retrieval (focused mode):
  - Always retrieves the 2 stock profile chunks for the ticker
  - Plus top 6 SEC filing chunks by vector similarity
  - LLM always has both scores/metrics AND filing content

Components:
  - SentenceTransformersTextEmbedder (local, bge-small-en-v1.5)
  - PgvectorEmbeddingRetriever (from Haystack pgvector integration)
  - PromptBuilder (constructs the LLM prompt with context)
  - OpenRouter LLM via OpenAI-compatible API (Mistral Medium 3.1)

Usage:
  from rag.pipelines import query_focused, query_global

  result = query_focused("What are key risks?", ticker="KRO", history=[])
  result = query_global("Which stocks have leadership changes?", history=[])
"""

import os
import sys
import pathlib
from typing import List, Dict, Optional
from collections import defaultdict

from dotenv import load_dotenv

# Load env
_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")

# --- Config ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium-3.1")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Retrieval settings
FOCUSED_SEC_TOP_K = 6       # SEC filing chunks per focused query
FOCUSED_PROFILE_TOP_K = 2   # Stock profile chunks (always included)
GLOBAL_TOP_K = 15


# ======================================================================
# SYSTEM PROMPT
# ======================================================================

SYSTEM_PROMPT = """You are Void AI's Analyst Copilot — an AI assistant that helps investors
understand under-covered stocks using SEC filings, market data, and proprietary
coverage gap scores.

Rules:
- Keep responses concise — aim for 3-5 short paragraphs maximum.
- Use bullet points sparingly, not for every detail.
- Lead with the most important insight first.
- When referencing data, briefly note the source (10-K, 10-Q, 8-K, or stock profile).
- Use specific numbers when available but don't list every number you find.
- If the user asks about scores, explain briefly what they mean.
- Do NOT use markdown headers (###). Use plain text with bold (**text**) for emphasis only.
- If the context doesn't have enough info, say so in one sentence.
"""

# ======================================================================
# PROMPT TEMPLATE
# ======================================================================

PROMPT_TEMPLATE = """{{system_prompt}}

{% if history %}
Previous conversation:
{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}
{% endif %}

Context documents:
{% for doc in documents %}
---
[Source: {{ doc.meta.source_type }} | Ticker: {{ doc.meta.ticker }} | Type: {{ doc.meta.form_type }} | Section: {{ doc.meta.section }}]
{{ doc.content }}
---
{% endfor %}

User question: {{query}}

Answer:"""


# ======================================================================
# COMPONENT INITIALIZATION (lazy, singleton)
# ======================================================================

_components = {}


def _init_components():
    """Initialize all components once and cache them."""
    if _components:
        return

    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators.openai import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
    from rag.document_store import get_document_store

    store = get_document_store()

    # Embedder (local, shared by both modes)
    embedder = SentenceTransformersTextEmbedder(model=EMBED_MODEL)
    embedder.warm_up()

    # SEC filing retriever (for focused mode — only SEC filings)
    sec_retriever = PgvectorEmbeddingRetriever(
        document_store=store,
        top_k=FOCUSED_SEC_TOP_K,
    )

    # Profile retriever (for focused mode — only stock profiles)
    profile_retriever = PgvectorEmbeddingRetriever(
        document_store=store,
        top_k=FOCUSED_PROFILE_TOP_K,
    )

    # Global retriever (no filter, higher top_k)
    global_retriever = PgvectorEmbeddingRetriever(
        document_store=store,
        top_k=GLOBAL_TOP_K,
    )

    # Prompt builder
    prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)

    # LLM via OpenRouter (OpenAI-compatible)
    llm = OpenAIGenerator(
        api_key=Secret.from_token(OPENROUTER_API_KEY),
        model=OPENROUTER_MODEL,
        api_base_url=OPENROUTER_BASE_URL,
        generation_kwargs={"max_tokens": 1000},
    )

    _components["embedder"] = embedder
    _components["sec_retriever"] = sec_retriever
    _components["profile_retriever"] = profile_retriever
    _components["global_retriever"] = global_retriever
    _components["prompt_builder"] = prompt_builder
    _components["llm"] = llm


# ======================================================================
# DIVERSITY HELPER (for global mode)
# ======================================================================

def diversify_results(documents, max_per_ticker=2, max_total=10):
    """
    Ensure global results span multiple tickers.
    Takes top max_per_ticker docs per ticker, up to max_total.
    """
    ticker_docs = defaultdict(list)
    for doc in documents:
        ticker = doc.meta.get("ticker", "unknown")
        if len(ticker_docs[ticker]) < max_per_ticker:
            ticker_docs[ticker].append(doc)

    diversified = []
    seen_ids = set()
    for doc in documents:
        ticker = doc.meta.get("ticker", "unknown")
        if doc.id not in seen_ids and doc in ticker_docs[ticker]:
            diversified.append(doc)
            seen_ids.add(doc.id)
        if len(diversified) >= max_total:
            break

    return diversified


# ======================================================================
# PUBLIC QUERY FUNCTIONS
# ======================================================================

def query_focused(
    query: str,
    ticker: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Run a ticker-specific RAG query with hybrid retrieval.

    Always includes:
      - 2 stock profile chunks (company overview + coverage analysis)
      - Top 6 SEC filing chunks by similarity

    Args:
        query: The user's question
        ticker: Stock ticker to filter on (e.g. "KRO")
        history: List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        dict with "reply" (str) and "documents" (list of retrieved docs)
    """
    _init_components()

    # Step 1: Embed the query
    embed_result = _components["embedder"].run(text=query)
    query_embedding = embed_result["embedding"]

    # Step 2a: Retrieve stock profile chunks for this ticker
    profile_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.ticker", "operator": "==", "value": ticker},
            {"field": "meta.source_type", "operator": "==", "value": "stock_profile"},
        ]
    }
    profile_result = _components["profile_retriever"].run(
        query_embedding=query_embedding,
        filters=profile_filters,
    )
    profile_docs = profile_result["documents"]

    # Step 2b: Retrieve SEC filing chunks for this ticker
    sec_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.ticker", "operator": "==", "value": ticker},
            {"field": "meta.source_type", "operator": "==", "value": "sec_filing"},
        ]
    }
    sec_result = _components["sec_retriever"].run(
        query_embedding=query_embedding,
        filters=sec_filters,
    )
    sec_docs = sec_result["documents"]

    # Step 2c: Combine — profiles first (always present), then SEC filings
    documents = profile_docs + sec_docs

    # Step 3: Build prompt
    prompt_result = _components["prompt_builder"].run(
        query=query,
        system_prompt=SYSTEM_PROMPT,
        history=history or [],
        documents=documents,
    )

    # Step 4: Generate with LLM
    llm_result = _components["llm"].run(prompt=prompt_result["prompt"])

    reply = llm_result["replies"][0] if llm_result["replies"] else ""
    return {"reply": reply, "documents": documents}


def query_global(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Run a cross-universe RAG query (no ticker filter).

    Args:
        query: The user's question
        history: List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        dict with "reply" (str) and "documents" (list of retrieved docs)
    """
    _init_components()

    # Step 1: Embed the query
    embed_result = _components["embedder"].run(text=query)
    query_embedding = embed_result["embedding"]

    # Step 2: Retrieve (no ticker filter)
    retriever_result = _components["global_retriever"].run(
        query_embedding=query_embedding,
    )
    raw_docs = retriever_result["documents"]

    # Step 3: Diversify results
    documents = diversify_results(raw_docs)

    # Step 4: Build prompt
    prompt_result = _components["prompt_builder"].run(
        query=query,
        system_prompt=SYSTEM_PROMPT,
        history=history or [],
        documents=documents,
    )

    # Step 5: Generate with LLM
    llm_result = _components["llm"].run(prompt=prompt_result["prompt"])

    reply = llm_result["replies"][0] if llm_result["replies"] else ""
    return {"reply": reply, "documents": documents}