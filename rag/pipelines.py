"""
Step 4: Haystack RAG Retrieval Pipelines

Two modes:
  1. Focused — ticker-specific, hybrid retrieval (stock profiles + SEC filings)
  2. Global — cross-universe, searches all chunks with diversity

Embeddings: HuggingFace Inference API (router.huggingface.co) for fast query-time embedding.
LLM: OpenRouter API (Mistral Medium 3.1)

Usage:
  from rag.pipelines import query_focused, query_global
"""

import os
import sys
import pathlib
import requests
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from dotenv import load_dotenv

# Load env
_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")

# --- Config ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_EMBED_URL = f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium-3.1")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Retrieval settings
FOCUSED_SEC_TOP_K = 6
FOCUSED_PROFILE_TOP_K = 2
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
# HF ROUTER API EMBEDDER (replaces local SentenceTransformers)
# ======================================================================

def embed_query(text: str) -> List[float]:
    """Embed a single query using HuggingFace Inference API (router endpoint).
    Returns a 384-dimensional normalized vector.
    ~0.5-1s per query vs ~24s with local model on Railway's CPU.
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": text},
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    # API returns [[...]] for single input or [...] directly
    vec = result[0] if isinstance(result[0], list) else result

    # Normalize (bge models perform best with normalized vectors)
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ======================================================================
# COMPONENT INITIALIZATION (lazy, singleton)
# ======================================================================

_components = {}


def _init_components():
    """Initialize retriever, prompt builder, and LLM components once."""
    if _components:
        return

    from haystack.components.builders import PromptBuilder
    from haystack.components.generators.openai import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
    from rag.document_store import get_document_store

    store = get_document_store()

    # SEC filing retriever (for focused mode)
    sec_retriever = PgvectorEmbeddingRetriever(
        document_store=store,
        top_k=FOCUSED_SEC_TOP_K,
    )

    # Profile retriever (for focused mode)
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

    # LLM via OpenRouter
    llm = OpenAIGenerator(
        api_key=Secret.from_token(OPENROUTER_API_KEY),
        model=OPENROUTER_MODEL,
        api_base_url=OPENROUTER_BASE_URL,
        generation_kwargs={"max_tokens": 600},
    )

    _components["sec_retriever"] = sec_retriever
    _components["profile_retriever"] = profile_retriever
    _components["global_retriever"] = global_retriever
    _components["prompt_builder"] = prompt_builder
    _components["llm"] = llm


# ======================================================================
# DIVERSITY HELPER (for global mode)
# ======================================================================

def diversify_results(documents, max_per_ticker=2, max_total=10):
    """Ensure global results span multiple tickers."""
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
    """
    _init_components()

    # Step 1: Embed the query via HF API
    query_embedding = embed_query(query)

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

    # Step 2c: Combine — profiles first, then SEC filings
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
    """
    _init_components()

    # Step 1: Embed the query via HF API
    query_embedding = embed_query(query)

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