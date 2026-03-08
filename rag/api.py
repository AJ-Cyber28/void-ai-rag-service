"""
Void AI RAG + CrewAI Analysis Service

Endpoints:
  POST /chat          — RAG chat (focused or global mode)
  POST /chat/stream   — RAG chat with SSE streaming
  GET  /analysis/{ticker}  — Fetch cached AI analysis
  POST /analyze/{ticker}   — Generate AI analysis via CrewAI agents
  GET  /health        — Health check

Called by the Next.js frontend (Vercel).
Deployed on Railway.

Usage:
  uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload

Env (.env.local):
  PG_CONN_STRING, OPENROUTER_API_KEY, OPENROUTER_MODEL,
  SUPABASE_URL, SUPABASE_ANON_KEY, FINNHUB_API_KEY
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import pathlib
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

# Path setup
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env.local")

# Import pipelines (lazy loaded on first request)
from rag.pipelines import query_focused, query_global, query_focused_stream, query_global_stream

# Import CrewAI analysis module
from rag.crew_analysis import (
    gather_analysis_context,
    run_analysis_crew,
    get_cached_analysis,
    upsert_analysis,
    format_analysis_response,
)


# ======================================================================
# APP SETUP
# ======================================================================

app = FastAPI(
    title="Void AI RAG Service",
    description="RAG-powered chat API + CrewAI analysis for Void AI's Analyst Copilot",
    version="2.0.0",
)

# CORS — allow your Vercel frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",             # Local dev
        "https://void-ai-nine.vercel.app",   # Your Vercel deployment
        "https://*.vercel.app",              # Any Vercel preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# REQUEST / RESPONSE MODELS
# ======================================================================

class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None    # None = global mode, "COKE" = focused mode
    history: List[ChatMessage] = []  # Previous messages in this session


class SourceDocument(BaseModel):
    ticker: str
    source_type: str        # "stock_profile" or "sec_filing"
    form_type: Optional[str]
    section: Optional[str]
    snippet: str            # First 200 chars of content


class ChatResponse(BaseModel):
    reply: str
    mode: str                       # "focused" or "global"
    ticker: Optional[str]
    sources: List[SourceDocument]   # Retrieved docs metadata for UI


# --- Analysis models ---

class DebateStep(BaseModel):
    agent: str
    role: str
    summary: str
    fullOutput: str


class NewsItem(BaseModel):
    headline: str
    summary: str = ""
    source: str = ""
    datetime: str = ""
    url: str = ""


class AnalysisResponse(BaseModel):
    ticker: str
    hypothesis: str
    confidence: float
    bullCase: dict
    baseCase: dict
    bearCase: dict
    catalysts: list
    risks: list
    debateTranscript: list = []
    newsContext: list = []
    generatedAt: str = ""
    modelUsed: str = ""
    isStale: bool = False


# ======================================================================
# CHAT ENDPOINTS (existing)
# ======================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    - If `ticker` is provided → focused mode (ticker-specific)
    - If `ticker` is null/empty → global mode (cross-universe)
    """
    try:
        # Convert history to list of dicts
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        if request.ticker:
            # Focused mode
            result = query_focused(
                query=request.message,
                ticker=request.ticker.upper(),
                history=history,
            )
            mode = "focused"
        else:
            # Global mode
            result = query_global(
                query=request.message,
                history=history,
            )
            mode = "global"

        # Build source documents for the response
        sources = []
        for doc in result["documents"]:
            sources.append(SourceDocument(
                ticker=doc.meta.get("ticker", "?"),
                source_type=doc.meta.get("source_type", "?"),
                form_type=doc.meta.get("form_type"),
                section=doc.meta.get("section"),
                snippet=doc.content[:200].replace("\n", " "),
            ))

        return ChatResponse(
            reply=result["reply"],
            mode=mode,
            ticker=request.ticker.upper() if request.ticker else None,
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    
    - If `ticker` is provided → focused mode (ticker-specific)
    - If `ticker` is null/empty → global mode (cross-universe)
    """
    try:
        # Convert history to list of dicts
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        if request.ticker:
            # Focused mode
            result = query_focused_stream(
                query=request.message,
                ticker=request.ticker.upper(),
                history=history,
            )
        else:
            # Global mode
            result = query_global_stream(
                query=request.message,
                history=history,
            )

        # Build source documents for the response
        sources = []
        for doc in result["documents"]:
            sources.append({
                "ticker": doc.meta.get("ticker", "?"),
                "source_type": doc.meta.get("source_type", "?"),
                "form_type": doc.meta.get("form_type"),
                "section": doc.meta.get("section"),
                "snippet": doc.content[:200].replace("\n", " "),
            })

        async def event_generator():
            try:
                # Stream LLM tokens
                for token in result["stream"]:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                # Send sources at the end
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# CREWAI ANALYSIS ENDPOINTS (new)
# ======================================================================

@app.get("/analysis/{ticker}")
async def get_analysis(ticker: str):
    """
    Retrieve cached AI analysis for a ticker.
    Returns 404 if no analysis exists.
    """
    ticker = ticker.upper()

    cached = get_cached_analysis(ticker)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for {ticker}. Use POST /analyze/{ticker} to generate."
        )

    return format_analysis_response(cached)


@app.post("/analyze/{ticker}")
async def analyze_stock(ticker: str, force: bool = Query(False)):
    """
    Generate AI analysis for a ticker using CrewAI agents.

    - If fresh cache exists and force=False, returns cached version.
    - Otherwise runs the full 5-agent crew (~15-30 seconds).

    Query params:
        force (bool): If true, bypass cache and regenerate.
    """
    ticker = ticker.upper()

    # Check cache first (unless forced)
    if not force:
        cached = get_cached_analysis(ticker)
        if cached and not cached.get("is_stale", False):
            return format_analysis_response(cached)

    # Gather all context data
    try:
        context = gather_analysis_context(ticker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to gather data for {ticker}: {str(e)}"
        )

    # Run CrewAI analysis
    try:
        analysis = run_analysis_crew(context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis generation failed for {ticker}: {str(e)}"
        )

    # Cache the result
    gap_score = context.get("scores", {}).get("gap_score")
    if gap_score is not None:
        gap_score = float(gap_score)
    upsert_analysis(ticker, analysis, gap_score)

    # Format and return
    response = format_analysis_response({
        "ticker": ticker,
        **analysis,
        "generated_at": datetime.utcnow().isoformat(),
        "model_used": os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium-3.1"),
        "is_stale": False,
    })

    return response


# ======================================================================
# UTILITY ENDPOINTS
# ======================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "void-ai-rag"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Void AI RAG + Analysis Service",
        "version": "2.0.0",
        "endpoints": {
            "POST /chat": "RAG chat endpoint (focused or global mode)",
            "POST /chat/stream": "RAG chat with SSE streaming",
            "GET /analysis/{ticker}": "Fetch cached AI analysis for a stock",
            "POST /analyze/{ticker}": "Generate AI analysis using CrewAI agents (force=true to bypass cache)",
            "GET /health": "Health check",
        }
    }