"""
Step 5: FastAPI RAG Service

Single endpoint that serves both focused (ticker-specific) and global queries.
Called by the Next.js frontend from:
  1. Explore & Chat page (focused or global mode)
  2. Individual stock page Chat tab (always focused)

Usage:
  uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload

Env (.env.local):
  PG_CONN_STRING, OPENROUTER_API_KEY, OPENROUTER_MODEL
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import pathlib
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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


# ======================================================================
# APP SETUP
# ======================================================================

app = FastAPI(
    title="Void AI RAG Service",
    description="RAG-powered chat API for Void AI's Analyst Copilot",
    version="1.0.0",
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


# ======================================================================
# ENDPOINTS
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "void-ai-rag"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Void AI RAG Service",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Main chat endpoint (focused or global mode)",
            "GET /health": "Health check",
        }
    }