"""
Tavily Search MCP Client for Void AI RAG pipeline.

Launches the tavily-mcp Node.js MCP server as a subprocess and exposes
a single sync function: search_sync(query) for use in the RAG pipeline.

Requires:
  - Node.js installed (npx available)
  - TAVILY_API_KEY in environment (free tier: 1000 req/month, no credit card)
    Sign up at: https://app.tavily.com
  - pip install mcp
"""

import os
import json
import asyncio
import concurrent.futures
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


async def tavily_web_search(query: str, count: int = 5) -> list[dict]:
    """
    Run a Tavily search query via MCP and return results.

    Returns:
        List of dicts with keys: title, url, description
    """
    if not TAVILY_API_KEY:
        return []

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@0.1.4"],
        env={**os.environ, "TAVILY_API_KEY": TAVILY_API_KEY},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "tavily-search",
                arguments={"query": query, "max_results": count},
            )

            raw = result.content[0].text if result.content else "{}"
            print(f"  [Tavily MCP] raw response type={type(raw)}, length={len(raw)}, preview={raw[:300]!r}")

            # Tavily MCP may return JSON or a plain-text formatted string
            try:
                data = json.loads(raw)
                results = data.get("results", [])
                print(f"  [Tavily MCP] parsed JSON, {len(results)} results")
            except json.JSONDecodeError:
                # Fallback: treat raw as a plain-text answer and return it as one result
                print(f"  [Tavily MCP] not JSON, returning raw text as single result")
                if raw.strip():
                    return [{"title": "Web Search Result", "url": "", "description": raw[:500]}]
                return []

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("content", r.get("snippet", r.get("description", ""))),
                }
                for r in results
            ]


def search_sync(query: str, count: int = 5) -> list[dict]:
    """
    Synchronous wrapper for tavily_web_search.

    Runs the async MCP call in a dedicated thread so it gets its own
    event loop, isolated from FastAPI's running loop.
    """
    def _run():
        return asyncio.run(tavily_web_search(query, count))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=30)
    except Exception as e:
        print(f"  Tavily Search MCP failed: {e}")
        return []
