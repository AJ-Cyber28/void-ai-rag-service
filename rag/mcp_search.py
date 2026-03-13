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
            # Tavily returns a JSON string with a "results" array
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return []

            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("content", r.get("snippet", "")),
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
