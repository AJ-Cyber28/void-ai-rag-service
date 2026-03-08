"""
CrewAI Investment Analysis System for Void AI.

Runs a crew of 5 specialized agents to generate structured investment analysis
for a given stock ticker using data from Supabase, RAG (SEC filings), and Finnhub news.

Agents (sequential):
  1. Data Analyst         — quantitative data brief
  2. Coverage Specialist  — gap score interpretation
  3. Fundamental Researcher — SEC filings + news insights
  4. Investment Strategist — synthesizes bull/base/bear cases
  5. Devil's Advocate      — challenges and refines the thesis

Usage:
    from rag.crew_analysis import gather_analysis_context, run_analysis_crew
    context = gather_analysis_context("MHK")
    result = run_analysis_crew(context)

Env:
    OPENROUTER_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, FINNHUB_API_KEY
"""

import os
import sys
import json
import pathlib
import requests
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# Path setup
_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")

# --- Config ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium-3.1")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Initialize Supabase client
_supabase: Optional[Client] = None


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL or SUPABASE_ANON_KEY not set")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ======================================================================
# FINNHUB NEWS FETCHER
# ======================================================================

def fetch_finnhub_news(ticker: str, days: int = 30, limit: int = 5) -> list:
    """
    Fetch recent company news from Finnhub.
    Returns a list of dicts: {headline, summary, source, datetime, url}
    Gracefully returns empty list on failure or no news.
    """
    if not FINNHUB_API_KEY:
        print(f"  ⚠️ FINNHUB_API_KEY not set — skipping news for {ticker}")
        return []

    today = datetime.utcnow()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/company-news",
            params={
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        articles = response.json()

        if not isinstance(articles, list):
            return []

        # Sort by datetime (most recent first), take top N
        articles.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        results = []
        for article in articles[:limit]:
            dt = article.get("datetime", 0)
            results.append({
                "headline": article.get("headline", ""),
                "summary": (article.get("summary", "") or "")[:300],
                "source": article.get("source", ""),
                "datetime": datetime.fromtimestamp(dt).strftime("%Y-%m-%d") if dt else "",
                "url": article.get("url", ""),
            })
        return results

    except Exception as e:
        print(f"  ⚠️ Finnhub news fetch failed for {ticker}: {e}")
        return []


# ======================================================================
# DATA GATHERING (PRE-CREW)
# ======================================================================

def _fetch_single(table: str, ticker: str, select: str = "*") -> dict:
    """Fetch a single row from Supabase by ticker. Returns {} on failure."""
    try:
        res = _get_supabase().table(table).select(select).eq("ticker", ticker).limit(1).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        print(f"  ⚠️ Failed to fetch {table} for {ticker}: {e}")
        return {}


def _fetch_many(table: str, select: str, filters: dict = None, limit: int = 10) -> list:
    """Fetch multiple rows from Supabase. Returns [] on failure."""
    try:
        q = _get_supabase().table(table).select(select)
        if filters:
            for key, val in filters.items():
                if key.startswith("neq:"):
                    q = q.neq(key[4:], val)
                else:
                    q = q.eq(key, val)
        res = q.limit(limit).execute()
        return res.data or []
    except Exception as e:
        print(f"  ⚠️ Failed to fetch {table}: {e}")
        return []


def _retrieve_rag_chunks(ticker: str, source_type: str, top_k: int) -> list:
    """
    Retrieve RAG chunks from Haystack document store for a ticker.
    Returns list of dicts with content and metadata.
    """
    try:
        from rag.pipelines import embed_query, _init_components, _components

        _init_components()

        # Use a generic query to get profile/SEC chunks
        query_text = f"investment analysis {ticker} coverage gap risks opportunities"
        query_embedding = embed_query(query_text)

        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.ticker", "operator": "==", "value": ticker},
                {"field": "meta.source_type", "operator": "==", "value": source_type},
            ]
        }

        retriever_key = "profile_retriever" if source_type == "stock_profile" else "sec_retriever"
        retriever = _components.get(retriever_key, _components.get("global_retriever"))

        result = retriever.run(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )

        chunks = []
        for doc in result.get("documents", []):
            chunks.append({
                "content": doc.content,
                "source_type": doc.meta.get("source_type", "unknown"),
                "form_type": doc.meta.get("form_type", ""),
                "section": doc.meta.get("section", "general"),
            })
        return chunks

    except Exception as e:
        print(f"  ⚠️ RAG retrieval failed for {ticker} ({source_type}): {e}")
        return []


def gather_analysis_context(ticker: str) -> dict:
    """
    Pull all data needed by the crew for a single ticker.
    Returns a dict with: company, metrics, coverage, scores, peers, sec_chunks, profile_chunks, news
    """
    print(f"\n📊 Gathering analysis context for {ticker}...")

    # 1. Company info
    company = _fetch_single("companies", ticker, "ticker, name, sector, industry, market_cap, cap_type")
    if not company:
        raise ValueError(f"Ticker {ticker} not found in companies table")
    print(f"  ✅ Company: {company.get('name', ticker)}")

    # 2. Stock metrics
    metrics = _fetch_single("stock_metrics", ticker)
    print(f"  ✅ Metrics: {'found' if metrics else 'missing'}")

    # 3. Analyst coverage
    coverage = _fetch_single("analyst_coverage", ticker)
    print(f"  ✅ Coverage: {coverage.get('analyst_count', 'N/A')} analysts")

    # 4. Gap scores
    scores = _fetch_single("coverage_gap_scores", ticker)
    print(f"  ✅ Gap Score: {scores.get('gap_score', 'N/A')}")

    # 5. Sector peers
    sector = company.get("sector", "")
    peers = []
    if sector:
        try:
            res = _get_supabase().table("companies") \
                .select("ticker, name, market_cap, sector") \
                .eq("sector", sector) \
                .neq("ticker", ticker) \
                .limit(10).execute()
            peers = res.data or []
        except Exception:
            pass

    peer_tickers = [p["ticker"] for p in peers[:5]]
    peer_coverage = []
    if peer_tickers:
        try:
            res = _get_supabase().table("analyst_coverage") \
                .select("ticker, analyst_count") \
                .in_("ticker", peer_tickers).execute()
            peer_coverage = res.data or []
        except Exception:
            pass
    print(f"  ✅ Peers: {len(peers)} sector peers")

    # 6. SEC filing chunks from RAG
    sec_chunks = _retrieve_rag_chunks(ticker, "sec_filing", top_k=8)
    print(f"  ✅ SEC chunks: {len(sec_chunks)}")

    # 7. Stock profile chunks from RAG
    profile_chunks = _retrieve_rag_chunks(ticker, "stock_profile", top_k=2)
    print(f"  ✅ Profile chunks: {len(profile_chunks)}")

    # 8. Recent news from Finnhub
    news = fetch_finnhub_news(ticker, days=30, limit=5)
    print(f"  ✅ News articles: {len(news)}")

    return {
        "company": company,
        "metrics": metrics,
        "coverage": coverage,
        "scores": scores,
        "peers": peers[:5],
        "peer_coverage": peer_coverage,
        "sec_chunks": sec_chunks,
        "profile_chunks": profile_chunks,
        "news": news,
    }


# ======================================================================
# CREWAI AGENT & TASK DEFINITIONS
# ======================================================================

def run_analysis_crew(context: dict) -> dict:
    """
    Execute the full 5-agent crew and return parsed analysis JSON + debate transcript.

    Returns a dict matching the ai_analyses schema:
    {
        hypothesis, confidence, bull_case, base_case, bear_case,
        catalysts, risks, debate_transcript, news_context
    }
    """
    from crewai import Agent, Task, Crew, Process, LLM

    ticker = context["company"]["ticker"]
    company_name = context["company"].get("name", ticker)
    print(f"\n🤖 Starting CrewAI analysis for {company_name} ({ticker})...")

    # --- LLM config ---
    llm = LLM(
        model=f"openrouter/{OPENROUTER_MODEL}",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
    )

    # --- Agent definitions ---
    data_analyst = Agent(
        role="Quantitative Data Analyst",
        goal=f"Produce a precise numerical brief for {ticker} using market data, peer comparisons, and recent news",
        backstory=(
            "You are a quantitative equity analyst who specializes in distilling raw market data "
            "into clear, actionable briefs. You focus on numbers, trends, peer context, and "
            "any recent news that explains price action — never opinions or recommendations."
        ),
        llm=llm,
        verbose=False,
    )

    coverage_specialist = Agent(
        role="Coverage Gap Specialist",
        goal=f"Interpret why {ticker} is under-covered and what that means for investors",
        backstory=(
            "You are Void AI's proprietary coverage gap expert. You deeply understand the "
            "Gap Score methodology (50% Coverage + 30% Activity + 20% Quality) and can explain "
            "what coverage gaps mean for price discovery and alpha opportunity."
        ),
        llm=llm,
        verbose=False,
    )

    fundamental_researcher = Agent(
        role="Fundamental Research Analyst",
        goal=f"Extract key business insights from SEC filings, news, and fundamentals for {ticker}",
        backstory=(
            "You are a fundamental equity research analyst who reads SEC filings (10-K, 10-Q, 8-K) "
            "and recent news to understand a company's business model, risks, competitive position, "
            "and recent material events."
        ),
        llm=llm,
        verbose=False,
    )

    investment_strategist = Agent(
        role="Senior Investment Strategist",
        goal=f"Synthesize all research into a structured investment analysis with bull/base/bear cases for {ticker}",
        backstory=(
            "You are a senior investment strategist who takes inputs from your research team "
            "and produces actionable investment theses. You think in scenarios (bull/base/bear) "
            "and always ground your analysis in specific data points."
        ),
        llm=llm,
        verbose=False,
    )

    devils_advocate = Agent(
        role="Devil's Advocate Analyst",
        goal=f"Challenge and refine the investment analysis for {ticker} to make it more balanced and robust",
        backstory=(
            "You are a contrarian analyst whose job is to stress-test investment theses. "
            "You look for over-optimism, under-appreciated risks, and logical gaps. "
            "You make analyses stronger, not weaker."
        ),
        llm=llm,
        verbose=False,
    )

    # --- Format context data for prompts ---
    news_text = json.dumps(context.get("news", []), indent=2) if context.get("news") else "No recent news available."

    sec_text = "\n\n".join([
        f"[{c.get('source_type', 'unknown')} | {c.get('form_type', 'N/A')} | Section: {c.get('section', 'general')}]\n{c.get('content', '')}"
        for c in context.get("sec_chunks", [])
    ]) or "No SEC filing data available for this ticker."

    profile_text = "\n\n".join([
        c.get("content", "") for c in context.get("profile_chunks", [])
    ]) or "No stock profile data available."

    peer_info = json.dumps(context.get("peers", [])[:5], indent=2)
    peer_cov = json.dumps(context.get("peer_coverage", []), indent=2)

    # --- Task definitions ---
    task_data_brief = Task(
        description=f"""Analyze the following market data for {company_name} ({ticker}) and produce a structured data brief.

COMPANY: {json.dumps(context['company'], indent=2, default=str)}
METRICS: {json.dumps(context['metrics'], indent=2, default=str)}
COVERAGE: {json.dumps(context['coverage'], indent=2, default=str)}
PEER COMPANIES: {peer_info}
PEER ANALYST COVERAGE: {peer_cov}

RECENT NEWS (last 30 days):
{news_text}

Produce a concise data brief covering:
1. Current price positioning (vs 52-week range)
2. Volume and trading activity trends
3. Price momentum (1-month and 3-month)
4. Volatility profile
5. How this stock compares to its sector peers on these metrics
6. Any notable recent news events that may explain recent price/volume action""",
        expected_output="A structured quantitative brief with specific numbers, peer comparisons, and relevant news context. Around 300-500 words.",
        agent=data_analyst,
    )

    task_gap_analysis = Task(
        description=f"""Interpret the coverage gap for {company_name} ({ticker}).

SCORES: {json.dumps(context['scores'], indent=2, default=str)}
COVERAGE: {json.dumps(context['coverage'], indent=2, default=str)}
PEER ANALYST COVERAGE: {peer_cov}

Scoring methodology:
- Coverage Score (50%): How under-covered vs sector/size peers. 0 analysts = highest score.
- Activity Score (30%): Volume, volatility, and momentum signals. High activity = market interest.
- Quality Score (20%): Market cap, liquidity, data completeness. Filters out untradeable names.
- Gap Score = weighted combination. Higher = bigger opportunity.
- Opportunity types: High Priority (75+), Strong Opportunity (60-74), Moderate (45-59), Low Priority (<45)

Using the data brief from your colleague, explain:
1. Why this stock is under-covered relative to its peers
2. Whether this is a sector-wide pattern or company-specific
3. What the high activity + low coverage implies for price discovery
4. How confident we should be in this thesis given the confidence score""",
        expected_output="An analytical interpretation of the coverage gap with specific score references. Around 200-400 words.",
        agent=coverage_specialist,
        context=[task_data_brief],
    )

    task_fundamental = Task(
        description=f"""Analyze the SEC filings, recent news, and fundamental data for {company_name} ({ticker}).

STOCK PROFILE:
{profile_text}

SEC FILING EXCERPTS:
{sec_text}

RECENT NEWS (last 30 days):
{news_text}

Extract and summarize:
1. Key business risks from risk factors
2. Management's forward-looking statements (MD&A)
3. Recent material events from BOTH filings AND news headlines (earnings, leadership, acquisitions, restructuring, partnerships, lawsuits)
4. Competitive positioning and business model strengths/weaknesses
5. If SEC data or news is limited, note what's missing and work with available data

Use news headlines to identify the most current developments that SEC filings (which are backward-looking) may not yet capture.""",
        expected_output="A qualitative fundamental research summary with filing citations and news references. Around 300-500 words.",
        agent=fundamental_researcher,
    )

    task_synthesize = Task(
        description=f"""Synthesize all research from your team into a complete investment analysis for {company_name} ({ticker}).

You have access to:
- The quantitative data brief (Agent 1)
- The coverage gap interpretation (Agent 2)
- The fundamental research summary (Agent 3)

You MUST output ONLY valid JSON (no markdown code fences, no preamble, no trailing text) with this EXACT structure:
{{
  "hypothesis": "2-4 sentence investment thesis specific to this stock's coverage gap opportunity",
  "confidence": <integer 0-100>,
  "bullCase": {{"title": "Bull Case", "points": ["point 1", "point 2", "point 3", "point 4"]}},
  "baseCase": {{"title": "Base Case", "points": ["point 1", "point 2", "point 3"]}},
  "bearCase": {{"title": "Bear Case", "points": ["point 1", "point 2", "point 3", "point 4"]}},
  "catalysts": [{{"event": "Catalyst name", "date": "Q1 2026"}}, {{"event": "Another", "date": "H1 2026"}}],
  "risks": [{{"risk": "Risk name", "severity": "high"}}, {{"risk": "Another", "severity": "medium"}}]
}}

Rules:
- hypothesis: Must reference the specific coverage gap opportunity — why this stock is under-covered and why that matters
- confidence: Reflect data quality, gap score confidence, and SEC data availability
- bullCase: 3-4 specific, data-backed points (each point should be 1 sentence)
- baseCase: 2-3 realistic continuation scenario points
- bearCase: 3-4 genuine risks, not generic warnings
- catalysts: 3-5 with realistic timeframes (Q1 2026, H1 2026, Monthly, etc.)
- risks: 3-5 with severity ratings (high/medium/low) based on likelihood and impact

CRITICAL: Output ONLY the JSON object. No markdown, no ```json, no explanation before or after.""",
        expected_output="A valid JSON object with hypothesis, confidence, bullCase, baseCase, bearCase, catalysts, and risks.",
        agent=investment_strategist,
        context=[task_data_brief, task_gap_analysis, task_fundamental],
    )

    task_refine = Task(
        description=f"""Review and refine the investment analysis JSON for {company_name} ({ticker}).

The Investment Strategist produced a JSON analysis. Your job:
1. Challenge overly optimistic bull case points — are they realistic given the data?
2. Ensure bear case points are genuinely concerning, not just generic "risks exist"
3. Adjust confidence if the thesis seems over/under-confident given the available data
4. Add any missing catalysts or risks you noticed from the research
5. Sharpen the hypothesis to be more specific and actionable

STRICT OUTPUT RULES:
- Output ONLY a valid JSON object. Nothing else.
- Do NOT wrap in markdown code fences (no ```json or ```).
- Do NOT add any text before or after the JSON.
- Do NOT use markdown formatting (no ** or ### or --- ) inside JSON string values.
- Do NOT add extra fields beyond what is specified below.
- Keep all string values as plain text without any markdown.

The JSON must have EXACTLY these 7 top-level keys and no others:
{{
  "hypothesis": "plain text, 2-4 sentences",
  "confidence": <integer 0-100>,
  "bullCase": {{"title": "Bull Case", "points": ["plain text point", ...]}},
  "baseCase": {{"title": "Base Case", "points": ["plain text point", ...]}},
  "bearCase": {{"title": "Bear Case", "points": ["plain text point", ...]}},
  "catalysts": [{{"event": "plain text", "date": "timeframe"}}, ...],
  "risks": [{{"risk": "plain text", "severity": "high|medium|low"}}, ...]
}}

Do NOT add fields like "keyMissingData", "revisedThesis", "alternativeViewpoints", "finalAssessment", "notes", "probability", "mitigation", or "counter". ONLY the 7 keys above.""",
        expected_output="A single valid JSON object with exactly 7 keys: hypothesis, confidence, bullCase, baseCase, bearCase, catalysts, risks. No markdown, no extra fields.",
        agent=devils_advocate,
        context=[task_data_brief, task_gap_analysis, task_fundamental, task_synthesize],
    )

    tasks = [task_data_brief, task_gap_analysis, task_fundamental, task_synthesize, task_refine]

    # --- Execute crew ---
    crew = Crew(
        agents=[data_analyst, coverage_specialist, fundamental_researcher, investment_strategist, devils_advocate],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    print("  🚀 Crew kickoff...")
    start_time = time.time()
    result = crew.kickoff(inputs={"ticker": ticker})
    elapsed = time.time() - start_time
    print(f"  ✅ Crew finished in {elapsed:.1f}s")

    # --- Parse final JSON output ---
    raw_output = str(result.raw) if hasattr(result, "raw") else str(result)
    analysis = _parse_crew_json(raw_output)

    # If Devil's Advocate output failed to parse, try the Strategist's output (task 4)
    if analysis.get("confidence", 0) == 0 and analysis.get("hypothesis", "").startswith("Analysis generation"):
        print("  ⚠️ Devil's Advocate output unparseable, falling back to Strategist output...")
        strategist_output = str(tasks[3].output) if tasks[3].output else ""
        if strategist_output:
            fallback = _parse_crew_json(strategist_output)
            if fallback.get("confidence", 0) > 0:
                analysis = fallback
                print("  ✅ Successfully parsed Strategist output as fallback")

    # --- Build debate transcript ---
    agent_labels = [
        ("Data Analyst", "Quantitative Brief"),
        ("Coverage Specialist", "Gap Interpretation"),
        ("Fundamental Analyst", "SEC & News Insights"),
        ("Strategist", "Initial Thesis"),
        ("Devil's Advocate", "Final Refinement"),
    ]

    debate_transcript = []
    for i, (agent_name, role_label) in enumerate(agent_labels):
        full_output = str(tasks[i].output) if tasks[i].output else "Output not captured."
        # Extract a clean summary (first ~200 chars, ending at a sentence boundary)
        summary = _make_summary(full_output)
        debate_transcript.append({
            "agent": agent_name,
            "role": role_label,
            "summary": summary,
            "fullOutput": full_output,
        })

    # --- Assemble final result ---
    analysis["debate_transcript"] = debate_transcript
    analysis["news_context"] = context.get("news", [])

    return analysis


# ======================================================================
# JSON PARSING HELPERS
# ======================================================================

def _parse_crew_json(raw: str) -> dict:
    """
    Parse JSON from CrewAI output, handling markdown fences, preamble, and
    markdown formatting inside string values (e.g., **bold**).
    Falls back to extracting JSON from mixed text if needed.
    """
    clean = raw.strip()

    # Strip markdown code fences
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = lines[1:]  # Remove first line (```json or ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    # Also strip trailing ``` if present anywhere
    if clean.endswith("```"):
        clean = clean[:clean.rfind("```")].strip()

    # Try direct parse first
    try:
        parsed = json.loads(clean)
        return _normalize_analysis(parsed)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text (brace matching)
    brace_start = clean.find("{")
    if brace_start != -1:
        # Find the matching closing brace by counting depth
        depth = 0
        for i in range(brace_start, len(clean)):
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
                if depth == 0:
                    json_str = clean[brace_start:i + 1]
                    try:
                        parsed = json.loads(json_str)
                        return _normalize_analysis(parsed)
                    except json.JSONDecodeError:
                        pass
                    break

    # Try stripping markdown bold (**text**) from the raw string, then parse
    import re
    stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)  # Remove **bold**
    stripped = re.sub(r'###?\s*', '', stripped)  # Remove ### headers
    stripped = re.sub(r'---+', '', stripped)  # Remove --- dividers

    brace_start = stripped.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    json_str = stripped[brace_start:i + 1]
                    try:
                        parsed = json.loads(json_str)
                        return _normalize_analysis(parsed)
                    except json.JSONDecodeError:
                        pass
                    break

    # Fallback: return a minimal valid structure
    print(f"  ⚠️ Failed to parse crew JSON, using fallback. Raw output length: {len(raw)}")
    return {
        "hypothesis": "Analysis generation encountered a parsing issue. Please regenerate.",
        "confidence": 0,
        "bullCase": {"title": "Bull Case", "points": ["Unable to parse — please regenerate"]},
        "baseCase": {"title": "Base Case", "points": ["Unable to parse — please regenerate"]},
        "bearCase": {"title": "Bear Case", "points": ["Unable to parse — please regenerate"]},
        "catalysts": [],
        "risks": [],
    }


def _normalize_analysis(parsed: dict) -> dict:
    """
    Normalize a parsed JSON analysis — strip extra fields, clean markdown from values,
    and ensure the expected structure is present.
    """
    import re

    def clean_str(s):
        """Remove markdown formatting from a string value."""
        if not isinstance(s, str):
            return s
        s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)  # **bold** → bold
        s = re.sub(r'\*([^*]+)\*', r'\1', s)  # *italic* → italic
        s = s.replace("###", "").replace("---", "").strip()
        return s

    def clean_list(lst):
        """Clean markdown from all strings in a list of dicts or strings."""
        if not isinstance(lst, list):
            return lst
        result = []
        for item in lst:
            if isinstance(item, str):
                result.append(clean_str(item))
            elif isinstance(item, dict):
                result.append({k: clean_str(v) if isinstance(v, str) else v for k, v in item.items()})
            else:
                result.append(item)
        return result

    # Extract only the fields we need
    result = {
        "hypothesis": clean_str(parsed.get("hypothesis", "")),
        "confidence": parsed.get("confidence", 0),
        "bullCase": parsed.get("bullCase", {"title": "Bull Case", "points": []}),
        "baseCase": parsed.get("baseCase", {"title": "Base Case", "points": []}),
        "bearCase": parsed.get("bearCase", {"title": "Bear Case", "points": []}),
        "catalysts": [],
        "risks": [],
    }

    # Clean case points
    for case_key in ["bullCase", "baseCase", "bearCase"]:
        case = result[case_key]
        if isinstance(case, dict) and "points" in case:
            case["points"] = [clean_str(p) for p in case["points"] if isinstance(p, str)]

    # Clean catalysts — normalize to {event, date} only
    raw_catalysts = parsed.get("catalysts", [])
    for cat in raw_catalysts:
        if isinstance(cat, dict):
            result["catalysts"].append({
                "event": clean_str(cat.get("event", "")),
                "date": clean_str(cat.get("date", "")),
            })

    # Clean risks — normalize to {risk, severity} only
    raw_risks = parsed.get("risks", [])
    for risk in raw_risks:
        if isinstance(risk, dict):
            severity = risk.get("severity", "medium")
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            result["risks"].append({
                "risk": clean_str(risk.get("risk", "")),
                "severity": severity,
            })

    return result


def _make_summary(text: str, max_len: int = 200) -> str:
    """Extract a clean summary from agent output (first ~200 chars at sentence boundary)."""
    import re

    if not text or text == "Output not captured.":
        return "Output not captured."

    # Clean up any JSON in the summary (for strategist/advocate)
    if text.strip().startswith("{"):
        return "Produced structured JSON analysis (see full output)."

    # Strip markdown formatting for a clean summary
    cleaned = text
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)          # ### headers
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned) # **bold** → bold
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)     # *italic* → italic
    cleaned = re.sub(r'---+', '', cleaned)                # --- dividers
    cleaned = re.sub(r'\|[^\n]+\|', '', cleaned)          # | table rows |
    cleaned = re.sub(r'^\s*[-•]\s+', '', cleaned, flags=re.MULTILINE)  # bullet points
    cleaned = re.sub(r'>\s*', '', cleaned)                # > blockquotes
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)  # [links](url) → links
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)       # `code` → code
    cleaned = re.sub(r'\n{2,}', ' ', cleaned)             # double newlines → space
    cleaned = re.sub(r'\n', ' ', cleaned)                 # single newlines → space
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)             # collapse whitespace
    cleaned = cleaned.strip()

    if not cleaned:
        return "Analysis output (see full output for details)."

    truncated = cleaned[:max_len]
    # Try to end at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > 50:
        return truncated[:last_period + 1].strip()
    return truncated.strip() + "..."


# ======================================================================
# CACHE HELPERS (Supabase ai_analyses table)
# ======================================================================

def get_cached_analysis(ticker: str) -> Optional[dict]:
    """
    Check if a fresh cached analysis exists for a ticker.
    Returns the analysis dict or None.
    """
    try:
        res = _get_supabase().table("ai_analyses").select("*").eq("ticker", ticker).limit(1).execute()
        if not res.data:
            return None

        record = res.data[0]

        # Check if expired
        expires_at = record.get("expires_at")
        if expires_at:
            from dateutil.parser import parse as parse_dt
            if parse_dt(expires_at) < datetime.utcnow().replace(tzinfo=parse_dt(expires_at).tzinfo):
                return {**record, "is_stale": True}

        # Check if gap score has shifted significantly
        current_scores = _fetch_single("coverage_gap_scores", ticker, "gap_score")
        current_gap = current_scores.get("gap_score")
        gen_gap = record.get("gap_score_at_generation")
        if current_gap is not None and gen_gap is not None:
            if abs(float(current_gap) - float(gen_gap)) > 5:
                return {**record, "is_stale": True}

        return {**record, "is_stale": False}

    except Exception as e:
        print(f"  ⚠️ Cache lookup failed for {ticker}: {e}")
        return None


def upsert_analysis(ticker: str, analysis: dict, gap_score: float = None):
    """Save analysis to ai_analyses table."""
    try:
        record = {
            "ticker": ticker,
            "hypothesis": analysis.get("hypothesis", ""),
            "confidence": analysis.get("confidence", 0),
            "bull_case": analysis.get("bullCase", {}),
            "base_case": analysis.get("baseCase", {}),
            "bear_case": analysis.get("bearCase", {}),
            "catalysts": analysis.get("catalysts", []),
            "risks": analysis.get("risks", []),
            "debate_transcript": analysis.get("debate_transcript", []),
            "news_context": analysis.get("news_context", []),
            "model_used": OPENROUTER_MODEL,
            "generated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "gap_score_at_generation": gap_score,
        }

        _get_supabase().table("ai_analyses").upsert(
            record, on_conflict="ticker"
        ).execute()

        print(f"  ✅ Analysis cached for {ticker}")

    except Exception as e:
        print(f"  ⚠️ Failed to cache analysis for {ticker}: {e}")


def format_analysis_response(record: dict) -> dict:
    """
    Convert a Supabase ai_analyses row to the API response format.
    Handles both fresh analysis dicts and cached DB rows.
    """
    return {
        "ticker": record.get("ticker", ""),
        "hypothesis": record.get("hypothesis", ""),
        "confidence": float(record.get("confidence", 0)),
        "bullCase": record.get("bull_case") or record.get("bullCase", {}),
        "baseCase": record.get("base_case") or record.get("baseCase", {}),
        "bearCase": record.get("bear_case") or record.get("bearCase", {}),
        "catalysts": record.get("catalysts", []),
        "risks": record.get("risks", []),
        "debateTranscript": record.get("debate_transcript") or record.get("debateTranscript", []),
        "newsContext": record.get("news_context") or record.get("newsContext", []),
        "generatedAt": record.get("generated_at", ""),
        "modelUsed": record.get("model_used", OPENROUTER_MODEL),
        "isStale": record.get("is_stale", False),
    }