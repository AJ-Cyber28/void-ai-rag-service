"""
CrewAI Investment Analysis System for Void AI.

Runs an autonomous crew of 5 specialized agents that debate and challenge
each other to produce robust investment analysis for a given stock ticker.

Agents (autonomous debate):
  1. Bull Analyst           — advocates for the investment, finds upside
  2. Bear Analyst           — argues against, finds risks and red flags
  3. Fundamental Researcher — neutral fact-finder (SEC filings, news, data)
  4. Debate Moderator       — summarizes key agreements/disagreements
  5. Investment Strategist  — synthesizes debate into final structured JSON

Debate Flow (9 tasks, 2 rounds):
  Phase 1 — Independent Research (T1-T3, run in PARALLEL via ThreadPoolExecutor)
  Phase 2 — Round 1: Bear challenges Bull, Bull defends (T4-T5)
  Phase 2 — Round 2: Bull challenges Bear, Bear defends (T6-T7)
  Phase 3 — Moderator summarizes, Strategist outputs final JSON (T8-T9)

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

    # 6. SEC filing chunks from SEC EDGAR API (on-demand)
    from rag.mcp_sec import get_sec_chunks
    sec_chunks = get_sec_chunks(ticker=ticker, query=f"investment analysis {ticker} risks opportunities revenue earnings", top_k=5)
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
    Execute the autonomous 5-agent debate crew and return parsed analysis JSON + debate transcript.

    Agents: Bull Analyst, Bear Analyst, Fundamental Researcher, Debate Moderator, Investment Strategist
    Flow: 3 phases, 9 tasks, 2 debate rounds.

    Returns a dict matching the ai_analyses schema:
    {
        hypothesis, confidence, bull_case, base_case, bear_case,
        catalysts, risks, debate_transcript, news_context
    }
    """
    from crewai import Agent, Task, Crew, Process, LLM

    ticker = context["company"]["ticker"]
    company_name = context["company"].get("name", ticker)
    print(f"\n🤖 Starting CrewAI autonomous debate for {company_name} ({ticker})...")

    # --- LLM config ---
    llm = LLM(
        model=f"openrouter/{OPENROUTER_MODEL}",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
    )

    # --- Format context data for prompts (compact to reduce tokens) ---
    import html as html_module

    news_text = json.dumps(
        [{"headline": n.get("headline",""), "date": n.get("datetime",""), "summary": n.get("summary","")[:150]}
         for n in context.get("news", [])],
        separators=(",", ":")
    ) if context.get("news") else "No recent news."

    sec_text = "\n\n".join([
        f"[{c.get('form_type','N/A')}|{c.get('section','general')}] {html_module.unescape(c.get('content',''))[:500]}"
        for c in context.get("sec_chunks", [])
    ]) or "No SEC filing data available."

    profile_text = "\n\n".join([
        c.get("content", "")[:400] for c in context.get("profile_chunks", [])
    ]) or "No stock profile data available."

    peer_names = ", ".join([f"{p.get('ticker','')} ({p.get('name','')})" for p in context.get("peers", [])[:5]])
    peer_cov_text = ", ".join([f"{p.get('ticker','')}: {p.get('analyst_count','?')} analysts" for p in context.get("peer_coverage", [])])

    data_block = f"""COMPANY: {json.dumps(context['company'], separators=(",",":"), default=str)}
METRICS: {json.dumps(context['metrics'], separators=(",",":"), default=str)}
COVERAGE: {json.dumps(context['coverage'], separators=(",",":"), default=str)}
SCORES: {json.dumps(context['scores'], separators=(",",":"), default=str)}
PEERS: {peer_names}
PEER COVERAGE: {peer_cov_text}

STOCK PROFILE:
{profile_text}

SEC FILING EXCERPTS:
{sec_text}

RECENT NEWS:
{news_text}"""

    # =====================================================================
    # AGENT DEFINITIONS (5 agents)
    # =====================================================================

    bull_analyst = Agent(
        role="Bull Analyst",
        goal=f"Build the strongest possible investment case FOR {ticker}, grounded in data",
        backstory=(
            "You are a conviction-driven equity analyst who specializes in finding upside "
            "in under-covered stocks. You look for catalysts, growth drivers, coverage gaps "
            "that create alpha opportunities, and reasons the market is undervaluing this stock. "
            "You argue passionately but always back your points with specific data."
        ),
        llm=llm,
        verbose=False,
    )

    bear_analyst = Agent(
        role="Bear Analyst",
        goal=f"Build the strongest possible case AGAINST investing in {ticker}, grounded in data",
        backstory=(
            "You are a skeptical risk analyst who specializes in finding what can go wrong. "
            "You look for overvaluation signals, business model weaknesses, competitive threats, "
            "governance red flags, and reasons the market might be right to ignore this stock. "
            "You challenge every bullish assumption with specific counter-evidence."
        ),
        llm=llm,
        verbose=False,
    )

    fundamental_researcher = Agent(
        role="Fundamental Research Analyst",
        goal=f"Provide a neutral, fact-based analysis of {ticker} using SEC filings, news, and market data",
        backstory=(
            "You are a neutral fundamental equity researcher who presents facts without bias. "
            "You read SEC filings (10-K, 10-Q, 8-K), analyze market data, and summarize news — "
            "always distinguishing between facts and opinions. You never take a bullish or bearish "
            "stance; you provide the evidence and let others draw conclusions."
        ),
        llm=llm,
        verbose=False,
    )

    debate_moderator = Agent(
        role="Debate Moderator",
        goal=f"Summarize the key agreements, disagreements, and unresolved questions from the {ticker} debate",
        backstory=(
            "You are a senior research director who moderates investment debates. "
            "You identify where the bull and bear analysts agree, where they fundamentally "
            "disagree, which arguments were strongest, and what questions remain unresolved. "
            "You are completely impartial and focus on the quality of arguments, not their direction."
        ),
        llm=llm,
        verbose=False,
    )

    investment_strategist = Agent(
        role="Senior Investment Strategist",
        goal=f"Synthesize the full debate into a structured investment analysis with bull/base/bear cases for {ticker}",
        backstory=(
            "You are a senior investment strategist who reads adversarial debate transcripts "
            "and produces balanced, actionable investment theses. You weigh bull and bear arguments "
            "based on evidence quality, not volume. You think in scenarios (bull/base/bear) "
            "and set confidence levels based on how well arguments survived challenge."
        ),
        llm=llm,
        verbose=False,
    )

    # =====================================================================
    # PHASE 1: INDEPENDENT RESEARCH (T1-T3, run in PARALLEL)
    # =====================================================================

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_single_task(agent, description, expected_output):
        """Run a single CrewAI task in its own crew and return output text."""
        single_task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
        single_crew = Crew(
            agents=[agent],
            tasks=[single_task],
            process=Process.sequential,
            verbose=False,
        )
        result = single_crew.kickoff()
        return str(result.raw) if hasattr(result, "raw") else str(result)

    t1_desc = f"""You are the neutral fact-finder for {company_name} ({ticker}). Analyze ALL available data and produce a comprehensive, unbiased research brief.

{data_block}

Coverage Gap Scoring methodology:
- Coverage Score (50%): How under-covered vs sector/size peers. 0 analysts = highest score.
- Activity Score (30%): Volume, volatility, and momentum signals.
- Quality Score (20%): Market cap, liquidity, data completeness.
- Gap Score = weighted combination. Higher = bigger opportunity.

Produce a neutral research brief covering:
1. Company overview and business model
2. Key financial metrics and how they compare to peers
3. Coverage gap interpretation (what the scores mean)
4. SEC filing highlights (risks, MD&A, material events)
5. Recent news summary and what it signals
6. What data is missing or limited

Be strictly factual. Do NOT take a bullish or bearish stance."""

    t2_desc = f"""You are the Bull Analyst for {company_name} ({ticker}). Build the strongest possible investment case.

{data_block}

Construct a compelling bull thesis covering:
1. Why this stock's coverage gap represents an alpha opportunity
2. Key growth catalysts and upside drivers
3. Why the market is undervaluing or ignoring this stock
4. Specific data points that support your bullish view
5. What would need to happen for the bull case to play out

Be specific and data-driven. Cite numbers from the metrics, SEC filings, and news.
Acknowledge weaknesses briefly but explain why the upside outweighs them."""

    t3_desc = f"""You are the Bear Analyst for {company_name} ({ticker}). Build the strongest possible case AGAINST investing.

{data_block}

Construct a compelling bear thesis covering:
1. Why the coverage gap might exist for good reasons (market is right to ignore)
2. Key risks, red flags, and downside scenarios
3. Business model weaknesses and competitive threats
4. Specific data points that support your bearish view
5. What could go wrong that bulls are overlooking

Be specific and data-driven. Cite numbers from the metrics, SEC filings, and news.
Acknowledge strengths briefly but explain why the risks outweigh them."""

    print("  🚀 Phase 1: Running 3 research tasks in PARALLEL...")
    p1_start = time.time()

    phase1_outputs = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _run_single_task, fundamental_researcher, t1_desc,
                "A comprehensive neutral research brief with specific numbers, filing citations, and news references. 175-220 words."
            ): "fundamental",
            executor.submit(
                _run_single_task, bull_analyst, t2_desc,
                "A passionate but data-backed bull thesis with specific price targets, catalysts, and growth arguments. 175-220 words."
            ): "bull",
            executor.submit(
                _run_single_task, bear_analyst, t3_desc,
                "A rigorous bear thesis with specific risks, red flags, and downside arguments. 175-220 words."
            ): "bear",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                phase1_outputs[label] = future.result()
                print(f"    ✅ {label} completed ({len(phase1_outputs[label])} chars)")
            except Exception as e:
                print(f"    ❌ {label} failed: {e}")
                phase1_outputs[label] = f"Research task failed: {e}"

    p1_elapsed = time.time() - p1_start
    print(f"  ✅ Phase 1 done in {p1_elapsed:.1f}s (parallel)")

    fundamental_output = phase1_outputs.get("fundamental", "No fundamental research available.")
    bull_output = phase1_outputs.get("bull", "No bull thesis available.")
    bear_output = phase1_outputs.get("bear", "No bear thesis available.")

    # =====================================================================
    # PHASE 2+3: DEBATE ROUNDS + SYNTHESIS (sequential, 6 tasks)
    # Phase 1 outputs are injected as text in task descriptions.
    # =====================================================================

    t4_bear_challenges_bull = Task(
        description=f"""The Bull Analyst has presented their investment thesis for {company_name} ({ticker}).

BULL THESIS:
{bull_output}

Your job as Bear Analyst: DIRECTLY challenge their specific arguments.
- Pick apart their strongest points with counter-evidence
- Identify assumptions they're making that may not hold
- Point out data they're ignoring or misinterpreting
- Explain why their catalysts may not materialize

Be specific — reference their actual arguments, don't just repeat your own thesis.""",
        expected_output="A point-by-point challenge of the bull thesis with specific counter-arguments. 150-200 words.",
        agent=bear_analyst,
    )

    t5_bull_defends = Task(
        description=f"""The Bear Analyst has challenged your bull thesis for {company_name} ({ticker}).

Your job as Bull Analyst: DIRECTLY respond to their specific challenges.
- Defend your strongest points with additional evidence
- Concede any valid criticisms honestly
- Explain why their counter-arguments don't invalidate your core thesis
- Strengthen any weak points they exposed

Be specific — address their actual challenges, don't just restate your thesis.""",
        expected_output="A direct defense responding to each bear challenge, conceding valid points and strengthening the thesis. 150-200 words.",
        agent=bull_analyst,
        context=[t4_bear_challenges_bull],
    )

    t6_bull_challenges_bear = Task(
        description=f"""The Bear Analyst has presented their case against {company_name} ({ticker}).

BEAR THESIS:
{bear_output}

Your job as Bull Analyst: DIRECTLY challenge their specific bear arguments.
- Explain why their risks are overstated or already priced in
- Point out positive developments they're ignoring
- Challenge their interpretation of the data
- Explain why their worst-case scenario is unlikely

Be specific — reference their actual arguments.""",
        expected_output="A point-by-point challenge of the bear thesis with specific counter-arguments. 150-200 words.",
        agent=bull_analyst,
    )

    t7_bear_defends = Task(
        description=f"""The Bull Analyst has challenged your bear thesis for {company_name} ({ticker}).

Your job as Bear Analyst: DIRECTLY respond to their specific challenges.
- Defend your risk assessment with additional evidence
- Concede any valid counter-points honestly
- Explain why their optimism doesn't address your core concerns
- Identify any new risks that emerged from the debate

Be specific — address their actual challenges, don't just restate your thesis.""",
        expected_output="A direct defense responding to each bull challenge, conceding valid points and reinforcing key risks. 150-200 words.",
        agent=bear_analyst,
        context=[t6_bull_challenges_bear],
    )

    t8_moderator_summary = Task(
        description=f"""As Debate Moderator, summarize the full investment debate for {company_name} ({ticker}).

FUNDAMENTAL RESEARCH:
{fundamental_output}

You also have access to the Bull and Bear debate round outputs via task context.

Produce a structured debate summary:
1. KEY AGREEMENTS: Where bull and bear analysts converged
2. KEY DISAGREEMENTS: Where they fundamentally disagree and why
3. STRONGEST BULL ARGUMENT: Which bull point was hardest for the bear to counter?
4. STRONGEST BEAR ARGUMENT: Which bear point was hardest for the bull to counter?
5. UNRESOLVED QUESTIONS: What information would resolve the debate?
6. RECOMMENDED CONFIDENCE RANGE: Based on argument quality, what confidence range (e.g., 40-60) is appropriate?

Be impartial. Judge argument quality, not direction.""",
        expected_output="A structured debate summary with agreements, disagreements, strongest arguments, and confidence recommendation. 175-220 words.",
        agent=debate_moderator,
        context=[t5_bull_defends, t7_bear_defends],
    )

    t9_final_synthesis = Task(
        description=f"""As Senior Investment Strategist, synthesize the full debate into a final investment analysis for {company_name} ({ticker}).

FUNDAMENTAL RESEARCH:
{fundamental_output}

You also have access to the debate round outputs and moderator summary via task context.

Use the debate outcomes to produce a BALANCED analysis. Arguments that survived challenge should carry more weight. Conceded points should be reflected honestly.

You MUST output ONLY valid JSON (no markdown code fences, no preamble, no trailing text) with this EXACT structure:
{{
  "hypothesis": "2-4 sentence investment thesis informed by the debate. Reference what the debate revealed.",
  "confidence": <integer 0-100, informed by the moderator's recommended range>,
  "bullCase": {{"title": "Bull Case", "points": ["point 1 (survived debate)", "point 2", "point 3", "point 4"]}},
  "baseCase": {{"title": "Base Case", "points": ["point 1 (where bull and bear agreed)", "point 2", "point 3"]}},
  "bearCase": {{"title": "Bear Case", "points": ["point 1 (survived debate)", "point 2", "point 3", "point 4"]}},
  "catalysts": [{{"event": "Catalyst name", "date": "Q1 2026"}}, {{"event": "Another", "date": "H1 2026"}}],
  "risks": [{{"risk": "Risk name", "severity": "high"}}, {{"risk": "Another", "severity": "medium"}}]
}}

Rules:
- hypothesis: Must reference the coverage gap AND what the debate revealed about this investment
- confidence: Use the moderator's recommended range as a guide. Reflect how well the bull case survived challenge.
- bullCase: 3-4 points that SURVIVED bear challenges (strongest arguments)
- baseCase: 2-3 points where BOTH sides agreed (consensus view)
- bearCase: 3-4 risks that SURVIVED bull challenges (genuine concerns)
- catalysts: 3-5 with realistic timeframes
- risks: 3-5 with severity ratings based on debate outcomes

CRITICAL: Output ONLY the JSON object. No markdown, no ```json, no explanation before or after.
Do NOT add extra fields beyond the 7 specified above.""",
        expected_output="A valid JSON object with exactly 7 keys: hypothesis, confidence, bullCase, baseCase, bearCase, catalysts, risks.",
        agent=investment_strategist,
        context=[t5_bull_defends, t7_bear_defends, t8_moderator_summary],
    )

    # =====================================================================
    # EXECUTE PHASE 2+3 CREW (6 sequential tasks)
    # =====================================================================

    debate_tasks = [
        t4_bear_challenges_bull, t5_bull_defends,                # Round 1
        t6_bull_challenges_bear, t7_bear_defends,                # Round 2
        t8_moderator_summary, t9_final_synthesis,                # Synthesis
    ]

    crew = Crew(
        agents=[bull_analyst, bear_analyst, debate_moderator, investment_strategist],
        tasks=debate_tasks,
        process=Process.sequential,
        verbose=False,
    )

    print("  🚀 Phase 2+3: Debate rounds + synthesis (6 tasks sequential)...")
    p2_start = time.time()
    result = crew.kickoff(inputs={"ticker": ticker})
    p2_elapsed = time.time() - p2_start
    total_elapsed = time.time() - p1_start
    print(f"  ✅ Phase 2+3 done in {p2_elapsed:.1f}s")
    print(f"  ✅ Total crew time: {total_elapsed:.1f}s")

    # =====================================================================
    # PARSE FINAL JSON OUTPUT (with logging)
    # =====================================================================

    raw_output = str(result.raw) if hasattr(result, "raw") else str(result)
    print(f"  📋 Final output length: {len(raw_output)} chars")
    print(f"  📋 Final output preview: {raw_output[:200]}...")

    analysis = _parse_crew_json(raw_output)

    if analysis.get("confidence", 0) > 0 and not analysis.get("hypothesis", "").startswith("Analysis generation"):
        print(f"  ✅ JSON PARSE SUCCESS — confidence: {analysis['confidence']}, hypothesis length: {len(analysis.get('hypothesis', ''))}")
    else:
        print(f"  ❌ JSON PARSE FAILED on Strategist output — attempting fallback...")
        # Fallback: try debate_tasks outputs (Strategist=idx 5, Moderator=idx 4)
        for fallback_idx, fallback_label in [(5, "Strategist (T9)"), (4, "Moderator (T8)")]:
            if fallback_idx < len(debate_tasks):
                fallback_raw = str(debate_tasks[fallback_idx].output) if debate_tasks[fallback_idx].output else ""
                if fallback_raw:
                    print(f"    Trying fallback: {fallback_label} (length: {len(fallback_raw)})...")
                    fallback = _parse_crew_json(fallback_raw)
                    if fallback.get("confidence", 0) > 0:
                        analysis = fallback
                        print(f"    ✅ Fallback {fallback_label} parsed successfully")
                        break
        else:
            print(f"  ❌ ALL FALLBACKS FAILED — returning placeholder analysis")

    # =====================================================================
    # BUILD DEBATE TRANSCRIPT (3 parallel + 6 sequential = 9 tasks)
    # =====================================================================

    # Phase 1 outputs are raw strings from parallel execution
    phase1_entries = [
        ("Fundamental Researcher", "Neutral Research Brief", "phase1", fundamental_output),
        ("Bull Analyst", "Initial Bull Thesis", "phase1", bull_output),
        ("Bear Analyst", "Initial Bear Thesis", "phase1", bear_output),
    ]

    # Phase 2+3 outputs come from debate_tasks
    phase23_labels = [
        ("Bear Analyst", "Challenges Bull Thesis (Round 1)", "round1"),
        ("Bull Analyst", "Defends Bull Thesis (Round 1)", "round1"),
        ("Bull Analyst", "Challenges Bear Thesis (Round 2)", "round2"),
        ("Bear Analyst", "Defends Bear Thesis (Round 2)", "round2"),
        ("Debate Moderator", "Debate Summary", "synthesis"),
        ("Investment Strategist", "Final Analysis", "synthesis"),
    ]

    debate_transcript = []
    for agent_name, role_label, phase, full_output in phase1_entries:
        debate_transcript.append({
            "agent": agent_name,
            "role": role_label,
            "phase": phase,
            "summary": _make_summary(full_output),
            "fullOutput": full_output,
        })
    for i, (agent_name, role_label, phase) in enumerate(phase23_labels):
        full_output = str(debate_tasks[i].output) if debate_tasks[i].output else "Output not captured."
        debate_transcript.append({
            "agent": agent_name,
            "role": role_label,
            "phase": phase,
            "summary": _make_summary(full_output),
            "fullOutput": full_output,
        })

    # --- Assemble final result ---
    analysis["debate_transcript"] = debate_transcript
    analysis["news_context"] = context.get("news", [])

    return analysis


# ======================================================================
# JSON PARSING HELPERS
# ======================================================================

def _find_json_object(text: str) -> str | None:
    """
    Extract the outermost JSON object from *text* using string-aware brace
    matching.  Skips braces that appear inside JSON string literals so that
    values like  "P/E < 10}"  don't break the depth counter.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_parse_json(text: str) -> dict | None:
    """Attempt json.loads; on failure fix common LLM quirks and retry."""
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fix trailing commas before } or ] — very common LLM mistake
    import re as _re
    fixed = _re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip control characters (except newline/tab) that break json.loads
    fixed = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _parse_crew_json(raw: str) -> dict:
    """
    Parse JSON from CrewAI output, handling markdown fences, preamble,
    markdown formatting inside string values, trailing commas, and
    braces embedded in string literals.
    """
    clean = raw.strip()

    # ── Strip markdown code fences ──────────────────────────────────
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = lines[1:]  # drop opening ```json / ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    if clean.endswith("```"):
        clean = clean[:clean.rfind("```")].strip()

    # ── 1. Direct parse ─────────────────────────────────────────────
    parsed = _try_parse_json(clean)
    if isinstance(parsed, dict):
        return _normalize_analysis(parsed)

    # ── 2. String-aware brace extraction ────────────────────────────
    json_str = _find_json_object(clean)
    if json_str:
        parsed = _try_parse_json(json_str)
        if isinstance(parsed, dict):
            return _normalize_analysis(parsed)

    # ── 3. Strip markdown formatting, then retry ────────────────────
    import re
    stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)   # **bold**
    stripped = re.sub(r'\*([^*]+)\*', r'\1', stripped)     # *italic*
    stripped = re.sub(r'###?\s*', '', stripped)             # ### headers
    stripped = re.sub(r'---+', '', stripped)                # --- dividers

    parsed = _try_parse_json(stripped)
    if isinstance(parsed, dict):
        return _normalize_analysis(parsed)

    json_str = _find_json_object(stripped)
    if json_str:
        parsed = _try_parse_json(json_str)
        if isinstance(parsed, dict):
            return _normalize_analysis(parsed)

    # ── 4. Last-resort: scan for every '{' and try each ─────────────
    for idx in range(len(stripped)):
        if stripped[idx] == "{":
            candidate = _find_json_object(stripped[idx:])
            if candidate:
                parsed = _try_parse_json(candidate)
                if isinstance(parsed, dict) and "hypothesis" in parsed:
                    return _normalize_analysis(parsed)

    # ── Fallback ────────────────────────────────────────────────────
    print(f"  ⚠️ Failed to parse crew JSON, using fallback. Raw output length: {len(raw)}")
    print(f"  ⚠️ Raw output (first 500 chars): {raw[:500]}")
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
# STREAMING ANALYSIS (yields SSE events as each task completes)
# ======================================================================

def run_analysis_crew_stream(context: dict):
    """
    Generator that yields SSE event dicts as each agent task completes.

    Yields dicts with type:
      {"type": "task", "agent": str, "role": str, "phase": str, "output": str, "taskIndex": int, "totalTasks": 9}
      {"type": "analysis", ...full analysis dict...}
      {"type": "error", "message": str}

    Same logic as run_analysis_crew but broken into individual task runs
    so we can stream after each one.
    """
    from crewai import Agent, Task, Crew, Process, LLM
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ticker = context["company"]["ticker"]
    company_name = context["company"].get("name", ticker)
    print(f"\n🤖 [STREAM] Starting CrewAI debate for {company_name} ({ticker})...")

    llm = LLM(
        model=f"openrouter/{OPENROUTER_MODEL}",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
    )

    # --- Compact data block (same as run_analysis_crew) ---
    import html as html_module

    news_text = json.dumps(
        [{"headline": n.get("headline",""), "date": n.get("datetime",""), "summary": n.get("summary","")[:150]}
         for n in context.get("news", [])],
        separators=(",", ":")
    ) if context.get("news") else "No recent news."

    sec_text = "\n\n".join([
        f"[{c.get('form_type','N/A')}|{c.get('section','general')}] {html_module.unescape(c.get('content',''))[:500]}"
        for c in context.get("sec_chunks", [])
    ]) or "No SEC filing data available."

    profile_text = "\n\n".join([
        c.get("content", "")[:400] for c in context.get("profile_chunks", [])
    ]) or "No stock profile data available."

    peer_names = ", ".join([f"{p.get('ticker','')} ({p.get('name','')})" for p in context.get("peers", [])[:5]])
    peer_cov_text = ", ".join([f"{p.get('ticker','')}: {p.get('analyst_count','?')} analysts" for p in context.get("peer_coverage", [])])

    data_block = f"""COMPANY: {json.dumps(context['company'], separators=(",",":"), default=str)}
METRICS: {json.dumps(context['metrics'], separators=(",",":"), default=str)}
COVERAGE: {json.dumps(context['coverage'], separators=(",",":"), default=str)}
SCORES: {json.dumps(context['scores'], separators=(",",":"), default=str)}
PEERS: {peer_names}
PEER COVERAGE: {peer_cov_text}

STOCK PROFILE:
{profile_text}

SEC FILING EXCERPTS:
{sec_text}

RECENT NEWS:
{news_text}"""

    # --- Agents ---
    bull_analyst = Agent(role="Bull Analyst", goal=f"Build the strongest possible investment case FOR {ticker}, grounded in data", backstory="You are a conviction-driven equity analyst who specializes in finding upside in under-covered stocks. You argue passionately but always back your points with specific data.", llm=llm, verbose=False)
    bear_analyst = Agent(role="Bear Analyst", goal=f"Build the strongest possible case AGAINST investing in {ticker}, grounded in data", backstory="You are a skeptical risk analyst who specializes in finding what can go wrong. You challenge every bullish assumption with specific counter-evidence.", llm=llm, verbose=False)
    fundamental_researcher = Agent(role="Fundamental Research Analyst", goal=f"Provide a neutral, fact-based analysis of {ticker} using SEC filings, news, and market data", backstory="You are a neutral fundamental equity researcher who presents facts without bias. You never take a bullish or bearish stance.", llm=llm, verbose=False)
    debate_moderator = Agent(role="Debate Moderator", goal=f"Summarize the key agreements, disagreements, and unresolved questions from the {ticker} debate", backstory="You are a senior research director who moderates investment debates. You are completely impartial and focus on argument quality.", llm=llm, verbose=False)
    investment_strategist = Agent(role="Senior Investment Strategist", goal=f"Synthesize the full debate into a structured investment analysis with bull/base/bear cases for {ticker}", backstory="You are a senior investment strategist who reads adversarial debate transcripts and produces balanced, actionable investment theses.", llm=llm, verbose=False)

    def _run_task(agent, description, expected_output):
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        return str(result.raw) if hasattr(result, "raw") else str(result)

    total_tasks = 9
    task_idx = 0
    debate_transcript = []
    p1_start = time.time()

    def _emit(agent_name, role_label, phase, output):
        nonlocal task_idx
        task_idx += 1
        entry = {
            "type": "task",
            "agent": agent_name,
            "role": role_label,
            "phase": phase,
            "output": output,
            "summary": _make_summary(output),
            "taskIndex": task_idx,
            "totalTasks": total_tasks,
        }
        debate_transcript.append({
            "agent": agent_name,
            "role": role_label,
            "phase": phase,
            "summary": _make_summary(output),
            "fullOutput": output,
        })
        return entry

    # ===== PHASE 1: PARALLEL RESEARCH =====
    print("  🚀 [STREAM] Phase 1: 3 parallel research tasks...")

    t1_desc = f"""You are the neutral fact-finder for {company_name} ({ticker}). Analyze ALL available data and produce a comprehensive, unbiased research brief.

{data_block}

Coverage Gap Scoring methodology:
- Coverage Score (50%): How under-covered vs sector/size peers. 0 analysts = highest score.
- Activity Score (30%): Volume, volatility, and momentum signals.
- Quality Score (20%): Market cap, liquidity, data completeness.
- Gap Score = weighted combination. Higher = bigger opportunity.

Produce a neutral research brief covering:
1. Company overview and business model
2. Key financial metrics and how they compare to peers
3. Coverage gap interpretation (what the scores mean)
4. SEC filing highlights (risks, MD&A, material events)
5. Recent news summary and what it signals
6. What data is missing or limited

Be strictly factual. Do NOT take a bullish or bearish stance."""

    t2_desc = f"""You are the Bull Analyst for {company_name} ({ticker}). Build the strongest possible investment case.

{data_block}

Construct a compelling bull thesis covering:
1. Why this stock's coverage gap represents an alpha opportunity
2. Key growth catalysts and upside drivers
3. Why the market is undervaluing or ignoring this stock
4. Specific data points that support your bullish view
5. What would need to happen for the bull case to play out

Be specific and data-driven. Cite numbers from the metrics, SEC filings, and news.
Acknowledge weaknesses briefly but explain why the upside outweighs them."""

    t3_desc = f"""You are the Bear Analyst for {company_name} ({ticker}). Build the strongest possible case AGAINST investing.

{data_block}

Construct a compelling bear thesis covering:
1. Why the coverage gap might exist for good reasons (market is right to ignore)
2. Key risks, red flags, and downside scenarios
3. Business model weaknesses and competitive threats
4. Specific data points that support your bearish view
5. What could go wrong that bulls are overlooking

Be specific and data-driven. Cite numbers from the metrics, SEC filings, and news.
Acknowledge strengths briefly but explain why the risks outweigh them."""

    phase1_outputs = {}
    phase1_meta = {
        "fundamental": ("Fundamental Research Analyst", fundamental_researcher, t1_desc, "A comprehensive neutral research brief with specific numbers, filing citations, and news references. 175-220 words."),
        "bull": ("Bull Analyst", bull_analyst, t2_desc, "A passionate but data-backed bull thesis with specific price targets, catalysts, and growth arguments. 175-220 words."),
        "bear": ("Bear Analyst", bear_analyst, t3_desc, "A rigorous bear thesis with specific risks, red flags, and downside arguments. 175-220 words."),
    }
    phase1_roles = {
        "fundamental": ("Neutral Research Brief", "phase1"),
        "bull": ("Initial Bull Thesis", "phase1"),
        "bear": ("Initial Bear Thesis", "phase1"),
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_task, meta[1], meta[2], meta[3]): key
            for key, meta in phase1_meta.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                output = future.result()
                phase1_outputs[key] = output
                agent_name = phase1_meta[key][0]
                role_label, phase = phase1_roles[key]
                print(f"    ✅ [STREAM] {key} completed ({len(output)} chars)")
                yield _emit(agent_name, role_label, phase, output)
            except Exception as e:
                phase1_outputs[key] = f"Research task failed: {e}"
                agent_name = phase1_meta[key][0]
                role_label, phase = phase1_roles[key]
                yield _emit(agent_name, role_label, phase, f"Task failed: {e}")

    p1_elapsed = time.time() - p1_start
    print(f"  ✅ [STREAM] Phase 1 done in {p1_elapsed:.1f}s")

    fundamental_output = phase1_outputs.get("fundamental", "No fundamental research available.")
    bull_output = phase1_outputs.get("bull", "No bull thesis available.")
    bear_output = phase1_outputs.get("bear", "No bear thesis available.")

    # ===== PHASE 2: DEBATE ROUNDS (sequential, individual tasks) =====
    phase2_tasks = [
        ("Bear Analyst", "Challenges Bull Thesis (Round 1)", "round1", bear_analyst,
         f"""The Bull Analyst has presented their investment thesis for {company_name} ({ticker}).

BULL THESIS:
{bull_output}

Your job as Bear Analyst: DIRECTLY challenge their specific arguments.
- Pick apart their strongest points with counter-evidence
- Identify assumptions they're making that may not hold
- Point out data they're ignoring or misinterpreting
- Explain why their catalysts may not materialize

Be specific — reference their actual arguments, don't just repeat your own thesis.""",
         "A point-by-point challenge of the bull thesis with specific counter-arguments. 150-200 words."),
    ]

    # T4: Bear challenges bull
    print("  🚀 [STREAM] T4: Bear challenges bull...")
    t4_output = _run_task(*[phase2_tasks[0][i] for i in [3, 4, 5]])
    yield _emit(phase2_tasks[0][0], phase2_tasks[0][1], phase2_tasks[0][2], t4_output)

    # T5: Bull defends
    print("  🚀 [STREAM] T5: Bull defends...")
    t5_output = _run_task(bull_analyst, f"""The Bear Analyst has challenged your bull thesis for {company_name} ({ticker}).

BEAR'S CHALLENGES:
{t4_output}

Your job as Bull Analyst: DIRECTLY respond to their specific challenges.
- Defend your strongest points with additional evidence
- Concede any valid criticisms honestly
- Explain why their counter-arguments don't invalidate your core thesis
- Strengthen any weak points they exposed

Be specific — address their actual challenges, don't just restate your thesis.""",
        "A direct defense responding to each bear challenge, conceding valid points and strengthening the thesis. 150-200 words.")
    yield _emit("Bull Analyst", "Defends Bull Thesis (Round 1)", "round1", t5_output)

    # T6: Bull challenges bear
    print("  🚀 [STREAM] T6: Bull challenges bear...")
    t6_output = _run_task(bull_analyst, f"""The Bear Analyst has presented their case against {company_name} ({ticker}).

BEAR THESIS:
{bear_output}

Your job as Bull Analyst: DIRECTLY challenge their specific bear arguments.
- Explain why their risks are overstated or already priced in
- Point out positive developments they're ignoring
- Challenge their interpretation of the data
- Explain why their worst-case scenario is unlikely

Be specific — reference their actual arguments.""",
        "A point-by-point challenge of the bear thesis with specific counter-arguments. 150-200 words.")
    yield _emit("Bull Analyst", "Challenges Bear Thesis (Round 2)", "round2", t6_output)

    # T7: Bear defends
    print("  🚀 [STREAM] T7: Bear defends...")
    t7_output = _run_task(bear_analyst, f"""The Bull Analyst has challenged your bear thesis for {company_name} ({ticker}).

BULL'S CHALLENGES:
{t6_output}

Your job as Bear Analyst: DIRECTLY respond to their specific challenges.
- Defend your risk assessment with additional evidence
- Concede any valid counter-points honestly
- Explain why their optimism doesn't address your core concerns
- Identify any new risks that emerged from the debate

Be specific — address their actual challenges, don't just restate your thesis.""",
        "A direct defense responding to each bull challenge, conceding valid points and reinforcing key risks. 150-200 words.")
    yield _emit("Bear Analyst", "Defends Bear Thesis (Round 2)", "round2", t7_output)

    # ===== PHASE 3: MODERATOR + STRATEGIST =====
    # T8: Moderator
    print("  🚀 [STREAM] T8: Moderator summary...")
    t8_output = _run_task(debate_moderator, f"""As Debate Moderator, summarize the full investment debate for {company_name} ({ticker}).

FUNDAMENTAL RESEARCH:
{fundamental_output}

BULL DEFENSE (Round 1):
{t5_output}

BEAR DEFENSE (Round 2):
{t7_output}

Produce a structured debate summary:
1. KEY AGREEMENTS: Where bull and bear analysts converged
2. KEY DISAGREEMENTS: Where they fundamentally disagree and why
3. STRONGEST BULL ARGUMENT: Which bull point was hardest for the bear to counter?
4. STRONGEST BEAR ARGUMENT: Which bear point was hardest for the bull to counter?
5. UNRESOLVED QUESTIONS: What information would resolve the debate?
6. RECOMMENDED CONFIDENCE RANGE: Based on argument quality, what confidence range (e.g., 40-60) is appropriate?

Be impartial. Judge argument quality, not direction.""",
        "A structured debate summary with agreements, disagreements, strongest arguments, and confidence recommendation. 175-220 words.")
    yield _emit("Debate Moderator", "Debate Summary", "synthesis", t8_output)

    # T9: Strategist
    print("  🚀 [STREAM] T9: Final synthesis...")
    t9_output = _run_task(investment_strategist, f"""As Senior Investment Strategist, synthesize the full debate into a final investment analysis for {company_name} ({ticker}).

FUNDAMENTAL RESEARCH:
{fundamental_output}

BULL DEFENSE (Round 1):
{t5_output}

BEAR DEFENSE (Round 2):
{t7_output}

MODERATOR SUMMARY:
{t8_output}

Use the debate outcomes to produce a BALANCED analysis. Arguments that survived challenge should carry more weight.

You MUST output ONLY valid JSON (no markdown code fences, no preamble, no trailing text) with this EXACT structure:
{{
  "hypothesis": "2-4 sentence investment thesis informed by the debate.",
  "confidence": <integer 0-100>,
  "bullCase": {{"title": "Bull Case", "points": ["point 1", "point 2", "point 3"]}},
  "baseCase": {{"title": "Base Case", "points": ["point 1", "point 2", "point 3"]}},
  "bearCase": {{"title": "Bear Case", "points": ["point 1", "point 2", "point 3"]}},
  "catalysts": [{{"event": "Catalyst name", "date": "Q1 2026"}}],
  "risks": [{{"risk": "Risk name", "severity": "high"}}]
}}

CRITICAL: Output ONLY the JSON object. No markdown, no explanation.""",
        "A valid JSON object with exactly 7 keys: hypothesis, confidence, bullCase, baseCase, bearCase, catalysts, risks.")
    yield _emit("Investment Strategist", "Final Analysis", "synthesis", t9_output)

    total_elapsed = time.time() - p1_start
    print(f"  ✅ [STREAM] Total crew time: {total_elapsed:.1f}s")

    # ===== PARSE & YIELD FINAL ANALYSIS =====
    analysis = _parse_crew_json(t9_output)
    if analysis.get("confidence", 0) == 0 or analysis.get("hypothesis", "").startswith("Analysis generation"):
        # Fallback to moderator
        fallback = _parse_crew_json(t8_output)
        if fallback.get("confidence", 0) > 0:
            analysis = fallback

    analysis["debate_transcript"] = debate_transcript
    analysis["news_context"] = context.get("news", [])

    yield {"type": "analysis", **analysis}


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