"""
SEC Edgar MCP Tool for Void AI.

Replaces the batch-ingested SEC filing chunks in Supabase PgVector.
Fetches filings on demand from the SEC EDGAR public REST API.

Tools exposed:
  - get_recent_filings(ticker, form_types) -> list of filing metadata
  - get_filing_text(document_url) -> raw text content
  - get_sec_chunks(ticker, query, form_types, top_k) -> chunked, query-relevant excerpts

No API key required — SEC EDGAR is a public API.
Rate limit: 10 requests/second (enforced via sleep).
"""

import re
import time
import requests
from typing import Optional

SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_HEADERS = {"User-Agent": "void-ai-research contact@voidai.com"}

_cik_cache: dict[str, str] = {}
_tickers_data: Optional[dict] = None


def _load_tickers() -> dict:
    """Load SEC company tickers JSON (cached after first call)."""
    global _tickers_data
    if _tickers_data is not None:
        return _tickers_data
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        _tickers_data = resp.json()
        return _tickers_data
    except Exception as e:
        print(f"  SEC tickers load failed: {e}")
        _tickers_data = {}
        return _tickers_data


def _get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to SEC CIK number (zero-padded to 10 digits)."""
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    tickers = _load_tickers()
    for entry in tickers.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[ticker_upper] = cik
            return cik

    print(f"  SEC CIK not found for {ticker}")
    return None


def get_recent_filings(ticker: str, form_types: list[str] = None) -> list[dict]:
    """
    Get recent filing metadata for a ticker.
    form_types: e.g. ["10-K", "10-Q", "8-K"] — defaults to all three.
    Returns list of {form_type, filing_date, accession_number, document_url}
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    cik = _get_cik(ticker)
    if not cik:
        return []

    try:
        time.sleep(0.1)  # SEC rate limit courtesy
        resp = requests.get(
            SEC_SUBMISSIONS.format(cik=cik),
            headers=EDGAR_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        results = []
        for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form in form_types:
                acc_clean = acc.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
                results.append({
                    "form_type": form,
                    "filing_date": date,
                    "accession_number": acc,
                    "document_url": url,
                })
                if len(results) >= 10:
                    break

        print(f"  [SEC Edgar] {ticker}: found {len(results)} filings ({', '.join(form_types)})")
        return results

    except Exception as e:
        print(f"  SEC filings fetch failed for {ticker}: {e}")
        return []


def get_filing_text(document_url: str, max_chars: int = 100000) -> str:
    """
    Fetch and clean the text of an SEC filing document.
    Returns plain text (HTML stripped), truncated to max_chars.
    """
    try:
        time.sleep(0.1)
        resp = requests.get(document_url, headers=EDGAR_HEADERS, timeout=20)
        resp.raise_for_status()
        raw = resp.text

        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', raw)
        # Collapse whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text[:max_chars]

    except Exception as e:
        print(f"  SEC filing text fetch failed ({document_url}): {e}")
        return ""


def get_sec_chunks(
    ticker: str,
    query: str,
    form_types: list[str] = None,
    top_k: int = 6,
) -> list[dict]:
    """
    Fetch SEC filings for a ticker and return the most query-relevant text chunks.

    This is the main function used by the RAG pipeline and CrewAI agents.
    It replaces the PgVector sec_filing retrieval.

    Returns list of dicts: {content, source_type, form_type, section, filing_date}
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    filings = get_recent_filings(ticker, form_types)
    if not filings:
        return []

    # Prioritize 10-K/10-Q over 8-K (8-Ks are often debt/corporate events, not financials)
    priority = [f for f in filings if f["form_type"] in ("10-K", "10-Q")]
    secondary = [f for f in filings if f["form_type"] not in ("10-K", "10-Q")]
    ordered = (priority + secondary)[:3]

    # Fetch text from the top 3 filings
    all_chunks = []
    for filing in ordered:
        text = get_filing_text(filing["document_url"])
        if not text:
            continue

        # Chunk into ~1000-char segments with 100-char overlap
        chunk_size = 1000
        overlap = 100
        for i in range(0, min(len(text), 80000), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 100:
                continue
            all_chunks.append({
                "content": chunk,
                "source_type": "sec_filing",
                "form_type": filing["form_type"],
                "section": "general",
                "filing_date": filing["filing_date"],
            })

    if not all_chunks:
        return []

    # Keyword-based relevance scoring (no embedding needed)
    query_words = set(query.lower().split())
    scored = []
    for chunk in all_chunks:
        chunk_words = set(chunk["content"].lower().split())
        overlap_count = len(query_words & chunk_words)
        scored.append((overlap_count, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [chunk for _, chunk in scored[:top_k]]
    print(f"  [SEC Edgar] {ticker}: returning {len(result)} chunks from {len(all_chunks)} total")
    return result
