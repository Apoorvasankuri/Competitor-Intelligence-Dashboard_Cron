"""
bse_filings_ingest.py  — Change 21, Step 2b
============================================================================

Fetches official BSE corporate filings for the dashboard's BSE-listed
competitors and returns them as article dicts shaped EXACTLY like the ones
scraper_production produces from Google News RSS — so they flow through the
existing save_to_database() insert and, downstream, the same relevance /
dedup / clustering / ranking / Executive-Brief path as every other article.

WHY THIS EXISTS
  The scraper's site:bseindia.com Google-News-RSS lens returned garbage title
  fragments ("Ravi Patidar", "Status of Issue Application") because RSS only
  sees a filing page's <title>, never the real disclosure. This module talks
  to BSE's filings API directly, so it gets the REAL headline (HEADLINE field)
  and, via pypdf, the actual PDF text — token-free, no LLM involved.

HOW IT MAPS ONTO raw_scraped_articles
  - news_title  <- HEADLINE (real disclosure headline)
  - competitor  <- competitor_scrip_map name for SCRIP_CD (KNOWN, not guessed)
  - content     <- extracted PDF text (pypdf), truncated; "" if no/failed PDF
  - link        <- the filing's BSE attachment URL (stable, unique per filing)
  - source      <- "BSE" ; source_type <- "official_exchange"
  - search_query_type <- "bse_filing" (tracked distinctly in pipeline logs)
  - accepted_by_gate  <- "accepted_bse_filing"
  - published_date    <- NEWS_DT (filtered to the lookback window)

DESIGN NOTES
  - Reads competitor_scrip_map (loaded by load_competitor_scrip_map.py) for the
    name<->code mapping. If that table is missing/empty, returns [] cleanly.
  - "Ingest all categories": one API call per company with a blank category
    (BSE's -1 sentinel), not 17 calls per company.
  - 1-day lookback by default (BSE_LOOKBACK_DAYS), matching the daily cron.
  - Every network op is wrapped; one bad company/PDF never aborts the batch.
  - Depends on: requests, psycopg (already used by the scraper) and pypdf (new).

This module has NO side effects on import. scraper_production calls
fetch_bse_filing_articles() during a run; save_to_database() persists the result.
"""

import io
import logging
import os
from datetime import datetime, timedelta

import psycopg
from psycopg.rows import dict_row

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - handled at runtime
    PdfReader = None

# ── Config ────────────────────────────────────────────────────────────────────

BSE_FILINGS_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
# Filing attachments live here (same base the existing BSE downloaders use).
BSE_ATTACH_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{filename}"

# 1 day: the scraper runs daily, so we only want yesterday/today's filings.
BSE_LOOKBACK_DAYS = int(os.getenv("BSE_LOOKBACK_DAYS", "1"))

# "Ingest all categories" -> BSE's sentinel for "no category filter".
BSE_ALL_CATEGORIES = "-1"
BSE_ALL_SUBCATEGORIES = "-1"

# Cap PDF work so a single 300-page annual report can't dominate a run.
BSE_PDF_MAX_BYTES = int(os.getenv("BSE_PDF_MAX_BYTES", str(8 * 1024 * 1024)))   # 8 MB
BSE_PDF_MAX_PAGES = int(os.getenv("BSE_PDF_MAX_PAGES", "15"))
BSE_CONTENT_MAX_CHARS = int(os.getenv("BSE_CONTENT_MAX_CHARS", "5000"))

# Safety cap on API pages per company (a company rarely files >1 page/day).
BSE_MAX_PAGES_PER_COMPANY = int(os.getenv("BSE_MAX_PAGES_PER_COMPANY", "3"))

BSE_REQUEST_TIMEOUT = int(os.getenv("BSE_REQUEST_TIMEOUT", "20"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
    "Connection": "keep-alive",
}


# ── Mapping table ─────────────────────────────────────────────────────────────

def _get_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(url, row_factory=dict_row)


def load_scrip_map() -> dict:
    """Return {scrip_code: competitor_name} for active rows. On any error
    (table missing, DB down), return {} so the caller degrades to 'no BSE
    ingestion this run' rather than crashing the scraper."""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT competitor_name, bse_scrip_code FROM competitor_scrip_map "
            "WHERE active = TRUE"
        )
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        logging.warning("BSE: could not read competitor_scrip_map (%s) — skipping BSE ingestion.", e)
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Multiple competitor names can share a code (dup spellings). Keep the first
    # (longest, i.e. most specific) name per code for tagging consistency.
    by_code = {}
    for r in rows:
        code = str(r["bse_scrip_code"]).strip()
        name = str(r["competitor_name"]).strip()
        if not code or not name:
            continue
        if code not in by_code or len(name) > len(by_code[code]):
            by_code[code] = name
    logging.info("BSE: loaded %s scrip codes (%s name rows) from competitor_scrip_map.",
                 len(by_code), len(rows))
    return by_code


# ── BSE API ───────────────────────────────────────────────────────────────────

def _parse_bse_datetime(value: str):
    """BSE dates look like '2026-07-06T23:44:22.95'. Return a date or None."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:26], fmt)
        except Exception:
            continue
    # last resort: first 10 chars as date
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except Exception:
        return None


def fetch_company_filings(session, scrip_code: str, from_date: datetime, to_date: datetime) -> list:
    """Return raw BSE filing records for one company across the date window,
    all categories. Paginated but capped. Never raises — returns [] on error."""
    records = []
    for page in range(1, BSE_MAX_PAGES_PER_COMPANY + 1):
        params = {
            "pageno": page,
            "strCat": BSE_ALL_CATEGORIES,
            "subcategory": BSE_ALL_SUBCATEGORIES,
            "strPrevDate": from_date.strftime("%Y%m%d"),
            "strToDate": to_date.strftime("%Y%m%d"),
            "strSearch": "P",
            "strscrip": scrip_code,
            "strType": "C",
        }
        try:
            r = session.get(BSE_FILINGS_URL, params=params, headers=HEADERS, timeout=BSE_REQUEST_TIMEOUT)
            if not r.ok:
                logging.warning("BSE: HTTP %s for scrip %s page %s", r.status_code, scrip_code, page)
                break
            data = r.json()
        except Exception as e:
            logging.warning("BSE: fetch failed for scrip %s page %s: %s", scrip_code, page, e)
            break

        table = data.get("Table", []) or []
        if not table:
            break
        records.extend(table)

        # Stop early if we've clearly got the whole (small) result set.
        if len(table) < 10:
            break
    return records


def extract_pdf_text(session, attachment_name: str) -> str:
    """Download a filing PDF and extract its text with pypdf. Returns "" on any
    problem (no attachment, too big, download error, parse error, pypdf missing)
    — token-free and never fatal."""
    if not attachment_name or PdfReader is None:
        return ""
    url = BSE_ATTACH_URL.format(filename=attachment_name)
    try:
        r = session.get(url, headers=HEADERS, timeout=BSE_REQUEST_TIMEOUT, stream=True)
        if not r.ok:
            return ""
        # Enforce size cap without reading unbounded data.
        chunks, total = [], 0
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > BSE_PDF_MAX_BYTES:
                logging.info("BSE: PDF %s exceeds %s bytes — skipping extraction.",
                             attachment_name, BSE_PDF_MAX_BYTES)
                return ""
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw[:5].startswith(b"%PDF"):
            return ""  # not actually a PDF (some attachments are xml/zip)

        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for i, page in enumerate(reader.pages):
            if i >= BSE_PDF_MAX_PAGES:
                break
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(p.strip() for p in parts if p and p.strip())
        text = " ".join(text.split())  # collapse whitespace
        return text[:BSE_CONTENT_MAX_CHARS]
    except Exception as e:
        logging.debug("BSE: PDF extraction failed for %s: %s", attachment_name, e)
        return ""


# ── Build article dicts ───────────────────────────────────────────────────────

def _filing_to_article(rec: dict, competitor_name: str, content: str) -> dict:
    """Map one BSE filing record onto the raw_scraped_articles article-dict
    shape used by scraper_production.save_to_database()."""
    headline = str(rec.get("HEADLINE") or rec.get("NEWSSUB") or "").strip()
    news_dt = _parse_bse_datetime(rec.get("NEWS_DT") or rec.get("DissemDT") or rec.get("DT_TM"))
    attachment = str(rec.get("ATTACHMENTNAME") or "").strip()
    category = str(rec.get("CATEGORYNAME") or "").strip()
    subcat = str(rec.get("SUBCATNAME") or "").strip()

    # Stable, unique link per filing (used by ON CONFLICT (link, published_date)).
    if attachment:
        link = BSE_ATTACH_URL.format(filename=attachment)
    else:
        newsid = str(rec.get("NEWSID") or rec.get("XML_NAME") or headline[:40])
        link = f"https://www.bseindia.com/corporates/anndata.aspx?newsid={newsid}"

    theme = ", ".join([x for x in (category, subcat) if x])

    return {
        "search_keyword": competitor_name,
        "search_query": competitor_name,
        "search_query_type": "bse_filing",
        "news_title": headline or f"{competitor_name} — BSE filing",
        "source": "BSE",
        "link": link,
        "published_date": news_dt or datetime.now(),
        "sbu": "General",
        "competitor": competitor_name,           # KNOWN from scrip map, not guessed
        "content": content or "",
        # Official-exchange authority metadata (top of the source hierarchy).
        "source_domain": "bseindia.com",
        "source_type": "official_exchange",
        "source_category": "official_exchange",
        "source_priority": 1,
        "source_authority_score": 60,
        "preferred_for_executive_summary": True,
        "source_notes": "BSE corporate filing (direct API)",
        "source_match_method": "bse_api",
        "detected_client_authority": "",
        "detected_strategic_theme": theme,
        "accepted_by_gate": "accepted_bse_filing",
    }


def _bse_headline_passes_gate(headline: str) -> bool:
    """Change 21 Option A: keep a BSE filing only if its headline carries real
    business-event language.

    Reuses scraper_production.has_strong_business_event_signal (Change 19)
    rather than inventing a parallel phrase list. Deliberately does NOT reuse
    is_pure_noise_article / the "OR high-quality-source" escape hatch from the
    competitor-led news gate: every BSE filing is ALREADY source_type=
    official_exchange by construction, so that escape hatch would always be
    true here and let everything through — meaningless in this context.

    It also does NOT help that Change 19's NOISE_PHRASES list is tuned for
    stock-market chatter (share price, valuation, bearish...) — none of that
    matches the actual junk seen here (SEBI compliance certificates, NAV
    disclosures, CSR-foundation incorporation, postal-ballot notices), so
    "not pure noise" alone would pass all of it straight through. Requiring a
    positive event-signal match is the correct filter for this source.

    Known limitation (flagged, not silently decided): headlines like
    "Larsen & Toubro Secures Moody's 'Baa1' Rating" will be DROPPED here,
    because "rating"/"credit rating"/"moody's"/"crisil" aren't currently in
    STRONG_EVENT_PHRASES. If credit-rating actions should count as a
    competitive event, that phrase list needs extending — a decision for the
    dashboard owner, not made unilaterally here.

    Deferred import (inside the function, not at module top) avoids a
    circular import with scraper_production, which imports this module in
    turn once wired into the daily run.
    """
    if not headline:
        return False
    try:
        from scraper_production import has_strong_business_event_signal
    except Exception as e:
        logging.warning(
            "BSE: could not import gate helper from scraper_production (%s) — "
            "keeping this filing unfiltered rather than silently dropping everything.", e
        )
        return True  # fail-open: an import problem shouldn't kill all BSE ingestion
    return has_strong_business_event_signal(headline, "BSE")


def fetch_bse_filing_articles() -> list:
    """Entry point. Return a list of article dicts (possibly empty) ready to
    hand to scraper_production.save_to_database(). Never raises."""
    if requests is None:
        logging.warning("BSE: 'requests' unavailable — skipping BSE ingestion.")
        return []
    if PdfReader is None:
        logging.warning("BSE: 'pypdf' not installed — filings will still ingest, "
                        "but with empty content (no PDF text). Run: pip install pypdf")

    scrip_map = load_scrip_map()
    if not scrip_map:
        return []

    to_date = datetime.now()
    from_date = to_date - timedelta(days=BSE_LOOKBACK_DAYS)
    logging.info("BSE: fetching filings for %s companies, %s..%s (all categories).",
                 len(scrip_map), from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

    session = requests.Session()
    session.headers.update(HEADERS)

    articles = []
    seen_links = set()
    companies_with_filings = 0
    pdf_ok = 0
    pdf_empty = 0
    dropped_no_event_signal = 0

    for scrip_code, competitor_name in scrip_map.items():
        records = fetch_company_filings(session, scrip_code, from_date, to_date)
        if not records:
            continue

        kept_for_company = 0
        for rec in records:
            # Date-guard: BSE occasionally returns edge rows outside the window.
            dt = _parse_bse_datetime(rec.get("NEWS_DT") or rec.get("DissemDT") or rec.get("DT_TM"))
            if dt is not None and dt.date() < from_date.date():
                continue

            headline = str(rec.get("HEADLINE") or rec.get("NEWSSUB") or "").strip()

            # Change 21 Option A: gate on the headline BEFORE spending any PDF
            # download/extraction effort — no point paying that cost on a
            # filing we're going to drop anyway (compliance certs, NAV
            # disclosures, CSR-foundation incorporation, postal-ballot
            # notices all showed up in real testing and carry no event signal).
            if not _bse_headline_passes_gate(headline):
                dropped_no_event_signal += 1
                continue

            content = extract_pdf_text(session, str(rec.get("ATTACHMENTNAME") or "").strip())
            if content:
                pdf_ok += 1
            else:
                pdf_empty += 1

            article = _filing_to_article(rec, competitor_name, content)
            if article["link"] in seen_links:
                continue
            seen_links.add(article["link"])
            articles.append(article)
            kept_for_company += 1

        if kept_for_company:
            companies_with_filings += 1

    logging.info(
        "BSE: produced %s filing articles from %s/%s companies "
        "(pdf_extracted=%s, pdf_empty=%s, dropped_no_event_signal=%s).",
        len(articles), companies_with_filings, len(scrip_map), pdf_ok, pdf_empty, dropped_no_event_signal
    )
    return articles


if __name__ == "__main__":
    # Manual smoke test: fetch and print a summary WITHOUT saving anything.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    items = fetch_bse_filing_articles()
    print(f"\nFetched {len(items)} BSE filing articles (NOT saved).")
    for a in items[:20]:
        has_pdf = "PDF" if a["content"] else "no-pdf"
        print(f"  [{has_pdf:6}] {a['competitor']:<38} | {a['news_title'][:70]}")
