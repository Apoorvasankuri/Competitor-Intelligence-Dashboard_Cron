import os
import uuid
import logging
import time
import json
import requests
import psycopg
from psycopg.rows import dict_row
import pandas as pd
import re
from dotenv import load_dotenv
from typing import Dict, List
from bs4 import BeautifulSoup
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
yesterday = date.today() - timedelta(days=1)
from difflib import SequenceMatcher

# Pipeline run ID - shared across scraper and processor if env var is set
PIPELINE_ID = os.getenv("PIPELINE_ID", f"run-{uuid.uuid4().hex[:8]}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - [{PIPELINE_ID}] - %(message)s"
)

# Load environment variables
load_dotenv()

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise Exception("CLAUDE_API_KEY environment variable not set")

client = Anthropic(api_key=CLAUDE_API_KEY)

# Model
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# Excel mapping file
EXCEL_MAPPING_FILE = "SBU_Competitor_Mapping.xlsx"

# Performance Configuration
STAGE1_BATCH_SIZE = 20
STAGE2_BATCH_SIZE = 5
MAX_WORKERS = 15
RATE_LIMIT_DELAY = 0.15

# Relevance threshold
RELEVANCE_THRESHOLD = 70
ACTIONABILITY_MIN_FOR_EXECBRIEF = 40
CONFIDENCE_MIN_FOR_EXECBRIEF = 50
SBU_FIT_MIN_FOR_EXECBRIEF = 40

# Change 16: backstop window for load_raw_articles(). processing_status is
# now the PRIMARY filter (only 'pending' rows are ever loaded); this date
# bound just prevents ever reprocessing something absurdly old if a status
# update silently failed. Independent of scraper_production.SAVE_WINDOW_DAYS.
LOAD_WINDOW_DAYS = 7

# Change 5 Part C: safety cap for O(n^2) in-memory event clustering.
MAX_EVENT_CLUSTERING_ARTICLES = int(os.getenv("MAX_EVENT_CLUSTERING_ARTICLES", "500"))

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection from environment variable"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    
    return psycopg.connect(database_url, row_factory=dict_row)

# ============================================================================
# PIPELINE RUN TRACKING
# ============================================================================
# SQL schema update required (run once in PostgreSQL):
#
# CREATE TABLE IF NOT EXISTS pipeline_runs (
#     id SERIAL PRIMARY KEY,
#     pipeline_id TEXT NOT NULL,
#     stage TEXT NOT NULL,
#     status TEXT NOT NULL,           -- 'started' | 'success' | 'failed'
#     articles_in INTEGER,
#     articles_out INTEGER,
#     error_message TEXT,
#     started_at TIMESTAMP DEFAULT NOW(),
#     ended_at TIMESTAMP
# );
#
# CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline_id ON pipeline_runs(pipeline_id);
# CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stage ON pipeline_runs(stage);

def log_pipeline_run(stage, status, articles_in=None, articles_out=None, error_message=None):
    """
    Insert a row into pipeline_runs for observability.
    Never raises exceptions - failure to log should not break the pipeline.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if status in ("success", "failed"):
            cur.execute("""
                INSERT INTO pipeline_runs
                    (pipeline_id, stage, status, articles_in, articles_out, error_message, started_at, ended_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (PIPELINE_ID, stage, status, articles_in, articles_out, error_message))
        else:
            cur.execute("""
                INSERT INTO pipeline_runs
                    (pipeline_id, stage, status, articles_in, articles_out, error_message, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (PIPELINE_ID, stage, status, articles_in, articles_out, error_message))

        conn.commit()
        cur.close()

    except Exception as e:
        logging.warning(f"log_pipeline_run failed for stage={stage}, status={status}: {e}")

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def load_raw_articles() -> pd.DataFrame:
    """Load unprocessed articles from raw_scraped_articles table.

    Change 16: filters on processing_status = 'pending' (not an exact
    single-day published_date match — see LOAD_WINDOW_DAYS above), and
    claims every returned row by flipping it to 'processed' before
    returning, so a widened window can never cause the same row to be
    re-scored on a later run.
    """
    conn = get_db_connection()
    
    query = f"""
        SELECT 
            id,
            published_date,
            news_title,
            competitor,
            sbu,
            source,
            search_keyword,
            link,
            content,
            source_domain,
            source_type,
            source_category,
            source_priority,
            source_authority_score,
            preferred_for_executive_summary,
            source_notes,
            source_match_method,
            search_query_type,
            detected_client_authority,
            detected_strategic_theme,
            search_query,
            accepted_by_gate
        FROM raw_scraped_articles
        WHERE processing_status = 'pending'
          AND published_date >= CURRENT_DATE - INTERVAL '{LOAD_WINDOW_DAYS} days'
        ORDER BY published_date DESC
        LIMIT 5000
    """
    
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    cur.close()

    if not results:
        conn.close()
        return pd.DataFrame()

    # Change 16: claim these rows immediately. Rows that don't end up in the
    # final high-relevance save still stay marked 'processed' — matching
    # today's effective behavior where a low-relevance article is scored
    # once and never revisited, just now explicit instead of an accident of
    # the old single-day filter aging it out of range.
    article_ids = [r['id'] for r in results if r.get('id') is not None]
    try:
        mark_cur = conn.cursor()
        mark_cur.execute(
            "UPDATE raw_scraped_articles SET processing_status = 'processed' WHERE id = ANY(%s)",
            (article_ids,)
        )
        conn.commit()
        mark_cur.close()
        logging.info(f"Claimed {len(article_ids)} raw articles (processing_status='processed') for this run")
    except Exception as e:
        conn.rollback()
        logging.warning(f"Could not mark raw articles as processed — they may be re-scored next run: {e}")

    conn.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'news_title': 'News Title',
        'link': 'Link',
        'competitor': 'Competitor',
        'sbu': 'SBU',
        'source': 'Source',
        'published_date': 'Published Date'
    })

    # Defensive: guarantee source-metadata columns exist even if the raw
    # table predates the source-registry migration or a value is NULL.
    source_defaults = {
        "source_domain": None,
        "source_type": "unknown",
        "source_category": "unknown",
        "source_priority": 8,
        "source_authority_score": 5,
        "preferred_for_executive_summary": False,
        "source_notes": None,
        "source_match_method": "default",
        "search_query_type": "competitor",
        "detected_client_authority": "",
        "detected_strategic_theme": "",
    }
    for col, default_value in source_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["search_query_type"] = df["search_query_type"].fillna("competitor")
    df["detected_client_authority"] = df["detected_client_authority"].fillna("")
    df["detected_strategic_theme"] = df["detected_strategic_theme"].fillna("")

    df["source_type"] = df["source_type"].fillna("unknown")
    df["source_category"] = df["source_category"].fillna("unknown")
    df["source_priority"] = df["source_priority"].fillna(8)
    df["source_authority_score"] = df["source_authority_score"].fillna(5)
    df["preferred_for_executive_summary"] = df["preferred_for_executive_summary"].fillna(False)
    df["source_match_method"] = df["source_match_method"].fillna("default")

    # Search-lens fields added in Change 4 Part B/C (search_query, accepted_by_gate).
    search_lens_defaults = {
        "search_query": None,
        "accepted_by_gate": "",
    }
    for col, default_value in search_lens_defaults.items():
        if col not in df.columns:
            df[col] = default_value
    df["accepted_by_gate"] = df["accepted_by_gate"].fillna("")

    logging.info(f"Loaded source metadata for {df['source_type'].notna().sum()} raw articles")
    logging.info("Search query type distribution: %s",
                 df["search_query_type"].value_counts(dropna=False).to_dict())
    if "accepted_by_gate" in df.columns:
        logging.info("Accepted-by-gate distribution: %s",
                     df["accepted_by_gate"].value_counts(dropna=False).to_dict())

    return df

def log_query_type_distribution(df, label):
    try:
        if df is None or df.empty:
            logging.info("%s: no articles", label)
            return
        if "search_query_type" not in df.columns:
            logging.info("%s: search_query_type column not found", label)
            return
        distribution = df["search_query_type"].fillna("unknown").value_counts(dropna=False).to_dict()
        logging.info("%s - search_query_type distribution: %s", label, distribution)
    except Exception as e:
        logging.warning("Could not log search_query_type distribution for %s: %s", label, e)


def log_relevance_yield_by_query_type(df, label):
    try:
        if df is None or df.empty:
            logging.info("%s: no articles", label)
            return
        if "search_query_type" not in df.columns or "relevance_score" not in df.columns:
            logging.info("%s: required columns not found for relevance yield", label)
            return
        temp = df.copy()
        temp["search_query_type"] = temp["search_query_type"].fillna("unknown")
        temp["relevance_score"] = pd.to_numeric(temp["relevance_score"], errors="coerce").fillna(0)
        grouped = temp.groupby("search_query_type").agg(
            article_count=("relevance_score", "count"),
            avg_relevance_score=("relevance_score", "mean"),
            above_threshold=("relevance_score", lambda x: int((x >= RELEVANCE_THRESHOLD).sum()))
        ).reset_index()
        grouped["pass_rate_pct"] = (
            grouped["above_threshold"] / grouped["article_count"] * 100
        ).round(2)
        logging.info("%s - relevance yield by query type:", label)
        for _, row in grouped.sort_values("article_count", ascending=False).iterrows():
            logging.info(
                "query_type=%s | count=%s | avg_score=%.2f | above_threshold=%s | pass_rate=%s%%",
                row["search_query_type"],
                int(row["article_count"]),
                float(row["avg_relevance_score"]),
                int(row["above_threshold"]),
                row["pass_rate_pct"]
            )
    except Exception as e:
        logging.warning("Could not log relevance yield for %s: %s", label, e)


def log_category_yield_by_query_type(df, label):
    try:
        if df is None or df.empty:
            logging.info("%s: no articles", label)
            return
        if "search_query_type" not in df.columns or "category_tag" not in df.columns:
            logging.info("%s: required columns not found for category yield", label)
            return
        temp = df.copy()
        temp["search_query_type"] = temp["search_query_type"].fillna("unknown")
        temp["category_tag"] = temp["category_tag"].fillna("unknown")
        pivot = (
            temp.groupby(["search_query_type", "category_tag"])
            .size()
            .reset_index(name="count")
            .sort_values(["search_query_type", "count"], ascending=[True, False])
        )
        logging.info("%s - category yield by query type:", label)
        for query_type in pivot["search_query_type"].unique():
            subset = pivot[pivot["search_query_type"] == query_type].head(5)
            summary = {row["category_tag"]: int(row["count"]) for _, row in subset.iterrows()}
            logging.info("query_type=%s | top_categories=%s", query_type, summary)
    except Exception as e:
        logging.warning("Could not log category yield for %s: %s", label, e)


def log_gate_distribution(df, label):
    try:
        if df is None or df.empty:
            logging.info("%s: no articles", label)
            return
        if "accepted_by_gate" not in df.columns:
            logging.info("%s: accepted_by_gate column not found", label)
            return
        distribution = df["accepted_by_gate"].fillna("unknown").value_counts(dropna=False).to_dict()
        logging.info("%s - accepted_by_gate distribution: %s", label, distribution)
    except Exception as e:
        logging.warning("Could not log accepted_by_gate distribution for %s: %s", label, e)


def log_source_type_distribution(df, label):
    try:
        if df is None or df.empty:
            logging.info("%s: no articles", label)
            return
        if "source_type" not in df.columns:
            logging.info("%s: source_type column not found", label)
            return
        distribution = df["source_type"].fillna("unknown").value_counts(dropna=False).to_dict()
        logging.info("%s - source_type distribution: %s", label, distribution)
    except Exception as e:
        logging.warning("Could not log source_type distribution for %s: %s", label, e)

# ============================================================
# Change 5 Part A: event-clustering scaffolding (safe, no behavior change yet).
# ============================================================
RELATIONSHIP_EXACT_DUPLICATE = "exact_duplicate"
RELATIONSHIP_SAME_EVENT = "same_event"
RELATIONSHIP_FOLLOW_ON_UPDATE = "follow_on_update"
RELATIONSHIP_COMMENTARY = "commentary_on_event"
RELATIONSHIP_RELATED_CONTEXT = "related_context"
RELATIONSHIP_SEPARATE_EVENT = "separate_event"

RELATIONSHIP_TYPES = [
    RELATIONSHIP_EXACT_DUPLICATE,
    RELATIONSHIP_SAME_EVENT,
    RELATIONSHIP_FOLLOW_ON_UPDATE,
    RELATIONSHIP_COMMENTARY,
    RELATIONSHIP_RELATED_CONTEXT,
    RELATIONSHIP_SEPARATE_EVENT,
]


def normalize_text_for_matching(value: str) -> str:
    """Normalize text for event matching."""
    if value is None:
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9\s&.-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_numeric_value(value):
    """Safely normalize numeric fields such as contract value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def values_close(v1, v2, tolerance_pct=10) -> bool:
    """Check if two numeric values are close within tolerance percentage."""
    n1 = normalize_numeric_value(v1)
    n2 = normalize_numeric_value(v2)
    if n1 is None or n2 is None:
        return False
    if n1 == 0 and n2 == 0:
        return True
    if max(abs(n1), abs(n2)) == 0:
        return False
    diff_pct = abs(n1 - n2) / max(abs(n1), abs(n2)) * 100
    return diff_pct <= tolerance_pct


def split_csv_field(value: str) -> list:
    """Split comma-separated fields like competitors or SBUs into a cleaned list."""
    if not value:
        return []
    return [
        item.strip()
        for item in str(value).split(",")
        if item and item.strip() and item.strip() != "-"
    ]


def assign_event_clusters_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder scaffold for Change 5 event clustering.

    Current behavior:
    - Does not change existing deduplication behavior.
    - Ensures cluster-related columns exist.
    - Sets cluster_id to None for now.
    - Sets relationship_type to separate_event for now.
    - Sets is_representative_article to True for all current final articles.

    Later parts will replace this scaffold with actual event matching and cluster assignment.
    """
    if df is None or df.empty:
        return df
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "relationship_type" not in df.columns:
        df["relationship_type"] = RELATIONSHIP_SEPARATE_EVENT
    if "is_representative_article" not in df.columns:
        df["is_representative_article"] = True
    logging.info("Event clustering scaffold applied: %s articles marked as separate_event", len(df))
    return df
# ============================================================
# Change 5 Part B: event fingerprint normalization + deterministic matching.
# Reusable helpers only — NOT wired into the pipeline yet.
# ============================================================

# Fingerprint key aliases (fingerprints come from different LLM prompt versions).
_FP_CLIENT_KEYS = ["client_or_authority", "client", "authority"]
_FP_PROJECT_KEYS = ["project_name", "project", "scope"]
_FP_VALUE_KEYS = ["contract_value_crore", "contract_value_inr_crore", "value_crore", "deal_value_crore"]
_FP_LOCATION_KEYS = ["location", "geography"]
_FP_BIDDER_KEYS = ["companies_bidding", "bidders"]
_FP_PERIOD_KEYS = ["period", "quarter", "year"]
_FP_REVENUE_KEYS = ["revenue", "revenue_crore", "total_income"]
_FP_PROFIT_KEYS = ["profit", "net_profit", "pat", "profit_crore"]
_FP_ORDERBOOK_KEYS = ["order_book", "order_book_crore", "orderbook"]
_FP_ACQUIRER_KEYS = ["acquirer", "buyer"]
_FP_TARGET_KEYS = ["target_company", "target", "asset"]
_FP_DEALTYPE_KEYS = ["deal_type", "type"]
_FP_SECTOR_KEYS = ["sector", "segment"]
_FP_AUTHORITY_KEYS = ["authority", "regulator", "ministry"]
_FP_POLICY_KEYS = ["policy_or_rule", "policy", "scheme", "topic"]
_FP_MILESTONE_KEYS = ["milestone", "status", "completion_status"]


def get_event_type(row) -> str:
    """Return normalized event/category type for clustering."""
    category = row.get("category_tag") if hasattr(row, "get") else getattr(row, "category_tag", None)
    if category is None:
        return "unknown"
    return normalize_text_for_matching(category)


def get_primary_competitor_set(row) -> set:
    """Return normalized competitor set from competitor_tagging."""
    value = row.get("competitor_tagging") if hasattr(row, "get") else None
    competitors = split_csv_field(value)
    return set(normalize_text_for_matching(c) for c in competitors if c)


def get_sbu_set(row) -> set:
    """Return normalized SBU set from sbu_tagging."""
    value = row.get("sbu_tagging") if hasattr(row, "get") else None
    sbus = split_csv_field(value)
    return set(normalize_text_for_matching(s) for s in sbus if s)


def get_fingerprint_dict(row) -> dict:
    """Safely return event fingerprint dictionary from a row."""
    raw = None
    for col in ["_fingerprint", "event_fingerprint", "fingerprint"]:
        try:
            if hasattr(row, "get") and row.get(col) is not None:
                raw = row.get(col)
                break
        except Exception:
            pass
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def normalized_fingerprint_value(fp: dict, keys: list) -> str:
    """Return first available normalized fingerprint value from a list of possible keys."""
    if not fp:
        return ""
    for key in keys:
        value = fp.get(key)
        if value not in [None, "", "-"]:
            return normalize_text_for_matching(value)
    return ""


def fingerprint_numeric_value(fp: dict, keys: list):
    """Return first available numeric fingerprint value from possible keys."""
    if not fp:
        return None
    for key in keys:
        value = fp.get(key)
        num = normalize_numeric_value(value)
        if num is not None:
            return num
    return None


def title_similarity(title1: str, title2: str) -> float:
    """Return SequenceMatcher similarity between normalized titles."""
    t1 = normalize_text_for_matching(title1)
    t2 = normalize_text_for_matching(title2)
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1, t2).ratio()


def token_overlap_score(text1: str, text2: str) -> float:
    """Return token overlap score based on non-stopword tokens."""
    stop_words = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "is", "was", "are", "were", "has", "have", "had", "this",
        "that", "these", "those", "ltd", "limited", "private", "company"
    }
    tokens1 = set(normalize_text_for_matching(text1).split()) - stop_words
    tokens2 = set(normalize_text_for_matching(text2).split()) - stop_words
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))


def _row_title(row) -> str:
    """Best-effort title accessor across dict / Series / naming variants."""
    if hasattr(row, "get"):
        return row.get("news_title") or row.get("News Title") or row.get("title") or ""
    return getattr(row, "news_title", "") or getattr(row, "title", "") or ""


def _text_signals(row1, row2):
    """Shared (title_similarity, token_overlap) for a row pair."""
    t1, t2 = _row_title(row1), _row_title(row2)
    return title_similarity(t1, t2), token_overlap_score(t1, t2)


def match_order_win_event(row1, row2) -> bool:
    """A. Order wins: competitor overlap AND >=2 corroborating signals."""
    if not (get_primary_competitor_set(row1) & get_primary_competitor_set(row2)):
        return False
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    ts, tok = _text_signals(row1, row2)
    signals = 0
    c1 = normalized_fingerprint_value(fp1, _FP_CLIENT_KEYS)
    c2 = normalized_fingerprint_value(fp2, _FP_CLIENT_KEYS)
    if c1 and c2 and (c1 == c2 or SequenceMatcher(None, c1, c2).ratio() > 0.75):
        signals += 1
    p1 = normalized_fingerprint_value(fp1, _FP_PROJECT_KEYS)
    p2 = normalized_fingerprint_value(fp2, _FP_PROJECT_KEYS)
    if p1 and p2 and (p1 == p2 or SequenceMatcher(None, p1, p2).ratio() > 0.70):
        signals += 1
    if values_close(fingerprint_numeric_value(fp1, _FP_VALUE_KEYS),
                    fingerprint_numeric_value(fp2, _FP_VALUE_KEYS), 10):
        signals += 1
    l1 = normalized_fingerprint_value(fp1, _FP_LOCATION_KEYS)
    l2 = normalized_fingerprint_value(fp2, _FP_LOCATION_KEYS)
    if l1 and l2 and (l1 == l2 or SequenceMatcher(None, l1, l2).ratio() > 0.75):
        signals += 1
    if ts > 0.55 or tok > 0.45:
        signals += 1
    return signals >= 2


def match_bidding_event(row1, row2) -> bool:
    """B. Bidding: client/authority matches AND >=2 corroborating signals."""
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    c1 = normalized_fingerprint_value(fp1, _FP_CLIENT_KEYS)
    c2 = normalized_fingerprint_value(fp2, _FP_CLIENT_KEYS)
    if not (c1 and c2 and (c1 == c2 or SequenceMatcher(None, c1, c2).ratio() > 0.75)):
        return False
    ts, _ = _text_signals(row1, row2)
    signals = 0
    p1 = normalized_fingerprint_value(fp1, _FP_PROJECT_KEYS)
    p2 = normalized_fingerprint_value(fp2, _FP_PROJECT_KEYS)
    if p1 and p2 and (p1 == p2 or SequenceMatcher(None, p1, p2).ratio() > 0.70):
        signals += 1
    if values_close(fingerprint_numeric_value(fp1, _FP_VALUE_KEYS),
                    fingerprint_numeric_value(fp2, _FP_VALUE_KEYS), 10):
        signals += 1
    l1 = normalized_fingerprint_value(fp1, _FP_LOCATION_KEYS)
    l2 = normalized_fingerprint_value(fp2, _FP_LOCATION_KEYS)
    if l1 and l2 and (l1 == l2 or SequenceMatcher(None, l1, l2).ratio() > 0.75):
        signals += 1
    bidders1 = get_primary_competitor_set(row1) | set(
        normalize_text_for_matching(x) for x in split_csv_field(
            fp1.get("companies_bidding") or fp1.get("bidders")))
    bidders2 = get_primary_competitor_set(row2) | set(
        normalize_text_for_matching(x) for x in split_csv_field(
            fp2.get("companies_bidding") or fp2.get("bidders")))
    if bidders1 & bidders2:
        signals += 1
    if ts > 0.55:
        signals += 1
    return signals >= 2


def match_financial_event(row1, row2) -> bool:
    """C. Financial: competitor overlap AND period matches AND one value/title signal."""
    if not (get_primary_competitor_set(row1) & get_primary_competitor_set(row2)):
        return False
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    per1 = normalized_fingerprint_value(fp1, _FP_PERIOD_KEYS)
    per2 = normalized_fingerprint_value(fp2, _FP_PERIOD_KEYS)
    if not (per1 and per2 and per1 == per2):
        return False
    ts, _ = _text_signals(row1, row2)
    if values_close(fingerprint_numeric_value(fp1, _FP_REVENUE_KEYS),
                    fingerprint_numeric_value(fp2, _FP_REVENUE_KEYS), 10):
        return True
    if values_close(fingerprint_numeric_value(fp1, _FP_PROFIT_KEYS),
                    fingerprint_numeric_value(fp2, _FP_PROFIT_KEYS), 10):
        return True
    if values_close(fingerprint_numeric_value(fp1, _FP_ORDERBOOK_KEYS),
                    fingerprint_numeric_value(fp2, _FP_ORDERBOOK_KEYS), 10):
        return True
    return ts > 0.55


def match_ma_event(row1, row2) -> bool:
    """D. M&A: (acquirer matches AND target matches) OR (title>0.70 AND competitor overlap)."""
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    a1 = normalized_fingerprint_value(fp1, _FP_ACQUIRER_KEYS)
    a2 = normalized_fingerprint_value(fp2, _FP_ACQUIRER_KEYS)
    t1 = normalized_fingerprint_value(fp1, _FP_TARGET_KEYS)
    t2 = normalized_fingerprint_value(fp2, _FP_TARGET_KEYS)
    acquirer_ok = a1 and a2 and (a1 == a2 or SequenceMatcher(None, a1, a2).ratio() > 0.75)
    target_ok = t1 and t2 and (t1 == t2 or SequenceMatcher(None, t1, t2).ratio() > 0.75)
    if acquirer_ok and target_ok:
        return True
    ts, _ = _text_signals(row1, row2)
    if ts > 0.70 and (get_primary_competitor_set(row1) & get_primary_competitor_set(row2)):
        return True
    return False


def match_partnership_event(row1, row2) -> bool:
    """E. Partnership: overlapping companies AND one of deal_type/sector/project/title."""
    if not (get_primary_competitor_set(row1) & get_primary_competitor_set(row2)):
        return False
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    ts, _ = _text_signals(row1, row2)
    dt1 = normalized_fingerprint_value(fp1, _FP_DEALTYPE_KEYS)
    dt2 = normalized_fingerprint_value(fp2, _FP_DEALTYPE_KEYS)
    if dt1 and dt2 and dt1 == dt2:
        return True
    s1 = normalized_fingerprint_value(fp1, _FP_SECTOR_KEYS)
    s2 = normalized_fingerprint_value(fp2, _FP_SECTOR_KEYS)
    if s1 and s2 and s1 == s2:
        return True
    p1 = normalized_fingerprint_value(fp1, _FP_PROJECT_KEYS)
    p2 = normalized_fingerprint_value(fp2, _FP_PROJECT_KEYS)
    if p1 and p2 and (p1 == p2 or SequenceMatcher(None, p1, p2).ratio() > 0.70):
        return True
    return ts > 0.60


def match_policy_event(row1, row2) -> bool:
    """F. Policy: (authority matches AND policy matches) OR (title>0.65 AND SBU overlap)."""
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    au1 = normalized_fingerprint_value(fp1, _FP_AUTHORITY_KEYS)
    au2 = normalized_fingerprint_value(fp2, _FP_AUTHORITY_KEYS)
    po1 = normalized_fingerprint_value(fp1, _FP_POLICY_KEYS)
    po2 = normalized_fingerprint_value(fp2, _FP_POLICY_KEYS)
    authority_ok = au1 and au2 and (au1 == au2 or SequenceMatcher(None, au1, au2).ratio() > 0.75)
    policy_ok = po1 and po2 and (po1 == po2 or SequenceMatcher(None, po1, po2).ratio() > 0.70)
    if authority_ok and policy_ok:
        return True
    ts, _ = _text_signals(row1, row2)
    if ts > 0.65 and (get_sbu_set(row1) & get_sbu_set(row2)):
        return True
    return False


def match_project_execution_event(row1, row2) -> bool:
    """G. Project execution: competitor overlap AND project/location overlap AND milestone/title."""
    if not (get_primary_competitor_set(row1) & get_primary_competitor_set(row2)):
        return False
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    p1 = normalized_fingerprint_value(fp1, _FP_PROJECT_KEYS)
    p2 = normalized_fingerprint_value(fp2, _FP_PROJECT_KEYS)
    l1 = normalized_fingerprint_value(fp1, _FP_LOCATION_KEYS)
    l2 = normalized_fingerprint_value(fp2, _FP_LOCATION_KEYS)
    project_ok = p1 and p2 and (p1 == p2 or SequenceMatcher(None, p1, p2).ratio() > 0.70)
    location_ok = l1 and l2 and (l1 == l2 or SequenceMatcher(None, l1, l2).ratio() > 0.75)
    if not (project_ok or location_ok):
        return False
    m1 = normalized_fingerprint_value(fp1, _FP_MILESTONE_KEYS)
    m2 = normalized_fingerprint_value(fp2, _FP_MILESTONE_KEYS)
    ts, _ = _text_signals(row1, row2)
    if m1 and m2 and m1 == m2:
        return True
    return ts > 0.60


def match_generic_event(row1, row2) -> bool:
    """H. Generic fallback."""
    comp_overlap = bool(get_primary_competitor_set(row1) & get_primary_competitor_set(row2))
    sbu_overlap = bool(get_sbu_set(row1) & get_sbu_set(row2))
    ts, tok = _text_signals(row1, row2)
    if comp_overlap and ts > 0.70:
        return True
    if tok > 0.60 and sbu_overlap:
        return True
    fp1, fp2 = get_fingerprint_dict(row1), get_fingerprint_dict(row2)
    if comp_overlap and values_close(fingerprint_numeric_value(fp1, _FP_VALUE_KEYS),
                                     fingerprint_numeric_value(fp2, _FP_VALUE_KEYS), 10):
        g1 = normalized_fingerprint_value(fp1, _FP_LOCATION_KEYS)
        g2 = normalized_fingerprint_value(fp2, _FP_LOCATION_KEYS)
        if g1 and g2 and (g1 == g2 or SequenceMatcher(None, g1, g2).ratio() > 0.75):
            return True
    return False


def compare_event_relationship(row1, row2) -> str:
    """Compare two processed article rows and return a relationship type."""
    event_type_1 = get_event_type(row1)
    event_type_2 = get_event_type(row2)

    if event_type_1 != event_type_2:
        # Different categories are usually separate,
        # but allow related_context if titles are highly similar.
        t1 = _row_title(row1)
        t2 = _row_title(row2)
        if title_similarity(t1, t2) > 0.75:
            return RELATIONSHIP_RELATED_CONTEXT
        return RELATIONSHIP_SEPARATE_EVENT

    if event_type_1 == "order wins":
        return RELATIONSHIP_SAME_EVENT if match_order_win_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "bidding activity":
        return RELATIONSHIP_SAME_EVENT if match_bidding_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "financial":
        return RELATIONSHIP_SAME_EVENT if match_financial_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "mergers & acquisitions":
        return RELATIONSHIP_SAME_EVENT if match_ma_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "partnerships & alliances":
        return RELATIONSHIP_SAME_EVENT if match_partnership_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "regulatory & policy":
        return RELATIONSHIP_SAME_EVENT if match_policy_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT
    if event_type_1 == "project execution":
        return RELATIONSHIP_SAME_EVENT if match_project_execution_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT

    return RELATIONSHIP_SAME_EVENT if match_generic_event(row1, row2) else RELATIONSHIP_SEPARATE_EVENT


def test_event_matching_on_dataframe(df: pd.DataFrame, max_pairs: int = 100) -> dict:
    """
    Developer diagnostic helper. Compares the first max_pairs row pairs and returns
    relationship counts. NOT called in the production pipeline yet.
    """
    counts = {rel: 0 for rel in RELATIONSHIP_TYPES}
    if df is None or df.empty or len(df) < 2:
        return counts
    checked = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if checked >= max_pairs:
                return counts
            rel = compare_event_relationship(df.iloc[i], df.iloc[j])
            counts[rel] = counts.get(rel, 0) + 1
            checked += 1
    return counts
# ============================================================
# Change 5 Part C: in-memory event clustering of the final dataframe.
# Annotates cluster_id / relationship_type / is_representative_article.
# Does NOT write to event_clusters table.
# ============================================================
def get_row_title(row) -> str:
    """Return best available title field for a row."""
    if hasattr(row, "get"):
        return (
            row.get("news_title")
            or row.get("News Title")
            or row.get("title")
            or row.get("newstitle")
            or ""
        )
    return ""


def get_row_date(row):
    """Return best available published date field."""
    if hasattr(row, "get"):
        return (
            row.get("published_date")
            or row.get("Published Date")
            or row.get("publishedate")
            or row.get("date")
        )
    return None


def safe_rank_score(row) -> int:
    """Safely return rank_score as integer."""
    try:
        value = row.get("rank_score") if hasattr(row, "get") else None
        if value is None:
            return 0
        return int(float(value))
    except Exception:
        return 0


def safe_source_score(row) -> int:
    """Safely return source_authority_score as integer."""
    try:
        value = row.get("source_authority_score") if hasattr(row, "get") else None
        if value is None:
            return 5
        return int(float(value))
    except Exception:
        return 5


def choose_representative_index(df: pd.DataFrame, indexes: list) -> int:
    """
    Choose representative article for a cluster.
    Priority: source_authority_score > rank_score > preferred_for_executive_summary
    > summary/title completeness > newer published_date.
    """
    if not indexes:
        return None

    def sort_key(idx):
        row = df.loc[idx]
        source_score = safe_source_score(row)
        rank_score = safe_rank_score(row)
        preferred = 0
        try:
            preferred = 1 if bool(row.get("preferred_for_executive_summary")) else 0
        except Exception:
            preferred = 0
        summary_len = 0
        try:
            summary = row.get("summary") or row.get("kec_business_summary") or ""
            title = get_row_title(row)
            summary_len = len(str(summary)) + min(len(str(title)), 200)
        except Exception:
            summary_len = 0
        date_value = get_row_date(row)
        date_sort = pd.Timestamp.min
        try:
            if date_value is not None:
                date_sort = pd.to_datetime(date_value, errors="coerce")
                if pd.isna(date_sort):
                    date_sort = pd.Timestamp.min
        except Exception:
            date_sort = pd.Timestamp.min
        return (source_score, rank_score, preferred, summary_len, date_sort)

    return sorted(indexes, key=sort_key, reverse=True)[0]


def get_next_global_cluster_id() -> int:
    """
    Change 15: cluster_id must be globally unique across pipeline runs, not a
    per-run local counter starting at 1 each time (the previous behavior,
    which meant Monday's cluster_id=5 and Tuesday's unrelated cluster_id=5
    were indistinguishable to any downstream code that groups/dedupes on
    cluster_id alone, e.g. /api/chat's source dedup key). Returns
    MAX(cluster_id) + 1 across processed_articles so a fresh run never mints
    ids that collide with a previous run's. Defensive: never raises, falls
    back to 1 on a fresh table or DB error.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(cluster_id), 0) + 1 AS next_id FROM processed_articles")
        row = cur.fetchone()
        cur.close()
        next_id = int(row["next_id"]) if row and row.get("next_id") is not None else 1
        return max(next_id, 1)
    except Exception as e:
        logging.warning(f"get_next_global_cluster_id failed, defaulting to 1: {e}")
        return 1
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def resolve_cross_run_cluster_continuations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change 15: fixes cluster fragmentation across pipeline runs.

    phase2_llm_dedup() already identifies cross-run duplicates via
    check_fingerprint_against_db() (is_duplicate=True, matched_article_id
    set) — but that match was previously discarded after the boolean flag.
    A follow-on article about an event from a PRIOR run therefore got its own
    new, locally-numbered cluster_id instead of joining the original
    cluster, and — being alone in that new cluster — was marked
    is_representative_article=True, resurfacing on the dashboard as a fake
    "new" event even though it had already been flagged as a duplicate.

    For every row with is_duplicate=True and a resolvable matched_article_id,
    this pulls the matched article's existing cluster_id and descriptive
    cluster_* fields from processed_articles and applies them directly,
    setting is_representative_article=False so the row is correctly excluded
    from Executive Brief / SBU Storylines / Competitor Strategy / Client-
    Authority Tracker / chat surfacing (which all filter on
    is_representative_article = TRUE OR cluster_id IS NULL).

    Rows that don't resolve (no matched_article_id, matched row has no
    cluster_id yet, or a DB error) are left untouched and fall through to
    normal fresh in-memory clustering — consistent with the pipeline's
    existing guardrail of never hiding rows on missing cluster metadata.

    Adds a transient `_cluster_continuation` column (dropped before save,
    see main()) so create_in_memory_event_clusters() and
    build_cluster_event_fields() know to skip these rows rather than
    recomputing them from only today's partial view. Defensive: never raises.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    if "_cluster_continuation" not in df.columns:
        df["_cluster_continuation"] = False

    if "matched_article_id" not in df.columns or "is_duplicate" not in df.columns:
        return df

    candidates = df[(df["is_duplicate"] == True) & (df["matched_article_id"].notna())]
    if candidates.empty:
        return df

    conn = None
    try:
        matched_ids = sorted({int(v) for v in candidates["matched_article_id"].tolist() if pd.notna(v)})
        if not matched_ids:
            return df

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, cluster_id, cluster_title, cluster_summary, cluster_article_count,
                   cluster_source_confidence, cluster_competitors, cluster_sbus,
                   cluster_categories, cluster_primary_source, cluster_primary_source_type,
                   cluster_primary_url
            FROM processed_articles
            WHERE id = ANY(%s) AND cluster_id IS NOT NULL
            """,
            (matched_ids,),
        )
        matches = {row["id"]: row for row in cur.fetchall()}
        cur.close()
    except Exception as e:
        logging.warning(f"resolve_cross_run_cluster_continuations DB lookup failed, skipping continuation resolution: {e}")
        return df
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    resolved_count = 0
    for idx in candidates.index:
        try:
            matched_id = int(df.at[idx, "matched_article_id"])
        except Exception:
            continue
        match = matches.get(matched_id)
        if not match:
            continue  # legacy row with no cluster_id yet — falls through to fresh clustering

        df.at[idx, "cluster_id"] = match["cluster_id"]
        df.at[idx, "is_representative_article"] = False
        df.at[idx, "relationship_type"] = RELATIONSHIP_SAME_EVENT
        df.at[idx, "cluster_title"] = match.get("cluster_title") or ""
        df.at[idx, "cluster_summary"] = match.get("cluster_summary") or ""
        df.at[idx, "cluster_source_confidence"] = match.get("cluster_source_confidence") or "Low"
        df.at[idx, "cluster_competitors"] = match.get("cluster_competitors") or ""
        df.at[idx, "cluster_sbus"] = match.get("cluster_sbus") or ""
        df.at[idx, "cluster_categories"] = match.get("cluster_categories") or ""
        df.at[idx, "cluster_primary_source"] = match.get("cluster_primary_source") or ""
        df.at[idx, "cluster_primary_source_type"] = match.get("cluster_primary_source_type") or ""
        df.at[idx, "cluster_primary_url"] = match.get("cluster_primary_url") or ""
        # Best-effort snapshot only — the OLD representative row's own stored
        # cluster_article_count is NOT updated (the pipeline is insert-only),
        # so it can undercount slightly until a future reconciliation pass.
        old_count = match.get("cluster_article_count")
        try:
            old_count = int(old_count) if old_count is not None else 1
        except Exception:
            old_count = 1
        df.at[idx, "cluster_article_count"] = old_count + 1
        df.at[idx, "_cluster_continuation"] = True
        resolved_count += 1

    logging.info(
        "Cross-run cluster continuations resolved: %s of %s cross-run duplicate candidates joined an existing cluster",
        resolved_count, len(candidates)
    )
    return df


def create_in_memory_event_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create event clusters in memory using compare_event_relationship().
    Assigns cluster_id (globally unique — see get_next_global_cluster_id,
    Change 15), marks one representative per cluster, sets relationship_type.
    Does NOT write to event_clusters table.

    Change 15: rows already resolved by resolve_cross_run_cluster_continuations()
    (marked _cluster_continuation=True) are left untouched here — they already
    carry a valid existing cluster_id and is_representative_article=False.
    Only the remaining ("fresh") rows go through the O(n^2) in-run matching
    below, and fresh clusters are numbered starting from
    get_next_global_cluster_id() so they can never collide with a previous
    run's cluster_id values.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    if "_cluster_continuation" not in df.columns:
        df["_cluster_continuation"] = False

    continuation_mask = df["_cluster_continuation"] == True
    continuation_df = df[continuation_mask].copy()
    fresh_df = df[~continuation_mask].copy()

    if fresh_df.empty:
        return continuation_df

    if len(fresh_df) > MAX_EVENT_CLUSTERING_ARTICLES:
        logging.warning(
            "Too many articles for in-memory O(n^2) clustering: %s. Falling back to scaffold for the fresh subset.",
            len(fresh_df)
        )
        fresh_df = assign_event_clusters_scaffold(fresh_df)
        return pd.concat([continuation_df, fresh_df]).sort_index()

    fresh_df["cluster_id"] = None
    fresh_df["relationship_type"] = RELATIONSHIP_SEPARATE_EVENT
    fresh_df["is_representative_article"] = False

    clusters = []
    cluster_id_counter = get_next_global_cluster_id()

    for idx, row in fresh_df.iterrows():
        assigned_cluster = None
        assigned_relationship = RELATIONSHIP_SEPARATE_EVENT

        for cluster in clusters:
            representative_row = fresh_df.loc[cluster["representative_idx"]]
            relationship = compare_event_relationship(row, representative_row)
            if relationship in [
                RELATIONSHIP_EXACT_DUPLICATE,
                RELATIONSHIP_SAME_EVENT,
                RELATIONSHIP_FOLLOW_ON_UPDATE,
                RELATIONSHIP_COMMENTARY,
            ]:
                assigned_cluster = cluster
                assigned_relationship = relationship
                break

        if assigned_cluster is None:
            assigned_cluster = {
                "cluster_id": cluster_id_counter,
                "article_indexes": [idx],
                "representative_idx": idx,
            }
            clusters.append(assigned_cluster)
            fresh_df.at[idx, "cluster_id"] = cluster_id_counter
            fresh_df.at[idx, "relationship_type"] = RELATIONSHIP_SEPARATE_EVENT
            cluster_id_counter += 1
        else:
            assigned_cluster["article_indexes"].append(idx)
            fresh_df.at[idx, "cluster_id"] = assigned_cluster["cluster_id"]
            fresh_df.at[idx, "relationship_type"] = assigned_relationship
            new_rep = choose_representative_index(fresh_df, assigned_cluster["article_indexes"])
            assigned_cluster["representative_idx"] = new_rep

    for cluster in clusters:
        rep_idx = cluster["representative_idx"]
        if rep_idx is not None:
            fresh_df.at[rep_idx, "is_representative_article"] = True

    for cluster in clusters:
        cluster_indexes = cluster["article_indexes"]
        if not any(bool(fresh_df.at[i, "is_representative_article"]) for i in cluster_indexes):
            fresh_df.at[cluster_indexes[0], "is_representative_article"] = True

    total_articles = len(fresh_df)
    total_clusters = len(clusters)
    duplicate_or_related = total_articles - total_clusters
    logging.info(
        "In-memory event clustering complete: fresh_articles=%s, new_clusters=%s, grouped_articles=%s, continuation_articles=%s",
        total_articles, total_clusters, duplicate_or_related, len(continuation_df)
    )
    try:
        relationship_counts = fresh_df["relationship_type"].value_counts(dropna=False).to_dict()
        logging.info("Event relationship distribution (fresh): %s", relationship_counts)
    except Exception:
        pass

    return pd.concat([continuation_df, fresh_df]).sort_index()
# ============================================================
# Change 5 Part D: build cluster-level event fields in memory.
# Deterministic only (no LLM). Does NOT write event_clusters table.
# ============================================================
def get_row_id(row):
    """Return best available article id."""
    if hasattr(row, "get"):
        return row.get("id") or row.get("article_id") or row.get("raw_article_id")
    return None


def get_row_summary(row) -> str:
    """Return best available summary field."""
    if hasattr(row, "get"):
        return (
            row.get("summary")
            or row.get("kec_business_summary")
            or row.get("executive_summary")
            or ""
        )
    return ""


def get_row_link(row) -> str:
    """Return best available article URL."""
    if hasattr(row, "get"):
        return row.get("link") or row.get("Link") or row.get("url") or row.get("article_url") or ""
    return ""


def derive_source_confidence(source_score: int) -> str:
    """Convert source authority score to confidence label."""
    try:
        score = int(source_score)
    except Exception:
        score = 5
    if score >= 50:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def join_unique_values(values) -> str:
    """Join unique non-empty values as a comma-separated string."""
    cleaned = []
    seen = set()
    for value in values:
        if value is None:
            continue
        parts = split_csv_field(value) if isinstance(value, str) else [value]
        for part in parts:
            part_clean = str(part).strip()
            if not part_clean or part_clean == "-":
                continue
            key = normalize_text_for_matching(part_clean)
            if key not in seen:
                seen.add(key)
                cleaned.append(part_clean)
    return ", ".join(cleaned)


def build_cluster_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster-level event fields to every row using deterministic data from the
    representative article and cluster members. No LLM. No event_clusters writes.

    Change 15: rows already resolved as cross-run continuations
    (_cluster_continuation=True) are skipped here — their cluster_* fields and
    is_representative_article were already set correctly by
    resolve_cross_run_cluster_continuations() from the ORIGINAL cluster's
    data. Recomputing them from only today's partial view (which typically
    contains none of the other members of that historical cluster) would
    incorrectly promote the duplicate row back to representative status.
    """
    if df is None or df.empty:
        return df

    if "cluster_id" not in df.columns:
        logging.warning("build_cluster_event_fields called without cluster_id; applying scaffold")
        df = assign_event_clusters_scaffold(df)

    df = df.copy()
    if "_cluster_continuation" not in df.columns:
        df["_cluster_continuation"] = False

    cluster_columns_defaults = {
        "cluster_title": "",
        "cluster_summary": "",
        "cluster_article_count": 1,
        "cluster_representative_article_id": None,
        "cluster_source_confidence": "Low",
        "cluster_rank_score": 0,
        "cluster_competitors": "",
        "cluster_sbus": "",
        "cluster_categories": "",
        "cluster_primary_source": "",
        "cluster_primary_source_type": "",
        "cluster_primary_url": "",
    }
    for col, default_value in cluster_columns_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    continuation_mask = df["_cluster_continuation"] == True
    continuation_df = df[continuation_mask]
    computable_df = df[~continuation_mask].copy()

    for cluster_id, cluster_df in computable_df.groupby("cluster_id", dropna=False):
        if cluster_df.empty:
            continue

        representative_rows = cluster_df[cluster_df.get("is_representative_article", False) == True]
        if representative_rows.empty:
            representative_idx = choose_representative_index(computable_df, list(cluster_df.index))
            computable_df.at[representative_idx, "is_representative_article"] = True
            representative_row = computable_df.loc[representative_idx]
        else:
            representative_idx = representative_rows.index[0]
            representative_row = computable_df.loc[representative_idx]

        cluster_title = get_row_title(representative_row)
        cluster_summary = get_row_summary(representative_row)
        cluster_article_count = len(cluster_df)
        cluster_representative_article_id = get_row_id(representative_row)
        cluster_rank_score = max(safe_rank_score(row) for _, row in cluster_df.iterrows())

        source_score = safe_source_score(representative_row)
        cluster_source_confidence = derive_source_confidence(source_score)

        cluster_competitors = join_unique_values(cluster_df.get("competitor_tagging", pd.Series(dtype=str)).tolist())
        cluster_sbus = join_unique_values(cluster_df.get("sbu_tagging", pd.Series(dtype=str)).tolist())
        cluster_categories = join_unique_values(cluster_df.get("category_tag", pd.Series(dtype=str)).tolist())

        cluster_primary_source = representative_row.get("Source") or representative_row.get("source") or ""
        cluster_primary_source_type = representative_row.get("source_type") or ""
        cluster_primary_url = get_row_link(representative_row)

        for idx in cluster_df.index:
            computable_df.at[idx, "cluster_title"] = cluster_title
            computable_df.at[idx, "cluster_summary"] = cluster_summary
            computable_df.at[idx, "cluster_article_count"] = cluster_article_count
            computable_df.at[idx, "cluster_representative_article_id"] = cluster_representative_article_id
            computable_df.at[idx, "cluster_source_confidence"] = cluster_source_confidence
            computable_df.at[idx, "cluster_rank_score"] = cluster_rank_score
            computable_df.at[idx, "cluster_competitors"] = cluster_competitors
            computable_df.at[idx, "cluster_sbus"] = cluster_sbus
            computable_df.at[idx, "cluster_categories"] = cluster_categories
            computable_df.at[idx, "cluster_primary_source"] = cluster_primary_source
            computable_df.at[idx, "cluster_primary_source_type"] = cluster_primary_source_type
            computable_df.at[idx, "cluster_primary_url"] = cluster_primary_url

    df = pd.concat([continuation_df, computable_df]).sort_index()

    try:
        unique_clusters = df["cluster_id"].nunique(dropna=True)
        logging.info("Cluster event fields built for %s clusters (%s continuation rows preserved as-is)",
                      unique_clusters, len(continuation_df))
    except Exception:
        pass

    return df
# ============================================================
# Change 15: event-impact weight tables.
#
# These were referenced throughout Change 5G's scoring helpers but never
# defined anywhere in this file — every get_*_weight() call raised NameError,
# was swallowed by that function's own except-block (which referenced the
# same undefined name), and propagated up to compute_event_impact_score()'s
# outer try/except, which silently returned 0. Net effect: event_impact_score
# was 0 for every article, cluster_rank_score got overwritten to 0 for every
# cluster, and Executive Brief / SBU Storylines / Competitor Strategy /
# Client-Authority Tracker (all filtering on eventImpactScore >= threshold)
# rendered permanently empty. This block is the fix.
#
# Design: each table is scaled so a maximally strong event (top category,
# tier-1 competitor, mega deal value, highest-authority source, well-
# corroborated cluster, published today) sums to exactly MAX_EVENT_IMPACT_SCORE
# alongside the multi-signal components (actionability/confidence/sbu_fit,
# already capped at 40/20/20 in compute_event_impact_score) — 90+80+100+60+
# 40+50+40+20+20 = 500. Tune freely; nothing else depends on these exact
# numbers, only on the tables existing and threshold lists staying sorted
# descending (each get_*_weight helper returns the first tier whose
# threshold is <= the row's value, so the largest qualifying tier must come
# first).
# ============================================================
MAX_EVENT_IMPACT_SCORE = 500

EVENT_CATEGORY_IMPACT_WEIGHTS = {
    "order wins": 90,
    "mergers & acquisitions": 85,
    "bidding activity": 70,
    "partnerships & alliances": 60,
    "project execution": 55,
    "regulatory & policy": 50,
    "financial": 45,
    "legal & disputes": 40,
    "leadership & management": 25,
    "industry trends": 20,
    "stock market": 15,
    "not_analyzed": 0,
    "unknown": 30,
}

# Keyed by the integer "Tier" column from the Competitor Excel sheet
# (load_competitor_tiers). Unmapped tiers fall back to 0 via .get(tier, 0).
COMPETITOR_TIER_WEIGHTS = {
    1: 80,
    2: 55,
    3: 30,
    4: 10,
}

# (min_value_inr_crore, weight) — sorted descending by threshold.
DEAL_VALUE_TIER_WEIGHTS = [
    (5000, 100),
    (2000, 85),
    (1000, 65),
    (500, 45),
    (100, 25),
    (10, 10),
]

# (min_source_authority_score, weight) — sorted descending by threshold.
# Matches the 5-60 scale in scraper_production.SOURCE_REGISTRY:
# official_exchange/client_authority/govt_policy ~58-60, company_official 50,
# specialist media 38, business media 30, press release 20, aggregator/
# unknown 5-10.
SOURCE_AUTHORITY_TIER_WEIGHTS = [
    (55, 60),
    (45, 45),
    (35, 30),
    (25, 18),
    (15, 8),
    (0, 2),
]

# (min_cluster_article_count, weight) — sorted descending by threshold.
# Rewards corroboration: an event independently reported by more sources is
# more likely to be real and more important.
CLUSTER_SIZE_TIER_WEIGHTS = [
    (5, 40),
    (3, 25),
    (2, 12),
    (1, 0),
]

# (min_days_ago, weight) — sorted descending by threshold. get_freshness_weight
# computes days_ago = today - published_date, so a LARGER threshold catches
# only OLDER articles; keeping the largest threshold first and smallest
# (0 = published today) last means the freshest articles fall through to the
# highest weight.
CLUSTER_FRESHNESS_WEIGHTS = [
    (14, 10),
    (7, 20),
    (3, 35),
    (1, 45),
    (0, 50),
]


# ============================================================
# Change 5 Part G: event-level executive impact scoring.
# Deterministic weights only. Defensive; never raises into the pipeline.
# ============================================================
def get_event_category_weight(category_tag):
    """Weight for a category (normalized), default to 'unknown'."""
    try:
        if category_tag is None:
            return EVENT_CATEGORY_IMPACT_WEIGHTS["unknown"]
        key = str(category_tag).strip().lower()
        return EVENT_CATEGORY_IMPACT_WEIGHTS.get(key, EVENT_CATEGORY_IMPACT_WEIGHTS["unknown"])
    except Exception:
        return EVENT_CATEGORY_IMPACT_WEIGHTS["unknown"]


def get_max_competitor_tier_weight(competitor_tier_map, competitor_tagging):
    """Max tier weight across a (possibly comma-separated) competitor field."""
    try:
        if not competitor_tagging or not competitor_tier_map:
            return 0
        best = 0
        for comp in split_csv_field(competitor_tagging):
            tier = competitor_tier_map.get(comp)
            if tier is None:
                tier = competitor_tier_map.get(str(comp).strip())
            if tier is None:
                continue
            weight = COMPETITOR_TIER_WEIGHTS.get(int(tier), 0)
            if weight > best:
                best = weight
        return best
    except Exception:
        return 0


def get_deal_value_weight(value_inr_crore):
    """First tier whose threshold <= value."""
    try:
        value = normalize_numeric_value(value_inr_crore)
        if value is None or value == 0:
            return 0
        for threshold, weight in DEAL_VALUE_TIER_WEIGHTS:
            if threshold <= value:
                return weight
        return 0
    except Exception:
        return 0


def get_source_authority_weight(source_authority_score):
    """First tier whose threshold <= source authority score."""
    try:
        score = normalize_numeric_value(source_authority_score)
        if score is None:
            score = 0
        for threshold, weight in SOURCE_AUTHORITY_TIER_WEIGHTS:
            if threshold <= score:
                return weight
        return 0
    except Exception:
        return 0


def get_cluster_size_weight(cluster_article_count):
    """First tier whose threshold <= cluster size."""
    try:
        count = normalize_numeric_value(cluster_article_count)
        if count is None:
            count = 1
        for threshold, weight in CLUSTER_SIZE_TIER_WEIGHTS:
            if threshold <= count:
                return weight
        return 0
    except Exception:
        return 0


def get_freshness_weight(published_date):
    """First tier whose threshold <= days_ago."""
    try:
        if published_date is None:
            return 0
        dt = pd.to_datetime(published_date, errors="coerce")
        if pd.isna(dt):
            return 0
        now = pd.Timestamp.now()
        if dt.tzinfo is not None:
            now = pd.Timestamp.now(tz=dt.tzinfo)
        days_ago = (now.normalize() - dt.normalize()).days
        if days_ago < 0:
            days_ago = 0
        for threshold, weight in CLUSTER_FRESHNESS_WEIGHTS:
            if threshold <= days_ago:
                return weight
        return 0
    except Exception:
        return 0


def compute_event_impact_score(row, competitor_tier_map):
    """Deterministic executive impact score for one row, capped at MAX_EVENT_IMPACT_SCORE."""
    try:
        get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
        actionability_component = min((normalize_numeric_value(get("actionability_score")) or 0) * 0.4, 40)
        confidence_component = min((normalize_numeric_value(get("confidence_score")) or 0) * 0.2, 20)
        sbu_fit_component = min((normalize_numeric_value(get("sbu_fit_score")) or 0) * 0.2, 20)
        score = (
            get_event_category_weight(get("category_tag"))
            + get_max_competitor_tier_weight(competitor_tier_map, get("competitor_tagging"))
            + get_deal_value_weight(get("contract_value_inr_crore"))
            + get_source_authority_weight(get("source_authority_score"))
            + get_cluster_size_weight(get("cluster_article_count"))
            + get_freshness_weight(get("published_date"))
            + actionability_component
            + confidence_component
            + sbu_fit_component
        )
        if score > MAX_EVENT_IMPACT_SCORE:
            score = MAX_EVENT_IMPACT_SCORE
        return int(score)
    except Exception:
        return 0


def assign_event_impact_scores(df, competitor_tier_map):
    """
    Compute event_impact_score per row and overwrite cluster_rank_score with the
    per-cluster max. Missing cluster_id is treated as its own single-article cluster.
    Defensive: never raises.
    """
    try:
        if df is None or df.empty:
            return df

        df = df.copy()
        df["event_impact_score"] = df.apply(
            lambda r: compute_event_impact_score(r, competitor_tier_map), axis=1
        )

        if "cluster_rank_score" not in df.columns:
            df["cluster_rank_score"] = 0

        has_cluster = df["cluster_id"].notna() if "cluster_id" in df.columns else pd.Series(False, index=df.index)

        if "cluster_id" in df.columns and has_cluster.any():
            cluster_max = df[has_cluster].groupby("cluster_id")["event_impact_score"].transform("max")
            df.loc[has_cluster, "cluster_rank_score"] = cluster_max

        if (~has_cluster).any():
            df.loc[~has_cluster, "cluster_rank_score"] = df.loc[~has_cluster, "event_impact_score"]

        try:
            number_of_unique_clusters = df["cluster_id"].nunique(dropna=True) if "cluster_id" in df.columns else 0
            max_score = int(df["event_impact_score"].max()) if len(df) else 0
            avg_score = round(float(df["event_impact_score"].mean()), 2) if len(df) else 0
            logging.info("Event impact scores assigned. Clusters: %s | Max score: %s | Avg score: %s",
                         number_of_unique_clusters, max_score, avg_score)
        except Exception:
            pass

        return df
    except Exception as e:
        logging.warning("assign_event_impact_scores failed: %s", e)
        return df

def save_to_processed_articles(df: pd.DataFrame):
    """Save processed articles to processed_articles table"""
    if df.empty:
        logging.info("No articles to save")
        return
    
    conn = get_db_connection()
    
    # ------------------------------------------------------------------
    # SQL schema update required (run once before deploying this change):
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_domain TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_type TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_category TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_priority INTEGER DEFAULT 8;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_authority_score INTEGER DEFAULT 5;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS preferred_for_executive_summary BOOLEAN DEFAULT FALSE;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_notes TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_match_method TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS search_query_type TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS detected_client_authority TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS detected_strategic_theme TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS search_query TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS accepted_by_gate TEXT;
    #
    # SQL schema update required for event clustering (Change 5 Part A):
    #   CREATE TABLE IF NOT EXISTS event_clusters (
    #       id SERIAL PRIMARY KEY,
    #       cluster_title TEXT,
    #       event_type TEXT,
    #       primary_sbu TEXT,
    #       secondary_sbus TEXT,
    #       competitors TEXT,
    #       client_or_authority TEXT,
    #       project_name TEXT,
    #       geography TEXT,
    #       contract_value_inr_crore NUMERIC,
    #       representative_article_id INTEGER,
    #       canonical_fingerprint JSONB,
    #       cluster_summary TEXT,
    #       why_it_matters TEXT,
    #       recommended_action TEXT,
    #       cluster_rank_score INTEGER DEFAULT 0,
    #       source_confidence TEXT,
    #       article_count INTEGER DEFAULT 1,
    #       first_seen TIMESTAMP,
    #       last_seen TIMESTAMP,
    #       created_at TIMESTAMP DEFAULT NOW(),
    #       updated_at TIMESTAMP DEFAULT NOW()
    #   );
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_id INTEGER;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS relationship_type TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS is_representative_article BOOLEAN DEFAULT FALSE;
    #
    # SQL schema update required for cluster event fields (Change 5 Part D):
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_title TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_summary TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_article_count INTEGER DEFAULT 1;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_representative_article_id INTEGER;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_source_confidence TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_rank_score INTEGER DEFAULT 0;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_competitors TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_sbus TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_categories TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_primary_source TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_primary_source_type TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS cluster_primary_url TEXT;
    # ------------------------------------------------------------------
    insert_query = """
    INSERT INTO processed_articles (
        published_date,
        news_title,
        link,
        "Source",
        relevance_score,
        competitor_tagging,
        sbu_tagging,
        category_tag,
        summary,
        scraped_content,
        contract_value_inr_crore,
        geography,
        competitor_tier,
        rank_score,
        fingerprint,
        is_duplicate,
        source_domain,
        source_type,
        source_category,
        source_priority,
        source_authority_score,
        preferred_for_executive_summary,
        source_notes,
        source_match_method,
        search_query_type,
        detected_client_authority,
        detected_strategic_theme,
        search_query,
        accepted_by_gate,
        cluster_id,
        relationship_type,
        is_representative_article,
        cluster_title,
        cluster_summary,
        cluster_article_count,
        cluster_representative_article_id,
        cluster_source_confidence,
        cluster_rank_score,
        cluster_competitors,
        cluster_sbus,
        cluster_categories,
        cluster_primary_source,
        cluster_primary_source_type,
        cluster_primary_url,
        event_impact_score,
        actionability_score,
        confidence_score,
        sbu_fit_score
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s,
        %s, %s, %s
    )
    ON CONFLICT (link, published_date) DO NOTHING
"""
    
    saved_count = 0
    failed_count = 0
    duplicate_count = 0
    
    for idx, row in df.iterrows():
        try:
            is_dup = row.get('is_duplicate', False)
            if is_dup:
                duplicate_count += 1

            # Convert fingerprint dict to JSON string for storage
            fp = row.get('_fingerprint') or row.get('fingerprint')
            fp_json = json.dumps(fp) if fp and isinstance(fp, dict) else None

            cur = conn.cursor()
            cur.execute(insert_query, (
                row.get('Published Date'),
                row.get('News Title'),
                row.get('Link'),
                row.get('Source', ''),
                row.get('relevance_score', 0),
                row.get('competitor_tagging', '-'),
                row.get('sbu_tagging', 'None'),
                row.get('category_tag', 'not_analyzed'),
                row.get('summary', ''),
                row.get('scraped_content', ''),
                row.get('contract_value_inr_crore'),
                row.get('geography'),
                row.get('competitor_tier'),
                row.get('rank_score', 0),
                fp_json,
                is_dup,
                row.get('source_domain'),
                row.get('source_type', 'unknown'),
                row.get('source_category', 'unknown'),
                row.get('source_priority', 8),
                row.get('source_authority_score', 5),
                row.get('preferred_for_executive_summary', False),
                row.get('source_notes'),
                row.get('source_match_method', 'default'),
                row.get('search_query_type', 'competitor'),
                row.get('detected_client_authority', ''),
                row.get('detected_strategic_theme', ''),
                row.get('search_query'),
                row.get('accepted_by_gate', ''),
                row.get('cluster_id'),
                row.get('relationship_type', RELATIONSHIP_SEPARATE_EVENT),
                row.get('is_representative_article', True),
                row.get('cluster_title', ''),
                row.get('cluster_summary', ''),
                row.get('cluster_article_count', 1),
                row.get('cluster_representative_article_id'),
                row.get('cluster_source_confidence', 'Low'),
                row.get('cluster_rank_score', 0),
                row.get('cluster_competitors', ''),
                row.get('cluster_sbus', ''),
                row.get('cluster_categories', ''),
                row.get('cluster_primary_source', ''),
                row.get('cluster_primary_source_type', ''),
                row.get('cluster_primary_url', ''),
                row.get('event_impact_score', 0),
                row.get('actionability_score', 0),
                row.get('confidence_score', 0),
                row.get('sbu_fit_score', 0),
            ))
            conn.commit()
            # Delete from raw after successful save
            cur.execute("DELETE FROM raw_scraped_articles WHERE id = %s", (row.get('id'),))
            conn.commit()
            cur.close()
            saved_count += 1
        except Exception as e:
            conn.rollback()
            failed_count += 1
            logging.error(f"Error saving article '{row.get('News Title', 'Unknown')[:50]}...': {e}")
    
    conn.close()
    
    logging.info(f"✅ Saved {saved_count} articles to processed_articles table")
    logging.info(f"   📌 {duplicate_count} flagged as cross-batch duplicates")
    if failed_count > 0:
        logging.warning(f"⚠️  Failed to save {failed_count} articles")


# ============================================================================
# LOAD DATA FROM EXCEL
# ============================================================================

def load_excel_data():
    """Load competitors, SBUs, and categories from Excel file"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found! Please ensure it's in the same directory.")
    
    logging.info(f"📂 Loading data from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    competitors_list = competitor_df['Competitor'].dropna().unique().tolist()
    
    # Read SBU sheet
    sbu_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='SBU', header=1)
    sbu_list = sbu_df['SBU'].dropna().unique().tolist()
    
    # Read Categories sheet
    categories_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Categories')
    categories_list = categories_df['Category'].dropna().tolist()
    
    logging.info(f"   ✅ Loaded {len(competitors_list)} competitors")
    logging.info(f"   ✅ Loaded {len(sbu_list)} SBUs")
    logging.info(f"   ✅ Loaded {len(categories_list)} categories")
    
    return {
        'competitors': competitors_list,
        'sbus': sbu_list,
        'categories': categories_list
    }


def load_competitor_tiers():
    """Load competitor tier mapping from Excel file"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found!")
    
    logging.info(f"📂 Loading competitor tiers from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    
    # Create tier mapping dictionary
    tier_map = {}
    for idx, row in competitor_df.iterrows():
        competitor = row.get('Competitor')
        tier = row.get('Tier')
        
        if pd.notna(competitor) and pd.notna(tier):
            tier_map[competitor.strip()] = int(tier)
    
    logging.info(f"   ✅ Loaded tiers for {len(tier_map)} competitors")
    
    return tier_map

def load_competitor_variations():
    """Load competitor variations from 'Competitor Key Words' column in Excel"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found!")
    
    logging.info(f"📂 Loading competitor variations from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    
    # Create mapping: variation (lowercase) → official name
    variation_to_official = {}
    official_names = []
    
    for idx, row in competitor_df.iterrows():
        official_name = row.get('Competitor')
        keywords_raw = row.get('Competitor Key Words', '')  # USE THIS COLUMN
        
        if pd.notna(official_name):
            official_names.append(official_name)
            
            # Map official name to itself
            variation_to_official[official_name.lower().strip()] = official_name
            
            # Map all keywords/variations to official name
            if pd.notna(keywords_raw):
                # Extract keywords between quotes
                variations = re.findall(r'"([^"]+)"', str(keywords_raw))
                
                for var in variations:
                    var_clean = var.strip()
                    if var_clean:
                        variation_to_official[var_clean.lower()] = official_name
                        
                        # Also add common case variations
                        variation_to_official[var_clean.upper()] = official_name
                        variation_to_official[var_clean.title()] = official_name
    
    logging.info(f"   ✅ Loaded {len(official_names)} official names with {len(variation_to_official)} total variations")
    
    # Log some examples for verification
    logging.info(f"   📝 Example mappings:")
    for var, official in list(variation_to_official.items())[:5]:
        logging.info(f"      '{var}' → '{official}'")
    
    return {
        'official_names': official_names,
        'variation_map': variation_to_official
    }

def normalize_competitors_to_official(competitor_string: str, variation_map: dict) -> str:
    """
    Normalize competitor names to official names from Excel
    Input: "L&T, Tata Projects Ltd, Kalpataru Power"
    Output: "Larsen & Toubro Limited, Tata Projects Limited, Kalpataru Projects International Limited"
    """
    
    if not competitor_string or competitor_string.strip() in ['-', '', 'None']:
        return '-'
    
    # Split by comma
    competitors = [c.strip() for c in competitor_string.split(',')]
    
    # Normalize each
    normalized = set()  # Use set to avoid duplicates
    
    for comp in competitors:
        comp_lower = comp.lower().strip()
        
        # Look up in variation map
        if comp_lower in variation_map:
            official_name = variation_map[comp_lower]
            normalized.add(official_name)
        else:
            # If not found, check for partial match (fallback)
            found = False
            for var, official in variation_map.items():
                if var in comp_lower or comp_lower in var:
                    normalized.add(official)
                    found = True
                    break
            
            if not found:
                # Keep original if no match (log warning)
                logging.warning(f"   ⚠️ Unknown competitor variation: '{comp}' - keeping as-is")
                normalized.add(comp)
    
    return ", ".join(sorted(normalized)) if normalized else '-'

# ============================================================================
# BUILD DYNAMIC PROMPT (SCRIPT 4 DETAILED VERSION)
# ============================================================================

def build_full_analysis_prompt(categories: List[str]) -> str:
    """Build the full analysis prompt with dynamic data"""
    
    # Format categories list with numbering
    categories_text = "\n".join([f"{i+1}. **{cat}**" for i, cat in enumerate(categories)])
    
    prompt = f"""You are a business intelligence analyst for KEC International analyzing competitor news articles.

====================
ABOUT KEC INTERNATIONAL
====================
KEC International is a global infrastructure EPC major with 80+ years of experience, executing large-scale projects across 110+ countries.
KEC operates through six main business verticals (SBUs):

**1. TRANSMISSION & DISTRIBUTION (T&D)**
   - INDIA T&D: Power transmission lines, substations, grid infrastructure within India
     • High voltage transmission lines (220 kV to 765 kV)
     • Substations and switchyards
     • HVDC (High Voltage Direct Current) systems
     • Digital substations
     • STATCOM (Static Synchronous Compensator)
     • Underground cabling
     • Towers, poles & hardware supplies
   
   - INTERNATIONAL T&D: Power transmission projects outside India
     • Cross-border transmission projects
     • Primary focus geographies: Middle East, Africa, and South East Asia (excluding China)
     • Same technical scope as India T&D but in these focus markets

**2. TRANSPORTATION**
   - Railways: Overhead electrification (OHE), signaling systems, TCAS Kavach, railway bridges, stations & platforms
   - Urban Infrastructure: Metro rail projects (viaducts, stations, tech-enabled areas), ropeways
   - Speed upgradation projects
   - Track laying, depot & workshops

**3. CIVIL**
   - Residential buildings and high-rise towers
   - Commercial buildings and office complexes
   - Factories and industrial facilities
   - Airports and aviation infrastructure
   - Hospitals and healthcare facilities
   - Data centers
   - Water pipeline projects and treatment plants
   - Warehouses & logistics facilities
   - Tunnel ventilation systems
   - Municipal waste-to-energy plants, FGD (Flue Gas Desulphurisation) units

**4. RENEWABLES**
   - Solar: Large-scale solar power plants (>500 MW capability), industrial solar solutions
   - Wind: Wind farm development and infrastructure
   - Green Hydrogen: Emerging capabilities
   - Hybrid renewable projects

**5. OIL & GAS PIPELINES**
   - Cross-country oil and gas pipelines
   - Slurry pipelines
   - Water pipelines (potable water supply projects)
   - Composite station works

**6. CABLES & CONDUCTORS** (Manufacturing)
   - Power cables
   - Control & instrumentation cables
   - Railway cables
   - Conductors (overhead line conductors)
   - Telecom cables
   - Special application cables
====================
COMPETITORS LIST
====================
Below are KEC's competitors. Use FUZZY MATCHING to recognize abbreviations, acronyms, partial names, and common variations (e.g. "L&T" → "Larsen & Toubro Limited", "KPTL" → "Kalpataru Projects International Limited", "HCC" → "Hindustan Construction Company Limited", "RVNL" → "Rail Vikas Nigam Limited").

Always return the FULL OFFICIAL NAME exactly as listed below:

- AFCONS Infrastructure Limited
- Ace Pipeline Contracts Private Limited
- Advance Infrastructures Pvt Ltd
- Ahluwalia Contracts (India) Limited
- Al Fanar Group
- Al Sharif Group Holding
- Algihaz Holdings
- Amara Raja Group
- Ashoka Buildcon Limited
- Bajaj Electricals Limited
- Bajel Projects Limited
- Bondada Engineering Limited
- Bridge & Roof Company(India) Limited
- CMEC (China)
- China Southern Power Grid Company Limited
- Corrtech International Limited
- Dilip Buildcon Limited
- Dineshchandra R. Agrawal Infracon Private Limited
- EnProCon Enterprise Limited
- Essens Renewable Private Limited
- Ever Renew Energy Pvt. Ltd.
- Everrenew Energy Private Limited
- GR Infraprojects Limited
- H.G. Infra Engineering Limited
- Hartek Group
- Hindustan Construction Company Limited
- Hitachi Energy India Limited
- Hyosung T&D India Private Limited
- Hyundai Engineering & Construction Co.
- IRCON International Limited
- ISC Projects Private Limited
- J. Kumar Infraprojects Limited
- JSIW Infrastructure Private Limited
- Jackson Electricals & Infrastructure Pvt. Ltd.
- Jackson Green Energy
- Jyoti Structures Limited
- KP Energy Ltd
- KPI Green Energy Limited
- Kalpataru Projects International Limited
- Kernex Microsystems Private Limited
- Kintech-Synergy
- Kiran Infrastructure Private Limited
- Konkan Railway Corporation Limited
- Larsen & Toubro Limited
- Likhitha Infrastructure Limited
- MKC Infrastructure Limited
- Mastek Group
- NCC Limited
- NRP Projects Private Limited
- Offshore Infrastructures Limited
- Oriana Power Limited
- PNC Infratech Limited
- Param Group
- Power Mech Projects Limited
- Pratham Engineering
- Preformed Line Products Company (PLP)
- Rail Vikas Nigam Limited
- RailTel Corporation of India Limited
- Rays Power Infra Limited
- ReNew Energy Global PLC
- Sadel Group
- Sangreen Future Renewables Private Limited
- Saudi Services For Electro Mechanic Works Company Limited
- Shapoorji Pallonji & Company Private Limited
- Siemens Energy India Limited
- Simplex Infrastructures Limited
- Sinohydro Corporation Limited
- Skipper Limited
- Solarworld Energy Solutions Limited
- State Grid Corporation of China
- Sterling and Wilson Renewable Energy Limited
- Sterlite Power Transmission Limited
- Tata Power Solar Systems Limited
- Tata Projects Limited
- Techno Electric & Engineering Company Limited
- Texmaco Rail & Engineering Limited
- Tolahi Projects Private Limited
- Transrail Lighting Limited

====================
CATEGORIES
====================
Classify into ONE category:

{categories_text}

**CATEGORY PRIORITIZATION RULES:**
- Contract WON → "order wins" (highest priority)
- Contract being bid → "bidding activity"
- Project completed/commissioned → "project execution"
- M&A announced → "mergers & acquisitions"
- JV/partnership → "partnerships & alliances"
- Quarterly results → "financial"
- Stock movement → "stock market"
- When uncertain, pick the PRIMARY business action

====================
YOUR TASK
====================
Analyze the following article and extract four fields:

**1. COMPETITOR TAGGING**
RULES:
- Read the FULL article content carefully
- Identify ALL competitors mentioned who are doing business activities relevant to KEC's sectors
- ONLY use competitor names from the COMPETITORS LIST above
- Match variations to the standard name (e.g., "Larsen & Toubro" → "L&T")
- Map subsidiaries to parent company IF doing EPC/infrastructure work
- If article mentions competitor but NOT for relevant business (e.g., "Siemens washing machines"), output "-"
- If multiple competitors, separate with commas: "L&T, Tata Projects"
- If NO competitor found, output "-"
- Do NOT include KEC itself in competitor tagging

EXAMPLES:
✓ "Larsen & Toubro bags metro contract" → "L&T"
✓ "L&T Construction and Tata Projects bid for project" → "L&T, Tata Projects"
✓ "Sterling and Wilson Renewable Energy wins solar EPC" → "Sterling & Wilson"
✗ "Siemens launches new home appliances" → "-" (not KEC-relevant business)
✗ "Tata Power distributes electricity in Mumbai" → "-" (distribution, not EPC)

**2. SBU TAGGING**
RULES:
- Identify which of KEC's SBUs this article is relevant to
- IGNORE any previous SBU detection - analyze from article content freshly
- Be STRICT: Most articles relate to ONLY ONE SBU
- Only assign multiple SBUs if article explicitly mentions multiple business areas
- Use these exact SBU names: "India T&D", "International T&D", "Transportation", "Civil", "Renewables", "Oil & Gas"
- If article is about international T&D projects, use "International T&D" (not "India T&D")
- If truly none match or too generic, use "General"

MULTI-SBU EXAMPLES (rare cases):
✓ "L&T wins integrated EPC for solar park with 400 kV transmission evacuation" → "Renewables, India T&D" (or "International T&D" if outside India)
✓ "Metro project includes OHE and civil viaduct work" → "Transportation, Civil"

SINGLE-SBU EXAMPLES (most common):
✓ "L&T completes 765 kV transmission line in Rajasthan" → "India T&D"
✓ "Tata Projects wins 500 MW solar EPC contract in Abu Dhabi" → "Renewables"
✓ "IRCON bags railway electrification project" → "Transportation"
✓ "NCC constructs residential towers in Bangalore" → "Civil"

**3. CATEGORY TAG**
RULES:
- Classify into ONE category from the list above
- Apply prioritization rules
- Focus on the PRIMARY business action in the article

====================
OUTPUT FORMAT
====================
Return ONLY valid JSON with these exact field names:

{{
  "competitor_tagging": "<comma-separated competitor names from list, or '-'>",
  "sbu_tagging": "<comma-separated SBU names from list, or 'General'>",
  "category_tag": "<single category from list>",
  "contract_value_inr_crore": <numeric value in INR crore, or null if not mentioned>,
  "geography": "<India/Middle East/Africa/South East Asia/Americas/SAARC/Other or null>"
}}

**EXTRACTION RULES FOR NEW FIELDS:**

**contract_value_inr_crore:**
- Extract ONLY if explicitly mentioned in article
- Convert to INR Crore:
  * ₹X crore → X
  * ₹X lakh → X/100
  * $X million → X × 85 (approx)
  * X MW solar → null (capacity, not contract value)
- For financial results, extract revenue/profit value
- For M&A, extract deal value
- If not mentioned, return null

**geography:**
- Identify primary location mentioned
- Map to regions:
  * "India" → Any Indian state/city
  * "Middle East" → UAE, Saudi, Qatar, Bahrain, Oman, Kuwait
  * "Africa" → Any African country
  * "South East Asia" → Indonesia, Vietnam, Malaysia, Thailand, Philippines, Singapore, Myanmar, Cambodia, Laos ((excludes China)
  * "Americas" → USA, Brazil, etc.
  * "SAARC" → Bangladesh, Sri Lanka, Nepal, etc.
  * "Other" → Rest of world
- If not clear or multiple regions, use primary project location
- If not mentioned, return null
====================
EXAMPLE 1: ORDER WIN
====================
Title: "L&T bags ₹1,200 crore metro project in Pune"
Content: "Larsen & Toubro has been awarded a major contract worth ₹1,200 crore for civil and station works for Pune Metro Line 4. The project includes construction of 8 elevated stations and 12 km viaduct. L&T will complete the work in 36 months..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "L&T",
  "sbu_tagging": "Transportation",
  "category_tag": "order wins",
  "contract_value_inr_crore": 1200,
  "geography": "India"
}}
====================
EXAMPLE 2: MULTI-COMPETITOR BIDDING
====================
Title: "Five companies bid for PGCIL's 400 kV transmission project"
Content: "Power Grid Corporation of India has received bids from L&T, Tata Projects, Kalpataru Power, KEC International and Sterlite Power for the 400 kV Bikaner-Merta transmission line project worth approximately ₹600 crore..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "L&T, Tata Projects, Kalpataru, Sterlite",
  "sbu_tagging": "India T&D",
  "category_tag": "bidding activity",
  "contract_value_inr_crore": 600,
  "geography": "India"
}}
====================
EXAMPLE 3: FINANCIAL RESULTS
====================
Title: "Kalpataru Power posts 22% growth in Q4 revenue"
Content: "Kalpataru Power Transmission reported strong Q4 results with consolidated revenue growing 22% to ₹4,200 crore. The company's order book stands at ₹28,000 crore with strong pipeline in T&D and urban infra segments. Margins improved to 8.2%..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "Kalpataru",
  "sbu_tagging": "General",
  "category_tag": "financial",
  "contract_value_inr_crore": 4200,
  "geography": null
}}
Now analyze the provided article."""

    return prompt


# ============================================================================
# STAGE 1: QUICK RELEVANCE SCORING
# ============================================================================

QUICK_SCORE_PROMPT = """You are an expert relevance scorer for KEC International's competitive and market intelligence system, serving senior management for strategic decision-making.

KEC's platform tracks TWO types of intelligence. Both are valuable. Do NOT require an article to name a competitor to be relevant.

Competitors: L&T, Kalpataru, Sterlite, Tata Projects, NCC, Siemens, ABB, IRCON, RVNL, Shapoorji, PNC, Simplex, Sterling & Wilson, ReNew, Hero Future, etc.

KEC'S CORE BUSINESSES:
- Transmission & Distribution (T&D): power lines, substations, grid infrastructure
- Transportation: railways, metro, monorail, signaling
- Civil: buildings, water treatment, industrial facilities, defense infrastructure
- Renewables: solar parks, wind farms, hybrid, BESS
- Oil & Gas: pipelines, terminals, storage facilities

TYPE A — COMPETITOR INTELLIGENCE:
- competitor order wins
- competitor bidding activity
- competitor project execution
- competitor M&A or partnerships
- competitor capacity expansion
- competitor financial / order-book commentary
- competitor new market entry

TYPE B — MARKET OPPORTUNITY INTELLIGENCE (often does NOT name a competitor):
- tenders from project authorities (e.g. PGCIL, NTPC, SECI, NHAI, metro/rail bodies)
- client / authority project announcements
- government policy approvals
- budget / capex allocations
- regulatory schemes
- transmission project pipeline
- renewable / BESS tenders
- metro / rail package announcements
- oil & gas pipeline tenders
- infrastructure project awards before the winner is known
- official filings that may mention orders, contracts, LoA, order book, M&A, or project updates

SCORING RULES (0-100):

90-100: CRITICAL INTELLIGENCE
- Major competitor order win or bid in KEC sectors
- Large tender / project package from a major client or authority
- Official filing announcing a major order, LoA, contract, order-book, M&A, or capex
- Government approval or policy materially affecting T&D, rail, metro, renewables, civil, or oil & gas
- Major M&A / JV / strategic partnership in relevant sectors
- Large BESS, Green Energy Corridor, transmission, metro, rail, civil, or pipeline opportunity

80-89: HIGH RELEVANCE
- Medium-sized tender or project announcement in KEC sectors
- Important client / authority update without a named winner
- Competitor financials with order book, guidance, project pipeline, or segment commentary
- Specialist media reporting a credible project pipeline or policy movement
- New market entry by a relevant competitor or client

70-79: USEFUL / MONITOR
- Sector trend with direct relevance to KEC SBUs
- Early-stage policy or project signal
- Smaller but relevant project or tender
- Company announcement that may require follow-up
- Official source with a vague title but where query / source context suggests possible relevance

40-69: WEAK / CONTEXTUAL
- Generic infrastructure or energy news with an indirect connection
- Stock / market article with limited operational detail
- Broad economy / capex commentary without specific project or SBU impact

0-39: LOW RELEVANCE / NOISE
- Stock price movement only
- CSR, awards, HR, careers, routine board / admin update
- Unrelated businesses
- Generic market updates with no KEC-sector relevance
- Unrelated subsidiaries or consumer / IT / finance activity

HANDLING OFFICIAL / SOURCE-SPECIFIC ARTICLES:
Each article includes context fields (Search Query, Search Query Type, Accepted By Gate, Source Type, Source Category, Source Authority Score, detected signals). For articles accepted through official / source-specific query types (search_query_type starting with "site_", or an accepted_by_gate indicating an official / exchange / authority / government / tender source):
- Do NOT penalize the article merely because the title is vague.
- Official filings and government announcements often have generic titles.
- Use search_query, search_query_type, accepted_by_gate, source_type, source_category, source_authority_score, and the detected signals as context.
- If the title is vague but the source / query context is strong, assign AT LEAST monitor-level relevance (70-79) unless the title clearly indicates routine noise.
- Routine noise includes generic board meetings, HR / careers, CSR, stock-only, unrelated investor admin, or governance updates with no order / project / policy / capex relevance.
- The later full-analysis stage can validate the details.

MULTI-SIGNAL SCORING (Change 9): For each article, output FOUR independent scores (0-100 each):

1) relevance_score — how directly the news impacts KEC's core competitive landscape and market opportunities.

2) actionability_score — how much this should trigger a concrete BD, tendering, strategy, or execution action at KEC.
   90-100: immediate action (large competitor order win, direct tender, live bid, PGCIL/NHAI/SECI/DMRC package, government approval affecting KEC pipeline).
   60-89: follow-up expected (partnerships, capacity expansion, new market entry, sector policy, project execution in KEC geography).
   30-59: monitor only.
   0-29: no action expected.

3) confidence_score — how trustworthy and specific the article is.
   90-100: named competitor, named client, specific value/scope/geography, clear category, credible source description.
   60-89: some specifics but partial or vague.
   30-59: generic without concrete facts.
   0-29: unreliable, promotional, or noise.

4) sbu_fit_score — alignment to KEC's SBU priorities: India T&D, International T&D, Transportation, Civil, Renewables, Oil & Gas.
   90-100: directly one of KEC's SBUs with strong sector fit.
   60-89: adjacent or overlapping sector.
   30-59: broadly related.
   0-29: unrelated.

When scoring all four, consider source_type, search_query_type, accepted_by_gate, detected_client_authority, and detected_strategic_theme. For site-specific official queries, do not penalize vague titles; use the query context and source_type.

You will be given a batch of articles. Return ONLY a JSON array of objects, each with:
"id", "relevance_score", "actionability_score", "confidence_score", "sbu_fit_score". No explanation."""

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_relevance_score(articles_batch: List[Dict]) -> List[Dict]:
    """Score a batch of articles in a single API call (multi-signal: relevance, actionability, confidence, sbu_fit)."""
    
    articles_text = ""
    for article in articles_batch:
        articles_text += f"""
ID: {article['id']}
Title: {article.get('title', '')}
Source: {article.get('source', '')}
Search Query: {article.get('search_query', '')}
Search Query Type: {article.get('search_query_type', 'competitor')}
Accepted By Gate: {article.get('accepted_by_gate', '')}
Detected Client/Authority: {article.get('detected_client_authority', '')}
Detected Strategic Theme: {article.get('detected_strategic_theme', '')}
Competitor: {article.get('competitor', '')}
Source Type: {article.get('source_type', 'unknown')}
Source Category: {article.get('source_category', 'unknown')}
Source Authority Score: {article.get('source_authority_score', 5)}
Needs LLM Relevance Validation: {article.get('needs_llm_relevance_validation', False)}
"""
    
    prompt = f"""Score these {len(articles_batch)} articles (0-100 each score):
{articles_text}

Return ONLY a JSON array like: [{{"id": 1, "relevance_score": 85, "actionability_score": 70, "confidence_score": 80, "sbu_fit_score": 75}}, ...]"""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 60,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": QUICK_SCORE_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens read from cache")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens written to cache")
        
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\[[\s\S]*\]', raw_response)
        if json_match:
            scores_list = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON array found in response")
        
        def _clamp(v, default):
            try:
                return max(0, min(100, int(v)))
            except Exception:
                return default

        score_map = {}
        for item in scores_list:
            article_id = item.get('id')
            # Backward compat: accept legacy "score" as relevance if present.
            rel = item.get('relevance_score', item.get('score', 0))
            score_map[article_id] = {
                'relevance_score': _clamp(rel, 0),
                'actionability_score': _clamp(item.get('actionability_score', 40), 40),
                'confidence_score': _clamp(item.get('confidence_score', 50), 50),
                'sbu_fit_score': _clamp(item.get('sbu_fit_score', 50), 50),
            }
        
        default = {'relevance_score': 0, 'actionability_score': 40, 'confidence_score': 50, 'sbu_fit_score': 50}
        return [score_map.get(article['id'], dict(default)) for article in articles_batch]
        
    except Exception as e:
        logging.warning(f"Batch scoring failed: {e}")
        default = {'relevance_score': 0, 'actionability_score': 40, 'confidence_score': 50, 'sbu_fit_score': 50}
        return [dict(default) for _ in articles_batch]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_full_analysis(articles_batch: List[Dict], full_prompt: str) -> List[Dict]:
    """Analyze a batch of articles in a single API call."""
    
    articles_text = ""
    for i, article in enumerate(articles_batch):
        content = article['content'][:2000] if article['content'] else article['title']
        articles_text += f"""
--- ARTICLE {i+1} (relevance: {article['relevance_score']}/100) ---
Title: {article['title']}
Content: {content}
"""
    
    prompt = f"""Analyze these {len(articles_batch)} articles. For EACH article, provide the full analysis.

{articles_text}

Return a JSON array with one object per article (in the same order), each containing:
competitor_tagging, sbu_tagging, category_tag, contract_value_inr_crore, geography

Example format:
[
  {{"competitor_tagging": "L&T", "sbu_tagging": "India T&D", "category_tag": "order wins", "contract_value_inr_crore": 1200, "geography": "India"}},
  {{"competitor_tagging": "Kalpataru", "sbu_tagging": "Renewables", "category_tag": "financial", "contract_value_inr_crore": null, "geography": null}}
]

Return ONLY the JSON array, no other text."""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 300,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": full_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens read from cache")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens written to cache")
        
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\[[\s\S]*\]', raw_response)
        if json_match:
            analyses = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON array found in response")
        
        results = []
        for i, analysis in enumerate(analyses):
            if i < len(articles_batch):
                analysis['relevance_score'] = articles_batch[i]['relevance_score']
            
            required = ["competitor_tagging", "sbu_tagging", "category_tag", "contract_value_inr_crore", "geography"]
            for field in required:
                if field not in analysis:
                    analysis[field] = None if field in ['contract_value_inr_crore', 'geography'] else '-'
            
            if analysis.get('contract_value_inr_crore') is not None:
                try:
                    analysis['contract_value_inr_crore'] = float(analysis['contract_value_inr_crore'])
                except:
                    analysis['contract_value_inr_crore'] = None
            
            results.append(analysis)
        
        while len(results) < len(articles_batch):
            idx = len(results)
            results.append({
                "relevance_score": articles_batch[idx]['relevance_score'] if idx < len(articles_batch) else 0,
                "competitor_tagging": "-",
                "sbu_tagging": "None",
                "category_tag": "error",
                "contract_value_inr_crore": None,
                "geography": None
            })
        
        return results
        
    except Exception as e:
        logging.error(f"Batch analysis failed: {e}")
        return [{
            "relevance_score": article.get('relevance_score', 0),
            "competitor_tagging": "-",
            "sbu_tagging": "None",
            "category_tag": "error",
            "contract_value_inr_crore": None,
            "geography": None
        } for article in articles_batch]

# ============================================================================
# STAGE 2: FULL ANALYSIS
# ============================================================================

def scrape_article(url: str, max_length: int = 3000) -> str:
    """Scrape article content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe"]):
            element.decompose()
        
        # Extract text
        text = ' '.join(soup.get_text(separator=' ', strip=True).split())
        
        return text[:max_length] if text else ""
        
    except Exception as e:
        logging.warning(f"Scraping failed for {url}: {e}")
        return ""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def full_analysis(title: str, content: str, relevance_score: int, full_prompt: str) -> Dict:
    """Full analysis with scraped content"""
    
    # Use content if available, otherwise fall back to title
    analysis_text = content[:2000] if content else title
    
    prompt = f"""Analyze this news (relevance score already determined: {relevance_score}/100):

Title: {title}
Content: {analysis_text}

Provide detailed analysis."""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            temperature=0,
            system=full_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        # Extract JSON
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        
        if json_match:
            analysis = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found")
        
        # Add relevance score from Stage 1
        analysis['relevance_score'] = relevance_score
        
        # Validate
        required = ["competitor_tagging", "sbu_tagging", "category_tag", "contract_value_inr_crore", "geography"]
        for field in required:
            if field not in analysis:
                raise ValueError(f"Missing field: {field}")

        # Ensure numeric fields are properly typed
        if analysis.get('contract_value_inr_crore') is not None:
            try:
                analysis['contract_value_inr_crore'] = float(analysis['contract_value_inr_crore'])
            except:
                analysis['contract_value_inr_crore'] = None

        return analysis
        
    except Exception as e:
        logging.error(f"Full analysis failed for '{title[:50]}...': {e}")
        return {
            "relevance_score": relevance_score,
            "competitor_tagging": "-",
            "sbu_tagging": "None",
            "category_tag": "error"
        }

# ============================================================================
# RANKING
# ============================================================================

def safe_source_authority_score(value):
    """Coerce a source_authority_score to a safe int, defaulting to 5."""
    try:
        if value is None:
            return 5
        return int(value)
    except Exception:
        return 5


def calculate_rank_score(row: pd.Series, competitor_tier_map: Dict[str, int]) -> Dict:
    """
    Calculate ranking score for an article

    Formula:
    Rank Score = (Category × 50) + (Relevance) + (Competitor Tier × 10) + (Geography × 5) + (Value Tier × 5)

    Returns dict with rank_score and component breakdowns
    """

    # 1. CATEGORY WEIGHT (0-3) × 50 = 0-150 points
    category = str(row.get('category_tag', '')).lower()

    category_weights = {
        'order wins': 3,
        'bidding activity': 3,
        'mergers & acquisitions': 2,
        'partnerships & alliances': 2,
        'project execution': 2,
        'financial': 1,
        'stock market': 1,
    }

    category_weight = category_weights.get(category, 0)
    category_points = category_weight * 50

    # 2. RELEVANCE SCORE (70-100) = 70-100 points
    relevance_points = int(row.get('relevance_score', 70))

    # 3. COMPETITOR TIER (1-3) × 10 = 10-30 points
    competitor_tagging = str(row.get('competitor_tagging', '-'))
    competitors = [c.strip() for c in competitor_tagging.split(',') if c.strip() != '-']

    # Get highest tier (Tier 1 is best, so lowest number)
    competitor_tier = 3  # Default to lowest tier
    for comp in competitors:
        tier = competitor_tier_map.get(comp, 3)
        if tier < competitor_tier:
            competitor_tier = tier

    # Invert: Tier 1 = 3 points, Tier 2 = 2 points, Tier 3 = 1 point
    competitor_tier_inverted = 4 - competitor_tier
    competitor_points = competitor_tier_inverted * 10

    # 4. GEOGRAPHY BONUS (0-2) × 5 = 0-10 points
    geography = str(row.get('geography', '')).lower() if pd.notna(row.get('geography')) else ''
    sbu = str(row.get('sbu_tagging', '')).lower()

    geography_bonus = 0

    if 'international t&d' in sbu:
        if any(region in geography for region in ['middle east', 'uae', 'saudi', 'qatar', 'bahrain', 'oman', 'kuwait']):
            geography_bonus = 2
        elif any(region in geography for region in ['africa', 'americas', 'saarc']):
            geography_bonus = 1
    elif any(s in sbu for s in ['india t&d', 'transportation', 'civil', 'renewables']):
        if 'india' in geography:
            geography_bonus = 2
    elif 'oil & gas' in sbu or 'oil and gas' in sbu:
        if 'india' in geography or 'middle east' in geography:
            geography_bonus = 2

    geography_points = geography_bonus * 5

    # 5. VALUE TIER (0-4) × 5 = 0-20 points
    contract_value = row.get('contract_value_inr_crore')

    value_tier = 0

    if pd.notna(contract_value) and contract_value > 0:
        if category in ['order wins', 'bidding activity']:
            if contract_value >= 1000:
                value_tier = 4
            elif contract_value >= 500:
                value_tier = 3
            elif contract_value >= 100:
                value_tier = 2
            else:
                value_tier = 1

        elif category == 'financial':
            if contract_value >= 5000:
                value_tier = 4
            elif contract_value >= 2000:
                value_tier = 3
            elif contract_value >= 500:
                value_tier = 2
            else:
                value_tier = 1

        elif category in ['mergers & acquisitions', 'partnerships & alliances']:
            if contract_value >= 500:
                value_tier = 4
            elif contract_value >= 200:
                value_tier = 3
            elif contract_value >= 50:
                value_tier = 2
            else:
                value_tier = 1

        elif category == 'project execution':
            if contract_value >= 1000:
                value_tier = 4
            elif contract_value >= 500:
                value_tier = 3
            elif contract_value >= 100:
                value_tier = 2
            else:
                value_tier = 1

    value_points = value_tier * 5

    # 6. SOURCE AUTHORITY (minimal, temporary integration — full ranking redesign comes later)
    #    Better sources add more points; low-authority sources add only a little.
    source_authority_score = safe_source_authority_score(row.get("source_authority_score", 5))

    # Change 9: multi-signal bonus (actionability / confidence / sbu_fit), capped at 60.
    def _sig(v):
        try:
            return max(0, min(100, int(float(v))))
        except Exception:
            return 0
    multi_signal_bonus = min(
        0.25 * _sig(row.get("actionability_score", 0))
        + 0.25 * _sig(row.get("confidence_score", 0))
        + 0.25 * _sig(row.get("sbu_fit_score", 0)),
        60
    )

    # TOTAL RANK SCORE
    total_rank = (
        category_points
        + relevance_points
        + competitor_points
        + geography_points
        + value_points
        + source_authority_score
        + multi_signal_bonus
    )

    return {
        'rank_score': total_rank,
        'competitor_tier': competitor_tier,
        'category_points': category_points,
        'relevance_points': relevance_points,
        'competitor_points': competitor_points,
        'geography_points': geography_points,
        'value_points': value_points,
        'source_authority_points': source_authority_score,
        'multi_signal_bonus': multi_signal_bonus
    }

# ============================================================================
# PIPELINE PROCESSING
# ============================================================================

def stage1_quick_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Batched relevance scoring."""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 1: BATCHED RELEVANCE SCORING (Optimized)")
    logging.info(f"  Batch size: {STAGE1_BATCH_SIZE} articles per API call")
    logging.info("="*60)
    
    relevance_scores = [0] * len(df)
    actionability_scores = [0] * len(df)
    confidence_scores = [0] * len(df)
    sbu_fit_scores = [0] * len(df)
    total = len(df)
    total_batches = (total + STAGE1_BATCH_SIZE - 1) // STAGE1_BATCH_SIZE
    
    # Prepare all batches
    all_batches = []
    for i in range(0, total, STAGE1_BATCH_SIZE):
        batch_df = df.iloc[i:i+STAGE1_BATCH_SIZE]
        articles_batch = []
        for local_idx, (df_idx, row) in enumerate(batch_df.iterrows()):
            articles_batch.append({
                'id': local_idx + 1,
                'title': str(row['News Title']),
                'competitor': str(row.get('Competitor', '')),
                # Change 4 Part F: pass search-lens + source context so Stage 1
                # can score non-competitor market-opportunity articles fairly.
                'source': str(row.get('Source', '')),
                'search_query': str(row.get('search_query') or ''),
                'search_query_type': str(row.get('search_query_type') or 'competitor'),
                'accepted_by_gate': str(row.get('accepted_by_gate') or ''),
                'detected_client_authority': str(row.get('detected_client_authority') or ''),
                'detected_strategic_theme': str(row.get('detected_strategic_theme') or ''),
                'source_type': str(row.get('source_type') or 'unknown'),
                'source_category': str(row.get('source_category') or 'unknown'),
                'source_authority_score': row.get('source_authority_score', 5),
                'needs_llm_relevance_validation': row.get('needs_llm_relevance_validation', False),
            })
        all_batches.append((i, articles_batch))
    
    # Run batches in parallel (4 concurrent)
    logging.info(f"📊 Scoring {len(all_batches)} batches (4 concurrent)...")
    
    def score_batch(batch_tuple):
        start_idx, articles_batch = batch_tuple
        return start_idx, batch_relevance_score(articles_batch)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(score_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            start_idx, batch_scores = future.result()
            batch_df = df.iloc[start_idx:start_idx+STAGE1_BATCH_SIZE]
            for j, scores in enumerate(batch_scores):
                if isinstance(scores, dict):
                    rel = scores.get('relevance_score', 0)
                    actionability_scores[start_idx + j] = scores.get('actionability_score', 40)
                    confidence_scores[start_idx + j] = scores.get('confidence_score', 50)
                    sbu_fit_scores[start_idx + j] = scores.get('sbu_fit_score', 50)
                else:
                    # Backward compatibility if a plain int slips through
                    rel = scores
                    actionability_scores[start_idx + j] = 40
                    confidence_scores[start_idx + j] = 50
                    sbu_fit_scores[start_idx + j] = 50
                relevance_scores[start_idx + j] = rel
                if rel >= RELEVANCE_THRESHOLD:
                    title = str(batch_df.iloc[j]['News Title'])
                    logging.info(f"   ✅ Score {rel}: {title[:60]}...")
    
    df['relevance_score'] = relevance_scores
    df['actionability_score'] = actionability_scores
    df['confidence_score'] = confidence_scores
    df['sbu_fit_score'] = sbu_fit_scores
    
    high_relevance = df[df['relevance_score'] >= RELEVANCE_THRESHOLD]
    
    logging.info(f"\n📈 Stage 1 Complete:")
    logging.info(f"   Total articles: {len(df)}")
    logging.info(f"   API calls made: {total_batches} (was {len(df)} before optimization)")
    logging.info(f"   High relevance (≥{RELEVANCE_THRESHOLD}): {len(high_relevance)} ({len(high_relevance)/len(df)*100:.1f}%)")
    logging.info(f"   Will proceed to full analysis: {len(high_relevance)} articles")

    # Change 4 Part D (optional): which lenses survive Stage 1 relevance scoring.
    # Diagnostics only — does not change filtering behavior.
    if "search_query_type" in df.columns:
        logging.info("Post-Stage-1 search_query_type distribution (survivors): %s",
                     high_relevance["search_query_type"].value_counts(dropna=False).to_dict())

    # Change 9: average multi-signal scores per query_type
    try:
        if "search_query_type" in df.columns:
            agg = df.groupby("search_query_type")[
                ["relevance_score", "actionability_score", "confidence_score", "sbu_fit_score"]
            ].mean().round(1)
            for qtype, r in agg.iterrows():
                logging.info("Stage1 avg [%s]: relevance=%.1f actionability=%.1f confidence=%.1f sbu_fit=%.1f",
                             qtype, r["relevance_score"], r["actionability_score"],
                             r["confidence_score"], r["sbu_fit_score"])
    except Exception as e:
        logging.warning("Could not log multi-signal averages: %s", e)

    return df

def stage2_full_analysis(df: pd.DataFrame, full_prompt: str, competitor_tier_map: Dict[str, int]) -> pd.DataFrame:
    """Stage 2: Batched full analysis with prompt caching."""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 2: BATCHED FULL ANALYSIS (Optimized)")
    logging.info(f"  Batch size: {STAGE2_BATCH_SIZE} articles per API call")
    logging.info(f"  System prompt caching: ENABLED")
    logging.info("="*60)
    
    competitor_data = load_competitor_variations()
    variation_map = competitor_data['variation_map']
    
    high_rel_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()
    
    if len(high_rel_df) == 0:
        logging.warning("No articles meet relevance threshold. Skipping Stage 2.")
        return df
    
    df['competitor_tagging'] = '-'
    df['sbu_tagging'] = 'None'
    df['category_tag'] = 'not_analyzed'
    df['summary'] = 'Not analyzed (low relevance)'
    df['scraped_content'] = ''
    df['rank_score'] = 0
    df['competitor_tier'] = 3
    df['contract_value_inr_crore'] = None
    df['geography'] = None
    
    total = len(high_rel_df)
    total_batches = (total + STAGE2_BATCH_SIZE - 1) // STAGE2_BATCH_SIZE
    
    logging.info(f"\n📥 Scraping {total} articles in parallel...")
    contents = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(scrape_article, row['Link']): idx
            for idx, row in high_rel_df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            contents[idx] = future.result()

    # Change 17: scrape success-rate observability. scrape_article() already
    # swallows every failure (bot-block, 403, timeout, consent wall) and
    # returns "" — previously that was invisible; Stage 2 would silently
    # fall back to title-only analysis for those rows with no signal
    # anywhere that it happened. An empty string is an imperfect proxy for
    # "failed" (a real article could theoretically have no extractable
    # text), but in practice is a reliable enough signal to size the problem.
    scrape_empty_count = sum(1 for v in contents.values() if not (v or "").strip())
    scrape_success_count = len(contents) - scrape_empty_count
    scrape_success_rate = (scrape_success_count / len(contents) * 100) if contents else 0.0
    logging.info(
        f"   ✅ Scraped {len(contents)} articles: {scrape_success_count} with content "
        f"({scrape_success_rate:.1f}%), {scrape_empty_count} empty/failed"
    )
    if scrape_empty_count > 0 and scrape_success_rate < 70:
        logging.warning(
            f"   ⚠️ Scrape success rate {scrape_success_rate:.1f}% is low this run — "
            f"Stage 2 analysis (competitor/value/geography extraction) for the "
            f"{scrape_empty_count} empty-content articles is falling back to title-only, "
            f"which is materially less reliable."
        )
    
    high_rel_indices = list(high_rel_df.index)
    
    # Prepare all batches
    all_batches = []
    for i in range(0, total, STAGE2_BATCH_SIZE):
        batch_indices = high_rel_indices[i:i+STAGE2_BATCH_SIZE]
        articles_batch = []
        for idx in batch_indices:
            row = df.loc[idx]
            articles_batch.append({
                'title': str(row['News Title']),
                'content': contents.get(idx, ''),
                'relevance_score': row['relevance_score']
            })
        all_batches.append((batch_indices, articles_batch))
    
    # Run batches in parallel (3 concurrent)
    logging.info(f"🔍 Analyzing {len(all_batches)} batches (3 concurrent)...")
    
    def analyze_batch(batch_tuple):
        batch_indices, articles_batch = batch_tuple
        return batch_indices, batch_full_analysis(articles_batch, full_prompt)
    
    unconfirmed_value_count = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            batch_indices, batch_results = future.result()
            for j, (idx, analysis) in enumerate(zip(batch_indices, batch_results)):
                raw_competitors = analysis.get('competitor_tagging', '-')
                official_competitors = normalize_competitors_to_official(raw_competitors, variation_map)
                
                df.at[idx, 'competitor_tagging'] = official_competitors
                df.at[idx, 'sbu_tagging'] = analysis.get('sbu_tagging', 'None')
                df.at[idx, 'category_tag'] = analysis.get('category_tag', 'not_analyzed')
                # Change 17: widened from [:500] to [:2000] to match the content
                # window full_analysis()/batch_full_analysis() actually give the
                # Stage 2 LLM (analysis_text = content[:2000]) — the stored value
                # is also reused by phase2_llm_dedup's fingerprinting (see below)
                # and by summary generation, both of which benefit from the same
                # depth Stage 2 itself read rather than a much shorter snippet.
                df.at[idx, 'scraped_content'] = (contents.get(idx, ''))[:2000]

                # Change 17: hallucination guardrail — don't persist a contract
                # value the source text doesn't actually support.
                extracted_value = analysis.get('contract_value_inr_crore')
                if extracted_value is not None and not value_confirmed_in_text(extracted_value, contents.get(idx, '')):
                    logging.warning(
                        "contract_value_inr_crore=%s not found in scraped content for '%s' — nulling as unconfirmed",
                        extracted_value, str(df.loc[idx, 'News Title'])[:60]
                    )
                    unconfirmed_value_count += 1
                    extracted_value = None
                df.at[idx, 'contract_value_inr_crore'] = extracted_value
                df.at[idx, 'geography'] = analysis.get('geography')

    if unconfirmed_value_count > 0:
        logging.warning(
            f"   ⚠️ {unconfirmed_value_count} contract_value_inr_crore figures could not be "
            f"confirmed against source text and were nulled out this run"
        )

    logging.info(f"\n📊 Calculating ranking scores...")
    for idx in high_rel_indices:
        rank_data = calculate_rank_score(df.loc[idx], competitor_tier_map)
        df.at[idx, 'rank_score'] = rank_data['rank_score']
        df.at[idx, 'competitor_tier'] = rank_data['competitor_tier']
        
        logging.info(f"   Rank {rank_data['rank_score']}: {df.loc[idx, 'News Title'][:60]}...")
    
    logging.info(f"\n✅ Stage 2 Complete:")
    logging.info(f"   Articles analyzed: {len(high_rel_df)}")
    logging.info(f"   API calls made: {total_batches} (was {len(high_rel_df)} before optimization)")
    
    return df

# ============================================================================
# DEDUPLICATION - PHASE 1: STRING-BASED (FAST)
# ============================================================================

# Change 17: tolerance for confirming an LLM-extracted contract value against
# the actual scraped article text. Matches the tolerance already used
# elsewhere for value comparison (values_close, has_similar_numbers).
CONTRACT_VALUE_CONFIRMATION_TOLERANCE_PCT = 10


def value_confirmed_in_text(value, text: str, tolerance_pct: int = CONTRACT_VALUE_CONFIRMATION_TOLERANCE_PCT) -> bool:
    """
    Change 17: guardrail against a hallucinated contract_value_inr_crore.

    The Stage 2 prompt already says "null if not mentioned," but nothing
    previously checked that the LLM actually complied — a number could be
    invented from the headline, general background knowledge, or simply
    misread, with no way to catch it before it reached the dashboard as an
    executive-facing figure.

    Returns True only if `value` (or something within tolerance_pct of it,
    after extract_numbers_from_text's crore/lakh/million unit normalization)
    literally appears among the numbers found in `text`. Empty text can
    never confirm a value — if we have no article content, we have no way to
    verify anything the LLM claims, regardless of how plausible it looks.
    """
    if value is None:
        return True  # nothing to confirm
    if not text or not text.strip():
        return False
    try:
        candidates = extract_numbers_from_text(text)
    except Exception:
        return False
    for candidate in candidates:
        if values_close(value, candidate, tolerance_pct=tolerance_pct):
            return True
    return False


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all contract values from text, handling Indian number formats"""
    numbers = []

    # Handle Indian format: Rs 35,54,82,968 → convert to crore
    indian_format = re.findall(r'(?:rs|₹|inr)\.?\s*([\d,]+)', text, re.IGNORECASE)
    for match in indian_format:
        num_str = match.replace(',', '')
        try:
            num = float(num_str)
            # If number looks like full rupees (>10 million), convert to crore
            if num > 10_000_000:
                num = num / 10_000_000
            elif num > 100_000:
                num = num / 10_000_000
            numbers.append(round(num, 2))
        except:
            pass

    # Handle crore/lakh/million explicitly stated
    patterns = [
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:crore|cr)', 1.0),
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:lakh|lac)', 0.01),
        (r'(\d+(?:[,.]\d+)*)\s*(?:million|mn)', 8.5),
    ]

    for pattern, multiplier in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                num = float(match.replace(',', '')) * multiplier
                numbers.append(round(num, 2))
            except:
                pass

    return numbers


def has_similar_numbers(numbers1: List[float], numbers2: List[float]) -> bool:
    """Check if two lists of numbers have similar values (within 10% tolerance)"""
    if not numbers1 or not numbers2:
        return False
    for n1 in numbers1:
        for n2 in numbers2:
            if n1 > 0 and n2 > 0:
                diff_pct = abs(n1 - n2) / max(n1, n2) * 100
                if diff_pct < 10:
                    return True
    return False


def has_core_content_match(title1: str, title2: str) -> bool:
    """Check if two titles share core content keywords"""
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'must', 'its', 'their',
        'worth', 'order', 'contract', 'project', 'wins', 'bags', 'secures',
        'gets', 'receives', 'awarded', 'adds', 'another', 'growing', 'book'
    }
    words1 = set(re.findall(r'\b\w+\b', title1.lower())) - stop_words
    words2 = set(re.findall(r'\b\w+\b', title2.lower())) - stop_words
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    total = min(len(words1), len(words2))
    return (overlap / total * 100) >= 60 if total > 0 else False


def phase1_string_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Fast string-based deduplication.
    Catches exact matches, fuzzy matches, and value+competitor matches.
    """
    logging.info("\n🔍 Phase 1: String-based deduplication...")

    if df.empty:
        return df

    df_reset = df.reset_index(drop=True)
    to_drop = set()

    # Strategy 1: Exact title duplicates
    seen_titles = {}
    for i in range(len(df_reset)):
        title = str(df_reset.iloc[i]['News Title']).lower().strip()
        if title in seen_titles:
            to_drop.add(i)
        else:
            seen_titles[title] = i

    exact_count = len(to_drop)
    logging.info(f"   Exact duplicates: {exact_count}")

    # Strategy 2: Fuzzy + value + core content matching
    for i in range(len(df_reset)):
        if i in to_drop:
            continue

        title_i = str(df_reset.iloc[i]['News Title']).lower()
        date_i = df_reset.iloc[i]['Published Date']
        competitor_i = str(df_reset.iloc[i].get('competitor_tagging') or df_reset.iloc[i].get('Competitor') or '').lower()
        numbers_i = extract_numbers_from_text(title_i)

        for j in range(i + 1, min(i + 100, len(df_reset))):
            if j in to_drop:
                continue

            title_j = str(df_reset.iloc[j]['News Title']).lower()
            date_j = df_reset.iloc[j]['Published Date']
            competitor_j = str(df_reset.iloc[j].get('competitor_tagging') or df_reset.iloc[j].get('Competitor') or '').lower()

            try:
                date_diff = abs((date_i - date_j).days)
            except:
                date_diff = 0

            if date_diff > 3:
                continue

            numbers_j = extract_numbers_from_text(title_j)
            similarity = SequenceMatcher(None, title_i, title_j).ratio()
            same_competitor = (competitor_i == competitor_j and competitor_i not in ('', '-'))
            same_value = has_similar_numbers(numbers_i, numbers_j)
            core_match = has_core_content_match(title_i, title_j)

            is_duplicate = False
            if similarity > 0.85:
                is_duplicate = True
            elif same_competitor and same_value and date_diff <= 1:
                is_duplicate = True
            elif same_competitor and core_match and date_diff <= 2:
                is_duplicate = True

            if is_duplicate:
                to_drop.add(j)

    fuzzy_count = len(to_drop) - exact_count
    logging.info(f"   Fuzzy/value duplicates: {fuzzy_count}")
    logging.info(f"   Phase 1 total removed: {len(to_drop)} | Remaining: {len(df_reset) - len(to_drop)}")

    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


# ============================================================================
# DEDUPLICATION - PHASE 2: LLM-BASED (SEMANTIC)
# ============================================================================

# Category-specific fields to extract for fingerprinting
CATEGORY_FINGERPRINT_FIELDS = {
    "order wins":               ["company", "client_or_authority", "contract_value_crore", "scope", "location"],
    "bidding activity":         ["companies_bidding", "client_or_authority", "project_value_crore", "scope", "location"],
    "financial":                ["company", "period", "revenue_crore", "profit_crore", "order_book_crore"],
    "project execution":        ["company", "project_name", "capacity_or_scale", "location", "milestone"],
    "mergers & acquisitions":   ["acquirer", "target_company", "deal_value_crore", "stake_percent"],
    "partnerships & alliances": ["companies_involved", "deal_type", "sector", "value_crore"],
    "stock market":             ["company", "price_movement_percent", "trigger_event"],
    "regulatory & policy":      ["authority", "policy_or_rule", "sector_affected"],
    "industry trends":          ["topic", "key_stat", "geography"],
    "legal & disputes":         ["company", "counterparty", "issue_type", "value_crore"],
}

FINGERPRINT_SYSTEM_PROMPT = """You are a news deduplication assistant for a competitive intelligence system.

Your job is to read a news article and extract a structured fingerprint of the KEY FACTS that identify this specific event.
The fingerprint will be used to detect if multiple articles are reporting the same underlying news event.

Extract ONLY facts explicitly stated in the article. Use null for anything not mentioned.
Return ONLY valid JSON, no explanation."""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def extract_fingerprint(title: str, content: str, category: str) -> Dict:
    """Extract a semantic fingerprint from an article based on its category"""

    fields = CATEGORY_FINGERPRINT_FIELDS.get(
        category.lower(),
        ["company", "event_type", "value", "location"]  # fallback
    )

    fields_desc = "\n".join([f'  "{f}": <extracted value or null>' for f in fields])

    prompt = f"""Article Title: {title}

Article Content: {content[:2000] if content else 'Not available'}

Category: {category}

Extract the following key facts from this article and return as JSON:
{{
{fields_desc}
}}

Rules:
- Extract ONLY facts explicitly stated
- For values/numbers: normalize to crore (e.g. Rs 35,54,82,968 = 35.55 crore, Rs 1200 crore = 1200)
- For company names: use the most common/standard form
- For scope: include MW/km/units as mentioned
- Use null if not mentioned"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": FINGERPRINT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group(0))
        return {}

    except Exception as e:
        logging.warning(f"Fingerprint extraction failed for '{title[:50]}': {e}")
        return {}

def check_fingerprint_against_db(fingerprint: Dict, category: str, competitor: str, published_date, lookback_days: int = 7) -> Dict:
    """
    Check if a fingerprint matches any existing article in the database.
    Returns {"is_duplicate": True/False, "matched_article_id": id or None}
    """
    if not fingerprint:
        return {"is_duplicate": False, "matched_article_id": None}

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get recent articles with fingerprints in the same category/competitor group
        cur.execute("""
            SELECT id, news_title, fingerprint, category_tag, competitor_tagging, published_date
            FROM processed_articles
            WHERE fingerprint IS NOT NULL
            AND category_tag = %s
            AND published_date >= %s - INTERVAL '%s days'
            AND is_duplicate = FALSE
            ORDER BY published_date DESC
            LIMIT 50
        """, (category, published_date, lookback_days))

        existing = cur.fetchall()
        cur.close()
        conn.close()

        for row in existing:
            existing_fp = row.get('fingerprint')
            if not existing_fp:
                continue

            # Parse if stored as string
            if isinstance(existing_fp, str):
                try:
                    existing_fp = json.loads(existing_fp)
                except:
                    continue

            if fingerprints_match(fingerprint, existing_fp, category):
                logging.info(f"   🔄 Cross-batch duplicate found! Matches article #{row['id']}: {row['news_title'][:60]}...")
                return {"is_duplicate": True, "matched_article_id": row['id']}

        return {"is_duplicate": False, "matched_article_id": None}

    except Exception as e:
        logging.warning(f"   ⚠️ DB fingerprint check failed: {e}")
        return {"is_duplicate": False, "matched_article_id": None}
    
def fingerprints_match(fp1: Dict, fp2: Dict, category: str) -> bool:
    """
    Compare two fingerprints to determine if they represent the same event.
    Uses category-specific matching logic.
    """
    if not fp1 or not fp2:
        return False

    cat = category.lower()

    def normalize(val):
        if val is None:
            return None
        return str(val).lower().strip()

    def values_similar(v1, v2, tolerance=0.10):
        try:
            n1, n2 = float(v1), float(v2)
            if n1 == 0 and n2 == 0:
                return True
            if n1 == 0 or n2 == 0:
                return False
            return abs(n1 - n2) / max(n1, n2) <= tolerance
        except:
            return False

    def company_match(c1, c2):
        if not c1 or not c2:
            return False
        c1, c2 = normalize(c1), normalize(c2)
        if c1 == c2 or c1 in c2 or c2 in c1:
            return True
        return SequenceMatcher(None, c1, c2).ratio() > 0.80

    if cat == "order wins":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        if not company_ok:
            return False
        v1, v2 = fp1.get('contract_value_crore'), fp2.get('contract_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        client_ok = company_match(fp1.get('client_or_authority'), fp2.get('client_or_authority'))
        scope1, scope2 = normalize(fp1.get('scope')), normalize(fp2.get('scope'))
        scope_ok = (scope1 and scope2 and (scope1 in scope2 or scope2 in scope1)) or (not scope1 or not scope2)
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        return company_ok and value_ok and (client_ok or scope_ok) and location_ok

    elif cat == "bidding activity":
        client_ok = company_match(fp1.get('client_or_authority'), fp2.get('client_or_authority'))
        v1, v2 = fp1.get('project_value_crore'), fp2.get('project_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        return client_ok and value_ok and location_ok

    elif cat == "financial":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        period_ok = normalize(fp1.get('period')) == normalize(fp2.get('period')) or not fp1.get('period') or not fp2.get('period')
        v1, v2 = fp1.get('revenue_crore'), fp2.get('revenue_crore')
        revenue_ok = values_similar(v1, v2) if (v1 and v2) else True
        return company_ok and period_ok and revenue_ok

    elif cat == "mergers & acquisitions":
        acquirer_ok = company_match(fp1.get('acquirer'), fp2.get('acquirer'))
        target_ok = company_match(fp1.get('target_company'), fp2.get('target_company'))
        v1, v2 = fp1.get('deal_value_crore'), fp2.get('deal_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        # If both companies null, fall back to competitor group match
        if not fp1.get('acquirer') and not fp1.get('target_company'):
            return True  # Same competitor group + same category = duplicate
        return (acquirer_ok or target_ok) and value_ok
    
    elif cat == "partnerships & alliances":
        companies1 = normalize(fp1.get('companies_involved') or '')
        companies2 = normalize(fp2.get('companies_involved') or '')
        companies_ok = SequenceMatcher(None, companies1, companies2).ratio() > 0.70 if companies1 and companies2 else False
        sector_ok = normalize(fp1.get('sector')) == normalize(fp2.get('sector')) or not fp1.get('sector') or not fp2.get('sector')
        return companies_ok and sector_ok

    elif cat == "project execution":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        scale1, scale2 = normalize(fp1.get('capacity_or_scale')), normalize(fp2.get('capacity_or_scale'))
        scale_ok = (scale1 and scale2 and SequenceMatcher(None, scale1, scale2).ratio() > 0.70) or not scale1 or not scale2
        return company_ok and location_ok and scale_ok

    elif cat == "stock market":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        trigger1, trigger2 = normalize(fp1.get('trigger_event')), normalize(fp2.get('trigger_event'))
        trigger_ok = (trigger1 and trigger2 and SequenceMatcher(None, trigger1, trigger2).ratio() > 0.70) or not trigger1 or not trigger2
        return company_ok and trigger_ok

    else:
        # Generic: fuzzy match on all string fields
        matches = 0
        total = 0
        for key in fp1:
            if fp1[key] and fp2.get(key):
                total += 1
                if normalize(fp1[key]) == normalize(fp2[key]) or SequenceMatcher(None, normalize(fp1[key]), normalize(fp2[key])).ratio() > 0.75:
                    matches += 1
        return (matches / total) >= 0.6 if total > 0 else False


def phase2_llm_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: LLM-based semantic deduplication.
    Groups articles by competitor, extracts fingerprints, compares within each group.
    Keeps the article with the highest rank_score from each duplicate group.
    """
    logging.info("\n🤖 Phase 2: LLM semantic deduplication...")

    if df.empty or len(df) <= 1:
        return df

    if 'rank_score' not in df.columns:
        df['rank_score'] = 0

    df_reset = df.reset_index(drop=True)

    # Step 1: Change 17 — reuse the content Stage 2 already scraped instead of
    # fetching every URL a second time. The old code always re-scraped live
    # here and only fell back to the stored scraped_content when the idx key
    # was entirely absent from the fresh results — which never happened for
    # any row with a Link, since every submitted future gets a dict entry
    # even on failure (""). In practice that meant: (a) every article was
    # fetched twice, doubling network cost and bot-block/rate-limit exposure,
    # and (b) a transient failure on this SECOND attempt silently threw away
    # perfectly good content Stage 2 had already captured. Only articles with
    # no stored content get a live re-scrape here.
    logging.info(f"   📥 Reusing Stage 2 scraped content for {len(df_reset)} articles...")

    contents = {}
    need_rescrape = []
    for idx, row in df_reset.iterrows():
        existing = str(row.get('scraped_content', '') or '')
        if existing.strip():
            contents[idx] = existing
        elif pd.notna(row.get('Link')):
            need_rescrape.append(idx)

    if need_rescrape:
        logging.info(f"   📥 Re-scraping {len(need_rescrape)} articles with no stored content...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(scrape_article, df_reset.loc[idx, 'Link']): idx
                for idx in need_rescrape
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                contents[idx] = future.result()

    # Step 2: Extract fingerprints via LLM
    logging.info(f"   🔑 Extracting fingerprints...")
    fingerprints = {}
    for idx, row in df_reset.iterrows():
        title = str(row['News Title'])
        content = contents.get(idx, str(row.get('scraped_content', '')))
        category = str(row.get('category_tag', 'order wins'))

        fp = extract_fingerprint(title, content, category)
        fingerprints[idx] = fp

        logging.info(f"   [{idx+1}/{len(df_reset)}] {title[:60]}...")
        time.sleep(RATE_LIMIT_DELAY)
    # Store fingerprints on DataFrame for later summary generation
    df_reset['_fingerprint'] = df_reset.index.map(lambda i: fingerprints.get(i, {}))

    # Step 2.5: Cross-batch dedup — check each fingerprint against DB history
    logging.info("\n   🗄️ Checking fingerprints against database history...")
    df_reset['is_duplicate'] = False
    df_reset['matched_article_id'] = None  # Change 15: needed to resolve cluster continuity
    cross_batch_dupes = 0

    for idx, row in df_reset.iterrows():
        fp = fingerprints.get(idx, {})
        if not fp:
            continue

        category = str(row.get('category_tag', '')).lower()
        competitor = str(row.get('competitor_tagging', '-'))
        pub_date = row.get('Published Date')

        result = check_fingerprint_against_db(fp, category, competitor, pub_date)
        if result['is_duplicate']:
            df_reset.at[idx, 'is_duplicate'] = True
            df_reset.at[idx, 'matched_article_id'] = result.get('matched_article_id')
            cross_batch_dupes += 1

    logging.info(f"   🗄️ Cross-batch duplicates found: {cross_batch_dupes}")

    # Step 3: Group by competitor, compare fingerprints within each group
    logging.info("\n   🔍 Comparing fingerprints within competitor groups...")

    to_drop = set()
    competitor_groups: Dict[str, List[int]] = {}

    for idx, row in df_reset.iterrows():
        competitor_raw = str(row.get('competitor_tagging') or row.get('Competitor') or 'General')
        competitors = [c.strip() for c in competitor_raw.split(',') if c.strip() and c.strip() != '-']
        if not competitors:
            competitors = ['General']
        for comp in competitors:
            if comp not in competitor_groups:
                competitor_groups[comp] = []
            competitor_groups[comp].append(idx)

    logging.info(f"   📦 {len(competitor_groups)} competitor groups to compare...")

    already_compared = set()

    for comp, indices in competitor_groups.items():
        if len(indices) <= 1:
            continue

        logging.info(f"   👥 {comp}: {len(indices)} articles")

        for i in range(len(indices)):
            idx_i = indices[i]
            if idx_i in to_drop:
                continue

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                if idx_j in to_drop:
                    continue

                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in already_compared:
                    continue
                already_compared.add(pair)

                row_i = df_reset.iloc[idx_i]
                row_j = df_reset.iloc[idx_j]

                # Only compare articles within 3 days of each other
                try:
                    date_diff = abs((row_i['Published Date'] - row_j['Published Date']).days)
                except:
                    date_diff = 0

                cat_i = str(row_i.get('category_tag', '')).lower()
                cat_j = str(row_j.get('category_tag', '')).lower()

                wide_window_cats = {'mergers & acquisitions', 'partnerships & alliances', 'legal & disputes'}
                max_days = 5 if cat_i in wide_window_cats else 3
                if date_diff > max_days:
                    continue
                if cat_i != cat_j:
                    continue

                fp_i = fingerprints.get(idx_i, {})
                fp_j = fingerprints.get(idx_j, {})

                if fingerprints_match(fp_i, fp_j, cat_i):
                    # Keep higher rank_score, drop the other
                    score_i = float(row_i.get('rank_score') or 0)
                    score_j = float(row_j.get('rank_score') or 0)

                    drop_idx = idx_j if score_i >= score_j else idx_i
                    to_drop.add(drop_idx)

                    keep_title = row_i['News Title'] if score_i >= score_j else row_j['News Title']
                    drop_title = row_j['News Title'] if score_i >= score_j else row_i['News Title']
                    logging.info(f"   🗑️  DUPLICATE: '{drop_title[:60]}...'")
                    logging.info(f"       KEEPING:   '{keep_title[:60]}...'")

    logging.info(f"\n   Phase 2 removed: {len(to_drop)} semantic duplicates")
    logging.info(f"   Final article count: {len(df_reset) - len(to_drop)}")

    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-phase deduplication:
    Phase 1 — Fast string-based (exact, fuzzy, value matching)
    Phase 2 — LLM semantic deduplication grouped by competitor
    """
    logging.info("\n" + "="*60)
    logging.info("DEDUPLICATION: TWO-PHASE")
    logging.info("="*60)

    initial_count = len(df)
    logging.info(f"   Starting with: {initial_count} articles")

    # Phase 1: String-based
    df = phase1_string_dedup(df)
    after_phase1 = len(df)
    logging.info(f"   After Phase 1: {after_phase1} articles ({initial_count - after_phase1} removed)")

    # Phase 2: LLM semantic
    df = phase2_llm_dedup(df)
    after_phase2 = len(df)
    logging.info(f"   After Phase 2: {after_phase2} articles ({after_phase1 - after_phase2} removed)")

    logging.info(f"\n✅ Dedup complete: {initial_count} → {after_phase2} articles ({initial_count - after_phase2} total removed)")

    return df

# ============================================================================
# LLM SUMMARY GENERATION
# ============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst for KEC International, an infrastructure EPC company.

Your job is to write concise 2-3 sentence executive summaries of competitor news articles.

Structure each summary as:
- Sentence 1: Who did what (the core event, using the competitor's exact full name)
- Sentence 2: Scale and context (contract value in ₹, geography, project scope/specs)
- Sentence 3: Strategic implication for KEC (which SBU is affected, competitive threat)

Rules:
- Use the EXACT competitor name from the "Competitor" field — if "-" or empty, infer from content
- NEVER write "-" or "Unknown" as a company name
- Be specific: include ₹ values, MW/km figures, location names wherever available
- Anchor on the pre-extracted facts (fingerprint) first, use raw content only to add colour
- Keep it under 60 words total
- Write in third person, present tense
- No filler phrases like "it is worth noting" or "this highlights"
- Return ONLY a JSON array of strings, no explanation, no markdown"""

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_generate_summaries(articles_batch: List[Dict]) -> List[str]:
    """Generate rich 2-3 sentence LLM summaries for a batch of articles."""

    articles_text = ""
    for i, article in enumerate(articles_batch):
        content = article.get('content', '')[:1500] or article.get('title', '')

        # Format fingerprint as clean key: value lines, skipping nulls
        fp = article.get('fingerprint', {})
        fp_text = ''
        if fp and isinstance(fp, dict):
            fp_lines = []
            for k, v in fp.items():
                if v is not None:
                    if isinstance(v, list):
                        v = ', '.join(str(x) for x in v)
                    fp_lines.append(f"  {k}: {v}")
            if fp_lines:
                fp_text = "Pre-extracted facts:\n" + "\n".join(fp_lines)

        articles_text += f"""
--- ARTICLE {i+1} ---
Title: {article['title']}
Competitor: {article['competitor_tagging']}
SBU: {article['sbu_tagging']}
Category: {article['category_tag']}
Geography: {article.get('geography') or 'Not specified'}
Contract Value (INR Crore): {article.get('contract_value_inr_crore') or 'Not specified'}
{fp_text}
Raw content: {content}
"""
    prompt = f"""Write a 2-3 sentence executive summary for each of these {len(articles_batch)} articles.

{articles_text}

Return a JSON array of strings, one summary per article, in the same order:
["Summary for article 1.", "Summary for article 2.", ...]

Remember:
- Use the exact competitor name from the "Competitor" field
- Anchor on the pre-extracted facts first
- Under 60 words per summary"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 150,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SUMMARY_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Log cache usage
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens")

        # Strip markdown fences
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        json_match = re.search(r'\[[\s\S]*\]', raw)
        if json_match:
            summaries = json.loads(json_match.group(0))
            # Pad with title fallbacks if model returned fewer than expected
            while len(summaries) < len(articles_batch):
                summaries.append(articles_batch[len(summaries)]['title'])
            return [str(s) for s in summaries[:len(articles_batch)]]

        raise ValueError("No JSON array found in response")

    except Exception as e:
        logging.error(f"Batch summary generation failed: {e}")
        return [a['title'] for a in articles_batch]


def generate_llm_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates batched LLM summary generation using fingerprints + scraped content.
    Call this BEFORE dropping the _fingerprint column.
    Runs 3 batches concurrently with prompt caching.
    """
    logging.info("\n📝 Generating LLM summaries (batched, fingerprint-anchored)...")

    if df.empty:
        return df

    SUMMARY_BATCH_SIZE = 5
    all_indices = list(df.index)
    total = len(all_indices)
    total_batches = (total + SUMMARY_BATCH_SIZE - 1) // SUMMARY_BATCH_SIZE

    logging.info(f"   Articles: {total} | Batches: {total_batches} | Concurrency: 3")

    # Build batches — include fingerprint from _fingerprint column
    all_batches = []
    for i in range(0, total, SUMMARY_BATCH_SIZE):
        batch_indices = all_indices[i:i + SUMMARY_BATCH_SIZE]
        articles_batch = []
        for idx in batch_indices:
            row = df.loc[idx]
            content = str(row.get('scraped_content', ''))
            fingerprint = row.get('_fingerprint', {})

            # If no content and no fingerprint, use title directly — skip LLM
            if not content.strip() and not fingerprint:
                df.at[idx, 'summary'] = str(row.get('News Title', ''))
                continue

            articles_batch.append({
                'title': str(row.get('News Title', '')),
                'competitor_tagging': str(row.get('competitor_tagging', '-')),
                'sbu_tagging': str(row.get('sbu_tagging', 'General')),
                'category_tag': str(row.get('category_tag', '')),
                'geography': row.get('geography'),
                'contract_value_inr_crore': row.get('contract_value_inr_crore'),
                'content': content,
                'fingerprint': fingerprint
            })        
    
        all_batches.append((batch_indices, articles_batch))

    def run_batch(batch_tuple):
        batch_indices, articles_batch = batch_tuple
        summaries = batch_generate_summaries(articles_batch)
        return batch_indices, summaries

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            batch_indices, summaries = future.result()
            for idx, summary in zip(batch_indices, summaries):
                df.at[idx, 'summary'] = summary
                logging.info(f"   ✅ {str(df.loc[idx, 'News Title'])[:50]}...")
                logging.info(f"      → {summary[:80]}...")

    logging.info(f"   ✅ Done: {total} summaries in {total_batches} API calls")
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline with structured stage tracking."""

    start_time = time.time()

    logging.info("=" * 60)
    logging.info("KEC INTERNATIONAL - COMPETITIVE INTELLIGENCE ANALYZER")
    logging.info("=" * 60)

    # Stage: load_raw_articles
    log_pipeline_run("load_raw_articles", "started")
    try:
        df = load_raw_articles()
        articles_in = len(df) if df is not None else 0
        log_pipeline_run("load_raw_articles", "success", articles_in=articles_in, articles_out=articles_in)
    except Exception as e:
        log_pipeline_run("load_raw_articles", "failed", error_message=str(e))
        logging.exception("load_raw_articles failed")
        raise

    if df is None or df.empty:
        logging.warning("No raw articles to process. Skipping remaining stages.")
        return

    logging.info(f"Loaded {len(df)} raw articles")

    log_query_type_distribution(df, "Raw articles loaded")
    log_gate_distribution(df, "Raw articles loaded")
    log_source_type_distribution(df, "Raw articles loaded")

    # Load Excel mapping data
    try:
        excel_data = load_excel_data()
        competitor_tier_map = load_competitor_tiers()
    except Exception as e:
        logging.error(f"Failed to load Excel data: {e}")
        raise

    full_prompt = build_full_analysis_prompt(categories=excel_data["categories"])

    # Stage: stage1_quick_scoring
    articles_in = len(df)
    log_pipeline_run("stage1_quick_scoring", "started", articles_in=articles_in)
    try:
        df = stage1_quick_scoring(df)
        log_pipeline_run("stage1_quick_scoring", "success", articles_in=articles_in, articles_out=len(df))
    except Exception as e:
        log_pipeline_run("stage1_quick_scoring", "failed", error_message=str(e))
        logging.exception("stage1_quick_scoring failed")
        raise

    if df is None or df.empty:
        logging.warning("No articles after Stage 1. Skipping remaining stages.")
        return

    log_query_type_distribution(df, "After Stage 1 scoring")
    log_relevance_yield_by_query_type(df, "After Stage 1 scoring")

    # Stage: stage2_full_analysis
    articles_in = len(df)
    log_pipeline_run("stage2_full_analysis", "started", articles_in=articles_in)
    try:
        df = stage2_full_analysis(df, full_prompt, competitor_tier_map)
        log_pipeline_run("stage2_full_analysis", "success", articles_in=articles_in, articles_out=len(df))
    except Exception as e:
        log_pipeline_run("stage2_full_analysis", "failed", error_message=str(e))
        logging.exception("stage2_full_analysis failed")
        raise

    if df is None or df.empty:
        logging.warning("No articles after Stage 2. Skipping remaining stages.")
        return

    log_query_type_distribution(df, "After Stage 2 full analysis")
    log_category_yield_by_query_type(df, "After Stage 2 full analysis")

    # Filter to high relevance before dedup / clustering / summaries / save
    df_final = df[df["relevance_score"] >= RELEVANCE_THRESHOLD].copy()

    if df_final.empty:
        logging.warning("No high-relevance articles after threshold filtering. Nothing to save.")
        return

    log_query_type_distribution(df_final, "After relevance threshold filtering")
    log_relevance_yield_by_query_type(df_final, "After relevance threshold filtering")
    log_gate_distribution(df_final, "After relevance threshold filtering")
    log_source_type_distribution(df_final, "After relevance threshold filtering")

    # Stage: deduplicate_articles
    articles_in = len(df_final)
    log_pipeline_run("deduplicate_articles", "started", articles_in=articles_in)
    try:
        df_final = deduplicate_articles(df_final)
        log_pipeline_run("deduplicate_articles", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("deduplicate_articles", "failed", error_message=str(e))
        logging.warning(f"deduplicate_articles failed, continuing without dedup: {e}")

    # Stage: generate_llm_summaries
    articles_in = len(df_final)
    log_pipeline_run("generate_llm_summaries", "started", articles_in=articles_in)
    try:
        non_dupes = df_final[df_final.get("is_duplicate", False) == False].copy()
        dupes = df_final[df_final.get("is_duplicate", False) == True].copy()

        if not non_dupes.empty:
            non_dupes = generate_llm_summaries(non_dupes)

        if not dupes.empty:
            dupes["summary"] = dupes["News Title"]

        df_final = pd.concat([non_dupes, dupes], ignore_index=True)
        df_final = df_final.drop(columns=["_fingerprint"], errors="ignore")

        log_pipeline_run("generate_llm_summaries", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("generate_llm_summaries", "failed", error_message=str(e))
        logging.warning(f"LLM summary generation failed, continuing with existing summaries: {e}")

    # Stage: resolve_cluster_continuations (Change 15)
    articles_in = len(df_final)
    log_pipeline_run("resolve_cluster_continuations", "started", articles_in=articles_in)
    try:
        df_final = resolve_cross_run_cluster_continuations(df_final)
        log_pipeline_run("resolve_cluster_continuations", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("resolve_cluster_continuations", "failed", error_message=str(e))
        logging.warning(f"resolve_cross_run_cluster_continuations failed, continuing without cross-run continuity: {e}")

    # Stage: in_memory_event_clustering
    articles_in = len(df_final)
    log_pipeline_run("in_memory_event_clustering", "started", articles_in=articles_in)
    try:
        df_final = create_in_memory_event_clusters(df_final)
        log_pipeline_run("in_memory_event_clustering", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("in_memory_event_clustering", "failed", error_message=str(e))
        logging.warning(f"Event clustering failed, falling back to scaffold: {e}")
        df_final = assign_event_clusters_scaffold(df_final)

    # Stage: build_cluster_event_fields
    articles_in = len(df_final)
    log_pipeline_run("build_cluster_event_fields", "started", articles_in=articles_in)
    try:
        df_final = build_cluster_event_fields(df_final)
        log_pipeline_run("build_cluster_event_fields", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("build_cluster_event_fields", "failed", error_message=str(e))
        logging.warning(f"build_cluster_event_fields failed: {e}")

    # Stage: assign_event_impact_scores
    articles_in = len(df_final)
    log_pipeline_run("assign_event_impact_scores", "started", articles_in=articles_in)
    try:
        df_final = assign_event_impact_scores(df_final, competitor_tier_map)
        log_pipeline_run("assign_event_impact_scores", "success", articles_in=articles_in, articles_out=len(df_final))
    except Exception as e:
        log_pipeline_run("assign_event_impact_scores", "failed", error_message=str(e))
        logging.warning(f"Event impact scoring failed: {e}")

    log_query_type_distribution(df_final, "Final articles before save")
    log_category_yield_by_query_type(df_final, "Final articles before save")
    log_gate_distribution(df_final, "Final articles before save")
    log_source_type_distribution(df_final, "Final articles before save")

    # Stage: save_processed_articles
    articles_in = len(df_final)
    log_pipeline_run("save_processed_articles", "started", articles_in=articles_in)
    try:
        # Change 15: drop transient bookkeeping columns before save (mirrors
        # the existing _fingerprint drop above).
        df_final = df_final.drop(columns=["_cluster_continuation", "matched_article_id"], errors="ignore")
        save_to_processed_articles(df_final)
        log_pipeline_run("save_processed_articles", "success", articles_in=articles_in, articles_out=articles_in)
    except Exception as e:
        log_pipeline_run("save_processed_articles", "failed", error_message=str(e))
        logging.exception("save_to_processed_articles failed")
        raise

    elapsed = time.time() - start_time
    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Time: {elapsed/60:.1f} minutes")
    logging.info(f"Total raw articles processed: {len(df)}")
    logging.info(f"Final saved candidate articles: {len(df_final)}")

if __name__ == "__main__":
    main()
