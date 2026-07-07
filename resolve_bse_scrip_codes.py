"""
resolve_bse_scrip_codes.py  — Change 21, Step 1 (standalone, run manually)
============================================================================

Builds a name -> BSE scrip-code mapping for the dashboard's competitor list, so
the scraper can later pull official BSE corporate filings directly (via the
per-scrip BSE filings API) instead of receiving mangled Google-News-RSS title
fragments for those filings.

HOW IT RESOLVES CODES
  It queries BSE's OWN live search endpoint (PeerSmartSearch — the same
  autocomplete that powers the search box on bseindia.com), once per competitor
  name. BSE itself returns the candidate matches (name + ISIN + 6-digit scrip
  code); we then apply conservative name-normalization + a similarity threshold
  to pick the best candidate and reject false matches (e.g. so "Power Mech"
  never latches onto "Power Grid Corporation").

  This replaced an earlier approach that downloaded BSE's "List_of_companies.csv"
  master and matched locally — that URL now serves a stale GSM-watchlist file
  with unusable headers, so the live per-name search is both more reliable and
  more accurate (BSE does the first-pass matching against its own index).

WHAT THIS SCRIPT DOES **NOT** DO
  - It does NOT touch the dashboard database, the scraper, or any live pipeline.
    It only reads the competitor Excel and writes two CSV files. Safe to run,
    inspect, re-run, and throw away.
  - It cannot invent scrip codes for companies that are not BSE-listed. Foreign
    firms (Hyundai E&C, Al Fanar, PowerChina), private companies (Megha, Kiran
    Construction) and keyword-variant duplicates simply have no code to find and
    will appear in the review file as unmatched. That is expected, not a bug.

USAGE (Windows cmd.exe, from the Cron repo folder where the Excel lives):
    pip install requests pandas openpyxl
    python resolve_bse_scrip_codes.py
    python resolve_bse_scrip_codes.py --threshold 0.90   # stricter matching
    python resolve_bse_scrip_codes.py --delay 0.5        # slower, gentler on BSE

OUTPUT (written next to this script):
    bse_scrip_matches.csv   -> confident matches
    bse_scrip_review.csv    -> everything to eyeball (unmatched + below-threshold best guesses)
"""

import argparse
import csv
import html
import logging
import re
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

# Same Excel the scraper reads (scraper_production.EXCEL_FILE_PATH).
EXCEL_FILE_PATH = "SBU_Competitor_Mapping.xlsx"
EXCEL_SHEET = "Competitor"
EXCEL_HEADER_ROW = 1          # matches scraper's pd.read_excel(..., header=1)
EXCEL_NAME_COLUMN = "Competitor"

# BSE's live search endpoint (verified from the maintained BseIndiaApi library).
# Returns HTML fragments; each candidate looks like:
#   <strong>HDFC</strong>   INE001A01036   500010
# i.e. matched-name-fragment, ISIN, 6-digit scrip code.
BSE_SEARCH_URL = "https://api.bseindia.com/BseIndiaAPI/api/PeerSmartSearch/w"

# BSE blocks bare requests; these headers (Origin/Referer = bseindia.com) are
# required — same approach the existing BSE downloaders use.
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

# A candidate row from BSE search: capture (name fragment, ISIN, 6-digit code).
# BSE wraps the matched substring of the name in a tag (<strong>/<b>/<span>),
# so we strip tags first, then pull "NAME  INExxxxxxxxx  DDDDDD".
_CANDIDATE_RE = re.compile(
    r"(?P<name>[A-Za-z0-9&.,\-()'/ ]+?)\s+(?P<isin>INE[0-9A-Z]{9})\s+(?P<code>\d{6})"
)
_TAG_RE = re.compile(r"<[^>]+>")

OUTPUT_MATCHES = "bse_scrip_matches.csv"
OUTPUT_REVIEW = "bse_scrip_review.csv"

DEFAULT_THRESHOLD = 0.88
DEFAULT_DELAY_SEC = 0.3       # polite pause between BSE search calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Name normalization + similarity (validated offline) ───────────────────────

_CORP_STOPWORDS_SET = {
    "limited", "ltd", "private", "pvt", "public", "plc", "corporation", "corp",
    "company", "co", "incorporated", "inc", "llp", "llc", "holdings", "holding",
    "india", "indian", "the",
}


def normalize_company_name(name: str) -> str:
    """Lowercase, strip punctuation, expand '&'->'and', drop corporate stopwords."""
    if name is None:
        return ""
    text = str(name).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split(" ") if t and t not in _CORP_STOPWORDS_SET]
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    """0..1 blended similarity between two normalized names (guards short-name
    false positives via token overlap/containment)."""
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return seq
    overlap = len(ta & tb) / len(ta | tb)
    contained = 1.0 if (ta <= tb or tb <= ta) else 0.0
    return max(
        seq,
        0.5 * seq + 0.5 * overlap,
        (0.4 * overlap + 0.6 * contained) if contained else 0.0,
    )


# ── Load competitor names from the scraper's Excel ────────────────────────────

def load_competitor_names(excel_path: str) -> list:
    """Return canonical competitor names (clean 'Competitor' column)."""
    log.info("Loading competitor names from %s ('%s' sheet)...", excel_path, EXCEL_SHEET)
    df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET, header=EXCEL_HEADER_ROW)
    if EXCEL_NAME_COLUMN not in df.columns:
        raise SystemExit(
            f"Column '{EXCEL_NAME_COLUMN}' not found in sheet '{EXCEL_SHEET}'. "
            f"Columns present: {list(df.columns)}"
        )
    names, seen = [], set()
    for value in df[EXCEL_NAME_COLUMN].tolist():
        if pd.isna(value):
            continue
        name = str(value).strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    log.info("Loaded %s unique competitor names.", len(names))
    return names


# ── BSE live search per name ──────────────────────────────────────────────────

def parse_candidates(raw_text: str) -> list:
    """Parse BSE search HTML into [{bse_name, isin, scrip_code}, ...]."""
    if not raw_text:
        return []
    # Decode HTML entities first (&amp;->&, &nbsp;->space) so names like
    # "Larsen &amp; Toubro" survive intact, THEN strip the <strong>/<b> tags
    # BSE wraps around the matched substring.
    text = html.unescape(raw_text)
    text = text.replace("\xa0", " ")
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    candidates = []
    for m in _CANDIDATE_RE.finditer(text):
        candidates.append({
            "bse_name": m.group("name").strip(" -"),
            "isin": m.group("isin"),
            "scrip_code": m.group("code"),
        })
    return candidates


def search_bse(session: requests.Session, name: str, timeout: int = 15) -> list:
    """Query BSE's live search for one competitor name; return parsed candidates.
    Returns [] on any error (caller records it as unmatched-needs-review)."""
    try:
        r = session.get(
            BSE_SEARCH_URL,
            params={"Type": "SS", "text": name},
            timeout=timeout,
        )
        if not r.ok:
            log.warning("  BSE search HTTP %s for '%s'", r.status_code, name)
            return []
        return parse_candidates(r.text)
    except Exception as e:
        log.warning("  BSE search failed for '%s': %s", name, e)
        return []


# ── Resolve ───────────────────────────────────────────────────────────────────

def resolve(competitor_names: list, threshold: float, delay: float) -> tuple:
    """Query BSE per name, score candidates, split into matches / review."""
    session = requests.Session()
    session.headers.update(HEADERS)

    matches, review = [], []
    total = len(competitor_names)

    for i, name in enumerate(competitor_names, 1):
        norm_query = normalize_company_name(name)
        candidates = search_bse(session, name)

        # Score every BSE-returned candidate against our normalized name.
        best, best_score = None, 0.0
        for cand in candidates:
            score = similarity(norm_query, normalize_company_name(cand["bse_name"]))
            if score > best_score:
                best_score, best = score, cand

        if best and best_score >= threshold:
            matches.append({
                "competitor_name": name,
                "scrip_code": best["scrip_code"],
                "matched_bse_name": best["bse_name"],
                "isin": best["isin"],
                "confidence": round(best_score, 3),
            })
            status = f"MATCH  [{best['scrip_code']}] {best['bse_name']} ({best_score:.2f})"
        else:
            review.append({
                "competitor_name": name,
                "scrip_code": best["scrip_code"] if best else "",
                "matched_bse_name": best["bse_name"] if best else "",
                "isin": best["isin"] if best else "",
                "confidence": round(best_score, 3),
                "candidates_returned": len(candidates),
                "reason": (
                    "below_threshold_best_guess" if best
                    else ("no_candidates_from_bse" if not candidates else "no_scoring_candidate")
                ),
            })
            status = (
                f"review ({len(candidates)} cand"
                + (f", best {best_score:.2f}" if best else "")
                + ")"
            )

        log.info("  [%3d/%d] %-45s -> %s", i, total, name[:45], status)
        if delay > 0:
            time.sleep(delay)

    return matches, review


# ── Output ────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resolve competitor names to BSE scrip codes via BSE live search.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confident-match similarity cutoff 0..1 (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--excel", default=EXCEL_FILE_PATH,
                        help=f"Competitor Excel path (default {EXCEL_FILE_PATH}).")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC,
                        help=f"Seconds to pause between BSE search calls (default {DEFAULT_DELAY_SEC}).")
    args = parser.parse_args()

    competitor_names = load_competitor_names(args.excel)

    log.info("Resolving %s names against BSE live search (threshold=%.2f)...",
             len(competitor_names), args.threshold)
    matches, review = resolve(competitor_names, args.threshold, args.delay)

    matches.sort(key=lambda r: r["confidence"], reverse=True)
    review.sort(key=lambda r: r["confidence"], reverse=True)

    write_csv(OUTPUT_MATCHES, matches,
              ["competitor_name", "scrip_code", "matched_bse_name", "isin", "confidence"])
    write_csv(OUTPUT_REVIEW, review,
              ["competitor_name", "scrip_code", "matched_bse_name", "isin",
               "confidence", "candidates_returned", "reason"])

    total = len(competitor_names)
    log.info("=" * 60)
    log.info("RESOLUTION COMPLETE (threshold=%.2f)", args.threshold)
    log.info("  Competitor names in         : %s", total)
    log.info("  Confident matches           : %s  -> %s", len(matches), OUTPUT_MATCHES)
    log.info("  Needs review / unmatched    : %s  -> %s", len(review), OUTPUT_REVIEW)
    if total:
        log.info("  Match rate                  : %.1f%%", 100.0 * len(matches) / total)
    log.info("=" * 60)
    log.info("NEXT: open %s and confirm the low-confidence matches near the", OUTPUT_MATCHES)
    log.info("      bottom look right, then skim %s for real companies that", OUTPUT_REVIEW)
    log.info("      SHOULD have matched. Nothing has touched the dashboard yet.")


if __name__ == "__main__":
    main()
