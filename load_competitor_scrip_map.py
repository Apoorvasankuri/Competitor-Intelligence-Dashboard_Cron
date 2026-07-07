"""
load_competitor_scrip_map.py  — Change 21, Step 2a (standalone, run manually ONCE)
==================================================================================

Loads the verified competitor-name -> BSE-scrip-code mapping into the DASHBOARD
database (the same Render Postgres the scraper/processor/backend use), so the
scraper's forthcoming BSE-filing ingestion path can look up which competitors
are BSE-listed and pull their official filings directly.

The mapping was verified by hand (BSE codes confirmed against bseindia.com /
screener.in), so there is no fuzzy matching here — this just writes known-good
rows. Three companies appear under two spellings each (both map to the same
code) so a BSE filing clusters with news regardless of which spelling the
scraper's detect_competitor tagged the news with.

SAFETY
  - Idempotent: creates the table IF NOT EXISTS and UPSERTs, so re-running is
    safe and simply refreshes the rows.
  - Touches ONLY the new competitor_scrip_map table. Does not read or modify
    raw_scraped_articles, processed_articles, pipeline_runs, users, or anything
    else. Nothing in the live pipeline reads this table yet — that comes in the
    next step.
  - Reads DATABASE_URL from the environment (same var the scraper uses). It does
    NOT hardcode credentials.

USAGE (from the Cron repo folder, with the dashboard DATABASE_URL set):
    Windows cmd.exe:
        set DATABASE_URL=postgresql://...your dashboard render url...
        python load_competitor_scrip_map.py
    Dry run (print what WOULD be written, touch nothing):
        python load_competitor_scrip_map.py --dry-run
    Verify what's currently in the table:
        python load_competitor_scrip_map.py --show
"""

import argparse
import logging
import os
import sys

# Use the same driver the scraper uses (psycopg v3). If this import fails,
# the scraper wouldn't run either, so it's the right dependency to assume.
import psycopg
from psycopg.rows import dict_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── The verified mapping ──────────────────────────────────────────────────────
# exact-Excel-competitor-name -> BSE 6-digit scrip code.
# Three companies (HG Infra, Hitachi Energy, KPI Green) intentionally appear
# under both of their Excel spellings, both pointing at the same code.
COMPETITOR_SCRIP_MAP = {
    "AFCONS Infrastructure Limited": "544280",
    "Ahluwalia Contracts (India) Limited": "532811",
    "Ashoka Buildcon Limited": "533271",
    "Bajaj Electricals Limited": "500031",
    "Dilip Buildcon Limited": "540047",
    "HG Infra Engineering Limited": "541019",
    "H.G. Infra Engineering Limited": "541019",
    "Hindustan Construction Company Limited": "500185",
    "Hitachi Energy India Limited": "543187",
    "Hitachi Energy": "543187",
    "IRCON International Limited": "541956",
    "Jyoti Structures Limited": "513250",
    "Kalpataru Projects International Limited": "522287",
    "Larsen & Toubro Limited": "500510",
    "Likhitha Infrastructure Limited": "543240",
    "NCC Limited": "500294",
    "PNC Infratech Limited": "539150",
    "Rail Vikas Nigam Limited": "542649",
    "Simplex Infrastructures Limited": "523838",
    "Skipper Limited": "538562",
    "Sterling and Wilson Renewable Energy Limited": "542760",
    "Techno Electric & Engineering Company Limited": "542141",
    "Texmaco Rail & Engineering Limited": "533326",
    "Transrail Lighting Limited": "544317",
    "J. Kumar Infraprojects Limited": "532940",
    "Kernex Microsystems Private Limited": "532686",
    "RailTel Corporation of India Limited": "543265",
    "Power Mech Projects Limited": "539302",
    "Capacite InfraProjects Limited": "540710",
    "GR Infraprojects Limited": "543317",
    "KPI Green Energy Limited": "542323",
    "KPI Green": "542323",
    "Bondada Engineering Limited": "543971",
    "Oriana Power Limited": "544136",
    # KP Energy Ltd intentionally removed: code 543710 turned out to be a
    # mutual fund (iSIF/MF NAV disclosures), not the infrastructure company.
    # Dropped rather than keep guessing at an unverified code (see --deactivate
    # below for how the already-loaded bad row gets turned off).
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS competitor_scrip_map (
    id              SERIAL PRIMARY KEY,
    competitor_name TEXT NOT NULL UNIQUE,
    bse_scrip_code  TEXT NOT NULL,
    active          BOOLEAN DEFAULT TRUE,
    updated_at      TIMESTAMP DEFAULT NOW()
);
"""
CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_competitor_scrip_map_code "
    "ON competitor_scrip_map(bse_scrip_code);"
)

UPSERT_SQL = """
INSERT INTO competitor_scrip_map (competitor_name, bse_scrip_code, active, updated_at)
VALUES (%s, %s, TRUE, NOW())
ON CONFLICT (competitor_name)
DO UPDATE SET bse_scrip_code = EXCLUDED.bse_scrip_code,
              active = TRUE,
              updated_at = NOW();
"""


def get_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL not set. Point it at the DASHBOARD Render Postgres "
            "(the same one the scraper/backend use), e.g.\n"
            "  set DATABASE_URL=postgresql://user:pass@host/dbname"
        )
    return psycopg.connect(url, row_factory=dict_row)


def show_current():
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT competitor_name, bse_scrip_code, active, updated_at "
                "FROM competitor_scrip_map ORDER BY competitor_name;"
            )
            rows = cur.fetchall()
        except Exception:
            log.warning("competitor_scrip_map table does not exist yet.")
            return
        log.info("competitor_scrip_map currently has %s rows:", len(rows))
        for r in rows:
            log.info("  %-48s %s  active=%s", r["competitor_name"], r["bse_scrip_code"], r["active"])
    finally:
        conn.close()


def load(dry_run: bool):
    distinct_codes = len(set(COMPETITOR_SCRIP_MAP.values()))
    log.info("Prepared %s name->code rows (%s distinct companies).",
             len(COMPETITOR_SCRIP_MAP), distinct_codes)

    if dry_run:
        log.info("DRY RUN — would upsert the following (nothing written):")
        for name, code in sorted(COMPETITOR_SCRIP_MAP.items()):
            log.info("  %-48s -> %s", name, code)
        log.info("DRY RUN complete. Re-run without --dry-run to write.")
        return

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(CREATE_INDEX_SQL)
        conn.commit()

        written = 0
        for name, code in COMPETITOR_SCRIP_MAP.items():
            cur.execute(UPSERT_SQL, (name, code))
            written += 1
        conn.commit()

        cur.execute("SELECT COUNT(*) AS n FROM competitor_scrip_map;")
        total = cur.fetchone()["n"]
        log.info("Upserted %s rows. competitor_scrip_map now holds %s rows total.", written, total)
    except Exception as e:
        conn.rollback()
        raise SystemExit(f"Load failed (rolled back): {e}")
    finally:
        conn.close()


def deactivate(name: str):
    """Set active=FALSE for one competitor_name (does not delete the row —
    keeps an audit trail of codes that were tried and found wrong)."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE competitor_scrip_map SET active = FALSE, updated_at = NOW() "
            "WHERE competitor_name = %s RETURNING competitor_name, bse_scrip_code",
            (name,),
        )
        row = cur.fetchone()
        conn.commit()
        if row:
            log.info("Deactivated: %s (was code %s). It will no longer be used for BSE ingestion.",
                     row["competitor_name"], row["bse_scrip_code"])
        else:
            log.warning("No row found for competitor_name=%r — nothing to deactivate.", name)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load competitor->BSE scrip-code map into the dashboard DB.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written; touch nothing.")
    parser.add_argument("--show", action="store_true",
                        help="Show the current contents of competitor_scrip_map and exit.")
    parser.add_argument("--deactivate", metavar="COMPETITOR_NAME",
                        help="Set active=FALSE for one existing row (e.g. a code later found to be wrong) and exit.")
    args = parser.parse_args()

    if args.show:
        show_current()
        return

    if args.deactivate:
        deactivate(args.deactivate)
        return

    load(args.dry_run)
    if not args.dry_run:
        log.info("Done. Nothing in the live pipeline reads this table yet — "
                 "that's the next step (scraper BSE-filing ingestion).")


if __name__ == "__main__":
    main()
