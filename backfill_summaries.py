"""
One-off backfill: regenerate summaries for existing processed_articles rows
using the corrected SUMMARY_SYSTEM_PROMPT (no strategic-implication sentence
— see llm_processor_production.py's SUMMARY_SYSTEM_PROMPT for the current,
bare-facts-only version).

Only touches rows from the last BACKFILL_WINDOW_DAYS days — the maximum
lookback window used anywhere on the dashboard (Client/Authority Tracker
and Competitor Strategy both use 90 days), so this covers everything a
user could actually see without paying to regenerate the full history.

Run once, from the Cron repo directory, with DATABASE_URL and CLAUDE_API_KEY
already set in the environment (same as llm_processor_production.py needs):

    python backfill_summaries.py

Safe to interrupt and rerun — each row is updated independently and this
only ever overwrites the `summary` column, nothing else.
"""

import json
import logging

from llm_processor_production import get_db_connection, batch_generate_summaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BACKFILL_WINDOW_DAYS = 15
BACKFILL_BATCH_SIZE = 5


def load_rows_to_backfill():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, news_title, competitor_tagging, sbu_tagging, category_tag,
               geography, contract_value_inr_crore, scraped_content, fingerprint
        FROM processed_articles
        WHERE published_date >= CURRENT_DATE - INTERVAL '%s days'
          AND summary IS NOT NULL
          AND summary != ''
        ORDER BY published_date DESC
        """
        % BACKFILL_WINDOW_DAYS
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def save_summary(row_id, new_summary):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE processed_articles SET summary = %s WHERE id = %s",
        (new_summary, row_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def main():
    rows = load_rows_to_backfill()
    total = len(rows)
    logging.info(
        f"Loaded {total} articles from the last {BACKFILL_WINDOW_DAYS} days to regenerate."
    )

    if total == 0:
        logging.info("Nothing to backfill.")
        return

    updated = 0
    failed = 0

    for i in range(0, total, BACKFILL_BATCH_SIZE):
        batch_rows = rows[i:i + BACKFILL_BATCH_SIZE]
        articles_batch = []
        for row in batch_rows:
            fp = row.get("fingerprint")
            if isinstance(fp, str):
                try:
                    fp = json.loads(fp)
                except Exception:
                    fp = {}
            articles_batch.append(
                {
                    "title": row.get("news_title", ""),
                    "competitor_tagging": row.get("competitor_tagging", "-"),
                    "sbu_tagging": row.get("sbu_tagging", "General"),
                    "category_tag": row.get("category_tag", ""),
                    "geography": row.get("geography"),
                    "contract_value_inr_crore": row.get("contract_value_inr_crore"),
                    "content": row.get("scraped_content", "") or "",
                    "fingerprint": fp or {},
                }
            )

        try:
            new_summaries = batch_generate_summaries(articles_batch)
            for row, new_summary in zip(batch_rows, new_summaries):
                save_summary(row["id"], new_summary)
                updated += 1
            logging.info(f"Progress: {min(i + BACKFILL_BATCH_SIZE, total)}/{total}")
        except Exception as e:
            failed += len(batch_rows)
            logging.error(f"Batch starting at {i} failed: {e}")

    logging.info(f"Backfill complete. Updated: {updated}, Failed: {failed}")


if __name__ == "__main__":
    main()
