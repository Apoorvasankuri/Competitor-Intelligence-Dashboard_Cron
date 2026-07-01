"""
Main runner for automated competitor intelligence pipeline.

Single entry point to run the complete pipeline:
  1. Scrape news from Google News RSS  (scraper_production.main)
  2. Process with Claude API for scoring, tagging, deduplication,
     ranking, and summarization  (llm_processor_production.main)

The scraper runs first. If scraping fails, the LLM processor is NOT run.
Exits with status 0 on success and 1 on any failure.
"""

import logging
import sys

from scraper_production import main as scraper_main
from llm_processor_production import main as llm_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main() -> int:
    """
    Run the complete pipeline end-to-end.

    Returns:
        0 if the full pipeline (scraper + processor) succeeds.
        1 if either stage fails.
    """
    logging.info("=" * 60)
    logging.info("🚀 Starting Competitor Intelligence Pipeline")
    logging.info("=" * 60)

    # ---- Step 1: Scraper ----
    logging.info("📰 STEP 1: Starting news scraper...")
    try:
        scraper_main()
    except Exception:
        logging.exception("❌ Scraper failed. Aborting pipeline; LLM processor will NOT run.")
        return 1
    logging.info("✅ STEP 1 complete: scraper finished successfully.")

    # ---- Step 2: LLM processor (only reached if scraper succeeded) ----
    logging.info("🤖 STEP 2: Starting LLM processor...")
    try:
        llm_main()
    except Exception:
        logging.exception("❌ LLM processor failed.")
        return 1
    logging.info("✅ STEP 2 complete: LLM processor finished successfully.")

    logging.info("=" * 60)
    logging.info("✅ Pipeline completed successfully!")
    logging.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
