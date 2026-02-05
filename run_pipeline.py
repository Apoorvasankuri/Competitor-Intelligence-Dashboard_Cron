"""
Main runner for automated competitor intelligence pipeline
Executes scraping followed by LLM processing
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

def main():
    """
    Run the complete pipeline:
    1. Scrape news from Google News RSS
    2. Process with Claude API for scoring, tagging, and summarization
    """
    try:
        logging.info("🚀 Starting Competitor Intelligence Pipeline")
        logging.info("")
        
        # Step 1: Scrape news
        logging.info("📰 STEP 1: Scraping news articles...")
        scraper_main()
        logging.info("")
        
        # Step 2: Process with LLM
        logging.info("🤖 STEP 2: Processing with Claude API...")
        llm_main()
        logging.info("")
        
        logging.info("✅ Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
