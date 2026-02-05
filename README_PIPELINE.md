# Competitor Intelligence Pipeline

Automated news scraping and LLM processing system that runs daily.

## Files

- `scraper_production.py` - Scrapes Google News RSS feeds
- `llm_processor_production.py` - Processes news with Claude API
- `run_pipeline.py` - Main runner (executes both scripts)
- `requirements_pipeline.txt` - Python dependencies

## What It Does

1. **Scrapes** competitor news from Google News (keywords: transmission, railways, solar, etc.)
2. **Filters** for articles mentioning competitors (L&T, Kalpataru, etc.)
3. **Saves** to PostgreSQL database
4. **Processes** with Claude API to:
   - Generate relevance scores (0-100)
   - Generate confidence scores (0-100)
   - Tag SBUs (India T&D, Transportation, Civil, etc.)
   - Tag categories (order wins, financial, M&A, etc.)
   - Create business summaries
5. **Updates** database with all analysis

## Environment Variables Needed

- `DATABASE_URL` - PostgreSQL connection string (from Render database)
- `CLAUDE_API_KEY` - Your Anthropic API key

## How to Deploy as Cron Job on Render

See deployment guide for step-by-step instructions.

## Improvements Over Original Scripts

### Scraper Improvements:
✅ No local file dependencies (no Excel)
✅ Writes directly to PostgreSQL
✅ Better error handling and retries
✅ Deduplication at database level
✅ Cleaner, more maintainable code
✅ Reduced keywords for efficiency (from 5000+ to ~20 core ones)

### LLM Processor Improvements:
✅ Reads from and writes to PostgreSQL
✅ No Selenium dependency (uses simple requests)
✅ Better rate limiting and retry logic
✅ Processes only unprocessed articles
✅ More robust error handling
✅ Cleaner prompt engineering
✅ JSON-based responses for reliability

## Local Testing (Optional)

```bash
# Install dependencies
pip install -r requirements_pipeline.txt

# Set environment variables
export DATABASE_URL="your_postgres_url"
export CLAUDE_API_KEY="your_api_key"

# Run pipeline
python run_pipeline.py
```

## Notes

- Pipeline processes up to 100 articles per run
- Articles with relevance score > 30 get full analysis
- Lower relevance articles get basic tagging only
- Duplicate detection handled by database constraint on (link, publishedate)
