"""
One-Time Run Script — Specific Competitors + Date Range
Runs the FULL pipeline: scrape → score → analyze → dedup → summarize → save to DB

Usage:
    Set TARGET_COMPETITORS and START_DATE / END_DATE below, then:
    python one_time_run.py
"""

import asyncio
import logging
from datetime import date, datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONFIGURE YOUR RUN HERE
# ============================================================

TARGET_COMPETITORS = [
    "Kalpataru Projects International Limited",
    "Capacite Infraprojects Limited",
    "NCC Limited",
    "J. Kumar Infraprojects Limited",
    "AFCONS Infrastructure Limited",
    "Dineshchandra R. Agrawal Infracon Private Limited",
    "Ashoka Buildcon Limited",
    "Hindustan Construction Company Limited",
    "H.G. Infra Engineering Limited",
    "Megha Engineering and Infrastructures Limited",
]

START_DATE = date(2025, 4, 1)   # inclusive
END_DATE   = date(2026, 6, 15)  # inclusive

# ============================================================
# IMPORTS (after config so errors are obvious)
# ============================================================

import pandas as pd
from scraper_production import (
    load_keywords_from_excel,
    scrape_news_async,
    get_db_connection,
)
from llm_processor_production import (
    load_excel_data,
    load_competitor_tiers,
    load_competitor_variations,
    build_full_analysis_prompt,
    stage1_quick_scoring,
    stage2_full_analysis,
    deduplicate_articles,
    generate_llm_summaries,
    RELEVANCE_THRESHOLD,
)


# ============================================================
# STEP 1: SCRAPE — only keywords for target competitors
# ============================================================

async def scrape_for_competitors():
    logging.info("=" * 60)
    logging.info(f"ONE-TIME SCRAPE: {len(TARGET_COMPETITORS)} competitors")
    logging.info(f"Date range: {START_DATE} → {END_DATE}")
    logging.info("=" * 60)

    kw_data = load_keywords_from_excel()

    # Filter keywords to only those that belong to target competitors
    target_set = set(TARGET_COMPETITORS)
    filtered_keywords = [
        kw for kw, mappings in kw_data['competitor_to_sbu'].items()
        if any(m['competitor'] in target_set for m in mappings)
    ]

    logging.info(f"Using {len(filtered_keywords)} keywords for {len(TARGET_COMPETITORS)} competitors")

    # Google News 'when:Nd' is relative to today — cover the full range
    lookback_days = (date.today() - START_DATE).days + 1
    logging.info(f"Lookback: {lookback_days} days from today to reach {START_DATE}")

    articles = await scrape_news_async(
        competitor_keywords=filtered_keywords,
        sbu_keywords=kw_data['sbu_keywords'],
        competitor_to_sbu=kw_data['competitor_to_sbu'],
        lookback_days=lookback_days,
    )

    # Filter to the date window
    filtered = [
        a for a in articles
        if START_DATE <= a['published_date'].date() <= END_DATE
    ]
    logging.info(f"After date filter ({START_DATE} → {END_DATE}): {len(filtered)} articles")
    return filtered


# ============================================================
# STEP 2: SAVE raw articles to DB (so LLM processor can read them)
# ============================================================

def save_raw_to_db(articles):
    if not articles:
        logging.info("No articles to save to raw table.")
        return

    conn = get_db_connection()
    insert_query = """
        INSERT INTO raw_scraped_articles (
            search_keyword, news_title, source, link, published_date,
            sbu, competitor, content
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (link, published_date) DO NOTHING
    """
    saved, skipped = 0, 0
    for a in articles:
        try:
            cur = conn.cursor()
            cur.execute(insert_query, (
                a['search_keyword'], a['news_title'], a['source'],
                a['link'], a['published_date'], a['sbu'],
                a['competitor'], a['content']
            ))
            conn.commit()
            cur.close()
            saved += 1
        except Exception as e:
            conn.rollback()
            skipped += 1
            logging.error(f"Raw save failed: {a.get('news_title','')[:60]} — {e}")

    conn.close()
    logging.info(f"✅ Raw articles saved: {saved} | skipped: {skipped}")


# ============================================================
# STEP 3: LOAD raw articles from DB for the date range
# ============================================================

def load_raw_from_db():
    conn = get_db_connection()
    query = f"""
        SELECT id, published_date, news_title, competitor, sbu,
               source, search_keyword, link, content
        FROM raw_scraped_articles
        WHERE published_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        AND competitor = ANY(%s)
        ORDER BY published_date DESC
        LIMIT 5000
    """
    cur = conn.cursor()
    cur.execute(query, (TARGET_COMPETITORS,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.rename(columns={
        'news_title': 'News Title',
        'link': 'Link',
        'competitor': 'Competitor',
        'sbu': 'SBU',
        'source': 'Source',
        'published_date': 'Published Date',
    })
    logging.info(f"Loaded {len(df)} raw articles from DB for processing")
    return df


# ============================================================
# STEP 4: SAVE processed articles to the MAIN DB table
# ============================================================

def save_processed_to_db(df: pd.DataFrame):
    if df.empty:
        logging.info("No processed articles to save.")
        return

    conn = get_db_connection()
    insert_query = """
        INSERT INTO processed_articles (
            published_date, news_title, link, "Source",
            relevance_score, competitor_tagging, sbu_tagging,
            category_tag, summary, scraped_content,
            contract_value_inr_crore, geography, competitor_tier, rank_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (link, published_date) DO NOTHING
    """
    saved, skipped = 0, 0
    for idx, row in df.iterrows():
        try:
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
            ))
            conn.commit()
            # Clean up raw table entry after successful processing
            if row.get('id'):
                cur.execute("DELETE FROM raw_scraped_articles WHERE id = %s", (row.get('id'),))
                conn.commit()
            cur.close()
            saved += 1
        except Exception as e:
            conn.rollback()
            skipped += 1
            logging.error(f"Processed save failed: {row.get('News Title','')[:60]} — {e}")

    conn.close()
    logging.info(f"✅ Saved to processed_articles: {saved} | skipped (conflict/error): {skipped}")


# ============================================================
# MAIN
# ============================================================

async def main():
    # Step 1: Scrape
    articles = await scrape_for_competitors()

    if not articles:
        logging.info("No articles found for the given competitors and date range. Exiting.")
        return

    # Step 2: Save to raw table
    save_raw_to_db(articles)

    # Step 3: Load back from DB (clean DataFrame with consistent schema)
    df = load_raw_from_db()
    if df.empty:
        logging.info("Nothing loaded from DB. Exiting.")
        return

    # Step 4: LLM pipeline — same as production
    try:
        excel_data = load_excel_data()
        competitor_tier_map = load_competitor_tiers()
    except Exception as e:
        logging.error(f"Failed to load Excel data: {e}")
        return

    full_prompt = build_full_analysis_prompt(categories=excel_data['categories'])

    df = stage1_quick_scoring(df)
    df = stage2_full_analysis(df, full_prompt, competitor_tier_map)

    high_rel_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()

    if len(high_rel_df) > 0:
        high_rel_df = deduplicate_articles(high_rel_df)
        high_rel_df = generate_llm_summaries(high_rel_df)
        high_rel_df = high_rel_df.drop(columns=['_fingerprint'], errors='ignore')

        # Step 5: Save to main processed_articles table
        save_processed_to_db(high_rel_df)

        logging.info("=" * 60)
        logging.info(f"✅ ONE-TIME RUN COMPLETE")
        logging.info(f"   Scraped:   {len(articles)} articles")
        logging.info(f"   Processed: {len(df)} articles")
        logging.info(f"   Saved:     {len(high_rel_df)} articles (relevance ≥ {RELEVANCE_THRESHOLD})")
        logging.info("=" * 60)
    else:
        logging.info("No articles met the relevance threshold. Nothing saved to processed_articles.")


if __name__ == "__main__":
    asyncio.run(main())
