import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote
import os
import logging
import re
import pandas as pd
from typing import List, Dict, Set

import psycopg
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOOKBACK_DAYS = 7
MAX_CONCURRENT_REQUESTS = 10
REQUEST_DELAY = 0.5
EXCEL_FILE_PATH = 'SBU_Competitor_Mapping.xlsx'

def load_keywords_from_excel():
    """Load SBUs and Competitors independently as keyword-to-name maps"""
    logging.info("Loading keywords from Excel file...")
    
    # 1. Load SBUs
    sbu_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='SBU', header=1)
    sbu_keyword_map = {}
    for _, row in sbu_df.iterrows():
        sbu_name = row['SBU']
        keywords_raw = row['Key Words']
        if pd.notna(sbu_name) and pd.notna(keywords_raw):
            keywords = re.findall(r'"([^"]+)"', str(keywords_raw))
            for kw in keywords:
                sbu_keyword_map[kw.lower()] = sbu_name
    
    # 2. Load Competitors
    competitor_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Competitor', header=1)
    comp_keyword_map = {}
    for _, row in competitor_df.iterrows():
        comp_name = row['Competitor']
        keywords_raw = row['Competitor Key Words']
        if pd.notna(comp_name) and pd.notna(keywords_raw):
            keywords = re.findall(r'"([^"]+)"', str(keywords_raw))
            for kw in keywords:
                comp_keyword_map[kw.lower()] = comp_name
    
    return {
        'sbu_keyword_map': sbu_keyword_map,
        'comp_keyword_map': comp_keyword_map,
        'search_terms': list(comp_keyword_map.keys())
    }

def detect_names(text: str, keyword_map: Dict) -> str:
    """Generic detection to find which names (SBU or Comp) match keywords in text"""
    text_lower = text.lower()
    matched_names = set()
    for kw, name in keyword_map.items():
        if kw in text_lower:
            matched_names.add(name)
    return ", ".join(sorted(matched_names))

async def fetch_feed_async(session: aiohttp.ClientSession, keyword: str, lookback_days: int) -> Dict:
    encoded_keyword = quote(keyword)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:{lookback_days}d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            content = await response.text()
            feed = feedparser.parse(content)
            return {'keyword': keyword, 'feed': feed, 'success': True}
    except Exception as e:
        logging.error(f"Error fetching feed for '{keyword}': {e}")
        return {'keyword': keyword, 'feed': None, 'success': False}

async def scrape_news_async(search_terms: List[str], sbu_map: Dict, comp_map: Dict, lookback_days: int) -> List[Dict]:
    all_articles = []
    seen_links = set()
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_feed_async(session, term, lookback_days) for term in search_terms]
        results = await asyncio.gather(*tasks)

    for result in results:
        if not result['success'] or not result['feed']: continue
        
        for entry in result['feed'].entries:
            link = entry.get("link", "")
            if link in seen_links: continue
            
            title = entry.get("title", "")
            source = ""
            if "description" in entry:
                soup = BeautifulSoup(entry.description, "html.parser")
                font_tag = soup.find("font")
                if font_tag: source = font_tag.text.strip()
            
            combined_text = f"{title} {source}"
            
            # Match independently
            matched_comps = detect_names(combined_text, comp_map)
            matched_sbus = detect_names(combined_text, sbu_map)
            
            # Filter: Both MUST be found
            if matched_comps and matched_sbus:
                try:
                    pubdate = datetime(*entry.published_parsed[:6])
                except:
                    pubdate = datetime.now()
                
                seen_links.add(link)
                all_articles.append({
                    "keyword": result['keyword'],
                    "newstitle": title,
                    "source": source,
                    "link": link,
                    "publishedate": pubdate,
                    "sbu": matched_sbus,
                    "competitor": matched_comps,
                    "scraped_content": ""
                })
    return all_articles

def get_db_connection():
    """Get database connection from environment variable"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    
    return psycopg.connect(database_url, row_factory=dict_row)

def save_to_database(articles: List[Dict]):
    """Save scraped articles to PostgreSQL database"""
    if not articles:
        logging.info("No articles to save")
        return
    
    conn = get_db_connection()
    
    insert_query = """
        INSERT INTO competitor_data (
            keyword, newstitle, source, link, publishedate, 
            sbu, competitor, scraped_content
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (link, publishedate) DO NOTHING
    """
    
    saved_count = 0
    failed_count = 0
    
    for article in articles:
        try:
            cur = conn.cursor()
            cur.execute(insert_query, (
                article['keyword'],
                article['newstitle'],
                article['source'],
                article['link'],
                article['publishedate'],
                article['sbu'],
                article['competitor'],
                article['scraped_content']
            ))
            conn.commit()  # Commit after each successful insert
            cur.close()
            saved_count += 1
        except Exception as e:
            conn.rollback()  # Rollback the failed transaction
            failed_count += 1
            logging.error(f"Error saving article '{article.get('newstitle', 'Unknown')[:50]}...': {e}")
    
    conn.close()
    
    logging.info(f"✅ Saved {saved_count} new articles to database")
    if failed_count > 0:
        logging.warning(f"⚠️  Failed to save {failed_count} articles")

async def main_async():
    """Main async scraping function"""
    logging.info("=" * 60)
    logging.info("Starting Competitor News Scraping Job (Async)")
    logging.info("=" * 60)
    
    # Load keywords from Excel
    data = load_keywords_from_excel()
    
    logging.info(f"Searching for {len(data['search_terms'])} competitor keywords")
    logging.info(f"Filtering by {len(data['sbu_keyword_map'])} SBU keywords")
    logging.info(f"Lookback period: {LOOKBACK_DAYS} days")
    
    # Scrape news
    articles = await scrape_news_async(
        search_terms=data['search_terms'],
        sbu_map=data['sbu_keyword_map'],
        comp_map=data['comp_keyword_map'],
        lookback_days=LOOKBACK_DAYS
    )
    
    # Save to database
    save_to_database(articles)
    
    logging.info("=" * 60)
    logging.info("Scraping Job Complete")
    logging.info("=" * 60)

def main():
    """Entry point for the scraper"""
    # Run async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
