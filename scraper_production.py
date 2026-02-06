"""
Production News Scraper for Competitor Intelligence (v2 - Async)
Scrapes Google News RSS feeds for competitor keywords and filters by SBU relevance
"""

import asyncio
import aiohttp
import feedparser
import psycopg
from psycopg.rows import dict_row
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote
import os
import logging
import re
import pandas as pd
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
LOOKBACK_DAYS = 7
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent requests
REQUEST_DELAY = 0.5  # Delay between requests in seconds
EXCEL_FILE_PATH = 'SBU_Competitor_Mapping.xlsx'


def load_keywords_from_excel():
    """Load SBU and Competitor keywords from Excel file"""
    logging.info("Loading keywords from Excel file...")
    
    # Read SBU sheet
    sbu_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='SBU', header=1)
    
    sbu_keywords_dict = {}
    all_sbu_keywords = set()
    
    for idx, row in sbu_df.iterrows():
        sbu_name = row['SBU']
        keywords_raw = row['Key Words']
        
        if pd.notna(sbu_name) and pd.notna(keywords_raw):
            # Extract keywords between quotes
            keywords = re.findall(r'"([^"]+)"', str(keywords_raw))
            sbu_keywords_dict[sbu_name] = keywords
            all_sbu_keywords.update(keywords)
    
    logging.info(f"Loaded {len(sbu_keywords_dict)} SBUs with {len(all_sbu_keywords)} unique keywords")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Competitor', header=1)
    
    competitor_keywords_list = []
    competitor_to_sbu = {}
    
    for idx, row in competitor_df.iterrows():
        sbu = row['SBU']
        competitor = row['Competitor']
        keywords_raw = row['Competitor Key Words']
        
        if pd.notna(competitor) and pd.notna(keywords_raw):
            # Extract keywords between quotes
            keywords = re.findall(r'"([^"]+)"', str(keywords_raw))
            competitor_keywords_list.extend(keywords)
            
            # Map each keyword to its SBU and competitor name
            for keyword in keywords:
                if keyword not in competitor_to_sbu:
                    competitor_to_sbu[keyword] = []
                competitor_to_sbu[keyword].append({
                    'sbu': sbu,
                    'competitor': competitor
                })
    
    # Get unique competitor keywords
    unique_competitor_keywords = list(set(competitor_keywords_list))
    
    logging.info(f"Loaded {len(unique_competitor_keywords)} unique competitor keywords")
    
    return {
        'sbu_keywords': list(all_sbu_keywords),
        'competitor_keywords': unique_competitor_keywords,
        'competitor_to_sbu': competitor_to_sbu,
        'sbu_keywords_dict': sbu_keywords_dict
    }


def detect_sbu(title: str, source: str, sbu_keywords: List[str]) -> str:
    """Detect relevant SBUs from title and source"""
    text = f"{title} {source}".lower()
    detected_sbus = set()
    
    for keyword in sbu_keywords:
        if keyword.lower() in text:
            detected_sbus.add(keyword)
    
    return ", ".join(sorted(detected_sbus)) if detected_sbus else ""


def detect_competitor(title: str, source: str, competitor_to_sbu: Dict, competitor_keywords: List[str]) -> str:
    """Detect competitors mentioned in title/source"""
    text = f"{title} {source}".lower()
    detected_competitors = set()
    
    for keyword in competitor_keywords:
        if keyword.lower() in text:
            # Get all competitor names associated with this keyword
            for mapping in competitor_to_sbu.get(keyword, []):
                detected_competitors.add(mapping['competitor'])
    
    return ", ".join(sorted(detected_competitors)) if detected_competitors else ""


async def fetch_feed_async(session: aiohttp.ClientSession, keyword: str, lookback_days: int) -> Dict:
    """Asynchronously fetch RSS feed for a given keyword"""
    encoded_keyword = quote(keyword)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:{lookback_days}d&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        # Use aiohttp to fetch the feed
        async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            content = await response.text()
            
            # Parse with feedparser (it's synchronous but fast)
            feed = feedparser.parse(content)
            
            if feed.bozo and hasattr(feed, 'bozo_exception'):
                logging.warning(f"Feed parse warning for '{keyword}': {feed.bozo_exception}")
            
            return {
                'keyword': keyword,
                'feed': feed,
                'success': True
            }
    
    except Exception as e:
        logging.error(f"Error fetching feed for '{keyword}': {e}")
        return {
            'keyword': keyword,
            'feed': None,
            'success': False
        }


async def scrape_news_async(competitor_keywords: List[str], sbu_keywords: List[str], 
                            competitor_to_sbu: Dict, lookback_days: int = LOOKBACK_DAYS) -> List[Dict]:
    """Scrape news asynchronously for all competitor keywords"""
    all_articles = []
    seen_links = set()
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks for all keywords
        tasks = []
        for keyword in competitor_keywords:
            task = fetch_feed_async(session, keyword, lookback_days)
            tasks.append(task)
            
            # Add small delay between task creation to avoid overwhelming the server
            await asyncio.sleep(REQUEST_DELAY / len(competitor_keywords))
        
        # Execute all tasks concurrently with progress tracking
        logging.info(f"Fetching {len(tasks)} RSS feeds concurrently...")
        results = await asyncio.gather(*tasks)
    
    # Process results
    successful_fetches = 0
    for result in results:
        if not result['success'] or not result['feed'] or not result['feed'].entries:
            continue
        
        successful_fetches += 1
        keyword = result['keyword']
        feed = result['feed']
        
        for entry in feed.entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            
            # Skip duplicates
            if link in seen_links:
                continue
            
            # Parse date
            try:
                pubdate = datetime(*entry.published_parsed[:6])
            except:
                pubdate = datetime.now()
            
            # Extract source
            source = ""
            if "description" in entry:
                soup = BeautifulSoup(entry.description, "html.parser")
                font_tag = soup.find("font")
                if font_tag:
                    source = font_tag.text.strip()
            
            # Detect competitor (must match to proceed)
            competitor = detect_competitor(title, source, competitor_to_sbu, competitor_keywords)
            if not competitor:
                continue
            
            # Detect SBU (must match at least one SBU keyword)
            sbu = detect_sbu(title, source, sbu_keywords)
            if not sbu:
                continue  # Skip articles without SBU relevance
            
            seen_links.add(link)
            
            all_articles.append({
                "keyword": keyword,
                "newstitle": title,
                "source": source,
                "link": link,
                "publishedate": pubdate,
                "sbu": sbu,
                "competitor": competitor,
                "scraped_content": ""
            })
    
    logging.info(f"Successfully fetched {successful_fetches}/{len(competitor_keywords)} feeds")
    logging.info(f"Found {len(all_articles)} relevant articles (with competitors AND SBU match)")
    
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
    keywords_data = load_keywords_from_excel()
    
    competitor_keywords = keywords_data['competitor_keywords']
    sbu_keywords = keywords_data['sbu_keywords']
    competitor_to_sbu = keywords_data['competitor_to_sbu']
    
    logging.info(f"Searching for {len(competitor_keywords)} competitor keywords")
    logging.info(f"Filtering by {len(sbu_keywords)} SBU keywords")
    logging.info(f"Lookback period: {LOOKBACK_DAYS} days")
    
    # Scrape news
    articles = await scrape_news_async(
        competitor_keywords=competitor_keywords,
        sbu_keywords=sbu_keywords,
        competitor_to_sbu=competitor_to_sbu,
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
