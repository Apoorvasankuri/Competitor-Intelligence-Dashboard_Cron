"""
Production News Scraper for Competitor Intelligence
Scrapes Google News RSS feeds and saves to PostgreSQL
"""

import feedparser
import psycopg
from psycopg.rows import dict_row
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import quote
import time
import random
import os
import logging
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
DELAY_BETWEEN_REQUESTS = 2  # seconds
RETRIES = 3
LOOKBACK_DAYS = 1  # How many days back to scrape

# SBU Detection Keywords (simplified - add more as needed)
SBU_KEYWORDS = {
    "India T&D": [
        "transmission", "distribution", "T&D", "power grid", "substation",
        "PGCIL", "PowerGrid", "NTPC", "switchgear", "transformer"
    ],
    "International T&D": [
        "international transmission", "overseas grid", "export contract"
    ],
    "Transportation": [
        "railway", "metro", "rail", "RVNL", "IRCON", "freight corridor"
    ],
    "Civil": [
        "building", "water infrastructure", "industrial construction",
        "defence infrastructure", "civil works"
    ],
    "Renewables": [
        "solar", "wind", "renewable energy", "solar EPC", "wind farm"
    ],
    "Oil & Gas": [
        "pipeline", "oil terminal", "gas pipeline", "petroleum"
    ]
}

# Competitor Detection Keywords
COMPETITOR_KEYWORDS = [
    "L&T", "Larsen & Toubro", "Kalpataru", "Sterlite Power",
    "Tata Projects", "NCC", "Siemens", "ABB", "Hitachi Energy",
    "IRCON", "RVNL", "Sterling and Wilson"
]

# Core Keywords for scraping
CORE_KEYWORDS = [
    "transmission", "distribution", "substation", "railway", "metro",
    "solar EPC", "wind power", "infrastructure", "civil construction",
    "pipeline", "L&T", "Kalpataru", "Sterlite Power", "Tata Projects"
]


def get_db_connection():
    """Get database connection from environment variable"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    
    return psycopg.connect(database_url, row_factory=dict_row)


def fetch_feed_safe(rss_url: str, keyword: str, retries: int = RETRIES):
    """Safely fetch RSS feed with retries"""
    for attempt in range(retries):
        try:
            feed = feedparser.parse(rss_url)
            if feed.bozo and hasattr(feed, 'bozo_exception'):
                logging.warning(f"Feed parse warning for '{keyword}': {feed.bozo_exception}")
            return feed
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for '{keyword}': {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None


def detect_sbu(title: str, source: str = "") -> str:
    """Detect relevant SBUs from title and source"""
    text = f"{title} {source}".lower()
    detected_sbus = []
    
    for sbu, keywords in SBU_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            detected_sbus.append(sbu)
    
    return ", ".join(detected_sbus) if detected_sbus else ""


def detect_competitor(title: str, source: str = "") -> str:
    """Detect competitors mentioned in title/source"""
    text = f"{title} {source}".lower()
    detected_competitors = []
    
    for competitor in COMPETITOR_KEYWORDS:
        if competitor.lower() in text:
            detected_competitors.append(competitor)
    
    return ", ".join(detected_competitors) if detected_competitors else ""


def scrape_news(keywords: List[str], lookback_days: int = LOOKBACK_DAYS) -> List[Dict]:
    """Scrape news from Google News RSS for given keywords"""
    all_articles = []
    seen_links = set()
    
    for i, keyword in enumerate(keywords):
        logging.info(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")
        
        encoded_keyword = quote(keyword)
        rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:{lookback_days}d&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = fetch_feed_safe(rss_url, keyword)
        if not feed or not feed.entries:
            logging.warning(f"No results for keyword: {keyword}")
            continue
        
        for entry in feed.entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            
            # Skip duplicates
            if link in seen_links:
                continue
            seen_links.add(link)
            
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
            
            # Verify keyword match
            text_lower = f"{title} {source}".lower()
            if keyword.lower() not in text_lower:
                continue
            
            # Detect SBU and Competitor
            sbu = detect_sbu(title, source)
            competitor = detect_competitor(title, source)
            
            all_articles.append({
                "keyword": keyword,
                "newstitle": title,
                "source": source,
                "link": link,
                "publishedate": pubdate,
                "sbu": sbu,
                "competitor": competitor,
                "scraped_content": ""  # Will be filled by LLM script
            })
        
        # Rate limiting
        time.sleep(random.uniform(DELAY_BETWEEN_REQUESTS, DELAY_BETWEEN_REQUESTS + 1))
    
    return all_articles


def save_to_database(articles: List[Dict]):
    """Save scraped articles to PostgreSQL database"""
    if not articles:
        logging.info("No articles to save")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
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
    for article in articles:
        try:
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
            saved_count += 1
        except Exception as e:
            logging.error(f"Error saving article: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    logging.info(f"✅ Saved {saved_count} new articles to database")


def main():
    """Main scraping function"""
    logging.info("=" * 60)
    logging.info("Starting News Scraping Job")
    logging.info("=" * 60)
    
    # Scrape news
    articles = scrape_news(CORE_KEYWORDS)
    logging.info(f"Scraped {len(articles)} unique articles")
    
    # Filter for articles with competitors
    articles_with_competitors = [a for a in articles if a['competitor']]
    logging.info(f"Found {len(articles_with_competitors)} articles with competitors")
    
    # Save to database
    save_to_database(articles_with_competitors)
    
    logging.info("=" * 60)
    logging.info("Scraping Job Complete")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
