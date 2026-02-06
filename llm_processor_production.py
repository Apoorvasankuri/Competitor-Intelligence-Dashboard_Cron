"""
Production LLM Processing for Competitor Intelligence (Optimized)
Processes scraped news with Claude API and updates PostgreSQL
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional
import psycopg
from psycopg.rows import dict_row
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError
import json
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise Exception("CLAUDE_API_KEY environment variable not set")

client = Anthropic(api_key=CLAUDE_API_KEY)
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Performance Configuration
BATCH_SIZE = 50  # Process articles in batches for better database performance
MAX_WORKERS = 5  # Parallel web scraping threads
PROCESS_LIMIT = 500  # Maximum articles to process per run
RATE_LIMIT_DELAY = 0.5  # Reduced from 1 second

# Combined prompt for efficiency - single API call instead of two
COMBINED_ANALYSIS_PROMPT = """You are a senior business analyst for KEC International Ltd evaluating news relevance.

KEC International operates in: India T&D, International T&D, Transportation (Rail/Metro), Civil (Infrastructure), Renewables (Solar/Wind), Oil & Gas (Pipelines).

Key Competitors: L&T, Kalpataru, Sterlite Power, Tata Projects, NCC, Siemens, ABB, Hitachi Energy, IRCON, RVNL.

Analyze this article and provide:

RELEVANCE SCORE (0-100):
90-100: Direct impact - contracts, orders, tenders, major projects
70-89: Strong sector news - policy, technology, market trends
30-69: Indirect relevance - adjacent infrastructure
1-29: Weak link - generic news
0: Not relevant

CONFIDENCE SCORE (0-100):
80-100: High confidence - detailed, explicit information
40-79: Medium confidence - some ambiguity
0-39: Low confidence - vague information

SBUs: India T&D, International T&D, Transportation, Civil, Renewables, Oil & Gas, General
Categories: order wins, new market entry, mergers & acquisitions, partnerships & alliances, financial, stock market, leadership/management, industry

Return ONLY valid JSON:
{
  "relevance_score": <int>,
  "confidence_score": <int>,
  "sbu_tagging": "<comma-separated SBUs>",
  "category_tag": "<single category>",
  "kec_business_summary": "<2-3 sentence summary>"
}"""


def get_db_connection():
    """Get database connection"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    return psycopg.connect(database_url, row_factory=dict_row)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),  # Reduced from 5
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def call_claude(prompt: str, system_prompt: str, max_tokens: int = 500) -> str:
    """Call Claude API with retry logic"""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def scrape_article_content(url: str, max_length: int = 3000) -> str:
    """Scrape article content from URL (reduced from 5000)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=8)  # Reduced timeout
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean and truncate
        text = ' '.join(text.split())
        return text[:max_length]
    
    except Exception as e:
        logging.debug(f"Failed to scrape {url}: {e}")
        return ""


def scrape_articles_parallel(articles: List[Dict]) -> Dict[int, str]:
    """Scrape multiple articles in parallel"""
    content_map = {}
    
    def scrape_single(article):
        article_id = article['id']
        content = article.get('scraped_content', '')
        if not content:
            content = scrape_article_content(article['link'])
        if not content:
            content = article['newstitle']  # Fallback to title
        return article_id, content
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_single, article): article for article in articles}
        
        for future in as_completed(futures):
            try:
                article_id, content = future.result()
                content_map[article_id] = content
            except Exception as e:
                article = futures[future]
                logging.warning(f"Scraping failed for article {article['id']}: {e}")
                content_map[article['id']] = article['newstitle']
    
    return content_map


def process_article_combined(title: str, content: str) -> Dict:
    """Process article with single Claude API call (more efficient)"""
    prompt = f"News Title: {title}\n\nContent: {content[:2500]}"
    
    try:
        response = call_claude(prompt, COMBINED_ANALYSIS_PROMPT, max_tokens=500)
        result = json.loads(response)
        
        return {
            'relevance_score': result.get('relevance_score', 0),
            'confidance_score': result.get('confidence_score', 0),  # Keeping typo for DB consistency
            'sbu_tagging': result.get('sbu_tagging', ''),
            'category_tag': result.get('category_tag', 'industry'),
            'kec_business_summary': result.get('kec_business_summary', title)
        }
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error: {e}, Response: {response[:200]}")
        return {
            'relevance_score': 0,
            'confidance_score': 0,
            'sbu_tagging': '',
            'category_tag': 'industry',
            'kec_business_summary': title
        }
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return {
            'relevance_score': 0,
            'confidance_score': 0,
            'sbu_tagging': '',
            'category_tag': 'industry',
            'kec_business_summary': title
        }


def get_unprocessed_articles(limit: int = PROCESS_LIMIT) -> List[Dict]:
    """Get articles that haven't been processed by LLM yet"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    query = """
        SELECT id, newstitle, link, scraped_content
        FROM competitor_data
        WHERE (relevance_score IS NULL OR relevance_score = 0)
        AND competitor IS NOT NULL
        AND competitor != ''
        ORDER BY publishedate DESC
        LIMIT %s
    """
    
    cur.execute(query, (limit,))
    articles = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return articles


def update_articles_batch(updates_list: List[tuple]):
    """Update multiple articles in a single transaction (much faster)"""
    if not updates_list:
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    update_query = """
        UPDATE competitor_data
        SET 
            scraped_content = %s,
            relevance_score = %s,
            confidance_score = %s,
            sbu_tagging = %s,
            category_tag = %s,
            kec_business_summary = %s,
            matched_sbu = %s
        WHERE id = %s
    """
    
    try:
        # Execute all updates in one transaction
        cur.executemany(update_query, updates_list)
        conn.commit()
        logging.info(f"✅ Batch updated {len(updates_list)} articles")
    except Exception as e:
        logging.error(f"Batch update failed: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def process_batch(articles: List[Dict], content_map: Dict[int, str]) -> List[tuple]:
    """Process a batch of articles and return updates"""
    updates_list = []
    
    for article in articles:
        article_id = article['id']
        title = article['newstitle']
        content = content_map.get(article_id, title)
        
        try:
            # Single API call for all analysis
            result = process_article_combined(title, content)
            
            # Prepare update tuple
            updates_list.append((
                content,
                result['relevance_score'],
                result['confidance_score'],
                result['sbu_tagging'],
                result['category_tag'],
                result['kec_business_summary'],
                result['sbu_tagging'],  # matched_sbu same as sbu_tagging
                article_id
            ))
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            logging.error(f"Failed to process article {article_id}: {e}")
            # Add failed article with default values
            updates_list.append((
                content,
                0, 0, '', 'industry', title, '', article_id
            ))
    
    return updates_list


def main():
    """Main LLM processing function"""
    logging.info("=" * 60)
    logging.info("Starting Optimized LLM Processing Job")
    logging.info("=" * 60)
    
    # Get unprocessed articles
    articles = get_unprocessed_articles()
    total_articles = len(articles)
    logging.info(f"Found {total_articles} articles to process")
    
    if not articles:
        logging.info("No articles to process. Exiting.")
        return
    
    # Process in batches
    all_updates = []
    
    for batch_num in range(0, total_articles, BATCH_SIZE):
        batch = articles[batch_num:batch_num + BATCH_SIZE]
        batch_size = len(batch)
        
        logging.info(f"\n--- Processing Batch {batch_num//BATCH_SIZE + 1} ({batch_size} articles) ---")
        
        # Step 1: Scrape all articles in parallel
        logging.info("Scraping article content in parallel...")
        content_map = scrape_articles_parallel(batch)
        
        # Step 2: Process with Claude (sequential for rate limiting)
        logging.info("Processing with Claude API...")
        batch_updates = process_batch(batch, content_map)
        
        # Step 3: Batch update database
        update_articles_batch(batch_updates)
        
        all_updates.extend(batch_updates)
        
        logging.info(f"Batch complete: {len(batch_updates)}/{batch_size} processed")
    
    # Summary
    successful = len([u for u in all_updates if u[1] > 0])  # relevance_score > 0
    
    logging.info("=" * 60)
    logging.info(f"LLM Processing Complete")
    logging.info(f"Total Processed: {len(all_updates)}/{total_articles}")
    logging.info(f"Successfully Analyzed: {successful}")
    logging.info(f"Failed/Skipped: {len(all_updates) - successful}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
