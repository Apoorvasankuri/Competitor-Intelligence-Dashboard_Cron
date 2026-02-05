"""
Production LLM Processing for Competitor Intelligence
Processes scraped news with Claude API and updates PostgreSQL
"""

import os
import logging
import time
from typing import Dict, List, Optional
import psycopg
from psycopg.rows import dict_row
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError
import json
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

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
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Using Sonnet for better quality

# System Prompts
SCORING_PROMPT = """You are a senior business analyst for KEC International Ltd evaluating news relevance.

KEC International operates in: India T&D, International T&D, Transportation (Rail/Metro), Civil (Infrastructure), Renewables (Solar/Wind), Oil & Gas (Pipelines).

Key Competitors: L&T, Kalpataru, Sterlite Power, Tata Projects, NCC, Siemens, ABB, Hitachi Energy, IRCON, RVNL.

RELEVANCE SCORE (0-100):
90-100: Direct impact - contracts, orders, tenders, major projects involving KEC or competitors
70-89: Strong sector news - policy, technology, market trends affecting the industry
30-69: Indirect relevance - adjacent infrastructure, enabling technologies
1-29: Weak link - generic news with loose connection
0: Not relevant

CONFIDENCE SCORE (0-100):
80-100: High confidence - detailed, explicit information
40-79: Medium confidence - some ambiguity or incomplete details
0-39: Low confidence - vague or unclear information

Return ONLY valid JSON:
{"relevance_score": <int>, "confidence_score": <int>}"""

ANALYSIS_PROMPT = """You are an expert analyst for KEC International Ltd. Analyze this news article.

SBUs: India T&D, International T&D, Transportation, Civil, Renewables, Oil & Gas, General

Categories: order wins, new market entry, mergers & acquisitions, partnerships & alliances, financial, stock market, leadership/management, industry

Return ONLY valid JSON:
{
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
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def call_claude(prompt: str, system_prompt: str, max_tokens: int = 1000) -> str:
    """Call Claude API with retry logic"""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def scrape_article_content(url: str, max_length: int = 5000) -> str:
    """Scrape article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean and truncate
        text = ' '.join(text.split())
        return text[:max_length]
    
    except Exception as e:
        logging.warning(f"Failed to scrape {url}: {e}")
        return ""


def process_article_scoring(title: str, content: str) -> Dict:
    """Get relevance and confidence scores"""
    prompt = f"News Title: {title}\n\nContent: {content[:2000]}"
    
    try:
        response = call_claude(prompt, SCORING_PROMPT, max_tokens=100)
        scores = json.loads(response)
        return {
            'relevance_score': scores.get('relevance_score', 0),
            'confidance_score': scores.get('confidence_score', 0)  # Note: keeping typo for DB consistency
        }
    except Exception as e:
        logging.error(f"Scoring failed: {e}")
        return {'relevance_score': 0, 'confidance_score': 0}


def process_article_analysis(title: str, content: str) -> Dict:
    """Get SBU tagging, category, and summary"""
    prompt = f"News Title: {title}\n\nContent: {content[:3000]}"
    
    try:
        response = call_claude(prompt, ANALYSIS_PROMPT, max_tokens=500)
        analysis = json.loads(response)
        return {
            'sbu_tagging': analysis.get('sbu_tagging', ''),
            'category_tag': analysis.get('category_tag', ''),
            'kec_business_summary': analysis.get('kec_business_summary', '')
        }
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return {
            'sbu_tagging': '',
            'category_tag': '',
            'kec_business_summary': ''
        }


def get_unprocessed_articles() -> List[Dict]:
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
        LIMIT 100
    """
    
    cur.execute(query)
    articles = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return articles


def update_article(article_id: int, updates: Dict):
    """Update article in database with LLM results"""
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
        cur.execute(update_query, (
            updates.get('scraped_content', ''),
            updates.get('relevance_score', 0),
            updates.get('confidance_score', 0),
            updates.get('sbu_tagging', ''),
            updates.get('category_tag', ''),
            updates.get('kec_business_summary', ''),
            updates.get('sbu_tagging', ''),  # matched_sbu same as sbu_tagging
            article_id
        ))
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to update article {article_id}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def main():
    """Main LLM processing function"""
    logging.info("=" * 60)
    logging.info("Starting LLM Processing Job")
    logging.info("=" * 60)
    
    # Get unprocessed articles
    articles = get_unprocessed_articles()
    logging.info(f"Found {len(articles)} articles to process")
    
    if not articles:
        logging.info("No articles to process. Exiting.")
        return
    
    processed_count = 0
    
    for i, article in enumerate(articles):
        article_id = article['id']
        title = article['newstitle']
        link = article['link']
        
        logging.info(f"Processing {i+1}/{len(articles)}: {title[:60]}...")
        
        # Scrape content if not already scraped
        content = article.get('scraped_content', '')
        if not content:
            content = scrape_article_content(link)
        
        if not content:
            content = title  # Use title if scraping failed
        
        # Process with Claude
        try:
            # Get scores
            scores = process_article_scoring(title, content)
            
            # Only do deep analysis if relevance > 30
            if scores['relevance_score'] > 30:
                analysis = process_article_analysis(title, content)
            else:
                analysis = {
                    'sbu_tagging': '',
                    'category_tag': 'industry',
                    'kec_business_summary': title
                }
            
            # Combine results
            updates = {
                'scraped_content': content,
                **scores,
                **analysis
            }
            
            # Update database
            update_article(article_id, updates)
            processed_count += 1
            
            logging.info(f"✅ Processed: Relevance={scores['relevance_score']}, Category={analysis.get('category_tag', 'N/A')}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Failed to process article {article_id}: {e}")
            continue
    
    logging.info("=" * 60)
    logging.info(f"LLM Processing Complete: {processed_count}/{len(articles)} articles processed")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
