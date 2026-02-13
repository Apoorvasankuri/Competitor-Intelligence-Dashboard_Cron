import os
import logging
import time
import json
import requests
import psycopg
from psycopg.rows import dict_row
import pandas as pd
import re
from dotenv import load_dotenv
from typing import Dict, List
from bs4 import BeautifulSoup
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise Exception("CLAUDE_API_KEY environment variable not set")

client = Anthropic(api_key=CLAUDE_API_KEY)

# Model
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# Excel mapping file
EXCEL_MAPPING_FILE = "SBU_Competitor_Mapping.xlsx"

# Performance Configuration
STAGE1_BATCH_SIZE = 50
STAGE2_BATCH_SIZE = 20
MAX_WORKERS = 15
RATE_LIMIT_DELAY = 0.15

# Relevance threshold
RELEVANCE_THRESHOLD = 70

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection from environment variable"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    
    return psycopg.connect(database_url, row_factory=dict_row)


def load_raw_articles() -> pd.DataFrame:
    """Load unprocessed articles from raw_scraped_articles table"""
    conn = get_db_connection()
    
    query = """
        SELECT 
            id,
            published_date,
            news_title,
            competitor,
            sbu,
            source,
            search_keyword,
            link,
            content
        FROM raw_scraped_articles
        ORDER BY published_date DESC
    """
    
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'news_title': 'News Title',
        'link': 'Link',
        'competitor': 'Competitor',
        'sbu': 'SBU',
        'source': 'Source',
        'published_date': 'Published Date'
    })
    
    return df


def save_to_processed_articles(df: pd.DataFrame):
    """Save processed articles to processed_articles table"""
    if df.empty:
        logging.info("No articles to save")
        return
    
    conn = get_db_connection()
    
    insert_query = """
        INSERT INTO processed_articles (
            published_date,
            news_title,
            link,
            relevance_score,
            competitor_tagging,
            sbu_tagging,
            category_tag,
            summary,
            scraped_content
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (link, published_date) DO NOTHING
    """
    
    saved_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        try:
            cur = conn.cursor()
            cur.execute(insert_query, (
                row.get('Published Date'),
                row.get('News Title'),
                row.get('Link'),
                row.get('relevance_score', 0),
                row.get('competitor_tagging', '-'),
                row.get('sbu_tagging', 'None'),
                row.get('category_tag', 'not_analyzed'),
                row.get('summary', ''),
                row.get('scraped_content', '')
            ))
            conn.commit()
            cur.close()
            saved_count += 1
        except Exception as e:
            conn.rollback()
            failed_count += 1
            logging.error(f"Error saving article '{row.get('News Title', 'Unknown')[:50]}...': {e}")
    
    conn.close()
    
    logging.info(f"✅ Saved {saved_count} articles to processed_articles table")
    if failed_count > 0:
        logging.warning(f"⚠️  Failed to save {failed_count} articles")


def clear_raw_articles():
    """Clear all records from raw_scraped_articles table"""
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM raw_scraped_articles")
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        logging.info(f"🗑️  Cleared {deleted_count} articles from raw_scraped_articles table")
    except Exception as e:
        conn.rollback()
        conn.close()
        logging.error(f"Error clearing raw_scraped_articles: {e}")


# ============================================================================
# LOAD DATA FROM EXCEL
# ============================================================================

def load_excel_data():
    """Load competitors, SBUs, and categories from Excel file"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found! Please ensure it's in the same directory.")
    
    logging.info(f"📂 Loading data from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    competitors_list = competitor_df['Competitor'].dropna().unique().tolist()
    
    # Read SBU sheet
    sbu_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='SBU', header=1)
    sbu_list = sbu_df['SBU'].dropna().unique().tolist()
    
    # Read Categories sheet
    categories_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Categories')
    categories_list = categories_df['Category'].dropna().tolist()
    
    logging.info(f"   ✅ Loaded {len(competitors_list)} competitors")
    logging.info(f"   ✅ Loaded {len(sbu_list)} SBUs")
    logging.info(f"   ✅ Loaded {len(categories_list)} categories")
    
    return {
        'competitors': competitors_list,
        'sbus': sbu_list,
        'categories': categories_list
    }


# ============================================================================
# BUILD DYNAMIC PROMPT
# ============================================================================

def build_full_analysis_prompt(competitors: List[str], categories: List[str]) -> str:
    """Build the full analysis prompt with dynamic data"""
    
    # Format competitors list
    competitors_text = "\n".join([f"- {comp}" for comp in competitors])
    
    # Format categories list with numbering
    categories_text = "\n".join([f"{i+1}. **{cat}**" for i, cat in enumerate(categories)])
    
    prompt = f"""You are a business intelligence analyst for KEC International analyzing competitor news articles.

====================
ABOUT KEC INTERNATIONAL
====================
KEC International is a global infrastructure EPC major with 80+ years of experience, executing large-scale projects across 110+ countries.
KEC operates through six main business verticals (SBUs):

**1. TRANSMISSION & DISTRIBUTION (T&D)**
   - INDIA T&D: Power transmission lines, substations, grid infrastructure within India
   - INTERNATIONAL T&D: Power transmission projects outside India

**2. TRANSPORTATION**
   - Railways: Overhead electrification (OHE), signaling systems
   - Urban Infrastructure: Metro rail projects

**3. CIVIL**
   - Buildings, industrial facilities, water treatment plants

**4. RENEWABLES**
   - Solar and wind power projects

**5. OIL & GAS PIPELINES**
   - Cross-country pipelines

**6. CABLES & CONDUCTORS** (Manufacturing)

====================
COMPETITORS LIST
====================
Use ONLY these competitor names from our database:

{competitors_text}

====================
CATEGORIES
====================
Classify into ONE category:

{categories_text}

====================
YOUR TASK
====================
Analyze the following article and extract four fields:

**1. COMPETITOR TAGGING**
- Identify ALL competitors mentioned doing KEC-relevant business
- ONLY use names from COMPETITORS LIST above
- If multiple, separate with commas: "L&T, Tata Projects"
- If NO competitor found, output "-"

**2. SBU TAGGING**
- Identify which of KEC's SBUs this is relevant to
- Use exact names: "India T&D", "International T&D", "Transportation", "Civil", "Renewables", "Oil & Gas"
- Most articles = ONE SBU only
- If truly none match, use "General"

**3. CATEGORY TAG**
- Classify into ONE category from list above

**4. SUMMARY**
- Write 2-3 sentences ONLY
- Include: WHO (competitor), WHAT (action), WHERE (location), VALUE (if mentioned)
- Focus on competitive impact to KEC

====================
OUTPUT FORMAT
====================
Return ONLY valid JSON:

{{
  "competitor_tagging": "<comma-separated or '-'>",
  "sbu_tagging": "<comma-separated or 'General'>",
  "category_tag": "<single category>",
  "summary": "<2-3 sentences>"
}}"""

    return prompt


# ============================================================================
# STAGE 1: QUICK RELEVANCE SCORING
# ============================================================================

QUICK_SCORE_PROMPT = """You are an expert relevance scorer for KEC International's competitive intelligence system.

KEC'S CORE BUSINESSES:
- Transmission & Distribution (T&D)
- Transportation (Railways, Metro)
- Civil (Buildings, Infrastructure)
- Renewables (Solar, Wind)
- Oil & Gas Pipelines

SCORING RULES (0-100):

85-100: MUST ANALYZE
- Competitor wins major EPC contract (₹500+ crore)
- Major M&A/JV in infrastructure
- Government policy/budget for T&D/Rail/Renewables

70-84: TANGENTIALLY USEFUL
- Quarterly results mentioning order book
- Industry sector commentary

20-39: WEAK RELEVANCE
- Stock price movements only
- Generic CSR announcements

0-19: IRRELEVANT
- Unrelated businesses (IT, FMCG, retail)
- Generic market news

Return ONLY an integer 0-100. No explanation."""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def quick_relevance_score(title: str, competitor: str) -> int:
    """Quick relevance scoring using title only"""
    
    prompt = f"""Title: {title}
Competitor: {competitor}

Relevance score (0-100):"""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            temperature=0,
            system=QUICK_SCORE_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        score_text = response.content[0].text.strip()
        score = int(re.search(r'\d+', score_text).group())
        return max(0, min(100, score))
        
    except Exception as e:
        logging.warning(f"Quick score failed for '{title[:50]}...': {e}")
        return 0


# ============================================================================
# STAGE 2: FULL ANALYSIS
# ============================================================================

def scrape_article(url: str, max_length: int = 3000) -> str:
    """Scrape article content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe"]):
            element.decompose()
        
        # Extract text
        text = ' '.join(soup.get_text(separator=' ', strip=True).split())
        
        return text[:max_length] if text else ""
        
    except Exception as e:
        logging.warning(f"Scraping failed for {url}: {e}")
        return ""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def full_analysis(title: str, content: str, relevance_score: int, full_prompt: str) -> Dict:
    """Full analysis with scraped content"""
    
    # Use content if available, otherwise fall back to title
    analysis_text = content[:2000] if content else title
    
    prompt = f"""Analyze this news (relevance score: {relevance_score}/100):

Title: {title}
Content: {analysis_text}

Provide detailed analysis."""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            temperature=0,
            system=full_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        # Extract JSON
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        
        if json_match:
            analysis = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found")
        
        # Add relevance score from Stage 1
        analysis['relevance_score'] = relevance_score
        
        # Validate
        required = ["competitor_tagging", "sbu_tagging", "category_tag", "summary"]
        for field in required:
            if field not in analysis:
                raise ValueError(f"Missing field: {field}")
        
        return analysis
        
    except Exception as e:
        logging.error(f"Full analysis failed for '{title[:50]}...': {e}")
        return {
            "relevance_score": relevance_score,
            "competitor_tagging": "-",
            "sbu_tagging": "None",
            "category_tag": "error",
            "summary": f"Analysis error: {str(e)[:100]}"
        }


# ============================================================================
# PIPELINE PROCESSING
# ============================================================================

def stage1_quick_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Quick relevance scoring for all articles"""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 1: QUICK RELEVANCE SCORING (Title Only)")
    logging.info("="*60)
    
    relevance_scores = []
    total = len(df)
    
    for i in range(0, total, STAGE1_BATCH_SIZE):
        batch_num = i // STAGE1_BATCH_SIZE + 1
        total_batches = (total + STAGE1_BATCH_SIZE - 1) // STAGE1_BATCH_SIZE
        
        batch_df = df.iloc[i:i+STAGE1_BATCH_SIZE]
        
        logging.info(f"\n📊 Scoring batch {batch_num}/{total_batches} ({len(batch_df)} articles)...")
        
        for idx, row in batch_df.iterrows():
            title = str(row['News Title'])
            competitor = str(row.get('Competitor', ''))
            
            score = quick_relevance_score(title, competitor)
            relevance_scores.append(score)
            
            if score >= RELEVANCE_THRESHOLD:
                logging.info(f"   ✅ Score {score}: {title[:60]}...")
            
            time.sleep(RATE_LIMIT_DELAY)
    
    df['relevance_score'] = relevance_scores
    
    high_relevance = df[df['relevance_score'] >= RELEVANCE_THRESHOLD]
    
    logging.info(f"\n📈 Stage 1 Complete:")
    logging.info(f"   Total articles: {len(df)}")
    logging.info(f"   High relevance (≥{RELEVANCE_THRESHOLD}): {len(high_relevance)} ({len(high_relevance)/len(df)*100:.1f}%)")
    
    return df


def stage2_full_analysis(df: pd.DataFrame, full_prompt: str) -> pd.DataFrame:
    """Stage 2: Full analysis only for high-relevance articles"""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 2: FULL ANALYSIS (Scraping + Deep Analysis)")
    logging.info("="*60)
    
    # Filter for high relevance
    high_rel_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()
    
    if len(high_rel_df) == 0:
        logging.warning("No articles meet relevance threshold. Skipping Stage 2.")
        return df
    
    # Initialize columns for all rows
    df['competitor_tagging'] = '-'
    df['sbu_tagging'] = 'None'
    df['category_tag'] = 'not_analyzed'
    df['summary'] = 'Not analyzed (low relevance)'
    df['scraped_content'] = ''
    
    total = len(high_rel_df)
    
    for i in range(0, total, STAGE2_BATCH_SIZE):
        batch_num = i // STAGE2_BATCH_SIZE + 1
        total_batches = (total + STAGE2_BATCH_SIZE - 1) // STAGE2_BATCH_SIZE
        
        batch_df = high_rel_df.iloc[i:i+STAGE2_BATCH_SIZE]
        
        logging.info(f"\n🔍 Analyzing batch {batch_num}/{total_batches} ({len(batch_df)} articles)...")
        
        # Parallel scraping
        logging.info(f"   📥 Scraping {len(batch_df)} articles...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(scrape_article, row['Link']): idx
                for idx, row in batch_df.iterrows()
            }
            
            contents = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                contents[idx] = future.result()
        
        # Sequential analysis
        logging.info(f"   🤖 Running full analysis...")
        for idx, row in batch_df.iterrows():
            title = str(row['News Title'])
            content = contents.get(idx, '')
            relevance = row['relevance_score']
            
            analysis = full_analysis(title, content, relevance, full_prompt)
            
            # Update dataframe
            df.at[idx, 'competitor_tagging'] = analysis['competitor_tagging']
            df.at[idx, 'sbu_tagging'] = analysis['sbu_tagging']
            df.at[idx, 'category_tag'] = analysis['category_tag']
            df.at[idx, 'summary'] = analysis['summary']
            df.at[idx, 'scraped_content'] = content[:500] if content else ''
            
            time.sleep(RATE_LIMIT_DELAY)
    
    logging.info(f"\n✅ Stage 2 Complete: Analyzed {len(high_rel_df)} high-relevance articles")
    
    return df


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Fast deduplication based on title similarity"""
    
    logging.info("\n🔍 Deduplicating articles...")
    
    df_reset = df.reset_index(drop=True)
    to_drop = set()
    
    for i in range(len(df_reset)):
        if i in to_drop:
            continue
        
        title_i = str(df_reset.iloc[i]['News Title']).lower()
        
        for j in range(i + 1, min(i + 50, len(df_reset))):
            if j in to_drop:
                continue
            
            title_j = str(df_reset.iloc[j]['News Title']).lower()
            similarity = SequenceMatcher(None, title_i, title_j).ratio()
            
            if similarity > 0.85:
                to_drop.add(j)
    
    logging.info(f"   🗑️ Removed {len(to_drop)} duplicates")
    
    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    start_time = time.time()
    
    logging.info("="*60)
    logging.info("KEC INTERNATIONAL - COMPETITIVE INTELLIGENCE ANALYZER")
    logging.info("="*60)
    
    # Load raw articles from database
    logging.info("📥 Loading articles from raw_scraped_articles table...")
    df = load_raw_articles()
    
    if df.empty:
        logging.info("ℹ️  No articles to process. Exiting.")
        return
    
    logging.info(f"📄 Loaded {len(df)} articles")
    
    # Deduplicate first (within current batch)
    df = deduplicate_articles(df)
    logging.info(f"📄 After deduplication: {len(df)} articles")
    
    # Load Excel mapping data
    try:
        excel_data = load_excel_data()
    except Exception as e:
        logging.error(f"❌ Failed to load Excel data: {e}")
        return
    
    # Build dynamic prompt
    logging.info("\n🔧 Building analysis prompt...")
    full_prompt = build_full_analysis_prompt(
        competitors=excel_data['competitors'],
        categories=excel_data['categories']
    )
    
    # Stage 1: Quick scoring
    df = stage1_quick_scoring(df)
    
    # Stage 2: Full analysis (only high-relevance)
    df = stage2_full_analysis(df, full_prompt)
    
    # Save to processed_articles table
    logging.info("\n💾 Saving to processed_articles table...")
    save_to_processed_articles(df)
    
    # Clear raw_scraped_articles table
    logging.info("\n🧹 Clearing raw_scraped_articles table...")
    clear_raw_articles()
    
    # Statistics
    elapsed = time.time() - start_time
    high_relevance = df[df['relevance_score'] >= RELEVANCE_THRESHOLD]
    
    logging.info("\n" + "="*60)
    logging.info("📈 PROCESSING COMPLETE")
    logging.info("="*60)
    logging.info(f"⏱️  Time: {elapsed/60:.1f} minutes")
    logging.info(f"📄 Total articles processed: {len(df)}")
    logging.info(f"⭐ High relevance: {len(high_relevance)} ({len(high_relevance)/len(df)*100:.1f}%)")
    
    if len(high_relevance) > 0:
        logging.info(f"\n📊 Top Categories:")
        for cat, count in high_relevance['category_tag'].value_counts().head(5).items():
            logging.info(f"   {cat}: {count}")
        
        logging.info(f"\n📁 Top SBUs:")
        for sbu, count in high_relevance['sbu_tagging'].value_counts().head(5).items():
            logging.info(f"   {sbu}: {count}")
    
    # Cost estimate
    stage1_calls = len(df)
    stage2_calls = len(high_relevance)
    total_calls = stage1_calls + stage2_calls
    est_tokens = (stage1_calls * 200) + (stage2_calls * 3000)
    est_cost = (est_tokens / 1_000_000) * 3.00
    
    logging.info(f"\n💰 API Usage:")
    logging.info(f"   Stage 1 calls: {stage1_calls}")
    logging.info(f"   Stage 2 calls: {stage2_calls}")
    logging.info(f"   Total calls: {total_calls}")
    logging.info(f"   Est. cost: ~${est_cost:.2f}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
