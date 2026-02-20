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
        "Source",
        relevance_score,
        competitor_tagging,
        sbu_tagging,
        category_tag,
        summary,
        scraped_content,
        contract_value_inr_crore,
        geography,
        competitor_tier,
        rank_score
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                row.get('rank_score', 0)
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
def load_competitor_tiers():
    """Load competitor tier mapping from Excel file"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found!")
    
    logging.info(f"📂 Loading competitor tiers from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    
    # Create tier mapping dictionary
    tier_map = {}
    for idx, row in competitor_df.iterrows():
        competitor = row.get('Competitor')
        tier = row.get('Tier')
        
        if pd.notna(competitor) and pd.notna(tier):
            tier_map[competitor.strip()] = int(tier)
    
    logging.info(f"   ✅ Loaded tiers for {len(tier_map)} competitors")
    
    return tier_map

# ============================================================================
# BUILD DYNAMIC PROMPT (SCRIPT 4 DETAILED VERSION)
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
     • High voltage transmission lines (220 kV to 765 kV)
     • Substations and switchyards
     • HVDC (High Voltage Direct Current) systems
     • Digital substations
     • STATCOM (Static Synchronous Compensator)
     • Underground cabling
     • Towers, poles & hardware supplies
   
   - INTERNATIONAL T&D: Power transmission projects outside India
     • Cross-border transmission projects
     • Grid infrastructure in Middle East, Americas, Africa, SAARC, Asia Pacific, CIS, Australia
     • Same technical scope as India T&D but in international markets

**2. TRANSPORTATION**
   - Railways: Overhead electrification (OHE), signaling systems, TCAS Kavach, railway bridges, stations & platforms
   - Urban Infrastructure: Metro rail projects (viaducts, stations, tech-enabled areas), ropeways
   - Speed upgradation projects
   - Track laying, depot & workshops

**3. CIVIL**
   - Residential buildings and high-rise towers
   - Commercial buildings and office complexes
   - Factories and industrial facilities
   - Airports and aviation infrastructure
   - Hospitals and healthcare facilities
   - Data centers
   - Water pipeline projects and treatment plants
   - Warehouses & logistics facilities
   - Tunnel ventilation systems
   - Municipal waste-to-energy plants, FGD (Flue Gas Desulphurisation) units

**4. RENEWABLES**
   - Solar: Large-scale solar power plants (>500 MW capability), industrial solar solutions
   - Wind: Wind farm development and infrastructure
   - Green Hydrogen: Emerging capabilities
   - Hybrid renewable projects

**5. OIL & GAS PIPELINES**
   - Cross-country oil and gas pipelines
   - Slurry pipelines
   - Water pipelines (potable water supply projects)
   - Composite station works

**6. CABLES & CONDUCTORS** (Manufacturing)
   - Power cables
   - Control & instrumentation cables
   - Railway cables
   - Conductors (overhead line conductors)
   - Telecom cables
   - Special application cables

====================
COMPETITORS LIST
====================
Use ONLY these competitor names from our database:

{competitors_text}

**VARIATION RECOGNITION:**
Recognize variations and map to the standard name above:
- "L&T" = "Larsen & Toubro" = "Larsen and Toubro" = "L&T Ltd" = "L&T Limited"
- "Tata Projects" = "Tata Projects Ltd" = "Tata Projects Limited"
- "Kalpataru" = "Kalpataru Power Transmission" = "Kalpataru Projects International" = "KPTL"
- "ABB" = "ABB India" = "ABB Limited" = "ABB Ltd"
- "Siemens" = "Siemens India" = "Siemens Limited" = "Siemens AG"
- "IRCON" = "IRCON International" = "IRCON International Ltd"
- "RVNL" = "Rail Vikas Nigam" = "Rail Vikas Nigam Limited"
- "Sterling & Wilson" = "Sterling and Wilson" = "S&W"
- "Sterlite" = "Sterlite Power" = "Sterlite Grid"

**SUBSIDIARY MAPPING:**
Map subsidiaries to parent company ONLY if doing EPC/infrastructure work:
- "L&T Construction" / "L&T Power" / "L&T Metro" → "L&T"
- "Tata Power" doing EPC → "Tata Projects", otherwise "-"

====================
CATEGORIES
====================
Classify into ONE category:

{categories_text}

**CATEGORY PRIORITIZATION RULES:**
- Contract WON → "order wins" (highest priority)
- Contract being bid → "bidding activity"
- Project completed/commissioned → "project execution"
- M&A announced → "mergers & acquisitions"
- JV/partnership → "partnerships & alliances"
- Quarterly results → "financial"
- Stock movement → "stock market"
- When uncertain, pick the PRIMARY business action

====================
YOUR TASK
====================
Analyze the following article and extract four fields:

**1. COMPETITOR TAGGING**
RULES:
- Read the FULL article content carefully
- Identify ALL competitors mentioned who are doing business activities relevant to KEC's sectors
- ONLY use competitor names from the COMPETITORS LIST above
- Match variations to the standard name (e.g., "Larsen & Toubro" → "L&T")
- Map subsidiaries to parent company IF doing EPC/infrastructure work
- If article mentions competitor but NOT for relevant business (e.g., "Siemens washing machines"), output "-"
- If multiple competitors, separate with commas: "L&T, Tata Projects"
- If NO competitor found, output "-"
- Do NOT include KEC itself in competitor tagging

EXAMPLES:
✓ "Larsen & Toubro bags metro contract" → "L&T"
✓ "L&T Construction and Tata Projects bid for project" → "L&T, Tata Projects"
✓ "Sterling and Wilson Renewable Energy wins solar EPC" → "Sterling & Wilson"
✗ "Siemens launches new home appliances" → "-" (not KEC-relevant business)
✗ "Tata Power distributes electricity in Mumbai" → "-" (distribution, not EPC)

**2. SBU TAGGING**
RULES:
- Identify which of KEC's SBUs this article is relevant to
- IGNORE any previous SBU detection - analyze from article content freshly
- Be STRICT: Most articles relate to ONLY ONE SBU
- Only assign multiple SBUs if article explicitly mentions multiple business areas
- Use these exact SBU names: "India T&D", "International T&D", "Transportation", "Civil", "Renewables", "Oil & Gas"
- If article is about international T&D projects, use "International T&D" (not "India T&D")
- If truly none match or too generic, use "General"

MULTI-SBU EXAMPLES (rare cases):
✓ "L&T wins integrated EPC for solar park with 400 kV transmission evacuation" → "Renewables, India T&D" (or "International T&D" if outside India)
✓ "Metro project includes OHE and civil viaduct work" → "Transportation, Civil"

SINGLE-SBU EXAMPLES (most common):
✓ "L&T completes 765 kV transmission line in Rajasthan" → "India T&D"
✓ "Tata Projects wins 500 MW solar EPC contract in Abu Dhabi" → "Renewables"
✓ "IRCON bags railway electrification project" → "Transportation"
✓ "NCC constructs residential towers in Bangalore" → "Civil"

**3. CATEGORY TAG**
RULES:
- Classify into ONE category from the list above
- Apply prioritization rules
- Focus on the PRIMARY business action in the article

**4. SUMMARY**
RULES:
- Write EXACTLY 1-2 sentences (maximum 40 words total)
- Be crisp and fact-dense - every word must add value
- MUST include: Competitor name, key numbers (contract value/revenue/percentage), specific action related to category
- Structure: "[Competitor] [action verb] [key number] [what/where] [category-specific detail]"
- NO generic phrases like "strengthening position" or "demonstrates dominance"
- Focus on FACTS, not implications

GOOD SUMMARY EXAMPLES:
✓ "L&T secured ₹850 crore contract for 220 kV transmission line in Uttar Pradesh, including 45 km overhead lines and 2 substations." (ORDER WIN - 24 words)
✓ "Kalpataru's Q4 revenue grew 22% to ₹4,200 crore with order book at ₹28,000 crore in T&D and urban infrastructure." (FINANCIAL - 22 words)
✓ "L&T, Tata Projects, Kalpataru, and Sterlite bid for PGCIL's ₹600 crore 400 kV Bikaner-Merta transmission project." (BIDDING - 17 words)

BAD SUMMARY EXAMPLES:
✗ "L&T secured a major contract, strengthening its position in India's grid expansion market where KEC also competes. This win demonstrates L&T's continued dominance in state utility projects." (too wordy, no specific details, 30 words)
✗ "The company won a project. This is good for them." (too vague, 9 words)
✗ "Kalpataru Power Transmission achieved impressive growth in the latest quarter, demonstrating strong operational efficiency and market traction across various business segments in competitive markets." (too generic, no numbers, 27 words)

CATEGORY-SPECIFIC REQUIREMENTS:
- ORDER WINS: Include contract value, scope (km/MW/stations), location
- FINANCIAL: Include revenue/profit figures, growth %, order book value
- BIDDING: List all competitors bidding, project value, scope
- PROJECT EXECUTION: Include capacity/scale, timeline, location
- M&A/PARTNERSHIPS: Include deal value, stake %, target company details
====================
OUTPUT FORMAT
====================
Return ONLY valid JSON with these exact field names:

{{
  "competitor_tagging": "<comma-separated competitor names from list, or '-'>",
  "sbu_tagging": "<comma-separated SBU names from list, or 'General'>",
  "category_tag": "<single category from list>",
  "summary": "<1-2 sentences, max 40 words with specific numbers>",
  "contract_value_inr_crore": <numeric value in INR crore, or null if not mentioned>,
  "geography": "<India/Middle East/Africa/Americas/SAARC/Other or null>"
}}

**EXTRACTION RULES FOR NEW FIELDS:**

**contract_value_inr_crore:**
- Extract ONLY if explicitly mentioned in article
- Convert to INR Crore:
  * ₹X crore → X
  * ₹X lakh → X/100
  * $X million → X × 85 (approx)
  * X MW solar → null (capacity, not contract value)
- For financial results, extract revenue/profit value
- For M&A, extract deal value
- If not mentioned, return null

**geography:**
- Identify primary location mentioned
- Map to regions:
  * "India" → Any Indian state/city
  * "Middle East" → UAE, Saudi, Qatar, Bahrain, Oman, Kuwait
  * "Africa" → Any African country
  * "Americas" → USA, Brazil, etc.
  * "SAARC" → Bangladesh, Sri Lanka, Nepal, etc.
  * "Other" → Rest of world
- If not clear or multiple regions, use primary project location
- If not mentioned, return null
====================
EXAMPLE 1: ORDER WIN
====================
Title: "L&T bags ₹1,200 crore metro project in Pune"
Content: "Larsen & Toubro has been awarded a major contract worth ₹1,200 crore for civil and station works for Pune Metro Line 4. The project includes construction of 8 elevated stations and 12 km viaduct. L&T will complete the work in 36 months..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "L&T",
  "sbu_tagging": "Transportation",
  "category_tag": "order wins",
  "summary": "L&T won ₹1,200 crore Pune Metro Line 4 contract covering 8 elevated stations and 12 km viaduct with 36-month timeline.",
  "contract_value_inr_crore": 1200,
  "geography": "India"
}}
====================
EXAMPLE 2: MULTI-COMPETITOR BIDDING
====================
Title: "Five companies bid for PGCIL's 400 kV transmission project"
Content: "Power Grid Corporation of India has received bids from L&T, Tata Projects, Kalpataru Power, KEC International and Sterlite Power for the 400 kV Bikaner-Merta transmission line project worth approximately ₹600 crore..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "L&T, Tata Projects, Kalpataru, Sterlite",
  "sbu_tagging": "India T&D",
  "category_tag": "bidding activity",
  "summary": "L&T, Tata Projects, Kalpataru, Sterlite competing for PGCIL's ₹600 crore 400 kV Bikaner-Merta transmission line.",
  "contract_value_inr_crore": 600,
  "geography": "India"
}}
====================
EXAMPLE 3: FINANCIAL RESULTS
====================
Title: "Kalpataru Power posts 22% growth in Q4 revenue"
Content: "Kalpataru Power Transmission reported strong Q4 results with consolidated revenue growing 22% to ₹4,200 crore. The company's order book stands at ₹28,000 crore with strong pipeline in T&D and urban infra segments. Margins improved to 8.2%..."

CORRECT OUTPUT:
{{
  "competitor_tagging": "Kalpataru",
  "sbu_tagging": "General",
  "category_tag": "financial",
  "summary": "Kalpataru Q4 revenue up 22% to ₹4,200 crore, order book ₹28,000 crore in T&D and urban infrastructure, margins 8.2%.",
  "contract_value_inr_crore": 4200,
  "geography": null
}}
Now analyze the provided article."""

    return prompt


# ============================================================================
# STAGE 1: QUICK RELEVANCE SCORING
# ============================================================================

QUICK_SCORE_PROMPT = """You are an expert relevance scorer for KEC International's competitive intelligence system, serving senior management for strategic decision-making.

Competitors: L&T, Kalpataru, Sterlite, Tata Projects, NCC, Siemens, ABB, IRCON, RVNL, Shapoorji, PNC, Simplex, Sterling & Wilson, ReNew, Hero Future, etc.

KEC'S CORE BUSINESSES:
- Transmission & Distribution (T&D): Power lines, substations, grid infrastructure
- Transportation: Railways, metro, monorail, signaling
- Civil: Buildings, water treatment, industrial facilities, defense infrastructure
- Renewables: Solar parks, wind farms, hybrid projects
- Oil & Gas: Pipelines, terminals, storage facilities

SCORING RULES (0-100):

85-100: MUST ANALYZE
- Competitor wins major EPC contract (₹500+ crore) in KEC sectors
- Major M&A/JV in EPC/infra sectors
- New market entry by competitor in KEC geographies
- Government policy/budget allocation for T&D/Rail/Renewables/Infra
- Technology developments in power transmission, rail systems

70-84: TANGENTIALLY USEFUL
- Competitor quarterly results IF they mention order book/projects
- General sector commentary by industry bodies
- Adjacent infrastructure if involves EPC work

20-39: WEAK RELEVANCE
- Stock price movements with no project/operational news
- Generic CSR/sustainability announcements
- Awards/rankings without business impact

0-19: IRRELEVANT
- Competitor's unrelated businesses (IT services, finance, FMCG, retail)
- Generic market/economy news with no sector specifics

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
    
    prompt = f"""Analyze this news (relevance score already determined: {relevance_score}/100):

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
        required = ["competitor_tagging", "sbu_tagging", "category_tag", "summary", "contract_value_inr_crore", "geography"]
        for field in required:
            if field not in analysis:
                raise ValueError(f"Missing field: {field}")

        # Ensure numeric fields are properly typed
        if analysis.get('contract_value_inr_crore') is not None:
            try:
                analysis['contract_value_inr_crore'] = float(analysis['contract_value_inr_crore'])
            except:
                analysis['contract_value_inr_crore'] = None

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
    logging.info(f"   Will proceed to full analysis: {len(high_relevance)} articles")
    
    return df


def stage2_full_analysis(df: pd.DataFrame, full_prompt: str, competitor_tier_map: Dict[str, int]) -> pd.DataFrame:
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
    
            # Update dataframe with analysis results
            df.at[idx, 'competitor_tagging'] = analysis['competitor_tagging']
            df.at[idx, 'sbu_tagging'] = analysis['sbu_tagging']
            df.at[idx, 'category_tag'] = analysis['category_tag']
            df.at[idx, 'summary'] = analysis['summary']
            df.at[idx, 'scraped_content'] = content[:500] if content else ''
            df.at[idx, 'contract_value_inr_crore'] = analysis.get('contract_value_inr_crore')
            df.at[idx, 'geography'] = analysis.get('geography')
    
            time.sleep(RATE_LIMIT_DELAY)
    
    # Calculate ranking for all high-relevance articles
    logging.info(f"\n📊 Calculating ranking scores...")
    for idx, row in high_rel_df.iterrows():
        rank_data = calculate_rank_score(df.loc[idx], competitor_tier_map)
        df.at[idx, 'rank_score'] = rank_data['rank_score']
        df.at[idx, 'competitor_tier'] = rank_data['competitor_tier']
        
        logging.info(f"   Rank {rank_data['rank_score']}: {df.loc[idx, 'News Title'][:60]}...")
    
    logging.info(f"\n✅ Stage 2 Complete: Analyzed {len(high_rel_df)} high-relevance articles")
    
    return df

def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced deduplication with multiple strategies:
    1. Exact title match
    2. Fuzzy title similarity (>85%)
    3. Same value + same competitor + similar date
    4. Core content extraction and matching
    """
    
    logging.info("\n🔍 Deduplicating articles...")
    
    if df.empty:
        return df
    
    df_reset = df.reset_index(drop=True)
    to_drop = set()
    
    # Strategy 1: Exact title duplicates
    seen_titles = {}
    for i in range(len(df_reset)):
        title = str(df_reset.iloc[i]['News Title']).lower().strip()
        if title in seen_titles:
            to_drop.add(i)
            logging.debug(f"   Exact duplicate: {title[:50]}...")
        else:
            seen_titles[title] = i
    
    logging.info(f"   Found {len(to_drop)} exact title duplicates")
    
    # Strategy 2 & 3: Fuzzy matching + value matching
    initial_drop_count = len(to_drop)
    
    for i in range(len(df_reset)):
        if i in to_drop:
            continue
        
        title_i = str(df_reset.iloc[i]['News Title']).lower()
        date_i = df_reset.iloc[i]['Published Date']
        competitor_i = str(df_reset.iloc[i].get('Competitor', '')).lower()
        
        # Extract numbers from title (contract values)
        numbers_i = extract_numbers_from_text(title_i)
        
        # Only compare with articles in similar time window (±3 days)
        for j in range(i + 1, min(i + 100, len(df_reset))):
            if j in to_drop:
                continue
            
            title_j = str(df_reset.iloc[j]['News Title']).lower()
            date_j = df_reset.iloc[j]['Published Date']
            competitor_j = str(df_reset.iloc[j].get('Competitor', '')).lower()
            
            # Check date proximity (within 3 days)
            try:
                date_diff = abs((date_i - date_j).days) if hasattr(date_i, 'days') else 0
            except:
                date_diff = 0
            
            if date_diff > 3:
                continue
            
            # Extract numbers from second title
            numbers_j = extract_numbers_from_text(title_j)
            
            # Check 1: Fuzzy title similarity
            similarity = SequenceMatcher(None, title_i, title_j).ratio()
            
            # Check 2: Same competitor
            same_competitor = (competitor_i == competitor_j and competitor_i != '')
            
            # Check 3: Similar contract value
            same_value = has_similar_numbers(numbers_i, numbers_j)
            
            # Check 4: Core content match (extract keywords)
            core_match = has_core_content_match(title_i, title_j)
            
            # Decision logic
            is_duplicate = False
            
            # Very high similarity = duplicate
            if similarity > 0.85:
                is_duplicate = True
                logging.debug(f"   High similarity ({similarity:.2f}): {title_j[:50]}...")
            
            # Same competitor + same value + same date = duplicate
            elif same_competitor and same_value and date_diff <= 1:
                is_duplicate = True
                logging.debug(f"   Same value+competitor: {title_j[:50]}...")
            
            # Same competitor + core content match + recent = duplicate
            elif same_competitor and core_match and date_diff <= 2:
                is_duplicate = True
                logging.debug(f"   Core content match: {title_j[:50]}...")
            
            if is_duplicate:
                to_drop.add(j)
    
    fuzzy_drop_count = len(to_drop) - initial_drop_count
    logging.info(f"   Found {fuzzy_drop_count} fuzzy/value duplicates")
    
    logging.info(f"   🗑️ Total removed: {len(to_drop)} duplicates ({len(to_drop)/len(df_reset)*100:.1f}%)")
    
    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers (contract values) from text"""
    # Match patterns like: 35.54, 35,54, 3554, Rs 35 crore, ₹35.6 crore
    numbers = []
    
    # Pattern 1: Direct numbers with crore/lakh
    patterns = [
        r'(?:rs|₹|inr)?\s*(\d+(?:[,.]\d+)*)\s*(?:crore|cr)',
        r'(?:rs|₹|inr)?\s*(\d+(?:[,.]\d+)*)\s*(?:lakh|lac)',
        r'(\d+(?:[,.]\d+)*)\s*(?:million|mn)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize: remove commas, convert to float
            num_str = match.replace(',', '')
            try:
                num = float(num_str)
                # Convert to crore for comparison
                if 'lakh' in text[text.find(match):text.find(match)+50].lower():
                    num = num / 100  # lakh to crore
                elif 'million' in text[text.find(match):text.find(match)+50].lower():
                    num = num * 0.85 / 10  # million to crore (approx)
                numbers.append(num)
            except:
                pass
    
    return numbers


def has_similar_numbers(numbers1: List[float], numbers2: List[float]) -> bool:
    """Check if two lists of numbers have similar values (within 5% tolerance)"""
    if not numbers1 or not numbers2:
        return False
    
    for n1 in numbers1:
        for n2 in numbers2:
            # Check if within 5% of each other
            if n1 > 0 and n2 > 0:
                diff_pct = abs(n1 - n2) / max(n1, n2) * 100
                if diff_pct < 5:  # Within 5%
                    return True
    
    return False


def has_core_content_match(title1: str, title2: str) -> bool:
    """
    Check if two titles have the same core content
    (same keywords, ignoring filler words)
    """
    # Remove common filler words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'must', 'its', 'their',
        'worth', 'order', 'contract', 'project', 'wins', 'bags', 'secures',
        'gets', 'receives', 'awarded', 'adds', 'another', 'growing', 'book'
    }
    
    # Extract keywords from both titles
    words1 = set(re.findall(r'\b\w+\b', title1.lower()))
    words2 = set(re.findall(r'\b\w+\b', title2.lower()))
    
    # Remove stop words
    keywords1 = words1 - stop_words
    keywords2 = words2 - stop_words
    
    # Calculate overlap
    if not keywords1 or not keywords2:
        return False
    
    overlap = len(keywords1 & keywords2)
    total = min(len(keywords1), len(keywords2))
    
    overlap_pct = overlap / total * 100 if total > 0 else 0
    
    # If 60%+ keywords overlap = same content
    return overlap_pct >= 60

# ADD THIS NEW FUNCTION HERE (after line 826)
def calculate_rank_score(row: pd.Series, competitor_tier_map: Dict[str, int]) -> Dict:
    """
    Calculate ranking score for an article
    
    Formula:
    Rank Score = (Category × 50) + (Relevance) + (Competitor Tier × 10) + (Geography × 5) + (Value Tier × 5)
    
    Returns dict with rank_score and component breakdowns
    """
    
    # 1. CATEGORY WEIGHT (0-3) × 50 = 0-150 points
    category = str(row.get('category_tag', '')).lower()
    
    category_weights = {
        'order wins': 3,
        'bidding activity': 3,
        'mergers & acquisitions': 2,
        'partnerships & alliances': 2,
        'project execution': 2,
        'financial': 1,
        'stock market': 1,
    }
    
    category_weight = category_weights.get(category, 0)
    category_points = category_weight * 50
    
    # 2. RELEVANCE SCORE (70-100) = 70-100 points
    relevance_points = int(row.get('relevance_score', 70))
    
    # 3. COMPETITOR TIER (1-3) × 10 = 10-30 points
    competitor_tagging = str(row.get('competitor_tagging', '-'))
    competitors = [c.strip() for c in competitor_tagging.split(',') if c.strip() != '-']
    
    # Get highest tier (Tier 1 is best, so lowest number)
    competitor_tier = 3  # Default to lowest tier
    for comp in competitors:
        tier = competitor_tier_map.get(comp, 3)
        if tier < competitor_tier:
            competitor_tier = tier
    
    # Invert: Tier 1 = 3 points, Tier 2 = 2 points, Tier 3 = 1 point
    competitor_tier_inverted = 4 - competitor_tier
    competitor_points = competitor_tier_inverted * 10
    
    # 4. GEOGRAPHY BONUS (0-2) × 5 = 0-10 points
    geography = str(row.get('geography', '')).lower() if pd.notna(row.get('geography')) else ''
    sbu = str(row.get('sbu_tagging', '')).lower()
    
    geography_bonus = 0
    
    if 'international t&d' in sbu:
        if any(region in geography for region in ['middle east', 'uae', 'saudi', 'qatar', 'bahrain', 'oman', 'kuwait']):
            geography_bonus = 2
        elif any(region in geography for region in ['africa', 'americas', 'saarc']):
            geography_bonus = 1
    elif any(s in sbu for s in ['india t&d', 'transportation', 'civil', 'renewables']):
        if 'india' in geography:
            geography_bonus = 2
    elif 'oil & gas' in sbu or 'oil and gas' in sbu:
        if 'india' in geography or 'middle east' in geography:
            geography_bonus = 2
    
    geography_points = geography_bonus * 5
    
    # 5. VALUE TIER (0-4) × 5 = 0-20 points
    contract_value = row.get('contract_value_inr_crore')
    
    value_tier = 0
    
    if pd.notna(contract_value) and contract_value > 0:
        if category in ['order wins', 'bidding activity']:
            if contract_value >= 1000:
                value_tier = 4
            elif contract_value >= 500:
                value_tier = 3
            elif contract_value >= 100:
                value_tier = 2
            else:
                value_tier = 1
        
        elif category == 'financial':
            if contract_value >= 5000:
                value_tier = 4
            elif contract_value >= 2000:
                value_tier = 3
            elif contract_value >= 500:
                value_tier = 2
            else:
                value_tier = 1
        
        elif category in ['mergers & acquisitions', 'partnerships & alliances']:
            if contract_value >= 500:
                value_tier = 4
            elif contract_value >= 200:
                value_tier = 3
            elif contract_value >= 50:
                value_tier = 2
            else:
                value_tier = 1
        
        elif category == 'project execution':
            if contract_value >= 1000:
                value_tier = 4
            elif contract_value >= 500:
                value_tier = 3
            elif contract_value >= 100:
                value_tier = 2
            else:
                value_tier = 1
    
    value_points = value_tier * 5
    
    # TOTAL RANK SCORE
    total_rank = category_points + relevance_points + competitor_points + geography_points + value_points
    
    return {
        'rank_score': total_rank,
        'competitor_tier': competitor_tier,
        'category_points': category_points,
        'relevance_points': relevance_points,
        'competitor_points': competitor_points,
        'geography_points': geography_points,
        'value_points': value_points
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

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
    
    # Load Excel mapping data
    try:
        excel_data = load_excel_data()
        competitor_tier_map = load_competitor_tiers()
    except Exception as e:
        logging.error(f"❌ Failed to load Excel data: {e}")
        return
    
    # Build dynamic prompt
    logging.info("\n🔧 Building enhanced analysis prompt...")
    full_prompt = build_full_analysis_prompt(
        competitors=excel_data['competitors'],
        categories=excel_data['categories']
    )
    logging.info(f"   ✅ Prompt built with {len(excel_data['competitors'])} competitors and {len(excel_data['categories'])} categories")
    
    # Stage 1: Quick scoring
    df = stage1_quick_scoring(df)
    
    # Stage 2: Full analysis (only high-relevance)
    df = stage2_full_analysis(df, full_prompt, competitor_tier_map)
    
    # Deduplicate high-relevance articles
    high_relevance_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()
    if len(high_relevance_df) > 0:
        high_relevance_df = deduplicate_articles(high_relevance_df)
    
    # Save to processed_articles table (only deduplicated high-relevance)
    logging.info("\n💾 Saving to processed_articles table...")
    save_to_processed_articles(high_relevance_df)
    
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
    logging.info(f"🎯 Final report (after dedup): {len(high_relevance_df)} articles")
    
    if len(high_relevance) > 0:
        logging.info(f"\n📊 Average Relevance Score: {high_relevance['relevance_score'].mean():.1f}")
        
        logging.info(f"\n📁 Top Categories:")
        for cat, count in high_relevance['category_tag'].value_counts().head(5).items():
            logging.info(f"   {cat}: {count}")
        
        logging.info(f"\n📁 Top SBUs:")
        for sbu, count in high_relevance['sbu_tagging'].value_counts().head(5).items():
            logging.info(f"   {sbu}: {count}")
        
        logging.info(f"\n🏢 Top Competitors:")
        for comp, count in high_relevance['competitor_tagging'].value_counts().head(5).items():
            if comp != '-':
                logging.info(f"   {comp}: {count}")
    
    # Cost estimate
    stage1_calls = len(df)
    stage2_calls = len(high_relevance)
    total_calls = stage1_calls + stage2_calls
    est_tokens = (stage1_calls * 200) + (stage2_calls * 7500)  # Updated for detailed prompt
    est_cost = (est_tokens / 1_000_000) * 3.00
    
    logging.info(f"\n💰 API Usage:")
    logging.info(f"   Stage 1 calls: {stage1_calls}")
    logging.info(f"   Stage 2 calls: {stage2_calls}")
    logging.info(f"   Total calls: {total_calls}")
    logging.info(f"   Est. tokens: ~{est_tokens:,}")
    logging.info(f"   Est. cost: ~${est_cost:.2f}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
