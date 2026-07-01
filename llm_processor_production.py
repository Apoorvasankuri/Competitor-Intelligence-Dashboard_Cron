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
from datetime import datetime, date, timedelta
yesterday = date.today() - timedelta(days=1)
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
STAGE1_BATCH_SIZE = 20
STAGE2_BATCH_SIZE = 5
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
    
    query = f"""
        SELECT 
            id,
            published_date,
            news_title,
            competitor,
            sbu,
            source,
            search_keyword,
            link,
            content,
            source_domain,
            source_type,
            source_category,
            source_priority,
            source_authority_score,
            preferred_for_executive_summary,
            source_notes,
            source_match_method,
            search_query_type,
            detected_client_authority,
            detected_strategic_theme,
            search_query,
            accepted_by_gate
        FROM raw_scraped_articles
        WHERE published_date = '{yesterday}'
        ORDER BY published_date DESC
        LIMIT 5000
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

    # Defensive: guarantee source-metadata columns exist even if the raw
    # table predates the source-registry migration or a value is NULL.
    source_defaults = {
        "source_domain": None,
        "source_type": "unknown",
        "source_category": "unknown",
        "source_priority": 8,
        "source_authority_score": 5,
        "preferred_for_executive_summary": False,
        "source_notes": None,
        "source_match_method": "default",
        "search_query_type": "competitor",
        "detected_client_authority": "",
        "detected_strategic_theme": "",
    }
    for col, default_value in source_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["search_query_type"] = df["search_query_type"].fillna("competitor")
    df["detected_client_authority"] = df["detected_client_authority"].fillna("")
    df["detected_strategic_theme"] = df["detected_strategic_theme"].fillna("")

    df["source_type"] = df["source_type"].fillna("unknown")
    df["source_category"] = df["source_category"].fillna("unknown")
    df["source_priority"] = df["source_priority"].fillna(8)
    df["source_authority_score"] = df["source_authority_score"].fillna(5)
    df["preferred_for_executive_summary"] = df["preferred_for_executive_summary"].fillna(False)
    df["source_match_method"] = df["source_match_method"].fillna("default")

    # Search-lens fields added in Change 4 Part B/C (search_query, accepted_by_gate).
    search_lens_defaults = {
        "search_query": None,
        "accepted_by_gate": "",
    }
    for col, default_value in search_lens_defaults.items():
        if col not in df.columns:
            df[col] = default_value
    df["accepted_by_gate"] = df["accepted_by_gate"].fillna("")

    logging.info(f"Loaded source metadata for {df['source_type'].notna().sum()} raw articles")
    logging.info("Search query type distribution: %s",
                 df["search_query_type"].value_counts(dropna=False).to_dict())
    if "accepted_by_gate" in df.columns:
        logging.info("Accepted-by-gate distribution: %s",
                     df["accepted_by_gate"].value_counts(dropna=False).to_dict())

    return df


def save_to_processed_articles(df: pd.DataFrame):
    """Save processed articles to processed_articles table"""
    if df.empty:
        logging.info("No articles to save")
        return
    
    conn = get_db_connection()
    
    # ------------------------------------------------------------------
    # SQL schema update required (run once before deploying this change):
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_domain TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_type TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_category TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_priority INTEGER DEFAULT 8;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_authority_score INTEGER DEFAULT 5;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS preferred_for_executive_summary BOOLEAN DEFAULT FALSE;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_notes TEXT;
    #   ALTER TABLE processed_articles ADD COLUMN IF NOT EXISTS source_match_method TEXT;
    # ------------------------------------------------------------------
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
        rank_score,
        fingerprint,
        is_duplicate,
        source_domain,
        source_type,
        source_category,
        source_priority,
        source_authority_score,
        preferred_for_executive_summary,
        source_notes,
        source_match_method,
        search_query_type,
        detected_client_authority,
        detected_strategic_theme
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s
    )
    ON CONFLICT (link, published_date) DO NOTHING
"""
    
    saved_count = 0
    failed_count = 0
    duplicate_count = 0
    
    for idx, row in df.iterrows():
        try:
            is_dup = row.get('is_duplicate', False)
            if is_dup:
                duplicate_count += 1

            # Convert fingerprint dict to JSON string for storage
            fp = row.get('_fingerprint') or row.get('fingerprint')
            fp_json = json.dumps(fp) if fp and isinstance(fp, dict) else None

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
                fp_json,
                is_dup,
                row.get('source_domain'),
                row.get('source_type', 'unknown'),
                row.get('source_category', 'unknown'),
                row.get('source_priority', 8),
                row.get('source_authority_score', 5),
                row.get('preferred_for_executive_summary', False),
                row.get('source_notes'),
                row.get('source_match_method', 'default'),
                row.get('search_query_type', 'competitor'),
                row.get('detected_client_authority', ''),
                row.get('detected_strategic_theme', ''),
            ))
            conn.commit()
            # Delete from raw after successful save
            cur.execute("DELETE FROM raw_scraped_articles WHERE id = %s", (row.get('id'),))
            conn.commit()
            cur.close()
            saved_count += 1
        except Exception as e:
            conn.rollback()
            failed_count += 1
            logging.error(f"Error saving article '{row.get('News Title', 'Unknown')[:50]}...': {e}")
    
    conn.close()
    
    logging.info(f"✅ Saved {saved_count} articles to processed_articles table")
    logging.info(f"   📌 {duplicate_count} flagged as cross-batch duplicates")
    if failed_count > 0:
        logging.warning(f"⚠️  Failed to save {failed_count} articles")


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

def load_competitor_variations():
    """Load competitor variations from 'Competitor Key Words' column in Excel"""
    
    if not os.path.exists(EXCEL_MAPPING_FILE):
        raise FileNotFoundError(f"❌ {EXCEL_MAPPING_FILE} not found!")
    
    logging.info(f"📂 Loading competitor variations from {EXCEL_MAPPING_FILE}...")
    
    # Read Competitor sheet
    competitor_df = pd.read_excel(EXCEL_MAPPING_FILE, sheet_name='Competitor', header=1)
    
    # Create mapping: variation (lowercase) → official name
    variation_to_official = {}
    official_names = []
    
    for idx, row in competitor_df.iterrows():
        official_name = row.get('Competitor')
        keywords_raw = row.get('Competitor Key Words', '')  # USE THIS COLUMN
        
        if pd.notna(official_name):
            official_names.append(official_name)
            
            # Map official name to itself
            variation_to_official[official_name.lower().strip()] = official_name
            
            # Map all keywords/variations to official name
            if pd.notna(keywords_raw):
                # Extract keywords between quotes
                variations = re.findall(r'"([^"]+)"', str(keywords_raw))
                
                for var in variations:
                    var_clean = var.strip()
                    if var_clean:
                        variation_to_official[var_clean.lower()] = official_name
                        
                        # Also add common case variations
                        variation_to_official[var_clean.upper()] = official_name
                        variation_to_official[var_clean.title()] = official_name
    
    logging.info(f"   ✅ Loaded {len(official_names)} official names with {len(variation_to_official)} total variations")
    
    # Log some examples for verification
    logging.info(f"   📝 Example mappings:")
    for var, official in list(variation_to_official.items())[:5]:
        logging.info(f"      '{var}' → '{official}'")
    
    return {
        'official_names': official_names,
        'variation_map': variation_to_official
    }

def normalize_competitors_to_official(competitor_string: str, variation_map: dict) -> str:
    """
    Normalize competitor names to official names from Excel
    Input: "L&T, Tata Projects Ltd, Kalpataru Power"
    Output: "Larsen & Toubro Limited, Tata Projects Limited, Kalpataru Projects International Limited"
    """
    
    if not competitor_string or competitor_string.strip() in ['-', '', 'None']:
        return '-'
    
    # Split by comma
    competitors = [c.strip() for c in competitor_string.split(',')]
    
    # Normalize each
    normalized = set()  # Use set to avoid duplicates
    
    for comp in competitors:
        comp_lower = comp.lower().strip()
        
        # Look up in variation map
        if comp_lower in variation_map:
            official_name = variation_map[comp_lower]
            normalized.add(official_name)
        else:
            # If not found, check for partial match (fallback)
            found = False
            for var, official in variation_map.items():
                if var in comp_lower or comp_lower in var:
                    normalized.add(official)
                    found = True
                    break
            
            if not found:
                # Keep original if no match (log warning)
                logging.warning(f"   ⚠️ Unknown competitor variation: '{comp}' - keeping as-is")
                normalized.add(comp)
    
    return ", ".join(sorted(normalized)) if normalized else '-'

# ============================================================================
# BUILD DYNAMIC PROMPT (SCRIPT 4 DETAILED VERSION)
# ============================================================================

def build_full_analysis_prompt(categories: List[str]) -> str:
    """Build the full analysis prompt with dynamic data"""
    
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
     • Primary focus geographies: Middle East, Africa, and South East Asia (excluding China)
     • Same technical scope as India T&D but in these focus markets

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
Below are KEC's competitors. Use FUZZY MATCHING to recognize abbreviations, acronyms, partial names, and common variations (e.g. "L&T" → "Larsen & Toubro Limited", "KPTL" → "Kalpataru Projects International Limited", "HCC" → "Hindustan Construction Company Limited", "RVNL" → "Rail Vikas Nigam Limited").

Always return the FULL OFFICIAL NAME exactly as listed below:

- AFCONS Infrastructure Limited
- Ace Pipeline Contracts Private Limited
- Advance Infrastructures Pvt Ltd
- Ahluwalia Contracts (India) Limited
- Al Fanar Group
- Al Sharif Group Holding
- Algihaz Holdings
- Amara Raja Group
- Ashoka Buildcon Limited
- Bajaj Electricals Limited
- Bajel Projects Limited
- Bondada Engineering Limited
- Bridge & Roof Company(India) Limited
- CMEC (China)
- China Southern Power Grid Company Limited
- Corrtech International Limited
- Dilip Buildcon Limited
- Dineshchandra R. Agrawal Infracon Private Limited
- EnProCon Enterprise Limited
- Essens Renewable Private Limited
- Ever Renew Energy Pvt. Ltd.
- Everrenew Energy Private Limited
- GR Infraprojects Limited
- H.G. Infra Engineering Limited
- Hartek Group
- Hindustan Construction Company Limited
- Hitachi Energy India Limited
- Hyosung T&D India Private Limited
- Hyundai Engineering & Construction Co.
- IRCON International Limited
- ISC Projects Private Limited
- J. Kumar Infraprojects Limited
- JSIW Infrastructure Private Limited
- Jackson Electricals & Infrastructure Pvt. Ltd.
- Jackson Green Energy
- Jyoti Structures Limited
- KP Energy Ltd
- KPI Green Energy Limited
- Kalpataru Projects International Limited
- Kernex Microsystems Private Limited
- Kintech-Synergy
- Kiran Infrastructure Private Limited
- Konkan Railway Corporation Limited
- Larsen & Toubro Limited
- Likhitha Infrastructure Limited
- MKC Infrastructure Limited
- Mastek Group
- NCC Limited
- NRP Projects Private Limited
- Offshore Infrastructures Limited
- Oriana Power Limited
- PNC Infratech Limited
- Param Group
- Power Mech Projects Limited
- Pratham Engineering
- Preformed Line Products Company (PLP)
- Rail Vikas Nigam Limited
- RailTel Corporation of India Limited
- Rays Power Infra Limited
- ReNew Energy Global PLC
- Sadel Group
- Sangreen Future Renewables Private Limited
- Saudi Services For Electro Mechanic Works Company Limited
- Shapoorji Pallonji & Company Private Limited
- Siemens Energy India Limited
- Simplex Infrastructures Limited
- Sinohydro Corporation Limited
- Skipper Limited
- Solarworld Energy Solutions Limited
- State Grid Corporation of China
- Sterling and Wilson Renewable Energy Limited
- Sterlite Power Transmission Limited
- Tata Power Solar Systems Limited
- Tata Projects Limited
- Techno Electric & Engineering Company Limited
- Texmaco Rail & Engineering Limited
- Tolahi Projects Private Limited
- Transrail Lighting Limited

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

====================
OUTPUT FORMAT
====================
Return ONLY valid JSON with these exact field names:

{{
  "competitor_tagging": "<comma-separated competitor names from list, or '-'>",
  "sbu_tagging": "<comma-separated SBU names from list, or 'General'>",
  "category_tag": "<single category from list>",
  "contract_value_inr_crore": <numeric value in INR crore, or null if not mentioned>,
  "geography": "<India/Middle East/Africa/South East Asia/Americas/SAARC/Other or null>"
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
  * "South East Asia" → Indonesia, Vietnam, Malaysia, Thailand, Philippines, Singapore, Myanmar, Cambodia, Laos ((excludes China)
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

You will be given a batch of articles. For each, return ONLY its relevance score (0-100).
Return a JSON array of objects with "id" and "score" fields. No explanation."""

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_relevance_score(articles_batch: List[Dict]) -> List[int]:
    """Score a batch of articles in a single API call."""
    
    articles_text = ""
    for article in articles_batch:
        articles_text += f"\n[{article['id']}] Title: {article['title']}\n    Competitor: {article['competitor']}\n"
    
    prompt = f"""Score these {len(articles_batch)} articles for relevance (0-100 each):
{articles_text}

Return ONLY a JSON array like: [{{"id": 1, "score": 85}}, {{"id": 2, "score": 30}}, ...]"""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 25,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": QUICK_SCORE_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens read from cache")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens written to cache")
        
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\[[\s\S]*\]', raw_response)
        if json_match:
            scores_list = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON array found in response")
        
        score_map = {}
        for item in scores_list:
            article_id = item.get('id')
            score = int(item.get('score', 0))
            score_map[article_id] = max(0, min(100, score))
        
        return [score_map.get(article['id'], 0) for article in articles_batch]
        
    except Exception as e:
        logging.warning(f"Batch scoring failed: {e}")
        return [0] * len(articles_batch)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_full_analysis(articles_batch: List[Dict], full_prompt: str) -> List[Dict]:
    """Analyze a batch of articles in a single API call."""
    
    articles_text = ""
    for i, article in enumerate(articles_batch):
        content = article['content'][:2000] if article['content'] else article['title']
        articles_text += f"""
--- ARTICLE {i+1} (relevance: {article['relevance_score']}/100) ---
Title: {article['title']}
Content: {content}
"""
    
    prompt = f"""Analyze these {len(articles_batch)} articles. For EACH article, provide the full analysis.

{articles_text}

Return a JSON array with one object per article (in the same order), each containing:
competitor_tagging, sbu_tagging, category_tag, contract_value_inr_crore, geography

Example format:
[
  {{"competitor_tagging": "L&T", "sbu_tagging": "India T&D", "category_tag": "order wins", "contract_value_inr_crore": 1200, "geography": "India"}},
  {{"competitor_tagging": "Kalpataru", "sbu_tagging": "Renewables", "category_tag": "financial", "contract_value_inr_crore": null, "geography": null}}
]

Return ONLY the JSON array, no other text."""
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 300,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": full_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens read from cache")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens written to cache")
        
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        
        json_match = re.search(r'\[[\s\S]*\]', raw_response)
        if json_match:
            analyses = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON array found in response")
        
        results = []
        for i, analysis in enumerate(analyses):
            if i < len(articles_batch):
                analysis['relevance_score'] = articles_batch[i]['relevance_score']
            
            required = ["competitor_tagging", "sbu_tagging", "category_tag", "contract_value_inr_crore", "geography"]
            for field in required:
                if field not in analysis:
                    analysis[field] = None if field in ['contract_value_inr_crore', 'geography'] else '-'
            
            if analysis.get('contract_value_inr_crore') is not None:
                try:
                    analysis['contract_value_inr_crore'] = float(analysis['contract_value_inr_crore'])
                except:
                    analysis['contract_value_inr_crore'] = None
            
            results.append(analysis)
        
        while len(results) < len(articles_batch):
            idx = len(results)
            results.append({
                "relevance_score": articles_batch[idx]['relevance_score'] if idx < len(articles_batch) else 0,
                "competitor_tagging": "-",
                "sbu_tagging": "None",
                "category_tag": "error",
                "contract_value_inr_crore": None,
                "geography": None
            })
        
        return results
        
    except Exception as e:
        logging.error(f"Batch analysis failed: {e}")
        return [{
            "relevance_score": article.get('relevance_score', 0),
            "competitor_tagging": "-",
            "sbu_tagging": "None",
            "category_tag": "error",
            "contract_value_inr_crore": None,
            "geography": None
        } for article in articles_batch]

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
        required = ["competitor_tagging", "sbu_tagging", "category_tag", "contract_value_inr_crore", "geography"]
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
            "category_tag": "error"
        }

# ============================================================================
# RANKING
# ============================================================================

def safe_source_authority_score(value):
    """Coerce a source_authority_score to a safe int, defaulting to 5."""
    try:
        if value is None:
            return 5
        return int(value)
    except Exception:
        return 5


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

    # 6. SOURCE AUTHORITY (minimal, temporary integration — full ranking redesign comes later)
    #    Better sources add more points; low-authority sources add only a little.
    source_authority_score = safe_source_authority_score(row.get("source_authority_score", 5))

    # TOTAL RANK SCORE
    total_rank = (
        category_points
        + relevance_points
        + competitor_points
        + geography_points
        + value_points
        + source_authority_score
    )

    return {
        'rank_score': total_rank,
        'competitor_tier': competitor_tier,
        'category_points': category_points,
        'relevance_points': relevance_points,
        'competitor_points': competitor_points,
        'geography_points': geography_points,
        'value_points': value_points,
        'source_authority_points': source_authority_score
    }

# ============================================================================
# PIPELINE PROCESSING
# ============================================================================

def stage1_quick_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Batched relevance scoring."""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 1: BATCHED RELEVANCE SCORING (Optimized)")
    logging.info(f"  Batch size: {STAGE1_BATCH_SIZE} articles per API call")
    logging.info("="*60)
    
    relevance_scores = [0] * len(df)
    total = len(df)
    total_batches = (total + STAGE1_BATCH_SIZE - 1) // STAGE1_BATCH_SIZE
    
    # Prepare all batches
    all_batches = []
    for i in range(0, total, STAGE1_BATCH_SIZE):
        batch_df = df.iloc[i:i+STAGE1_BATCH_SIZE]
        articles_batch = []
        for local_idx, (df_idx, row) in enumerate(batch_df.iterrows()):
            articles_batch.append({
                'id': local_idx + 1,
                'title': str(row['News Title']),
                'competitor': str(row.get('Competitor', ''))
            })
        all_batches.append((i, articles_batch))
    
    # Run batches in parallel (4 concurrent)
    logging.info(f"📊 Scoring {len(all_batches)} batches (4 concurrent)...")
    
    def score_batch(batch_tuple):
        start_idx, articles_batch = batch_tuple
        return start_idx, batch_relevance_score(articles_batch)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(score_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            start_idx, batch_scores = future.result()
            batch_df = df.iloc[start_idx:start_idx+STAGE1_BATCH_SIZE]
            for j, score in enumerate(batch_scores):
                relevance_scores[start_idx + j] = score
                if score >= RELEVANCE_THRESHOLD:
                    title = str(batch_df.iloc[j]['News Title'])
                    logging.info(f"   ✅ Score {score}: {title[:60]}...")
    
    df['relevance_score'] = relevance_scores
    
    high_relevance = df[df['relevance_score'] >= RELEVANCE_THRESHOLD]
    
    logging.info(f"\n📈 Stage 1 Complete:")
    logging.info(f"   Total articles: {len(df)}")
    logging.info(f"   API calls made: {total_batches} (was {len(df)} before optimization)")
    logging.info(f"   High relevance (≥{RELEVANCE_THRESHOLD}): {len(high_relevance)} ({len(high_relevance)/len(df)*100:.1f}%)")
    logging.info(f"   Will proceed to full analysis: {len(high_relevance)} articles")
    
    return df

def stage2_full_analysis(df: pd.DataFrame, full_prompt: str, competitor_tier_map: Dict[str, int]) -> pd.DataFrame:
    """Stage 2: Batched full analysis with prompt caching."""
    
    logging.info("\n" + "="*60)
    logging.info("STAGE 2: BATCHED FULL ANALYSIS (Optimized)")
    logging.info(f"  Batch size: {STAGE2_BATCH_SIZE} articles per API call")
    logging.info(f"  System prompt caching: ENABLED")
    logging.info("="*60)
    
    competitor_data = load_competitor_variations()
    variation_map = competitor_data['variation_map']
    
    high_rel_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()
    
    if len(high_rel_df) == 0:
        logging.warning("No articles meet relevance threshold. Skipping Stage 2.")
        return df
    
    df['competitor_tagging'] = '-'
    df['sbu_tagging'] = 'None'
    df['category_tag'] = 'not_analyzed'
    df['summary'] = 'Not analyzed (low relevance)'
    df['scraped_content'] = ''
    df['rank_score'] = 0
    df['competitor_tier'] = 3
    df['contract_value_inr_crore'] = None
    df['geography'] = None
    
    total = len(high_rel_df)
    total_batches = (total + STAGE2_BATCH_SIZE - 1) // STAGE2_BATCH_SIZE
    
    logging.info(f"\n📥 Scraping {total} articles in parallel...")
    contents = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(scrape_article, row['Link']): idx
            for idx, row in high_rel_df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            contents[idx] = future.result()
    
    logging.info(f"   ✅ Scraped {len(contents)} articles")
    
    high_rel_indices = list(high_rel_df.index)
    
    # Prepare all batches
    all_batches = []
    for i in range(0, total, STAGE2_BATCH_SIZE):
        batch_indices = high_rel_indices[i:i+STAGE2_BATCH_SIZE]
        articles_batch = []
        for idx in batch_indices:
            row = df.loc[idx]
            articles_batch.append({
                'title': str(row['News Title']),
                'content': contents.get(idx, ''),
                'relevance_score': row['relevance_score']
            })
        all_batches.append((batch_indices, articles_batch))
    
    # Run batches in parallel (3 concurrent)
    logging.info(f"🔍 Analyzing {len(all_batches)} batches (3 concurrent)...")
    
    def analyze_batch(batch_tuple):
        batch_indices, articles_batch = batch_tuple
        return batch_indices, batch_full_analysis(articles_batch, full_prompt)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            batch_indices, batch_results = future.result()
            for j, (idx, analysis) in enumerate(zip(batch_indices, batch_results)):
                raw_competitors = analysis.get('competitor_tagging', '-')
                official_competitors = normalize_competitors_to_official(raw_competitors, variation_map)
                
                df.at[idx, 'competitor_tagging'] = official_competitors
                df.at[idx, 'sbu_tagging'] = analysis.get('sbu_tagging', 'None')
                df.at[idx, 'category_tag'] = analysis.get('category_tag', 'not_analyzed')
                df.at[idx, 'scraped_content'] = (contents.get(idx, ''))[:500]
                df.at[idx, 'contract_value_inr_crore'] = analysis.get('contract_value_inr_crore')
                df.at[idx, 'geography'] = analysis.get('geography')
    
    logging.info(f"\n📊 Calculating ranking scores...")
    for idx in high_rel_indices:
        rank_data = calculate_rank_score(df.loc[idx], competitor_tier_map)
        df.at[idx, 'rank_score'] = rank_data['rank_score']
        df.at[idx, 'competitor_tier'] = rank_data['competitor_tier']
        
        logging.info(f"   Rank {rank_data['rank_score']}: {df.loc[idx, 'News Title'][:60]}...")
    
    logging.info(f"\n✅ Stage 2 Complete:")
    logging.info(f"   Articles analyzed: {len(high_rel_df)}")
    logging.info(f"   API calls made: {total_batches} (was {len(high_rel_df)} before optimization)")
    
    return df

# ============================================================================
# DEDUPLICATION - PHASE 1: STRING-BASED (FAST)
# ============================================================================

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all contract values from text, handling Indian number formats"""
    numbers = []

    # Handle Indian format: Rs 35,54,82,968 → convert to crore
    indian_format = re.findall(r'(?:rs|₹|inr)\.?\s*([\d,]+)', text, re.IGNORECASE)
    for match in indian_format:
        num_str = match.replace(',', '')
        try:
            num = float(num_str)
            # If number looks like full rupees (>10 million), convert to crore
            if num > 10_000_000:
                num = num / 10_000_000
            elif num > 100_000:
                num = num / 10_000_000
            numbers.append(round(num, 2))
        except:
            pass

    # Handle crore/lakh/million explicitly stated
    patterns = [
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:crore|cr)', 1.0),
        (r'(?:rs|₹|inr)?\.?\s*(\d+(?:[,.]\d+)*)\s*(?:lakh|lac)', 0.01),
        (r'(\d+(?:[,.]\d+)*)\s*(?:million|mn)', 8.5),
    ]

    for pattern, multiplier in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                num = float(match.replace(',', '')) * multiplier
                numbers.append(round(num, 2))
            except:
                pass

    return numbers


def has_similar_numbers(numbers1: List[float], numbers2: List[float]) -> bool:
    """Check if two lists of numbers have similar values (within 10% tolerance)"""
    if not numbers1 or not numbers2:
        return False
    for n1 in numbers1:
        for n2 in numbers2:
            if n1 > 0 and n2 > 0:
                diff_pct = abs(n1 - n2) / max(n1, n2) * 100
                if diff_pct < 10:
                    return True
    return False


def has_core_content_match(title1: str, title2: str) -> bool:
    """Check if two titles share core content keywords"""
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'must', 'its', 'their',
        'worth', 'order', 'contract', 'project', 'wins', 'bags', 'secures',
        'gets', 'receives', 'awarded', 'adds', 'another', 'growing', 'book'
    }
    words1 = set(re.findall(r'\b\w+\b', title1.lower())) - stop_words
    words2 = set(re.findall(r'\b\w+\b', title2.lower())) - stop_words
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    total = min(len(words1), len(words2))
    return (overlap / total * 100) >= 60 if total > 0 else False


def phase1_string_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Fast string-based deduplication.
    Catches exact matches, fuzzy matches, and value+competitor matches.
    """
    logging.info("\n🔍 Phase 1: String-based deduplication...")

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
        else:
            seen_titles[title] = i

    exact_count = len(to_drop)
    logging.info(f"   Exact duplicates: {exact_count}")

    # Strategy 2: Fuzzy + value + core content matching
    for i in range(len(df_reset)):
        if i in to_drop:
            continue

        title_i = str(df_reset.iloc[i]['News Title']).lower()
        date_i = df_reset.iloc[i]['Published Date']
        competitor_i = str(df_reset.iloc[i].get('competitor_tagging') or df_reset.iloc[i].get('Competitor') or '').lower()
        numbers_i = extract_numbers_from_text(title_i)

        for j in range(i + 1, min(i + 100, len(df_reset))):
            if j in to_drop:
                continue

            title_j = str(df_reset.iloc[j]['News Title']).lower()
            date_j = df_reset.iloc[j]['Published Date']
            competitor_j = str(df_reset.iloc[j].get('competitor_tagging') or df_reset.iloc[j].get('Competitor') or '').lower()

            try:
                date_diff = abs((date_i - date_j).days)
            except:
                date_diff = 0

            if date_diff > 3:
                continue

            numbers_j = extract_numbers_from_text(title_j)
            similarity = SequenceMatcher(None, title_i, title_j).ratio()
            same_competitor = (competitor_i == competitor_j and competitor_i not in ('', '-'))
            same_value = has_similar_numbers(numbers_i, numbers_j)
            core_match = has_core_content_match(title_i, title_j)

            is_duplicate = False
            if similarity > 0.85:
                is_duplicate = True
            elif same_competitor and same_value and date_diff <= 1:
                is_duplicate = True
            elif same_competitor and core_match and date_diff <= 2:
                is_duplicate = True

            if is_duplicate:
                to_drop.add(j)

    fuzzy_count = len(to_drop) - exact_count
    logging.info(f"   Fuzzy/value duplicates: {fuzzy_count}")
    logging.info(f"   Phase 1 total removed: {len(to_drop)} | Remaining: {len(df_reset) - len(to_drop)}")

    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


# ============================================================================
# DEDUPLICATION - PHASE 2: LLM-BASED (SEMANTIC)
# ============================================================================

# Category-specific fields to extract for fingerprinting
CATEGORY_FINGERPRINT_FIELDS = {
    "order wins":               ["company", "client_or_authority", "contract_value_crore", "scope", "location"],
    "bidding activity":         ["companies_bidding", "client_or_authority", "project_value_crore", "scope", "location"],
    "financial":                ["company", "period", "revenue_crore", "profit_crore", "order_book_crore"],
    "project execution":        ["company", "project_name", "capacity_or_scale", "location", "milestone"],
    "mergers & acquisitions":   ["acquirer", "target_company", "deal_value_crore", "stake_percent"],
    "partnerships & alliances": ["companies_involved", "deal_type", "sector", "value_crore"],
    "stock market":             ["company", "price_movement_percent", "trigger_event"],
    "regulatory & policy":      ["authority", "policy_or_rule", "sector_affected"],
    "industry trends":          ["topic", "key_stat", "geography"],
    "legal & disputes":         ["company", "counterparty", "issue_type", "value_crore"],
}

FINGERPRINT_SYSTEM_PROMPT = """You are a news deduplication assistant for a competitive intelligence system.

Your job is to read a news article and extract a structured fingerprint of the KEY FACTS that identify this specific event.
The fingerprint will be used to detect if multiple articles are reporting the same underlying news event.

Extract ONLY facts explicitly stated in the article. Use null for anything not mentioned.
Return ONLY valid JSON, no explanation."""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def extract_fingerprint(title: str, content: str, category: str) -> Dict:
    """Extract a semantic fingerprint from an article based on its category"""

    fields = CATEGORY_FINGERPRINT_FIELDS.get(
        category.lower(),
        ["company", "event_type", "value", "location"]  # fallback
    )

    fields_desc = "\n".join([f'  "{f}": <extracted value or null>' for f in fields])

    prompt = f"""Article Title: {title}

Article Content: {content[:2000] if content else 'Not available'}

Category: {category}

Extract the following key facts from this article and return as JSON:
{{
{fields_desc}
}}

Rules:
- Extract ONLY facts explicitly stated
- For values/numbers: normalize to crore (e.g. Rs 35,54,82,968 = 35.55 crore, Rs 1200 crore = 1200)
- For company names: use the most common/standard form
- For scope: include MW/km/units as mentioned
- Use null if not mentioned"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": FINGERPRINT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group(0))
        return {}

    except Exception as e:
        logging.warning(f"Fingerprint extraction failed for '{title[:50]}': {e}")
        return {}

def check_fingerprint_against_db(fingerprint: Dict, category: str, competitor: str, published_date, lookback_days: int = 7) -> Dict:
    """
    Check if a fingerprint matches any existing article in the database.
    Returns {"is_duplicate": True/False, "matched_article_id": id or None}
    """
    if not fingerprint:
        return {"is_duplicate": False, "matched_article_id": None}

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get recent articles with fingerprints in the same category/competitor group
        cur.execute("""
            SELECT id, news_title, fingerprint, category_tag, competitor_tagging, published_date
            FROM processed_articles
            WHERE fingerprint IS NOT NULL
            AND category_tag = %s
            AND published_date >= %s - INTERVAL '%s days'
            AND is_duplicate = FALSE
            ORDER BY published_date DESC
            LIMIT 50
        """, (category, published_date, lookback_days))

        existing = cur.fetchall()
        cur.close()
        conn.close()

        for row in existing:
            existing_fp = row.get('fingerprint')
            if not existing_fp:
                continue

            # Parse if stored as string
            if isinstance(existing_fp, str):
                try:
                    existing_fp = json.loads(existing_fp)
                except:
                    continue

            if fingerprints_match(fingerprint, existing_fp, category):
                logging.info(f"   🔄 Cross-batch duplicate found! Matches article #{row['id']}: {row['news_title'][:60]}...")
                return {"is_duplicate": True, "matched_article_id": row['id']}

        return {"is_duplicate": False, "matched_article_id": None}

    except Exception as e:
        logging.warning(f"   ⚠️ DB fingerprint check failed: {e}")
        return {"is_duplicate": False, "matched_article_id": None}
    
def fingerprints_match(fp1: Dict, fp2: Dict, category: str) -> bool:
    """
    Compare two fingerprints to determine if they represent the same event.
    Uses category-specific matching logic.
    """
    if not fp1 or not fp2:
        return False

    cat = category.lower()

    def normalize(val):
        if val is None:
            return None
        return str(val).lower().strip()

    def values_similar(v1, v2, tolerance=0.10):
        try:
            n1, n2 = float(v1), float(v2)
            if n1 == 0 and n2 == 0:
                return True
            if n1 == 0 or n2 == 0:
                return False
            return abs(n1 - n2) / max(n1, n2) <= tolerance
        except:
            return False

    def company_match(c1, c2):
        if not c1 or not c2:
            return False
        c1, c2 = normalize(c1), normalize(c2)
        if c1 == c2 or c1 in c2 or c2 in c1:
            return True
        return SequenceMatcher(None, c1, c2).ratio() > 0.80

    if cat == "order wins":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        if not company_ok:
            return False
        v1, v2 = fp1.get('contract_value_crore'), fp2.get('contract_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        client_ok = company_match(fp1.get('client_or_authority'), fp2.get('client_or_authority'))
        scope1, scope2 = normalize(fp1.get('scope')), normalize(fp2.get('scope'))
        scope_ok = (scope1 and scope2 and (scope1 in scope2 or scope2 in scope1)) or (not scope1 or not scope2)
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        return company_ok and value_ok and (client_ok or scope_ok) and location_ok

    elif cat == "bidding activity":
        client_ok = company_match(fp1.get('client_or_authority'), fp2.get('client_or_authority'))
        v1, v2 = fp1.get('project_value_crore'), fp2.get('project_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        return client_ok and value_ok and location_ok

    elif cat == "financial":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        period_ok = normalize(fp1.get('period')) == normalize(fp2.get('period')) or not fp1.get('period') or not fp2.get('period')
        v1, v2 = fp1.get('revenue_crore'), fp2.get('revenue_crore')
        revenue_ok = values_similar(v1, v2) if (v1 and v2) else True
        return company_ok and period_ok and revenue_ok

    elif cat == "mergers & acquisitions":
        acquirer_ok = company_match(fp1.get('acquirer'), fp2.get('acquirer'))
        target_ok = company_match(fp1.get('target_company'), fp2.get('target_company'))
        v1, v2 = fp1.get('deal_value_crore'), fp2.get('deal_value_crore')
        value_ok = values_similar(v1, v2) if (v1 and v2) else True
        # If both companies null, fall back to competitor group match
        if not fp1.get('acquirer') and not fp1.get('target_company'):
            return True  # Same competitor group + same category = duplicate
        return (acquirer_ok or target_ok) and value_ok
    
    elif cat == "partnerships & alliances":
        companies1 = normalize(fp1.get('companies_involved') or '')
        companies2 = normalize(fp2.get('companies_involved') or '')
        companies_ok = SequenceMatcher(None, companies1, companies2).ratio() > 0.70 if companies1 and companies2 else False
        sector_ok = normalize(fp1.get('sector')) == normalize(fp2.get('sector')) or not fp1.get('sector') or not fp2.get('sector')
        return companies_ok and sector_ok

    elif cat == "project execution":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        location_ok = normalize(fp1.get('location')) == normalize(fp2.get('location')) or not fp1.get('location') or not fp2.get('location')
        scale1, scale2 = normalize(fp1.get('capacity_or_scale')), normalize(fp2.get('capacity_or_scale'))
        scale_ok = (scale1 and scale2 and SequenceMatcher(None, scale1, scale2).ratio() > 0.70) or not scale1 or not scale2
        return company_ok and location_ok and scale_ok

    elif cat == "stock market":
        company_ok = company_match(fp1.get('company'), fp2.get('company'))
        trigger1, trigger2 = normalize(fp1.get('trigger_event')), normalize(fp2.get('trigger_event'))
        trigger_ok = (trigger1 and trigger2 and SequenceMatcher(None, trigger1, trigger2).ratio() > 0.70) or not trigger1 or not trigger2
        return company_ok and trigger_ok

    else:
        # Generic: fuzzy match on all string fields
        matches = 0
        total = 0
        for key in fp1:
            if fp1[key] and fp2.get(key):
                total += 1
                if normalize(fp1[key]) == normalize(fp2[key]) or SequenceMatcher(None, normalize(fp1[key]), normalize(fp2[key])).ratio() > 0.75:
                    matches += 1
        return (matches / total) >= 0.6 if total > 0 else False


def phase2_llm_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: LLM-based semantic deduplication.
    Groups articles by competitor, extracts fingerprints, compares within each group.
    Keeps the article with the highest rank_score from each duplicate group.
    """
    logging.info("\n🤖 Phase 2: LLM semantic deduplication...")

    if df.empty or len(df) <= 1:
        return df

    if 'rank_score' not in df.columns:
        df['rank_score'] = 0

    df_reset = df.reset_index(drop=True)

    # Step 1: Scrape articles in parallel for content
    logging.info(f"   📥 Scraping content for {len(df_reset)} articles...")

    contents = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(scrape_article, row['Link']): idx
            for idx, row in df_reset.iterrows()
            if pd.notna(row.get('Link'))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            contents[idx] = future.result()

    # Step 2: Extract fingerprints via LLM
    logging.info(f"   🔑 Extracting fingerprints...")
    fingerprints = {}
    for idx, row in df_reset.iterrows():
        title = str(row['News Title'])
        content = contents.get(idx, str(row.get('scraped_content', '')))
        category = str(row.get('category_tag', 'order wins'))

        fp = extract_fingerprint(title, content, category)
        fingerprints[idx] = fp

        logging.info(f"   [{idx+1}/{len(df_reset)}] {title[:60]}...")
        time.sleep(RATE_LIMIT_DELAY)
    # Store fingerprints on DataFrame for later summary generation
    df_reset['_fingerprint'] = df_reset.index.map(lambda i: fingerprints.get(i, {}))

    # Step 2.5: Cross-batch dedup — check each fingerprint against DB history
    logging.info("\n   🗄️ Checking fingerprints against database history...")
    df_reset['is_duplicate'] = False
    cross_batch_dupes = 0

    for idx, row in df_reset.iterrows():
        fp = fingerprints.get(idx, {})
        if not fp:
            continue

        category = str(row.get('category_tag', '')).lower()
        competitor = str(row.get('competitor_tagging', '-'))
        pub_date = row.get('Published Date')

        result = check_fingerprint_against_db(fp, category, competitor, pub_date)
        if result['is_duplicate']:
            df_reset.at[idx, 'is_duplicate'] = True
            cross_batch_dupes += 1

    logging.info(f"   🗄️ Cross-batch duplicates found: {cross_batch_dupes}")

    # Step 3: Group by competitor, compare fingerprints within each group
    logging.info("\n   🔍 Comparing fingerprints within competitor groups...")

    to_drop = set()
    competitor_groups: Dict[str, List[int]] = {}

    for idx, row in df_reset.iterrows():
        competitor_raw = str(row.get('competitor_tagging') or row.get('Competitor') or 'General')
        competitors = [c.strip() for c in competitor_raw.split(',') if c.strip() and c.strip() != '-']
        if not competitors:
            competitors = ['General']
        for comp in competitors:
            if comp not in competitor_groups:
                competitor_groups[comp] = []
            competitor_groups[comp].append(idx)

    logging.info(f"   📦 {len(competitor_groups)} competitor groups to compare...")

    already_compared = set()

    for comp, indices in competitor_groups.items():
        if len(indices) <= 1:
            continue

        logging.info(f"   👥 {comp}: {len(indices)} articles")

        for i in range(len(indices)):
            idx_i = indices[i]
            if idx_i in to_drop:
                continue

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                if idx_j in to_drop:
                    continue

                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in already_compared:
                    continue
                already_compared.add(pair)

                row_i = df_reset.iloc[idx_i]
                row_j = df_reset.iloc[idx_j]

                # Only compare articles within 3 days of each other
                try:
                    date_diff = abs((row_i['Published Date'] - row_j['Published Date']).days)
                except:
                    date_diff = 0

                cat_i = str(row_i.get('category_tag', '')).lower()
                cat_j = str(row_j.get('category_tag', '')).lower()

                wide_window_cats = {'mergers & acquisitions', 'partnerships & alliances', 'legal & disputes'}
                max_days = 5 if cat_i in wide_window_cats else 3
                if date_diff > max_days:
                    continue
                if cat_i != cat_j:
                    continue

                fp_i = fingerprints.get(idx_i, {})
                fp_j = fingerprints.get(idx_j, {})

                if fingerprints_match(fp_i, fp_j, cat_i):
                    # Keep higher rank_score, drop the other
                    score_i = float(row_i.get('rank_score') or 0)
                    score_j = float(row_j.get('rank_score') or 0)

                    drop_idx = idx_j if score_i >= score_j else idx_i
                    to_drop.add(drop_idx)

                    keep_title = row_i['News Title'] if score_i >= score_j else row_j['News Title']
                    drop_title = row_j['News Title'] if score_i >= score_j else row_i['News Title']
                    logging.info(f"   🗑️  DUPLICATE: '{drop_title[:60]}...'")
                    logging.info(f"       KEEPING:   '{keep_title[:60]}...'")

    logging.info(f"\n   Phase 2 removed: {len(to_drop)} semantic duplicates")
    logging.info(f"   Final article count: {len(df_reset) - len(to_drop)}")

    return df_reset.drop(index=list(to_drop)).reset_index(drop=True)


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-phase deduplication:
    Phase 1 — Fast string-based (exact, fuzzy, value matching)
    Phase 2 — LLM semantic deduplication grouped by competitor
    """
    logging.info("\n" + "="*60)
    logging.info("DEDUPLICATION: TWO-PHASE")
    logging.info("="*60)

    initial_count = len(df)
    logging.info(f"   Starting with: {initial_count} articles")

    # Phase 1: String-based
    df = phase1_string_dedup(df)
    after_phase1 = len(df)
    logging.info(f"   After Phase 1: {after_phase1} articles ({initial_count - after_phase1} removed)")

    # Phase 2: LLM semantic
    df = phase2_llm_dedup(df)
    after_phase2 = len(df)
    logging.info(f"   After Phase 2: {after_phase2} articles ({after_phase1 - after_phase2} removed)")

    logging.info(f"\n✅ Dedup complete: {initial_count} → {after_phase2} articles ({initial_count - after_phase2} total removed)")

    return df

# ============================================================================
# LLM SUMMARY GENERATION
# ============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst for KEC International, an infrastructure EPC company.

Your job is to write concise 2-3 sentence executive summaries of competitor news articles.

Structure each summary as:
- Sentence 1: Who did what (the core event, using the competitor's exact full name)
- Sentence 2: Scale and context (contract value in ₹, geography, project scope/specs)
- Sentence 3: Strategic implication for KEC (which SBU is affected, competitive threat)

Rules:
- Use the EXACT competitor name from the "Competitor" field — if "-" or empty, infer from content
- NEVER write "-" or "Unknown" as a company name
- Be specific: include ₹ values, MW/km figures, location names wherever available
- Anchor on the pre-extracted facts (fingerprint) first, use raw content only to add colour
- Keep it under 60 words total
- Write in third person, present tense
- No filler phrases like "it is worth noting" or "this highlights"
- Return ONLY a JSON array of strings, no explanation, no markdown"""

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def batch_generate_summaries(articles_batch: List[Dict]) -> List[str]:
    """Generate rich 2-3 sentence LLM summaries for a batch of articles."""

    articles_text = ""
    for i, article in enumerate(articles_batch):
        content = article.get('content', '')[:1500] or article.get('title', '')

        # Format fingerprint as clean key: value lines, skipping nulls
        fp = article.get('fingerprint', {})
        fp_text = ''
        if fp and isinstance(fp, dict):
            fp_lines = []
            for k, v in fp.items():
                if v is not None:
                    if isinstance(v, list):
                        v = ', '.join(str(x) for x in v)
                    fp_lines.append(f"  {k}: {v}")
            if fp_lines:
                fp_text = "Pre-extracted facts:\n" + "\n".join(fp_lines)

        articles_text += f"""
--- ARTICLE {i+1} ---
Title: {article['title']}
Competitor: {article['competitor_tagging']}
SBU: {article['sbu_tagging']}
Category: {article['category_tag']}
Geography: {article.get('geography') or 'Not specified'}
Contract Value (INR Crore): {article.get('contract_value_inr_crore') or 'Not specified'}
{fp_text}
Raw content: {content}
"""
    prompt = f"""Write a 2-3 sentence executive summary for each of these {len(articles_batch)} articles.

{articles_text}

Return a JSON array of strings, one summary per article, in the same order:
["Summary for article 1.", "Summary for article 2.", ...]

Remember:
- Use the exact competitor name from the "Competitor" field
- Anchor on the pre-extracted facts first
- Under 60 words per summary"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=len(articles_batch) * 150,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SUMMARY_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Log cache usage
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
        if cache_read > 0:
            logging.info(f"   💾 Cache HIT: {cache_read} tokens")
        elif cache_create > 0:
            logging.info(f"   💾 Cache WRITE: {cache_create} tokens")

        # Strip markdown fences
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        json_match = re.search(r'\[[\s\S]*\]', raw)
        if json_match:
            summaries = json.loads(json_match.group(0))
            # Pad with title fallbacks if model returned fewer than expected
            while len(summaries) < len(articles_batch):
                summaries.append(articles_batch[len(summaries)]['title'])
            return [str(s) for s in summaries[:len(articles_batch)]]

        raise ValueError("No JSON array found in response")

    except Exception as e:
        logging.error(f"Batch summary generation failed: {e}")
        return [a['title'] for a in articles_batch]


def generate_llm_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates batched LLM summary generation using fingerprints + scraped content.
    Call this BEFORE dropping the _fingerprint column.
    Runs 3 batches concurrently with prompt caching.
    """
    logging.info("\n📝 Generating LLM summaries (batched, fingerprint-anchored)...")

    if df.empty:
        return df

    SUMMARY_BATCH_SIZE = 5
    all_indices = list(df.index)
    total = len(all_indices)
    total_batches = (total + SUMMARY_BATCH_SIZE - 1) // SUMMARY_BATCH_SIZE

    logging.info(f"   Articles: {total} | Batches: {total_batches} | Concurrency: 3")

    # Build batches — include fingerprint from _fingerprint column
    all_batches = []
    for i in range(0, total, SUMMARY_BATCH_SIZE):
        batch_indices = all_indices[i:i + SUMMARY_BATCH_SIZE]
        articles_batch = []
        for idx in batch_indices:
            row = df.loc[idx]
            content = str(row.get('scraped_content', ''))
            fingerprint = row.get('_fingerprint', {})

            # If no content and no fingerprint, use title directly — skip LLM
            if not content.strip() and not fingerprint:
                df.at[idx, 'summary'] = str(row.get('News Title', ''))
                continue

            articles_batch.append({
                'title': str(row.get('News Title', '')),
                'competitor_tagging': str(row.get('competitor_tagging', '-')),
                'sbu_tagging': str(row.get('sbu_tagging', 'General')),
                'category_tag': str(row.get('category_tag', '')),
                'geography': row.get('geography'),
                'contract_value_inr_crore': row.get('contract_value_inr_crore'),
                'content': content,
                'fingerprint': fingerprint
            })        
    
        all_batches.append((batch_indices, articles_batch))

    def run_batch(batch_tuple):
        batch_indices, articles_batch = batch_tuple
        summaries = batch_generate_summaries(articles_batch)
        return batch_indices, summaries

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_batch, b): b for b in all_batches}
        for future in as_completed(futures):
            batch_indices, summaries = future.result()
            for idx, summary in zip(batch_indices, summaries):
                df.at[idx, 'summary'] = summary
                logging.info(f"   ✅ {str(df.loc[idx, 'News Title'])[:50]}...")
                logging.info(f"      → {summary[:80]}...")

    logging.info(f"   ✅ Done: {total} summaries in {total_batches} API calls")
    return df


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
        competitor_data = load_competitor_variations()
    except Exception as e:
        logging.error(f"❌ Failed to load Excel data: {e}")
        return
    
    # Build dynamic prompt
    logging.info("\n🔧 Building enhanced analysis prompt...")
    full_prompt = build_full_analysis_prompt(
        categories=excel_data['categories']
    )
    logging.info(f"   ✅ Prompt built with {len(excel_data['competitors'])} competitors and {len(excel_data['categories'])} categories")

    # Stage 1: Quick scoring
    df = stage1_quick_scoring(df)

    # Stage 2: Full analysis (only high-relevance)
    df = stage2_full_analysis(df, full_prompt, competitor_tier_map)

    # Two-phase deduplication on high-relevance articles
    high_relevance_df = df[df['relevance_score'] >= RELEVANCE_THRESHOLD].copy()
    if len(high_relevance_df) > 0:
        non_dupes = high_relevance_df[high_relevance_df.get('is_duplicate', False) == False].copy()
        dupes = high_relevance_df[high_relevance_df.get('is_duplicate', False) == True].copy()

        if len(non_dupes) > 0:
            non_dupes = generate_llm_summaries(non_dupes)

        if len(dupes) > 0:
            dupes['summary'] = dupes['News Title']

        high_relevance_df = pd.concat([non_dupes, dupes], ignore_index=True)
        high_relevance_df = high_relevance_df.drop(columns=['_fingerprint'], errors='ignore')

    # Save to processed_articles table (only deduplicated high-relevance)

    # Save to processed_articles table (only deduplicated high-relevance)
    logging.info("\n💾 Saving to processed_articles table...")
    save_to_processed_articles(high_relevance_df)

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
    dedup_calls = len(high_relevance_df)
    total_calls = stage1_calls + stage2_calls + dedup_calls
    est_tokens = (stage1_calls * 200) + (stage2_calls * 7500) + (dedup_calls * 500)
    est_cost = (est_tokens / 1_000_000) * 3.00

    logging.info(f"\n💰 API Usage:")
    logging.info(f"   Stage 1 calls: {stage1_calls}")
    logging.info(f"   Stage 2 calls: {stage2_calls}")
    logging.info(f"   Dedup fingerprint calls: {dedup_calls}")
    logging.info(f"   Total calls: {total_calls}")
    logging.info(f"   Est. tokens: ~{est_tokens:,}")
    logging.info(f"   Est. cost: ~${est_cost:.2f}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
