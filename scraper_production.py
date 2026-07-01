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
from urllib.parse import quote, urlparse
import os
import logging
import random
import re
import pandas as pd
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
LOOKBACK_DAYS = 15
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent requests
REQUEST_DELAY = 1  # Delay between requests in seconds
EXCEL_FILE_PATH = 'SBU_Competitor_Mapping.xlsx'
# Rotate User-Agents to avoid fingerprinting
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


# ============================================================
# Source Registry
# Classifies article sources by authority and reliability.
# Supports both Google News RSS publisher display names (source_names)
# and real publisher domains (domains).
# Lower source_priority = better source.
# Higher source_authority_score = more trustworthy source.
# ============================================================
SOURCE_REGISTRY = [
    {
        "source_type": "official_exchange",
        "source_category": "official_disclosure",
        "source_names": ["BSE India", "NSE India", "BSE", "NSE"],
        "domains": ["bseindia.com", "nseindia.com"],
        "source_priority": 1,
        "source_authority_score": 60,
        "preferred_for_executive_summary": True,
        "source_notes": "Official exchange filings and corporate disclosures"
    },
    {
        "source_type": "government_policy",
        "source_category": "policy_and_regulatory",
        "source_names": [
            "Press Information Bureau", "PIB", "Ministry of Power",
            "Ministry of Railways", "Ministry of Road Transport and Highways",
            "MNRE", "CEA", "CERC", "NITI Aayog"
        ],
        "domains": [
            "pib.gov.in", "powermin.gov.in", "railways.gov.in", "morth.nic.in",
            "mnre.gov.in", "cea.nic.in", "cercind.gov.in", "niti.gov.in"
        ],
        "source_priority": 1,
        "source_authority_score": 58,
        "preferred_for_executive_summary": True,
        "source_notes": "Government ministries, regulators, and policy sources"
    },
    {
        "source_type": "client_or_authority",
        "source_category": "primary_project_authority",
        "source_names": [
            "Power Grid Corporation of India", "POWERGRID", "PGCIL", "NTPC",
            "NHAI", "SECI", "DMRC", "Delhi Metro Rail Corporation",
            "Indian Railways", "NHSRCL", "GAIL", "ONGC", "Indian Oil", "BPCL", "HPCL"
        ],
        "domains": [
            "powergrid.in", "pgcilindia.com", "ntpc.co.in", "nhai.gov.in",
            "seci.co.in", "dmrc.org", "indianrailways.gov.in", "nhsrcl.in",
            "gailonline.com", "ongcindia.com", "iocl.com",
            "bharatpetroleum.in", "hindustanpetroleum.com"
        ],
        "source_priority": 1,
        "source_authority_score": 60,
        "preferred_for_executive_summary": True,
        "source_notes": "Project owners, PSUs, authorities, and infrastructure clients"
    },
    {
        "source_type": "company_official",
        "source_category": "primary_company_source",
        "source_names": [
            "Larsen & Toubro", "L&T", "Tata Projects",
            "Kalpataru Projects International", "Kalpataru Projects",
            "Sterlite Power", "Techno Electric", "Skipper", "IRCON", "RVNL",
            "Rail Vikas Nigam", "Afcons Infrastructure",
            "Hindustan Construction Company", "HCC", "NCC", "PNC Infratech",
            "HG Infra", "Dilip Buildcon", "Ashoka Buildcon", "GR Infraprojects",
            "Shapoorji Pallonji", "Siemens Energy", "Hitachi Energy", "GE Vernova",
            "Sterling and Wilson Renewable Energy", "ReNew", "Tata Power Solar",
            "Bajel Projects", "Transrail Lighting"
        ],
        "domains": [
            "larsentoubro.com", "tataprojects.com", "kalpataruprojects.com",
            "kpil.co.in", "sterlitepower.com", "technoelectric.com",
            "skipperlimited.com", "ircon.org", "rvnl.org", "afcons.com",
            "hccindia.com", "ncclimited.com", "pncinfra.com", "hginfra.com",
            "dilipbuildcon.com", "ashokabuildcon.com", "grinfra.com",
            "shapoorjipallonji.com", "siemens-energy.com", "hitachienergy.com",
            "gevernova.com", "sterlingandwilsonre.com", "renew.com",
            "tatapowersolar.com", "bajelprojects.com", "transrail.in", "transrail.co.in"
        ],
        "source_priority": 2,
        "source_authority_score": 50,
        "preferred_for_executive_summary": True,
        "source_notes": "Primary company websites, press releases, investor pages, and order announcements"
    },
    {
        "source_type": "specialist_infrastructure_media",
        "source_category": "sector_media",
        "source_names": [
            "Construction World", "Projects Today", "India Infrastructure",
            "Infrastructure Today", "Construction Week India", "ET Infra",
            "Urban Transport News", "Rail Analysis", "Metro Rail News",
            "Railway Technology", "Global Railway Review", "International Railway Journal"
        ],
        "domains": [
            "constructionworld.in", "projectstoday.com", "indiainfrastructure.com",
            "infrastructuretoday.co.in", "constructionweekonline.in",
            "infra.economictimes.indiatimes.com", "urbantransportnews.com",
            "railanalysis.com", "metrorailnews.in", "railway-technology.com",
            "globalrailwayreview.com", "railjournal.com"
        ],
        "source_priority": 3,
        "source_authority_score": 38,
        "preferred_for_executive_summary": True,
        "source_notes": "Specialist infrastructure, metro, rail, and project intelligence media"
    },
    {
        "source_type": "specialist_power_and_energy_media",
        "source_category": "sector_media",
        "source_names": [
            "ET EnergyWorld", "Power Line", "Power Today", "Power Technology",
            "T&D World", "PV Magazine India", "Mercom India", "Saur Energy",
            "Renewable Watch", "SolarQuarter", "Energy Storage News"
        ],
        "domains": [
            "energy.economictimes.indiatimes.com", "powerline.net.in",
            "powertoday.in", "power-technology.com", "tdworld.com",
            "pv-magazine-india.com", "mercomindia.com", "saurenergy.com",
            "renewablewatch.in", "solarquarter.com", "energy-storage.news"
        ],
        "source_priority": 3,
        "source_authority_score": 38,
        "preferred_for_executive_summary": True,
        "source_notes": "Specialist power, transmission, renewables, BESS, and grid media"
    },
    {
        "source_type": "business_media",
        "source_category": "mainstream_business_news",
        "source_names": [
            "Economic Times", "The Economic Times", "Business Standard", "Mint",
            "Livemint", "Moneycontrol", "Financial Express", "The Hindu BusinessLine",
            "BusinessLine", "CNBCTV18", "Zee Business", "Business Today",
            "NDTV Profit", "Reuters", "Bloomberg", "Financial Times"
        ],
        "domains": [
            "economictimes.indiatimes.com", "business-standard.com", "livemint.com",
            "moneycontrol.com", "financialexpress.com", "thehindubusinessline.com",
            "businessline.com", "cnbctv18.com", "zeebiz.com", "businesstoday.in",
            "ndtvprofit.com", "reuters.com", "bloomberg.com", "ft.com"
        ],
        "source_priority": 4,
        "source_authority_score": 30,
        "preferred_for_executive_summary": True,
        "source_notes": "Mainstream business and financial media"
    },
    {
        "source_type": "press_release_distribution",
        "source_category": "press_release",
        "source_names": [
            "PR Newswire", "Business Wire", "GlobeNewswire", "EIN Presswire",
            "ANI", "PTI", "NewsVoir"
        ],
        "domains": [
            "prnewswire.com", "businesswire.com", "globenewswire.com",
            "einpresswire.com", "aninews.in", "ptinews.com", "newsvoir.com"
        ],
        "source_priority": 5,
        "source_authority_score": 20,
        "preferred_for_executive_summary": False,
        "source_notes": "Press release distribution; useful for discovery but should be corroborated"
    },
    {
        "source_type": "aggregator_or_syndication",
        "source_category": "aggregator",
        "source_names": [
            "Google News", "Yahoo Finance", "Yahoo News", "MSN", "AOL",
            "Devdiscourse", "LatestLY", "Big News Network"
        ],
        "domains": [
            "news.google.com", "yahoo.com", "msn.com", "aol.com",
            "devdiscourse.com", "latestly.com", "bignewsnetwork.com"
        ],
        "source_priority": 6,
        "source_authority_score": 10,
        "preferred_for_executive_summary": False,
        "source_notes": "Aggregated or syndicated news; use only if no better source exists"
    },
    {
        "source_type": "low_authority_or_noise",
        "source_category": "low_authority",
        "source_names": [
            "openPR.com", "EIN News", "SBWire", "Digital Journal", "MENAFN",
            "Benzinga", "Simply Wall St", "StockTitan", "MarketsMojo", "Equity Bulls"
        ],
        "domains": [
            "openpr.com", "einnews.com", "sbwire.com", "digitaljournal.com",
            "menafn.com", "benzinga.com", "simplywall.st", "stocktitan.net",
            "marketsmojo.com", "equitybulls.com"
        ],
        "source_priority": 7,
        "source_authority_score": 5,
        "preferred_for_executive_summary": False,
        "source_notes": "Low-authority, SEO-heavy, stock-only, or republished sources"
    }
]

# Redirect / wrapper hosts that are NOT the real publisher.
# Google News RSS links resolve to news.google.com redirects, so at scrape
# time the extracted domain is almost always one of these.
REDIRECT_HOSTS = {"news.google.com"}

DEFAULT_SOURCE_TYPE = "unknown"
DEFAULT_SOURCE_CATEGORY = "unknown"
DEFAULT_SOURCE_PRIORITY = 8
DEFAULT_SOURCE_AUTHORITY_SCORE = 5
DEFAULT_PREFERRED_FOR_EXECUTIVE_SUMMARY = False
DEFAULT_SOURCE_NOTES = "Unclassified source"

def normalize_source_name(source_name: str) -> str:
    """Normalize Google News publisher display names for registry matching."""
    if not source_name:
        return ""
    name = source_name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("&amp;", "&")
    return name


def extract_domain(url: str) -> str:
    """
    Extract clean lowercase domain from a URL.
    Works for real publisher URLs and Google News redirect URLs.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def domain_matches(article_domain: str, registry_domain: str) -> bool:
    """
    Match exact domain or subdomain.
    e.g. infra.economictimes.indiatimes.com matches economictimes.indiatimes.com
    """
    if not article_domain or not registry_domain:
        return False
    article_domain = article_domain.lower().strip()
    registry_domain = registry_domain.lower().strip()
    return article_domain == registry_domain or article_domain.endswith("." + registry_domain)


def source_name_matches(article_source_name: str, registry_source_name: str) -> bool:
    """
    Match Google News publisher display names.
    Normalized exact match first, then a conservative contains match
    (only for registry names >= 5 chars, to avoid short-name false positives).
    """
    if not article_source_name or not registry_source_name:
        return False
    article_name = normalize_source_name(article_source_name)
    registry_name = normalize_source_name(registry_source_name)
    if article_name == registry_name:
        return True
    if len(registry_name) >= 5 and registry_name in article_name:
        return True
    return False


def classify_source(url: str, source_name: str = None) -> dict:
    """
    Classify an article's source using publisher display name first,
    then the URL domain.

    Order of precedence:
      1. source_name match  -> Google News gives us the publisher name even
                               when the link is a news.google.com redirect.
      2. domain match       -> only trustworthy when a real publisher URL is
                               available; SKIPPED for redirect hosts when a
                               source_name is present, so an unrecognized
                               publisher does not get mislabeled as aggregator.
      3. unknown defaults.
    """
    domain = extract_domain(url)

    # 1. Prefer source_name matching (Google News links are redirects).
    if source_name:
        for entry in SOURCE_REGISTRY:
            for registered_source_name in entry.get("source_names", []):
                if source_name_matches(source_name, registered_source_name):
                    return {
                        "source_domain": domain,
                        "source_type": entry["source_type"],
                        "source_category": entry["source_category"],
                        "source_priority": entry["source_priority"],
                        "source_authority_score": entry["source_authority_score"],
                        "preferred_for_executive_summary": entry["preferred_for_executive_summary"],
                        "source_notes": entry["source_notes"],
                        "source_match_method": "source_name"
                    }

    # 2. Domain fallback — only when we actually have a real publisher domain.
    #    Skip redirect hosts (e.g. news.google.com) when a source_name exists,
    #    so an unknown publisher falls through to unknown rather than aggregator.
    skip_domain = (domain in REDIRECT_HOSTS) and bool(source_name)
    if domain and not skip_domain:
        for entry in SOURCE_REGISTRY:
            for registered_domain in entry.get("domains", []):
                if domain_matches(domain, registered_domain):
                    return {
                        "source_domain": domain,
                        "source_type": entry["source_type"],
                        "source_category": entry["source_category"],
                        "source_priority": entry["source_priority"],
                        "source_authority_score": entry["source_authority_score"],
                        "preferred_for_executive_summary": entry["preferred_for_executive_summary"],
                        "source_notes": entry["source_notes"],
                        "source_match_method": "domain"
                    }

    # 3. Unknown fallback.
    return {
        "source_domain": domain,
        "source_type": DEFAULT_SOURCE_TYPE,
        "source_category": DEFAULT_SOURCE_CATEGORY,
        "source_priority": DEFAULT_SOURCE_PRIORITY,
        "source_authority_score": DEFAULT_SOURCE_AUTHORITY_SCORE,
        "preferred_for_executive_summary": DEFAULT_PREFERRED_FOR_EXECUTIVE_SUMMARY,
        "source_notes": DEFAULT_SOURCE_NOTES,
        "source_match_method": "default"
    }

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
                    'sbu': 'General',
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


async def fetch_feed_async(session: aiohttp.ClientSession, keyword: str, lookback_days: int, semaphore: asyncio.Semaphore) -> Dict:
    """Asynchronously fetch RSS feed with rate limiting and retry logic"""
    query = f"{keyword} when:{lookback_days}d"
    encoded_query = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    logging.info(f"Fetching Google News RSS for keyword: {keyword}")
    logging.debug(f"RSS URL: {rss_url}")

    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    async with semaphore:
        for attempt in range(3):
            try:
                jitter = random.uniform(0.5, 1.5)
                await asyncio.sleep(REQUEST_DELAY * jitter)

                async with session.get(rss_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 503:
                        wait = (attempt + 1) * 10
                        logging.warning(f"503 for '{keyword}' (attempt {attempt+1}/3), waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    
                    if response.status == 429:
                        wait = (attempt + 1) * 30
                        logging.warning(f"429 rate limited for '{keyword}', waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    if feed.bozo and hasattr(feed, 'bozo_exception'):
                        logging.warning(f"Feed parse warning for '{keyword}': {feed.bozo_exception}")
                    
                    return {'keyword': keyword, 'feed': feed, 'success': True}

            except Exception as e:
                logging.error(f"Error fetching feed for '{keyword}' (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep((attempt + 1) * 5)

    return {'keyword': keyword, 'feed': None, 'success': False}


async def scrape_news_async(competitor_keywords: List[str], sbu_keywords: List[str], 
                            competitor_to_sbu: Dict, lookback_days: int = LOOKBACK_DAYS) -> List[Dict]:
    """Scrape news asynchronously for all competitor keywords"""
    all_articles = []
    seen_links = set()
    
    # Create aiohttp session
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_feed_async(session, kw, lookback_days, semaphore) for kw in competitor_keywords]
        logging.info(f"Fetching {len(tasks)} RSS feeds (max {MAX_CONCURRENT_REQUESTS} concurrent)...")
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
            raw_title = entry.get("title", "")
            link = entry.get("link", "")
            
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
            # Clean title by removing source suffix
            title = raw_title
            if source:
                # Try exact match patterns first
                patterns = [
                    f' - {source}',
                    f' | {source}',
                    f' – {source}',
                    f'- {source}',
                    f'| {source}'
                ]
                for pattern in patterns:
                    if title.endswith(pattern):
                        title = title[:-len(pattern)].strip()
                        break
    
            # Fallback: remove anything after last separator
            if title == raw_title:  # If no exact match found
                title = raw_title.rsplit(' - ', 1)[0].rsplit(' | ', 1)[0].rsplit(' – ', 1)[0].strip()
    
            # Skip duplicates
            if link in seen_links:
                continue
            
            # Detect competitor (must match to proceed)
            competitor = detect_competitor(title, source, competitor_to_sbu, competitor_keywords)
            if not competitor:
                continue
            
            # Detect SBU (must match at least one SBU keyword)
            sbu = detect_sbu(title, source, sbu_keywords)
            if not sbu:
                sbu = "General"  # Let LLM decide relevance instead of dropping
            
            seen_links.add(link)

            # Classify source authority/reliability (metadata only — never filters articles)
            source_metadata = classify_source(link, source)
            logging.debug(
                f"Source classified: source='{source}', domain='{source_metadata['source_domain']}', "
                f"type='{source_metadata['source_type']}', score={source_metadata['source_authority_score']}, "
                f"match_method='{source_metadata['source_match_method']}'"
            )

            all_articles.append({
                "search_keyword": keyword,
                "news_title": title,
                "source": source,
                "link": link,
                "published_date": pubdate,
                "sbu": sbu,
                "competitor": competitor,
                "content": "",
                "source_domain": source_metadata["source_domain"],
                "source_type": source_metadata["source_type"],
                "source_category": source_metadata["source_category"],
                "source_priority": source_metadata["source_priority"],
                "source_authority_score": source_metadata["source_authority_score"],
                "preferred_for_executive_summary": source_metadata["preferred_for_executive_summary"],
                "source_notes": source_metadata["source_notes"],
                "source_match_method": source_metadata["source_match_method"],
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
    """Save scraped articles to PostgreSQL database - raw_scraped_articles table"""
    if not articles:
        logging.info("No articles to save")
        return
    
    # Filter to only keep articles between Mar 8 and Mar 9
    from datetime import date, timedelta
    yesterday = date.today()-timedelta(days=1)
    start_date = yesterday
    end_date = yesterday
    articles = [a for a in articles if start_date <= a['published_date'].date() <= end_date]
    logging.info(f"After date filter ({yesterday}): {len(articles)} articles")
    
    if not articles:
        logging.info("No articles in date range")
        return    
    conn = get_db_connection()

    # ------------------------------------------------------------------
    # SQL migration required (run once before deploying this change):
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_domain TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_type TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_category TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_priority INTEGER DEFAULT 8;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_authority_score INTEGER DEFAULT 5;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS preferred_for_executive_summary BOOLEAN DEFAULT FALSE;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_notes TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS source_match_method TEXT;
    # ------------------------------------------------------------------
    insert_query = """
        INSERT INTO raw_scraped_articles (
            search_keyword, news_title, source, link, published_date,
            sbu, competitor, content,
            source_domain, source_type, source_category, source_priority,
            source_authority_score, preferred_for_executive_summary,
            source_notes, source_match_method
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s
        )
        ON CONFLICT (link, published_date) DO NOTHING
    """
    
    saved_count = 0
    failed_count = 0
    
    for article in articles:
        try:
            cur = conn.cursor()
            cur.execute(insert_query, (
                article['search_keyword'],
                article['news_title'],
                article['source'],
                article['link'],
                article['published_date'],
                article['sbu'],
                article['competitor'],
                article['content'],
                article.get("source_domain"),
                article.get("source_type", "unknown"),
                article.get("source_category", "unknown"),
                article.get("source_priority", 8),
                article.get("source_authority_score", 5),
                article.get("preferred_for_executive_summary", False),
                article.get("source_notes"),
                article.get("source_match_method", "default"),
            ))
            conn.commit()  # Commit after each successful insert
            cur.close()
            saved_count += 1
        except Exception as e:
            conn.rollback()  # Rollback the failed transaction
            failed_count += 1
            logging.error(f"Error saving article '{article.get('news_title', 'Unknown')[:50]}...': {e}")
    
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
