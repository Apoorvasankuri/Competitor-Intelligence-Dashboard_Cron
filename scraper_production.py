"""
Production News Scraper for Competitor Intelligence (v2 - Async)
Scrapes Google News RSS feeds for competitor keywords and filters by SBU relevance
"""

import asyncio
import uuid
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
from collections import defaultdict
from bse_filings_ingest import fetch_bse_filing_articles

PIPELINE_ID = os.getenv("PIPELINE_ID", f"run-{uuid.uuid4().hex[:8]}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - [{PIPELINE_ID}] - %(message)s"

)

# Configuration
LOOKBACK_DAYS = 15

# Change 16: rolling save window (fix for the single-day data-loss bug).
# The old save_to_database() filter kept only articles published on exactly
# "yesterday" (one calendar day) — a missed, delayed, or retried cron run
# permanently lost anything outside that one day, even though RSS queries
# already fetch a LOOKBACK_DAYS=15 window. SAVE_WINDOW_DAYS widens the save
# filter so a missed run gets caught by the next one. Kept narrower than the
# full RSS lookback so we're not re-attempting inserts for 15 days of
# history on every run — ON CONFLICT (link, published_date) DO NOTHING in
# save_to_database() makes any repeat inserts cheap no-ops regardless.
SAVE_WINDOW_DAYS = 15

MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent requests
REQUEST_DELAY = 1  # Delay between requests in seconds
EXCEL_FILE_PATH = 'SBU_Competitor_Mapping.xlsx'

# ------------------------------------------------------------
# Change 4 Part H: dry-run mode (safe test run, no DB writes).
# Dry-run usage:
#   PowerShell: $env:DRY_RUN="true"; $env:DRY_RUN_MAX_QUERIES="30"; python scraper_production.py
#   CMD:        set DRY_RUN=true && set DRY_RUN_MAX_QUERIES=30 && python scraper_production.py
#   Bash:       DRY_RUN=true DRY_RUN_MAX_QUERIES=30 python scraper_production.py
# ------------------------------------------------------------
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
DRY_RUN_MAX_QUERIES = int(os.getenv("DRY_RUN_MAX_QUERIES", "30"))
DRY_RUN_SAMPLE_ARTICLES = int(os.getenv("DRY_RUN_SAMPLE_ARTICLES", "15"))
# Rotate User-Agents to avoid fingerprinting
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# ============================================================
# Multi-lens search keyword lists (Change 4: exhaustiveness)
# These drive non-competitor query lenses so we catch tender / policy /
# authority / strategic-theme news that may not name a competitor.
# ============================================================
CLIENT_AUTHORITY_KEYWORDS = [
    # ---- Original list (kept) ----
    "Power Grid Corporation of India", "POWERGRID", "PGCIL", "NTPC", "SECI",
    "NHAI", "MoRTH", "Indian Railways", "Railway Board", "RVNL", "IRCON",
    "DMRC", "Delhi Metro", "MMRDA", "Mumbai Metro", "BMRCL", "Bangalore Metro",
    "Chennai Metro", "CMRL", "NHSRCL", "GAIL", "ONGC", "Indian Oil", "IOCL",
    "BPCL", "HPCL", "REC", "PFC", "CEA", "CERC",

    # ---- India - Power T&D (client list expansion) ----
    "Power Grid Corporation of India Limited",
    "TANTRANSCO", "Tamil Nadu Green Energy Corporation", "Tamil Nadu Transmission Corporation",
    "KPTCL", "Karnataka Power Transmission Corporation",
    "MSETCL", "Maharashtra State Electricity Transmission Company",
    "RVPNL", "Rajasthan Rajya Vidyut Prasaran Nigam",
    "GETCO", "Gujarat Energy Transmission Corporation",
    "MPPKVVCL", "Madhya Pradesh Poorv Kshetra Vidyut Vitaran Company",
    "WBSETCL", "West Bengal State Electricity Transmission Company",
    "GRIDCO", "Grid Corporation of Odisha",
    "Adani Energy Solutions", "Adani Transmission",
    "Sterlite Power Grid Ventures", "Sterlite Grid 32", "Sterlite Power Transmission",
    "IndiGrid", "India Grid Trust",
    "Torrent Power",
    "Tata Power Company", "The Tata Power Company",
    "VISA Power",
    "Resonia",
    "Adani Green Energy",
    "Greenko", "Greenko Energy Holdings",
    "ReNew Power", "ReNew Energy Global",
    "Solar Energy Corporation of India",

    # ---- India - Railways & Metro (client list expansion) ----
    "Ministry of Railways",
    "Rail Vikas Nigam",
    "Delhi Metro Rail Corporation",
    "Mumbai Metropolitan Region Development Authority",
    "DFCCIL", "Dedicated Freight Corridor Corporation",
    "BMRC", "Bangalore Metro Rail Corporation",
    "KMRL", "Kochi Metro Rail",
    "NCRTC", "National Capital Region Transport Corporation", "Meerut RRTS",
    "Konkan Railway Corporation",

    # ---- India - Civil / Smart Infra (client list expansion) ----
    "National Highways Authority of India",
    "AAI", "Airports Authority of India",
    "Hindalco Industries",
    "Water Resources Department Madhya Pradesh",
    "Ministry of Defence",
    "BEL", "Bharat Electronics Limited",

    # ---- SAARC ----
    "Bhutan Power Corporation",
    "Druk Green Power Corporation",
    "Nepal Electricity Authority",
    "Power Grid Company of Bangladesh",
    "Power Development Board Bangladesh",
    "Ceylon Electricity Board",
    "Lanka Electricity Company",
    "DABS", "Da Afghanistan Breshna Sherkat",
    "STELCO", "Maldives State Electric Company",

    # ---- Middle East (client list expansion) ----
    "Saudi Electricity Company", "SEC",
    "Saudi Aramco",
    "Oman Electricity Transmission Company", "OETC",
    "Abu Dhabi Distribution Company", "ADDC",
    "TRANSCO Abu Dhabi", "Abu Dhabi Transmission and Despatch Company",
    "Dubai Electricity and Water Authority", "DEWA",
    "Kuwait Ministry of Electricity and Water",
    "Qatar General Electricity and Water Corporation", "Kahramaa",
    "Bahrain Electricity and Water Authority", "EWA",
    "Iraq Ministry of Electricity",
    "National Electric Power Company Jordan", "NEPCO",

    # ---- Africa (client list expansion) ----
    "Eskom", "Eskom Holdings",
    "Ethiopia Electric Power", "EEP",
    "Kenya Power and Lighting Company", "KPLC",
    "Nigeria Transmission Company", "TCN",
    "SNEL", "Société Nationale d'Électricité",
    "SONABEL",
    "REGIDESO",
    "Electricité du Mali", "EDM",
    "STEG", "Société Tunisienne d'Électricité et du Gaz",
    "Electricidade de Moçambique",
    "Ghana Grid Company", "GRIDCo",
    "Libyan General Electricity Company", "GECOL",
    "Sonelgaz",
    "Sudan Electricity Transmission Company", "SETCO",
    "NamPower",

    # ---- East Asia Pacific (client list expansion) ----
    "Tenaga Nasional Berhad", "TNB",
    "Sarawak Energy Berhad",
    "Sabah Electricity", "SESB",
    "Electricity Generating Authority of Thailand", "EGAT",
    "Metropolitan Electricity Authority Thailand", "MEA",
    "Provincial Electricity Authority Thailand", "PEA",
    "National Grid Corporation of the Philippines", "NGCP",
    "National Electrification Administration Philippines",
    "National Power Corporation Philippines", "NPC",
    "PT PLN", "Perusahaan Listrik Negara",
    "PT Perusahaan Gas Negara", "PGN",
    "Vietnam Electricity", "EVN", "EVNNPT",
    "SP Group", "Singapore Power",
    "Energy Market Authority Singapore",
    "Électricité du Cambodge", "EDC",
    "Électricité du Laos", "EDL",
    "ElectraNet",
    "Transgrid",
    "AusNet Services",
    "Western Power Australia",
    "Powerlink Queensland",
    "TasNetworks",
    "Transpower New Zealand",
    "PNG Power",
    "Taiwan Power Company", "Taipower",
    "Korea Electric Power Corporation", "KEPCO",
    "CLP Power Hong Kong",

    # ---- CIS / Central Asia ----
    "KEGOC", "Kazakhstan Electricity Grid Operating Company",
    "Tajiktransenergy", "Barki Tojik",
    "Georgian State Electrosystem",
    "Uzbekistan National Grid", "UzTransEnergo",
    "Moldelectrica",
    "Kazakhstan Temir Zholy",
    "National Dispatch Centre Azerbaijan",
    "Armenian Energy Networks",

    # ---- Americas ----
    "American Electric Power", "AEP",
    "Duke Energy",
    "NextEra Energy", "FPL",
    "Pacific Gas and Electric", "PG&E",
    "Comisión Federal de Electricidad", "CFE",
    "Eletrobras", "CHESF",
    "CTEEP", "Companhia de Transmissão de Energia Elétrica Paulista",
    "Hydro-Québec",
    "Interconexión Eléctrica", "ISA Colombia",

    # ---- Europe ----
    "Red Eléctrica de España", "REE",
    "Réseau de Transport d'Électricité", "RTE",

    # ---- Financing / Strategic Partners ----
    "State Bank of India", "SBI",
    "HDFC Bank",
    "Axis Bank",
    "Export-Import Bank of India", "EXIM Bank",
    "African Development Bank", "AfDB",
    "World Bank", "International Finance Corporation", "IFC",
    "Asian Development Bank", "ADB",
]

STRATEGIC_THEME_KEYWORDS = [
    "transmission tender", "transmission project", "TBCB transmission",
    "tariff based competitive bidding", "ISTS project", "inter-state transmission",
    "substation contract", "HVDC project", "765 kV transmission", "400 kV transmission",
    "Green Energy Corridor", "renewable energy evacuation", "BESS tender",
    "battery energy storage tender", "solar EPC contract", "wind EPC contract",
    "railway electrification contract", "Kavach tender", "metro rail contract",
    "metro civil package", "metro depot contract", "high speed rail package",
    "oil pipeline contract", "gas pipeline EPC", "water pipeline EPC",
    "data center construction contract", "airport construction contract",
    "industrial construction EPC"
]

# ------------------------------------------------------------
# Query-generation caps (Change 4: controlled rollout)
# Conservative for the first production run to avoid Google News
# throttling and to measure yield before scaling up. Tune here — do NOT
# hardcode these deep inside generate_search_queries().
# ------------------------------------------------------------
MAX_INDIVIDUAL_SBU_QUERIES = 75
MAX_COMPETITOR_SBU_QUERIES = 100
MAX_COMPETITOR_CLIENT_QUERIES = 75
MAX_SBU_CLIENT_QUERIES = 60
MAX_TOTAL_SEARCH_QUERIES = 400

SITE_OFFICIAL_QUERY_TYPES = (
    "site_official_exchange",
    "site_company_official",
    "site_client_authority",
    "site_government_policy",
    "site_tender",
)
SITE_SPECIALIST_QUERY_TYPE = "site_specialist_media"

SITE_QUERY_GATE_LABELS = {
    "site_official_exchange": "site_official_exchange_query",
    "site_company_official": "site_company_official_query",
    "site_client_authority": "site_client_authority_query",
    "site_government_policy": "site_government_policy_query",
    "site_tender": "site_tender_query",
    "site_specialist_media": "site_specialist_media_query",
}

# High-authority domains targeted via site: filters (Change 4 Part E).
OFFICIAL_EXCHANGE_DOMAINS = [
    "bseindia.com",
    "nseindia.com",
]
COMPANY_OFFICIAL_DOMAINS = [
    "larsentoubro.com", "tataprojects.com", "kalpataruprojects.com",
    "sterlitepower.com", "ircon.org", "rvnl.org", "afcons.com",
    "hccindia.com", "ncclimited.com", "pncinfra.com", "dilipbuildcon.com",
    "ashokabuildcon.com", "grinfra.com", "siemens-energy.com",
    "hitachienergy.com", "sterlingandwilsonre.com", "tatapower.com",
    "transrail.in",
]
CLIENT_AUTHORITY_DOMAINS = [
    "powergrid.in", "pgcilindia.com", "ntpc.co.in", "nhai.gov.in",
    "seci.co.in", "dmrc.org", "indianrailways.gov.in", "nhsrcl.in",
    "gailonline.com", "ongcindia.com", "iocl.com",
]
GOVERNMENT_POLICY_DOMAINS = [
    "pib.gov.in", "powermin.gov.in", "mnre.gov.in", "morth.nic.in",
    "railways.gov.in", "cea.nic.in", "cercind.gov.in",
]
TENDER_DOMAINS = [
    "eprocure.gov.in", "gem.gov.in", "ireps.gov.in", "etenders.gov.in",
]
SPECIALIST_MEDIA_DOMAINS = [
    "constructionworld.in", "projectstoday.com",
    "infra.economictimes.indiatimes.com", "energy.economictimes.indiatimes.com",
    "railanalysis.com", "metrorailnews.in", "mercomindia.com",
    "pv-magazine-india.com", "renewablewatch.in",
]

# Cap so site-specific lenses cannot crowd out base lenses under the global budget.
MAX_SITE_SPECIFIC_QUERIES = 80



# ============================================================
# Source Registry
# Classifies article sources by authority and reliability.
# Supports both Google News RSS publisher display names (source_names)
# and real publisher domains (domains).
# Lower source_priority = better source.
# Higher source_authority_score = more trustworthy source.
# ============================================================
#
# SOURCE REGISTRY DESIGN
# Google News RSS often returns news.google.com redirect URLs.
# Therefore, source classification must prefer publisher display name first,
# then fall back to real publisher domain if available.
# Do NOT classify every news.google.com link as aggregator if source_name is available.
# Lower source_priority means better source.
# Higher source_authority_score means more trusted source.
# Source authority should influence ranking and representative-source selection,
# but should NOT be used to discard articles at scraping stage.
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
            "Railway Technology", "Global Railway Review", "International Railway Journal",
            "News on Projects", "New Civil Engineer", "Construction Enquirer"
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
            "T&D World", "T&D India", "PV Magazine India", "Mercom India", "Saur Energy",
            "Renewable Watch", "SolarQuarter", "Energy Storage News",
            "Transformer Magazine", "Energetica India Magazine", "Solarbytes",
            "Power Peak Digest", "CleanTechnica", "Electrek"
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
            "BusinessLine", "CNBCTV18", "CNBC TV18", "Zee Business", "Business Today",
            "NDTV Profit", "ET Now", "Rediff MoneyWiz", "Reuters", "Bloomberg", "Financial Times"
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
            "Devdiscourse", "LatestLY", "Big News Network", "Dailyhunt", "inkl"
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
            "Benzinga", "Simply Wall St", "StockTitan", "MarketsMojo", "Equity Bulls",
            "Whalesbook", "TradingView", "Upstox", "HDFC Sky", "Trade Brains",
            "Goodreturns", "TipRanks", "ClearTax", "IndexBox", "AD HOC NEWS",
            "MarketWatch", "Business Upturn", "Equitypandit"
        ],
        "domains": [
            "openpr.com", "einnews.com", "sbwire.com", "digitaljournal.com",
            "menafn.com", "benzinga.com", "simplywall.st", "stocktitan.net",
            "marketsmojo.com", "equitybulls.com",
            "scanx.trade", "sahi.com", "marketscreener.com", "biginfo.in",
            "megaproject.com", "energynews.pro"
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


def looks_like_domain(name: str) -> bool:
    """
    True if a publisher 'name' is actually a bare domain, e.g. 'financialexpress.com'
    or 'Mercomindia.com'. Google News sometimes puts the domain in the publisher slot
    instead of a display name.
    """
    if not name:
        return False
    name = name.strip().lower()
    return ("." in name) and (" " not in name)


def get_default_source_metadata(url: str = "") -> dict:
    """
    Central definition of 'unknown source' defaults.
    Used by classify_source() so the fallback dict lives in exactly one place.
    """
    domain = extract_domain(url)
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

    # GUARDRAIL: classification order is intentional and must not be reordered.
    #   1. source_name match  -> primary, because Google News links are redirects
    #                            and the publisher display name is the reliable signal.
    #   1b. domain-shaped name -> some feeds put a bare domain in the name slot.
    #   2. URL domain match    -> fallback, only meaningful for real publisher URLs.
    #   3. unknown default     -> NEVER blocks ingestion; it is metadata only.

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

    # 1b. Domain-shaped publisher name (e.g. "financialexpress.com", "Mercomindia.com").
    #     Google News sometimes returns the bare domain as the publisher name. Match it
    #     against the registry's DOMAINS list so that data does work even though the
    #     article link is a news.google.com redirect. Also recovers the real domain.
    if source_name and looks_like_domain(source_name):
        name_as_domain = source_name.strip().lower()
        if name_as_domain.startswith("www."):
            name_as_domain = name_as_domain[4:]
        for entry in SOURCE_REGISTRY:
            for registered_domain in entry.get("domains", []):
                if domain_matches(name_as_domain, registered_domain):
                    return {
                        "source_domain": name_as_domain,
                        "source_type": entry["source_type"],
                        "source_category": entry["source_category"],
                        "source_priority": entry["source_priority"],
                        "source_authority_score": entry["source_authority_score"],
                        "preferred_for_executive_summary": entry["preferred_for_executive_summary"],
                        "source_notes": entry["source_notes"],
                        "source_match_method": "domain_from_source_name"
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

    # 3. Unknown fallback — metadata only. An unclassified source is still ingested;
    #    source authority affects ranking/representative selection, never inclusion.
    return get_default_source_metadata(url)

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


def detect_client_authority(text: str, client_keywords: List[str]) -> str:
    """Return comma-separated client/authority keywords found in text (case-insensitive)."""
    if not text:
        return ""
    lowered = text.lower()
    matched = set()
    for kw in client_keywords:
        if kw and kw.lower() in lowered:
            matched.add(kw)
    return ", ".join(sorted(matched)) if matched else ""


def detect_strategic_theme(text: str, theme_keywords: List[str]) -> str:
    """Return comma-separated strategic-theme keywords found in text (case-insensitive)."""
    if not text:
        return ""
    lowered = text.lower()
    matched = set()
    for kw in theme_keywords:
        if kw and kw.lower() in lowered:
            matched.add(kw)
    return ", ".join(sorted(matched)) if matched else ""



# ============================================================
# Change 19: scraper-stage quality gate.
#
# Dry run (DRY_RUN=true, DRY_RUN_MAX_QUERIES=100, LOOKBACK_DAYS=7) showed the
# "competitor" query_type branch accepted an article purely because a
# competitor name was detected in the title — no check on whether the
# article was actually about a business event at all. Result: 1001 accepted
# articles dominated by stock-price/valuation/board-meeting/investor-call
# chatter, alongside genuine signal like "Likhitha Infrastructure secures
# Rs 510 crore pipeline contract". This block adds a deterministic,
# no-LLM-cost gate so a competitor name alone is never sufficient.
# ============================================================

NOISE_PHRASES = [
    "share price",
    "stock price",
    "live stock price",
    "upper circuit",
    "lower circuit",
    "valuation",
    "valuation attractiveness",
    "technical shift",
    "bearish",
    "bullish",
    "momentum",
    "buyers queue",
    "sellers absent",
    "promoter group",
    "promoter shares",
    "pledged shares",
    "encumbrance",
    "investor call",
    "earnings call",
    "q4 results calendar",
    "results calendar",
    "board meeting",
    "trading update",
    "price target",
    "brokerage",
    "recommendation",
    "buy call",
    "sell call",
    "hold call",
    "market cap",
    "intraday",
    "multibagger",
    "dividend record date",
    "ex-dividend",
    "bonus issue",
    "stock split",
    "rights issue",
    "warrants",
    "appointment of directors",
    "resignation",
    "agm",
    "annual general meeting",
    "analyst rating",
    "stock alert",
    "shareholding pattern",
]

STRONG_EVENT_PHRASES = [
    "wins",
    "bags",
    "secures",
    "receives",
    "awarded",
    "award",
    "order",
    "contract",
    "project",
    "tender",
    "bid",
    "bidding",
    "emerges l1",
    "l1 bidder",
    "lowest bidder",
    "loa",
    "letter of award",
    "work order",
    "epc",
    "transmission",
    "substation",
    "hvdc",
    "765 kv",
    "400 kv",
    "pipeline",
    "gas pipeline",
    "oil pipeline",
    "water pipeline",
    "metro",
    "rail",
    "railway",
    "station",
    "depot",
    "civil package",
    "highway",
    "expressway",
    "road project",
    "solar",
    "wind",
    "bess",
    "renewable",
    "green energy corridor",
    "commissioned",
    "commissioning",
    "completed",
    "launched",
    "approved",
    "approval",
    "capex",
    "investment",
    "acquisition",
    "divestment",
    "stake sale",
    "merger",
    "joint venture",
    "jv",
    "partnership",
    "consortium",
    "new market",
    "expansion",
    "capacity expansion",
    # Credit-rating actions — competitive intelligence (affects bidding capacity,
    # consortium eligibility, capital access). Added after real BSE run showed
    # "L&T Secures Moody's Baa1 Rating" was being filtered by the noise gate.
    "credit rating",
    "rating upgrade",
    "rating downgrade",
    "rating reaffirmed",
    "rating affirmed",
    "rating outlook",
    "moody",
    "crisil",
    "icra",
    "care ratings",
    "india ratings",
    "fitch",
]

# Source-type trust tiers, matching the exact source_type strings produced by
# classify_source() / SOURCE_REGISTRY above.
HIGH_MEDIUM_QUALITY_SOURCE_TYPES = {
    "official_exchange",
    "government_policy",
    "client_or_authority",
    "company_official",
    "specialist_infrastructure_media",
    "specialist_power_and_energy_media",
    "business_media",
}

LOWER_TRUST_SOURCE_TYPES = {
    "low_authority_or_noise",
    "unknown",
    "aggregator_or_syndication",
    "press_release_distribution",
}


def normalize_for_gate(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for phrase matching.
    Keeps alphanumerics, spaces, '%' and '.' (so "765 kv" and "400 kv"
    survive normalization intact)."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s%.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any_phrase(text: str, phrases: List[str]) -> bool:
    """True if any phrase in `phrases` appears as a substring of normalized `text`."""
    if not text:
        return False
    norm = normalize_for_gate(text)
    for phrase in phrases:
        if phrase in norm:
            return True
    return False


def has_strong_business_event_signal(title: str, source: str) -> bool:
    """True if the title (or source line) carries language for an actual
    business event — order win, tender, project, M&A, commissioning, etc."""
    combined = f"{title} {source}"
    return contains_any_phrase(combined, STRONG_EVENT_PHRASES)


def is_high_quality_source_type(source_type: str) -> bool:
    """True for source types classify_source() considers high/medium
    authority (official exchange filings, government policy, client/
    authority portals, company IR pages, specialist trade media, business
    media)."""
    return (source_type or "").strip() in HIGH_MEDIUM_QUALITY_SOURCE_TYPES


def is_pure_noise_article(title: str, source: str, source_type: str) -> bool:
    """True only when the title/source carries noise language (stock price,
    valuation, board-meeting boilerplate, etc.) AND no strong business-event
    phrase rescues it. "Pure" is the key word: a title with BOTH a stock
    phrase and a real event phrase (e.g. "Ashoka Buildcon shares jump after
    bagging Guyana highway order") is NOT pure noise and must not be dropped
    here — has_strong_business_event_signal is checked before returning True.
    source_type is accepted for signature parity with the other gate helpers
    and potential future tuning, but the noise/not-noise decision itself is
    text-based; source-type trust is applied separately via
    is_high_quality_source_type in the calling gate logic."""
    combined = f"{title} {source}"
    if not contains_any_phrase(combined, NOISE_PHRASES):
        return False
    if has_strong_business_event_signal(title, source):
        return False
    return True


def select_dry_run_queries(search_queries: List[Dict], max_queries: int) -> List[Dict]:
    """
    Change 19: representative dry-run sampling across query_type.

    generate_search_queries() emits "competitor" queries first (see its
    PRIORITY ORDER docstring) — a plain search_queries[:max_queries] slice
    therefore always sampled the first N as 100% "competitor" type and zero
    of anything else, which is exactly why a DRY_RUN_MAX_QUERIES=100 dry run
    generated 100 competitor queries and no site_official_exchange /
    site_tender / site_client_authority / etc. queries at all.

    Round-robins one query at a time across each query_type bucket (in the
    order each type first appears in `search_queries`) until max_queries is
    reached or every bucket is exhausted, guaranteeing every available
    query_type gets representation instead of being starved by ordering.
    """
    if max_queries is None or max_queries <= 0 or max_queries >= len(search_queries):
        return list(search_queries)

    buckets: Dict[str, List[Dict]] = {}
    for q in search_queries:
        buckets.setdefault(q["query_type"], []).append(q)

    selected: List[Dict] = []
    while len(selected) < max_queries and any(buckets.values()):
        for qtype in list(buckets.keys()):
            if len(selected) >= max_queries:
                break
            bucket = buckets.get(qtype)
            if bucket:
                selected.append(bucket.pop(0))

    return selected


def generate_search_queries(competitor_keywords: List[str], sbu_keywords: List[str]) -> List[Dict]:
    """
    Build multi-lens Google News search queries for broader recall.

    Returns a list of {"query": str, "query_type": str} objects.
    query_type is one of:
        competitor, client_authority, strategic_theme, sbu,
        sbu_client, competitor_client, competitor_sbu

    PRIORITY ORDER (matters because of the global MAX_TOTAL_SEARCH_QUERIES cap):
        1. competitor          2. client_authority    3. strategic_theme
        4. sbu                 5. sbu_client          6. competitor_client
        7. competitor_sbu
    High-value non-competitor lenses come before narrow competitor combos, so
    if the total budget is exhausted, the low-yield combos get truncated first.

    Per-type caps are the module-level MAX_* constants. Queries are de-duplicated
    case-insensitively; the first (higher-priority) lens to produce a given query
    string wins its query_type.
    """
    comp = [c.strip() for c in competitor_keywords if c and c.strip()]
    sbu = [s.strip() for s in sbu_keywords if s and s.strip() and len(s.strip()) >= 3]

    queries = []
    seen = set()

    def add(q: str, qtype: str) -> bool:
        """Add one query if unique and under the global cap. Returns False if the
        global cap is reached (signal to stop generating)."""
        if len(queries) >= MAX_TOTAL_SEARCH_QUERIES:
            return False
        if not q:
            return True
        key = q.strip().lower()
        if not key or key in seen:
            return True
        seen.add(key)
        queries.append({"query": q.strip(), "query_type": qtype})
        return True

    # Each lens is a generator of (query, type); we consume them in priority order
    # and stop entirely once the global cap is hit.
    def lens_competitor():
        for c in comp:
            yield (c, "competitor")

    def lens_client_authority():
        for ca in CLIENT_AUTHORITY_KEYWORDS:
            yield (ca, "client_authority")

    def lens_strategic_theme():
        for th in STRATEGIC_THEME_KEYWORDS:
            yield (th, "strategic_theme")

    def lens_sbu():
        for s in sbu[:MAX_INDIVIDUAL_SBU_QUERIES]:
            yield (s, "sbu")

    def lens_sbu_client():
        n = 0
        for s in sbu:
            for ca in CLIENT_AUTHORITY_KEYWORDS:
                if n >= MAX_SBU_CLIENT_QUERIES:
                    return
                yield (f"{s} {ca}", "sbu_client")
                n += 1

    def lens_competitor_client():
        n = 0
        for c in comp:
            for ca in CLIENT_AUTHORITY_KEYWORDS:
                if n >= MAX_COMPETITOR_CLIENT_QUERIES:
                    return
                yield (f"{c} {ca}", "competitor_client")
                n += 1

    def lens_competitor_sbu():
        n = 0
        for c in comp:
            for s in sbu:
                if n >= MAX_COMPETITOR_SBU_QUERIES:
                    return
                yield (f"{c} {s}", "competitor_sbu")
                n += 1

    # Consume lenses strictly in priority order.
    prioritized_lenses = [
        lens_competitor,          # 1
        lens_client_authority,    # 2
        lens_strategic_theme,     # 3
        lens_sbu,                 # 4
        lens_sbu_client,          # 5
        lens_competitor_client,   # 6
        lens_competitor_sbu,      # 7
    ]

    for lens in prioritized_lenses:
        stopped = False
        for q, qtype in lens():
            if not add(q, qtype):
                stopped = True
                break
        if stopped:
            logging.info("Query generation hit MAX_TOTAL_SEARCH_QUERIES=%s at lens '%s'",
                         MAX_TOTAL_SEARCH_QUERIES, qtype)
            break

    return queries

def generate_site_specific_queries() -> List[Dict]:
    """
    Change 4 Part E: build site-targeted Google News RSS queries against
    high-authority domains. Each query pairs a single domain with a compact
    intent-term OR-group so recall stays high while article volume stays sane
    (one query per domain, NOT a domain x keyword cross-product).

    Returns a list of {"query": str, "query_type": str} objects using the
    six site_* query types the Part E gate expects.

    Controlled by MAX_SITE_SPECIFIC_QUERIES so these lenses cannot crowd out
    the base lenses under the global budget.
    """
    # Broad intent OR-group: infra/EPC business events. Kept generic on purpose
    # so official filings/press releases are not excluded by narrow phrasing;
    # the LLM processor validates true relevance downstream.
    intent = ('contract OR order OR tender OR bid OR award OR project OR '
              'EPC OR transmission OR substation OR pipeline OR "letter of award" OR '
              'commissioning OR "financial results" OR acquisition')

    tiers = [
        (OFFICIAL_EXCHANGE_DOMAINS, "site_official_exchange"),
        (COMPANY_OFFICIAL_DOMAINS, "site_company_official"),
        (CLIENT_AUTHORITY_DOMAINS, "site_client_authority"),
        (GOVERNMENT_POLICY_DOMAINS, "site_government_policy"),
        (TENDER_DOMAINS, "site_tender"),
        (SPECIALIST_MEDIA_DOMAINS, "site_specialist_media"),
    ]

    site_queries = []
    seen = set()
    for domains, qtype in tiers:
        for domain in domains:
            if len(site_queries) >= MAX_SITE_SPECIFIC_QUERIES:
                logging.info("Site-query generation hit MAX_SITE_SPECIFIC_QUERIES=%s at tier '%s'",
                             MAX_SITE_SPECIFIC_QUERIES, qtype)
                return site_queries
            d = (domain or "").strip().lower()
            if not d or d in seen:
                continue
            seen.add(d)
            site_queries.append({
                "query": f"site:{d} ({intent})",
                "query_type": qtype,
            })
    return site_queries

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

    # Run-level yield analytics by query_type (Change 4 Part D) — diagnostics only,
    # does not affect acceptance/ranking/dedup decisions.
    query_stats = defaultdict(lambda: {
        "queries_generated": 0,
        "fetch_attempts": 0,
        "fetch_success": 0,
        "fetch_failed": 0,
        "raw_items_seen": 0,
        "accepted": 0,
        "dropped": 0,
        "duplicate_link_skips": 0
    })
    
    # Create aiohttp session
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=120)

    # Build multi-lens search queries (competitor + SBU + client + theme + combos)
    base_queries = generate_search_queries(competitor_keywords, sbu_keywords)

    # Change 4 Part E: append site-targeted high-authority queries, de-duplicated
    # against the base set (base wins on any collision, keeping its query_type).
    site_queries = generate_site_specific_queries()
    search_queries = list(base_queries)
    _seen_queries = {q["query"].strip().lower() for q in search_queries}
    for sq in site_queries:
        key = sq["query"].strip().lower()
        if key and key not in _seen_queries:
            _seen_queries.add(key)
            search_queries.append(sq)
    logging.info("Query mix: base=%s, site_specific=%s, total=%s",
                 len(base_queries), len(site_queries), len(search_queries))

    if DRY_RUN:
        logging.info("DRY RUN MODE ENABLED — database writes will be skipped")

        # Change 19: representative sampling across query_type instead of a
        # naive search_queries[:DRY_RUN_MAX_QUERIES] slice. generate_search_queries()
        # emits "competitor" queries first, so the old slice always sampled
        # 100% competitor-type queries and none of the site-specific/
        # authority/theme lenses — exactly what the bug report's dry run showed.
        original_mix = {}
        for q in search_queries:
            original_mix[q["query_type"]] = original_mix.get(q["query_type"], 0) + 1
        total_generated = sum(original_mix.values())
        logging.info("Dry-run: original query mix before sampling: %s", original_mix)
        logging.info("Dry-run: total generated queries=%s", total_generated)

        search_queries = select_dry_run_queries(search_queries, DRY_RUN_MAX_QUERIES)

        sampled_mix = {}
        for q in search_queries:
            sampled_mix[q["query_type"]] = sampled_mix.get(q["query_type"], 0) + 1
        logging.info("Dry-run: sampled query mix (DRY_RUN_MAX_QUERIES=%s): %s", DRY_RUN_MAX_QUERIES, sampled_mix)
        logging.info("Dry-run: total used queries=%s", len(search_queries))

    query_to_type = {}
    for q in search_queries:
        query_to_type.setdefault(q["query"], q["query_type"])

    query_type_counts = {}
    for q in search_queries:
        query_type_counts[q["query_type"]] = query_type_counts.get(q["query_type"], 0) + 1
        query_stats[q["query_type"]]["queries_generated"] += 1
    logging.info("Generated %s Google News search queries", len(search_queries))
    logging.info("Search query type distribution: %s", query_type_counts)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_feed_async(session, q["query"], lookback_days, semaphore) for q in search_queries]
        for q in search_queries:
            query_stats[q["query_type"]]["fetch_attempts"] += 1
        logging.info(f"Fetching {len(tasks)} RSS feeds (max {MAX_CONCURRENT_REQUESTS} concurrent)...")
        results = await asyncio.gather(*tasks)
    # Process results
    successful_fetches = 0
    accepted_competitor_led_articles = 0
    accepted_non_competitor_signal_articles = 0
    dropped_no_competitor_for_competitor_query = 0
    dropped_no_signal_for_non_competitor_query = 0
    # Change 4 Part E: site-specific / high-authority query counters.
    accepted_by_site_query_without_signal = 0
    accepted_by_site_query_with_signal = 0
    dropped_site_specialist_no_signal = 0
    accepted_by_type = {}
    dropped_by_type = {}
    dropped_by_reason = {}  # Change 19: run-level drop-reason tally
    for result in results:
        keyword = result['keyword']
        query_type = query_to_type.get(keyword, "competitor")

        if result['success']:
            query_stats[query_type]["fetch_success"] += 1
        else:
            query_stats[query_type]["fetch_failed"] += 1

        if not result['success'] or not result['feed'] or not result['feed'].entries:
            continue
        
        successful_fetches += 1
        feed = result['feed']
        query_stats[query_type]["raw_items_seen"] += len(feed.entries)
        
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
            desc_text = ""
            if "description" in entry:
                soup = BeautifulSoup(entry.description, "html.parser")
                font_tag = soup.find("font")
                if font_tag:
                    source = font_tag.text.strip()
                desc_text = soup.get_text(" ", strip=True)
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
                query_stats[query_type]["duplicate_link_skips"] += 1
                continue
            
            # ---- Detect all signals (competitor detection still runs; it is now
            #      conditional per query type, not a hard gate for every lens) ----
            competitor = detect_competitor(title, source, competitor_to_sbu, competitor_keywords)
            sbu = detect_sbu(title, source, sbu_keywords)
            searchable_text = f"{title} {source} {desc_text}"
            client_authority = detect_client_authority(searchable_text, CLIENT_AUTHORITY_KEYWORDS)
            strategic_theme = detect_strategic_theme(searchable_text, STRATEGIC_THEME_KEYWORDS)

            # Classify source authority/reliability. Change 19 moved this up from
            # after the gate below (it used to run only once acceptance was
            # already decided) because the competitor-led gate now needs
            # source_type to evaluate is_high_quality_source_type /
            # is_pure_noise_article before it can decide accept-vs-drop.
            source_metadata = classify_source(link, source)
            logging.debug(
                f"Source classified: source='{source}', domain='{source_metadata['source_domain']}', "
                f"type='{source_metadata['source_type']}', score={source_metadata['source_authority_score']}, "
                f"match_method='{source_metadata['source_match_method']}'"
            )

            # ---- Change 19: universal noise gate ----
            # A title that reads as pure stock-market/admin chatter (share
            # price, valuation, board-meeting boilerplate, promoter-share
            # filings, etc.) with no accompanying business-event language is
            # dropped regardless of query_type — including site-specific/
            # official/authority queries, since even an authoritative domain
            # can occasionally surface a syndicated stock-alert item via a
            # broad site: search. is_pure_noise_article already exempts any
            # title that ALSO carries a strong business-event phrase (e.g.
            # "shares jump after bagging Guyana highway order" is kept).
            if is_pure_noise_article(title, source, source_metadata["source_type"]):
                dropped_by_type[query_type] = dropped_by_type.get(query_type, 0) + 1
                query_stats[query_type]["dropped"] += 1
                dropped_by_reason["dropped_noise_title"] = dropped_by_reason.get("dropped_noise_title", 0) + 1
                continue

            # ---- Query-type-aware acceptance ----
            accepted_by_gate = ""
            needs_llm_relevance_validation = False  # set True by site-specific gates (Part E)
            if query_type in ("competitor", "competitor_sbu", "competitor_client"):
                # Competitor-led lenses: a competitor MUST be present, AND
                # (Change 19) competitor presence alone is no longer
                # sufficient on its own — the dry run showed this exact gap:
                # 1001 accepted articles, 100% from "competitor" queries,
                # dominated by stock-price/valuation/board-meeting noise that
                # happened to mention a competitor name. Now requires a
                # strong business-event signal OR a high/medium-quality
                # source_type in addition to the competitor match.
                if not competitor:
                    dropped_no_competitor_for_competitor_query += 1
                    dropped_by_type[query_type] = dropped_by_type.get(query_type, 0) + 1
                    query_stats[query_type]["dropped"] += 1
                    dropped_by_reason["dropped_no_competitor"] = dropped_by_reason.get("dropped_no_competitor", 0) + 1
                    continue

                strong_event = has_strong_business_event_signal(title, source)
                high_quality_source = is_high_quality_source_type(source_metadata["source_type"])

                if not (strong_event or high_quality_source):
                    dropped_by_type[query_type] = dropped_by_type.get(query_type, 0) + 1
                    query_stats[query_type]["dropped"] += 1
                    dropped_by_reason["dropped_low_quality_no_event_signal"] = dropped_by_reason.get("dropped_low_quality_no_event_signal", 0) + 1
                    continue

                accepted_competitor_led_articles += 1
                accepted_by_type[query_type] = accepted_by_type.get(query_type, 0) + 1
                query_stats[query_type]["accepted"] += 1
                accepted_by_gate = "accepted_competitor_event" if strong_event else "accepted_high_quality_competitor_source"
            elif query_type in SITE_OFFICIAL_QUERY_TYPES:
                # Change 4 Part E — official/high-authority site: lenses.
                # Recall-first: the query already targets an authoritative domain,
                # so accept by DEFAULT even when the title carries no competitor/
                # SBU/client/theme keyword (official titles are often generic, e.g.
                # "Outcome of Board Meeting", "Cabinet approves"). Relevance is
                # validated downstream by the LLM processor.
                has_signal = bool(
                    (competitor and competitor != "-")
                    or (sbu and sbu != "General")
                    or client_authority or strategic_theme
                )

                # Per-type competitor/SBU defaults.
                if query_type == "site_government_policy":
                    competitor = "-"  # policy news is not competitor-attributed
                elif not competitor:
                    competitor = "-"
                if not sbu:
                    sbu = "General"

                base_label = SITE_QUERY_GATE_LABELS[query_type]
                if has_signal:
                    accepted_by_gate = base_label + "_with_signal"
                    accepted_by_site_query_with_signal += 1
                else:
                    accepted_by_gate = base_label + "_no_title_signal"
                    accepted_by_site_query_without_signal += 1

                needs_llm_relevance_validation = True
                accepted_by_type[query_type] = accepted_by_type.get(query_type, 0) + 1
                query_stats[query_type]["accepted"] += 1
            elif query_type == SITE_SPECIALIST_QUERY_TYPE:
                # Change 4 Part E — specialist media: useful but less authoritative
                # than official sources, so require at least one title-level signal.
                has_signal = bool(
                    (competitor and competitor != "-")
                    or (sbu and sbu != "General")
                    or client_authority or strategic_theme
                )
                if not has_signal:
                    dropped_site_specialist_no_signal += 1
                    dropped_by_type[query_type] = dropped_by_type.get(query_type, 0) + 1
                    query_stats[query_type]["dropped"] += 1
                    dropped_by_reason["dropped_site_specialist_no_signal"] = dropped_by_reason.get("dropped_site_specialist_no_signal", 0) + 1
                    continue
                if not competitor:
                    competitor = "-"
                if not sbu:
                    sbu = "General"
                accepted_by_gate = SITE_QUERY_GATE_LABELS[query_type] + "_with_signal"
                accepted_by_site_query_with_signal += 1
                needs_llm_relevance_validation = True
                accepted_by_type[query_type] = accepted_by_type.get(query_type, 0) + 1
                query_stats[query_type]["accepted"] += 1
            else:
                # Non-competitor lenses (client_authority, strategic_theme, sbu, sbu_client):
                # accept on ANY relevance signal — SBU, client/authority, or strategic theme.
                # These carry tender / policy / authority / pipeline intelligence and must
                # NOT be dropped just because no competitor is named.
                if not (sbu or client_authority or strategic_theme):
                    dropped_no_signal_for_non_competitor_query += 1
                    dropped_by_type[query_type] = dropped_by_type.get(query_type, 0) + 1
                    query_stats[query_type]["dropped"] += 1
                    dropped_by_reason["dropped_no_signal"] = dropped_by_reason.get("dropped_no_signal", 0) + 1
                    continue
                if not competitor:
                    competitor = "-"
                accepted_non_competitor_signal_articles += 1
                accepted_by_type[query_type] = accepted_by_type.get(query_type, 0) + 1
                query_stats[query_type]["accepted"] += 1

                # Record WHICH non-competitor signal(s) let this article through.
                non_competitor_signals = []
                if sbu and sbu != "General":
                    non_competitor_signals.append("sbu_detected")
                if client_authority:
                    non_competitor_signals.append("client_authority_detected")
                if strategic_theme:
                    non_competitor_signals.append("strategic_theme_detected")
                if len(non_competitor_signals) > 1:
                    accepted_by_gate = "multiple_non_competitor_signals"
                elif len(non_competitor_signals) == 1:
                    accepted_by_gate = non_competitor_signals[0]
                else:
                    accepted_by_gate = ""

            if not sbu:
                sbu = "General"  # Let LLM decide relevance instead of dropping

            seen_links.add(link)

            # TODO: Add search_query_type / detected_client_authority /
            #       detected_strategic_theme columns to raw_scraped_articles in a
            #       later schema update. For now these ride on the dict and are
            #       ignored by the (explicit-column) insert in save_to_database().
            all_articles.append({
                "search_keyword": keyword,
                "search_query": keyword,
                "search_query_type": query_type,
                "detected_client_authority": client_authority or "",
                "detected_strategic_theme": strategic_theme or "",
                "accepted_by_gate": accepted_by_gate,
                # TODO: Persist needs_llm_relevance_validation if used downstream
                # (add column to raw_scraped_articles + insert + processor SELECT).
                "needs_llm_relevance_validation": needs_llm_relevance_validation,
                "news_title": title,
                "source": source,
                "link": link,
                "published_date": pubdate,
                "sbu": sbu or "General",
                "competitor": competitor or "-",
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
    logging.info(f"Found {len(all_articles)} relevant articles (multi-lens acceptance)")
    logging.info(
        "Acceptance summary: competitor_led=%s, non_competitor_signal=%s, "
        "dropped_no_competitor=%s, dropped_no_signal=%s",
        accepted_competitor_led_articles,
        accepted_non_competitor_signal_articles,
        dropped_no_competitor_for_competitor_query,
        dropped_no_signal_for_non_competitor_query,
    )
    logging.info(
        "Site-query acceptance (Part E): accepted_without_signal=%s, "
        "accepted_with_signal=%s, dropped_specialist_no_signal=%s",
        accepted_by_site_query_without_signal,
        accepted_by_site_query_with_signal,
        dropped_site_specialist_no_signal,
    )
    logging.info("Accepted by query_type: %s", accepted_by_type)
    logging.info("Dropped by query_type: %s", dropped_by_type)

    # Change 19: accepted-by-gate-label / dropped-by-reason summaries.
    accepted_by_gate_counts = {}
    for a in all_articles:
        g = a.get("accepted_by_gate") or "unlabeled"
        accepted_by_gate_counts[g] = accepted_by_gate_counts.get(g, 0) + 1
    logging.info("Accepted by gate label: %s", accepted_by_gate_counts)
    logging.info("Dropped by reason: %s", dropped_by_reason)

    total_duplicate_skips = sum(s["duplicate_link_skips"] for s in query_stats.values())
    logging.info("Total duplicate link skips: %s", total_duplicate_skips)
    logging.info("DRY_RUN mode: %s", DRY_RUN)
    # Yield per 100 queries generated, by query_type — the key rollout signal:
    # tells you which lenses are worth their fetch budget.
    yield_per_100 = {}
    for qtype, qcount in query_type_counts.items():
        acc = accepted_by_type.get(qtype, 0)
        yield_per_100[qtype] = round((acc / qcount) * 100, 1) if qcount else 0.0
    logging.info("Accepted per 100 queries by query_type: %s", yield_per_100)

    # Do not filter articles based on source authority here.
    # Low-authority sources are still useful for recall.
    # Ranking and executive display will decide priority later.

    # Per-run source diagnostics (once per run, not per article).
    source_type_counts = {}
    source_match_method_counts = {}
    for a in all_articles:
        st = a.get("source_type", "unknown")
        mm = a.get("source_match_method", "default")
        source_type_counts[st] = source_type_counts.get(st, 0) + 1
        source_match_method_counts[mm] = source_match_method_counts.get(mm, 0) + 1
    logging.info("Source type distribution: %s", source_type_counts)
    logging.info("Source match method distribution: %s", source_match_method_counts)

    # ── Change 4 Part D: run-level yield analytics by query_type ──────────────
    # Diagnostics only — used to tune query caps, not to filter/rank/dedup.
    for qtype, stats in query_stats.items():
        raw_seen = stats["raw_items_seen"]
        accepted = stats["accepted"]
        acceptance_rate = round((accepted / raw_seen) * 100, 2) if raw_seen else 0

        logging.info(
            "Query type '%s': generated=%s, fetch_attempts=%s, success=%s, failed=%s, "
            "raw_seen=%s, accepted=%s, dropped=%s, duplicate_skips=%s, acceptance_rate=%s%%",
            qtype,
            stats["queries_generated"],
            stats["fetch_attempts"],
            stats["fetch_success"],
            stats["fetch_failed"],
            stats["raw_items_seen"],
            stats["accepted"],
            stats["dropped"],
            stats["duplicate_link_skips"],
            acceptance_rate
        )

    total_queries = sum(s["queries_generated"] for s in query_stats.values())
    total_accepted = sum(s["accepted"] for s in query_stats.values())
    total_dropped = sum(s["dropped"] for s in query_stats.values())
    top_3_by_accepted = sorted(
        query_stats.items(), key=lambda kv: kv[1]["accepted"], reverse=True
    )[:3]

    logging.info(
        "Run-level yield summary: total_queries=%s, total_accepted=%s, total_dropped=%s",
        total_queries, total_accepted, total_dropped
    )
    logging.info(
        "Top 3 query types by accepted count: %s",
        [(qtype, stats["accepted"]) for qtype, stats in top_3_by_accepted]
    )

    return all_articles


def get_db_connection():
    """Get database connection from environment variable"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    
    return psycopg.connect(database_url, row_factory=dict_row)

def log_pipeline_run(stage, status, articles_in=None, articles_out=None, error_message=None):
    """Insert a row into pipeline_runs for observability. Never raises."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if status in ("success", "failed"):
            cur.execute("""
                INSERT INTO pipeline_runs
                    (pipeline_id, stage, status, articles_in, articles_out, error_message, started_at, ended_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (PIPELINE_ID, stage, status, articles_in, articles_out, error_message))
        else:
            cur.execute("""
                INSERT INTO pipeline_runs
                    (pipeline_id, stage, status, articles_in, articles_out, error_message, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (PIPELINE_ID, stage, status, articles_in, articles_out, error_message))

        conn.commit()
        cur.close()

    except Exception as e:
        logging.warning(f"log_pipeline_run failed for stage={stage}, status={status}: {e}")

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def log_dry_run_samples(articles):
    """Print a small sample of accepted articles for dry-run review (Change 4 Part H)."""
    try:
        if not articles:
            logging.info("DRY RUN: no accepted articles to sample")
            return
        logging.info("DRY RUN: showing up to %s accepted article samples", DRY_RUN_SAMPLE_ARTICLES)
        for i, article in enumerate(articles[:DRY_RUN_SAMPLE_ARTICLES], start=1):
            logging.info(
                "DRY RUN SAMPLE %s | title=%s | source=%s | query_type=%s | accepted_by=%s | competitor=%s | sbu=%s | source_type=%s",
                i,
                article.get("news_title"),
                article.get("source"),
                article.get("search_query_type"),
                article.get("accepted_by_gate"),
                article.get("competitor"),
                article.get("sbu"),
                article.get("source_type"),
            )
    except Exception as e:
        logging.warning("Could not log dry-run samples: %s", e)


def save_to_database(articles: List[Dict]):
    """Save scraped articles to PostgreSQL database - raw_scraped_articles table"""
    if not articles:
        logging.info("No articles to save")
        return
    
    # Change 16: rolling window instead of an exact single-day match (was
    # hardcoded to "yesterday" only — see SAVE_WINDOW_DAYS above for why).
    from datetime import date, timedelta
    window_start = date.today() - timedelta(days=SAVE_WINDOW_DAYS)
    window_end = date.today()
    articles = [a for a in articles if window_start <= a['published_date'].date() <= window_end]
    logging.info(f"After date filter ({window_start} to {window_end}): {len(articles)} articles")
    
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
    # ------------------------------------------------------------------
    # SQL schema update required (run once before deploying this change):
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS search_query TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS search_query_type TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS detected_client_authority TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS detected_strategic_theme TEXT;
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS accepted_by_gate TEXT;
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SQL schema update required (Change 16 — run once before deploying):
    #   ALTER TABLE raw_scraped_articles ADD COLUMN IF NOT EXISTS processing_status TEXT DEFAULT 'pending';
    #   UPDATE raw_scraped_articles SET processing_status = 'pending' WHERE processing_status IS NULL;
    #   CREATE INDEX IF NOT EXISTS idx_raw_scraped_articles_processing_status ON raw_scraped_articles(processing_status);
    # ------------------------------------------------------------------
    insert_query = """
        INSERT INTO raw_scraped_articles (
            search_keyword, news_title, source, link, published_date,
            sbu, competitor, content,
            source_domain, source_type, source_category, source_priority,
            source_authority_score, preferred_for_executive_summary,
            source_notes, source_match_method,
            search_query_type, detected_client_authority, detected_strategic_theme,
            search_query, accepted_by_gate
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
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
                article.get("search_query_type", "unknown"),
                article.get("detected_client_authority", ""),
                article.get("detected_strategic_theme", ""),
                article.get("search_query") or article.get("search_keyword"),
                article.get("accepted_by_gate", ""),
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
    """Main async scraping function with stage tracking."""
    logging.info("=" * 60)
    logging.info("Starting Competitor News Scraping Job (Async)")
    logging.info("=" * 60)

    log_pipeline_run("scrape_started", "started")

    try:
        # Load keywords from Excel
        keywords_data = load_keywords_from_excel()

        competitor_keywords = keywords_data["competitor_keywords"]
        sbu_keywords = keywords_data["sbu_keywords"]
        competitor_to_sbu = keywords_data["competitor_to_sbu"]

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

        articles_count = len(articles) if articles else 0

        if articles_count == 0:
            logging.warning("Google News returned zero items across all queries")
            log_pipeline_run(
                "scrape_completed",
                "failed",
                articles_out=0,
                error_message="zero_items_all_queries"
            )
            return

        log_pipeline_run("scrape_completed", "success", articles_out=articles_count)

        # Change 21: BSE official-filing ingestion (runs after RSS scraping).
        # Fetches real corporate filings for BSE-listed competitors via the
        # direct BSE API — proper headlines, PDF-extracted content, zero LLM
        # tokens. Articles are shaped identically to RSS articles so they flow
        # through save_to_database, then Stage 1/2, clustering, and ranking
        # unchanged. On any error the scraper continues normally (BSE ingestion
        # is additive, not load-bearing).
        if not DRY_RUN:
            try:
                bse_articles = fetch_bse_filing_articles()
                if bse_articles:
                    logging.info("BSE: adding %s filing articles to save batch.", len(bse_articles))
                    articles.extend(bse_articles)
            except Exception as e:
                logging.warning("BSE filing ingestion failed (non-fatal, scraper continues): %s", e)

        if DRY_RUN:
            logging.info("DRY RUN MODE: skipping database save")
            log_dry_run_samples(articles)
            logging.info("=" * 70)
            logging.info("DRY RUN COMPLETE")
            logging.info("Accepted articles: %s", len(articles))
            logging.info("Database write: skipped")
            logging.info("=" * 70)
            return

        # Save to database
        log_pipeline_run("save_raw_articles", "started", articles_in=articles_count)

        try:
            save_to_database(articles)
            log_pipeline_run(
                "save_raw_articles",
                "success",
                articles_in=articles_count,
                articles_out=articles_count
            )   
            logging.info("Database write: completed")  # Change 19: symmetric with the DRY_RUN "skipped" log
        except Exception as e:
            log_pipeline_run("save_raw_articles", "failed", error_message=str(e))
            logging.exception("save_to_database failed")
            raise

        logging.info("=" * 60)
        logging.info("Scraping Job Complete")
        logging.info("=" * 60)

    except Exception as e:
        log_pipeline_run("scrape_completed", "failed", error_message=str(e))
        logging.exception("Scraper main_async failed")
        raise

def main():
    """Entry point for the scraper"""
    # Run async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
