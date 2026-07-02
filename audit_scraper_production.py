from pathlib import Path
import re
import sys

FILE = Path("scraper_production.py")


SECTIONS = {
    "Change 2 - Google News Query Logic": [
        ("Google News RSS URL has q=", r"news\.google\.com/rss/search\?q="),
        ("Uses encoded query", ["encoded_query", "quote(query)", "quote(keyword)"]),
        ("Uses lookback_days / when:", ["lookback_days", "when:"]),
        ("Preserves India English locale", ["hl=en-IN", "gl=IN", "ceid=IN:en"]),
    ],

    "Change 3 - Source Registry": [
        ("SOURCE_REGISTRY exists", "SOURCE_REGISTRY"),
        ("Registry supports source_names", "source_names"),
        ("Registry supports domains", "domains"),
        ("source_type exists", "source_type"),
        ("source_category exists", "source_category"),
        ("source_priority exists", "source_priority"),
        ("source_authority_score exists", "source_authority_score"),
        ("preferred_for_executive_summary exists", "preferred_for_executive_summary"),
        ("source_notes exists", "source_notes"),
        ("normalize_source_name() exists", "def normalize_source_name"),
        ("extract_domain() exists", "def extract_domain"),
        ("domain_matches() exists", "def domain_matches"),
        ("source_name_matches() exists", "def source_name_matches"),
        ("classify_source() exists", "def classify_source"),
        ("source_match_method exists", "source_match_method"),
        ("Default source metadata helper exists", "get_default_source_metadata"),
    ],

    "Change 3 - Source Metadata Persistence": [
        ("source_domain attached/saved", "source_domain"),
        ("source_type attached/saved", "source_type"),
        ("source_category attached/saved", "source_category"),
        ("source_priority attached/saved", "source_priority"),
        ("source_authority_score attached/saved", "source_authority_score"),
        ("preferred_for_executive_summary attached/saved", "preferred_for_executive_summary"),
        ("source_notes attached/saved", "source_notes"),
        ("source_match_method attached/saved", "source_match_method"),
        ("raw_scraped_articles referenced", "raw_scraped_articles"),
    ],

    "Change 4A - Multi-lens Search": [
        ("CLIENT_AUTHORITY_KEYWORDS exists", "CLIENT_AUTHORITY_KEYWORDS"),
        ("STRATEGIC_THEME_KEYWORDS exists", "STRATEGIC_THEME_KEYWORDS"),
        ("generate_search_queries() exists", "def generate_search_queries"),
        ("detect_client_authority() exists", "def detect_client_authority"),
        ("detect_strategic_theme() exists", "def detect_strategic_theme"),
        ("competitor_sbu query type exists", "competitor_sbu"),
        ("competitor_client query type exists", "competitor_client"),
        ("sbu_client query type exists", "sbu_client"),
        ("client_authority query type exists", "client_authority"),
        ("strategic_theme query type exists", "strategic_theme"),
        ("accepted_by_gate exists", "accepted_by_gate"),
        ("competitor_detected gate exists", "competitor_detected"),
    ],

    "Change 4A - Controlled Query Caps": [
        ("MAX_INDIVIDUAL_SBU_QUERIES exists", "MAX_INDIVIDUAL_SBU_QUERIES"),
        ("MAX_COMPETITOR_SBU_QUERIES exists", "MAX_COMPETITOR_SBU_QUERIES"),
        ("MAX_COMPETITOR_CLIENT_QUERIES exists", "MAX_COMPETITOR_CLIENT_QUERIES"),
        ("MAX_SBU_CLIENT_QUERIES exists", "MAX_SBU_CLIENT_QUERIES"),
        ("MAX_TOTAL_SEARCH_QUERIES exists", "MAX_TOTAL_SEARCH_QUERIES"),
    ],

    "Change 4B - Search Lens Metadata": [
        ("search_query exists", "search_query"),
        ("search_query_type exists", "search_query_type"),
        ("detected_client_authority exists", "detected_client_authority"),
        ("detected_strategic_theme exists", "detected_strategic_theme"),
        ("accepted_by_gate exists", "accepted_by_gate"),
    ],

    "Change 4D - Yield Analytics": [
        ("query_stats exists", "query_stats"),
        ("queries_generated exists", "queries_generated"),
        ("fetch_attempts exists", "fetch_attempts"),
        ("fetch_success exists", "fetch_success"),
        ("fetch_failed exists", "fetch_failed"),
        ("raw_items_seen exists", "raw_items_seen"),
        ("accepted counter exists", "accepted"),
        ("dropped counter exists", "dropped"),
        ("duplicate_link_skips exists", "duplicate_link_skips"),
        ("acceptance_rate exists", "acceptance_rate"),
    ],

    "Change 4E - Site-specific Queries": [
        ("SITE_OFFICIAL_QUERY_TYPES exists", "SITE_OFFICIAL_QUERY_TYPES"),
        ("SITE_SPECIALIST_QUERY_TYPE exists", "SITE_SPECIALIST_QUERY_TYPE"),
        ("site_official_exchange exists", "site_official_exchange"),
        ("site_company_official exists", "site_company_official"),
        ("site_client_authority exists", "site_client_authority"),
        ("site_government_policy exists", "site_government_policy"),
        ("site_tender exists", "site_tender"),
        ("site_specialist_media exists", "site_specialist_media"),
        ("generate_site_specific_queries() exists", "def generate_site_specific_queries"),
        ("OFFICIAL_EXCHANGE_DOMAINS exists", "OFFICIAL_EXCHANGE_DOMAINS"),
        ("COMPANY_OFFICIAL_DOMAINS exists", "COMPANY_OFFICIAL_DOMAINS"),
        ("CLIENT_AUTHORITY_DOMAINS exists", "CLIENT_AUTHORITY_DOMAINS"),
        ("GOVERNMENT_POLICY_DOMAINS exists", "GOVERNMENT_POLICY_DOMAINS"),
        ("TENDER_DOMAINS exists", "TENDER_DOMAINS"),
        ("SPECIALIST_MEDIA_DOMAINS exists", "SPECIALIST_MEDIA_DOMAINS"),
        ("needs_llm_relevance_validation exists", "needs_llm_relevance_validation"),
        ("with signal gate exists", "_with_signal"),
        ("no title signal gate exists", "_no_title_signal"),
        ("specialist no-signal counter exists", "dropped_site_specialist_no_signal"),
    ],

    "Change 4E - Site Query Merge": [
        ("base_queries generated", r"base_queries\s*=.*generate_search_queries"),
        ("site_queries generated", r"site_queries\s*=.*generate_site_specific_queries"),
    ],
}


def read():
    if not FILE.exists():
        print(f"❌ Missing file: {FILE}")
        sys.exit(1)
    return FILE.read_text(encoding="utf-8", errors="ignore")


def match(text, pattern):
    if isinstance(pattern, list):
        return any(p in text for p in pattern)

    if "\\" in pattern or pattern.startswith("^") or ".*" in pattern or r"\s" in pattern:
        return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None

    return pattern in text


def main():
    text = read()
    total_failures = 0

    print("\n=== Audit: scraper_production.py ===")

    for section, checks in SECTIONS.items():
        print(f"\n{section}")
        print("-" * len(section))

        failures = 0

        for label, pattern in checks:
            ok = match(text, pattern)
            if ok:
                print(f"✅ {label}")
            else:
                print(f"❌ {label}")
                failures += 1

        total_failures += failures

    print("\nSummary:")
    print(f"Total failures: {total_failures}")

    if total_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()