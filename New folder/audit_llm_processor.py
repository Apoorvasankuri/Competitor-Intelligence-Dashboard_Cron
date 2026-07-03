from pathlib import Path
import re
import sys

FILE = Path("llm_processor_production.py")

CHECKS = {
    "Source Metadata Propagation": [
        "source_domain",
        "source_type",
        "source_category",
        "source_priority",
        "source_authority_score",
        "preferred_for_executive_summary",
        "source_notes",
        "source_match_method",
    ],

    "Source Score Ranking": [
        "source_authority_score",
        "rank_score",
    ],

    "Search Lens Metadata Propagation": [
        "search_query",
        "search_query_type",
        "detected_client_authority",
        "detected_strategic_theme",
        "accepted_by_gate",
    ],

    "Search Lens Analytics": [
        "value_counts",
        "search_query_type",
    ],

    "Existing Core Pipeline Still Present": [
        "QUICK_SCORE_PROMPT",
        "batch_relevance_score",
        "batch_full_analysis",
        "calculate_rank_score",
        "deduplicate_articles",
        "generate_llm_summaries",
        "processed_articles",
    ],
}


def read():
    if not FILE.exists():
        print(f"❌ Missing file: {FILE}")
        sys.exit(1)
    return FILE.read_text(encoding="utf-8", errors="ignore")


def main():
    text = read()
    total_failures = 0

    print("\n=== Audit: llm_processor_production.py ===")

    for section, checks in CHECKS.items():
        print(f"\n{section}")
        print("-" * len(section))

        failures = 0
        for item in checks:
            if item in text:
                print(f"✅ {item}")
            else:
                print(f"❌ {item}")
                failures += 1

        total_failures += failures

    # Extra softer check
    print("\nRanking Pattern Check")
    print("-" * 21)
    if re.search(r"rank_score.*source_authority_score|source_authority_score.*rank_score", text, re.DOTALL):
        print("✅ source_authority_score appears connected to rank_score")
    else:
        print("⚠️ source_authority_score exists but may not be added to rank_score")

    print("\nSummary:")
    print(f"Total failures: {total_failures}")

    if total_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()