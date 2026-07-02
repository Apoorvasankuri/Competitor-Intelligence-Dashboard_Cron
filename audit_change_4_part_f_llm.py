from pathlib import Path
import re
import sys

FILE = Path("llm_processor_production.py")


CHECKS = {
    "Part F - QUICK_SCORE_PROMPT updated for two intelligence types": [
        "Competitor intelligence",
        "Market opportunity",
        "tender",
        "client/authority",
        "government policy",
        "official filings",
        "project pipeline",
    ],

    "Part F - Updated scoring bands": [
        "90-100",
        "80-89",
        "70-79",
        "40-69",
        "0-39",
        "Critical intelligence",
        "High relevance",
        "Useful",
        "monitor",
        "Weak",
        "noise",
    ],

    "Part F - Non-competitor examples included": [
        "PGCIL",
        "SECI",
        "BESS",
        "Green Energy Corridor",
        "metro",
        "transmission",
        "tender",
        "policy",
    ],

    "Part F - Search/source context included in batch_relevance_score": [
        "search_query",
        "search_query_type",
        "accepted_by_gate",
        "detected_client_authority",
        "detected_strategic_theme",
        "source_type",
        "source_authority_score",
        "source_category",
    ],

    "Part F - Site-specific / official-source instruction": [
        "source-specific",
        "official",
        "vague",
        "generic titles",
        "query context",
        "monitor-level",
        "routine noise",
    ],

    "Part F - Threshold unchanged": [
        "RELEVANCE_THRESHOLD = 70",
    ],

    "Core processor still intact": [
        "QUICK_SCORE_PROMPT",
        "batch_relevance_score",
        "stage1_quick_scoring",
        "stage2_full_analysis",
        "calculate_rank_score",
        "deduplicate_articles",
        "generate_llm_summaries",
    ],
}


def read_file():
    if not FILE.exists():
        print(f"❌ Missing file: {FILE}")
        sys.exit(1)
    return FILE.read_text(encoding="utf-8", errors="ignore")


def contains_case_insensitive(text, needle):
    return needle.lower() in text.lower()


def main():
    text = read_file()
    total_failures = 0
    total_warnings = 0

    print("\n=== Audit: Change 4 Part F — llm_processor_production.py ===\n")

    for section, checks in CHECKS.items():
        print(section)
        print("-" * len(section))

        section_failures = 0

        for item in checks:
            if contains_case_insensitive(text, item):
                print(f"✅ {item}")
            else:
                print(f"❌ {item}")
                section_failures += 1

        total_failures += section_failures
        print()

    # Additional structural check: article block should include query/source context near Stage 1 prompt assembly.
    print("Structural checks")
    print("-----------------")

    context_terms = [
        "Search Query",
        "Search Query Type",
        "Accepted By Gate",
        "Detected Client",
        "Detected Strategic Theme",
        "Source Type",
        "Source Authority Score",
    ]

    structural_hits = sum(1 for term in context_terms if contains_case_insensitive(text, term))

    if structural_hits >= 5:
        print("✅ Stage 1 article formatting appears to include search/source context")
    else:
        print("⚠️ Stage 1 article formatting may not include enough search/source context")
        total_warnings += 1

    # Check that the relevance threshold wasn't changed downward/upward accidentally.
    threshold_match = re.search(r"RELEVANCE_THRESHOLD\s*=\s*(\d+)", text)
    if threshold_match:
        threshold_value = int(threshold_match.group(1))
        if threshold_value == 70:
            print("✅ RELEVANCE_THRESHOLD remains 70")
        else:
            print(f"⚠️ RELEVANCE_THRESHOLD is {threshold_value}, expected 70 for Part F")
            total_warnings += 1
    else:
        print("⚠️ Could not find RELEVANCE_THRESHOLD")
        total_warnings += 1

    # Check that ranking/dedup functions still exist.
    for fn in ["calculate_rank_score", "deduplicate_articles", "generate_llm_summaries"]:
        if f"def {fn}" in text:
            print(f"✅ {fn}() still exists")
        else:
            print(f"❌ {fn}() missing")
            total_failures += 1

    print("\nSummary")
    print("-------")
    print(f"Failures: {total_failures}")
    print(f"Warnings: {total_warnings}")

    if total_failures:
        print("\n❌ Audit failed. Fix missing Part F items first.")
        sys.exit(1)

    if total_warnings:
        print("\n⚠️ Audit passed with warnings. Review the warning items.")
        sys.exit(0)

    print("\n✅ Audit passed. Change 4 Part F appears structurally complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()