"""
Throwaway test: verify source classification on a small live sample.
Runs the REAL scraper fetch + classify_source, prints results, writes nothing to the DB.
Run:  python test_classify.py
"""

import asyncio
import aiohttp
import logging

from scraper_production import (
    fetch_feed_async,
    classify_source,
    LOOKBACK_DAYS,
)
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# A few keywords likely to return well-known publishers + some long-tail ones
TEST_KEYWORDS = ["Larsen & Toubro", "RVNL order", "Power Grid transmission"]


async def run():
    semaphore = asyncio.Semaphore(3)
    results_rows = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_feed_async(session, kw, LOOKBACK_DAYS, semaphore)
            for kw in TEST_KEYWORDS
        ]
        feed_results = await asyncio.gather(*tasks)

    for result in feed_results:
        if not result["success"] or not result["feed"] or not result["feed"].entries:
            print(f"⚠️  No entries for keyword: {result['keyword']}")
            continue

        keyword = result["keyword"]
        # Only look at the first few entries per keyword to keep output short
        for entry in result["feed"].entries[:5]:
            raw_title = entry.get("title", "")
            link = entry.get("link", "")

            # Extract publisher display name exactly like the real scraper does
            source = ""
            if "description" in entry:
                soup = BeautifulSoup(entry.description, "html.parser")
                font_tag = soup.find("font")
                if font_tag:
                    source = font_tag.text.strip()

            meta = classify_source(link, source)
            results_rows.append({
                "keyword": keyword,
                "source": source,
                "domain": meta["source_domain"],
                "type": meta["source_type"],
                "score": meta["source_authority_score"],
                "method": meta["source_match_method"],
            })

    # Pretty print
    print("\n" + "=" * 110)
    print(f"{'PUBLISHER':<28}{'TYPE':<32}{'SCORE':<7}{'METHOD':<14}{'DOMAIN'}")
    print("=" * 110)
    for r in results_rows:
        print(f"{(r['source'] or '(none)'):<28}{r['type']:<32}{str(r['score']):<7}{r['method']:<14}{r['domain']}")
    print("=" * 110)

    # Quick summary so you can see match rate at a glance
    total = len(results_rows)
    matched = sum(1 for r in results_rows if r["method"] != "default")
    print(f"\nTotal sampled: {total}   |   Classified (non-default): {matched}   |   Unknown/default: {total - matched}")


if __name__ == "__main__":
    asyncio.run(run())