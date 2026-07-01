"""
Harvest the long tail: run more keywords, list ONLY unclassified publishers by frequency.
Writes nothing to the DB. Run:  python harvest_unknowns.py
"""
import asyncio, aiohttp, logging
from collections import Counter
from bs4 import BeautifulSoup
from scraper_production import fetch_feed_async, classify_source, LOOKBACK_DAYS

logging.basicConfig(level=logging.WARNING)  # quiet the per-keyword INFO spam

TEST_KEYWORDS = [
    "Larsen & Toubro", "RVNL order", "Power Grid transmission", "Kalpataru projects",
    "Tata Projects", "NCC order win", "Siemens Energy India", "Hitachi Energy",
    "SECI solar tender", "NHAI highway award", "DMRC metro contract", "NTPC order",
    "IRCON railway", "Sterlite Power", "Afcons infrastructure", "transmission line India",
    "substation project", "railway electrification", "solar EPC India", "HVDC project",
]

async def run():
    sem = asyncio.Semaphore(4)
    unknown_counter = Counter()
    known_counter = Counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed_async(session, kw, LOOKBACK_DAYS, sem) for kw in TEST_KEYWORDS]
        results = await asyncio.gather(*tasks)

    for result in results:
        if not result["success"] or not result["feed"]:
            continue
        for entry in result["feed"].entries:
            source = ""
            if "description" in entry:
                soup = BeautifulSoup(entry.description, "html.parser")
                font_tag = soup.find("font")
                if font_tag:
                    source = font_tag.text.strip()
            meta = classify_source(entry.get("link", ""), source)
            if meta["source_match_method"] == "default":
                unknown_counter[source or "(no publisher name)"] += 1
            else:
                known_counter[meta["source_type"]] += 1

    total_known = sum(known_counter.values())
    total_unknown = sum(unknown_counter.values())
    grand = total_known + total_unknown

    print("\n=== UNKNOWN PUBLISHERS (by frequency) ===")
    for name, cnt in unknown_counter.most_common(60):
        print(f"{cnt:>4}  {name}")

    print("\n=== KNOWN, by type ===")
    for t, cnt in known_counter.most_common():
        print(f"{cnt:>4}  {t}")

    print(f"\nTotal articles: {grand}  |  classified: {total_known} "
          f"({(100*total_known/grand if grand else 0):.0f}%)  |  unknown: {total_unknown}")

if __name__ == "__main__":
    asyncio.run(run())