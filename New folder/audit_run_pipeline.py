from pathlib import Path
import re
import sys

FILE = Path("run_pipeline.py")

CHECKS = [
    ("Imports/uses scraper_main", "scraper_main"),
    ("Imports/uses llm_main", "llm_main"),
    ("Calls scraper_main()", r"scraper_main\s*\("),
    ("Calls llm_main()", r"llm_main\s*\("),
    ("Uses logging.exception", "logging.exception"),
    ("Uses sys.exit(main())", r"sys\.exit\s*\(\s*main\s*\(\s*\)\s*\)"),
]


def read():
    if not FILE.exists():
        print(f"❌ Missing file: {FILE}")
        sys.exit(1)
    return FILE.read_text(encoding="utf-8", errors="ignore")


def check_regex(text, pattern):
    return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None


def main():
    text = read()

    print("\n=== Audit: run_pipeline.py ===\n")
    failures = 0

    for label, pattern in CHECKS:
        if pattern.startswith("r\""):
            ok = check_regex(text, pattern)
        elif "\\" in pattern or pattern.endswith(r"\("):
            ok = check_regex(text, pattern)
        else:
            ok = pattern in text

        if ok:
            print(f"✅ {label}")
        else:
            print(f"❌ {label}")
            failures += 1

    print("\nSummary:")
    print(f"Failures: {failures}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()