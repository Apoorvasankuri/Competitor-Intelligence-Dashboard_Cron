import psycopg
from psycopg.rows import dict_row
import pandas as pd
import os
import requests

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable not set")

conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

cur = conn.cursor()
cur.execute("""
    SELECT * FROM processed_articles 
    WHERE published_date >= '2026-02-25' 
    AND published_date <= '2026-02-28 23:59:59'
    ORDER BY published_date DESC
""")
results = cur.fetchall()
cur.close()
conn.close()

df = pd.DataFrame(results)
csv_string = df.to_csv(index=False)

# Try transfer.sh
response = requests.put(
    "https://transfer.sh/results_new.csv",
    data=csv_string.encode('utf-8'),
    headers={"Content-Type": "text/csv"}
)

if response.status_code == 200:
    print(f"✅ Exported {len(df)} rows")
    print(f"📥 Download link: {response.text.strip()}")
else:
    # Fallback: try 0x0.st
    response2 = requests.post(
        "https://0x0.st",
        files={"file": ("results_new.csv", csv_string.encode('utf-8'), "text/csv")}
    )
    if response2.status_code == 200:
        print(f"✅ Exported {len(df)} rows")
        print(f"📥 Download link: {response2.text.strip()}")
    else:
        print(f"❌ All uploads failed")
        print(f"Transfer.sh: {response.status_code} - {response.text[:200]}")
        print(f"0x0.st: {response2.status_code} - {response2.text[:200]}")