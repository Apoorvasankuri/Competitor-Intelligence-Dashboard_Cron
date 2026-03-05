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

response = requests.post(
    "https://file.io",
    files={"file": ("results_new.csv", csv_string, "text/csv")}
)

if response.status_code == 200:
    data = response.json()
    print(f"✅ Exported {len(df)} rows")
    print(f"📥 Download link: {data.get('link')}")
    print(f"⚠️  Link works ONCE — download immediately!")
else:
    print(f"❌ Upload failed: {response.text}")