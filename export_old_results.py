import psycopg
from psycopg.rows import dict_row
import os

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable not set")

conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS results_new")
cur.execute("""
    CREATE TABLE results_new AS
    SELECT * FROM processed_articles 
    WHERE published_date >= '2026-02-25' 
    AND published_date <= '2026-02-28 23:59:59'
""")

conn.commit()

cur.execute("SELECT COUNT(*) as cnt FROM results_new")
count = cur.fetchone()['cnt']

print(f"✅ Copied {count} rows to results_new table")

cur.close()
conn.close()