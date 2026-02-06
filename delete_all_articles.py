import os
import psycopg

DATABASE_URL = os.environ.get('DATABASE_URL')

conn = psycopg.connect(DATABASE_URL)
cur = conn.cursor()

# Delete all articles
cur.execute("DELETE FROM competitor_data;")
rows_deleted = cur.rowcount

conn.commit()
cur.close()
conn.close()

print(f"✅ Deleted {rows_deleted} articles from database")