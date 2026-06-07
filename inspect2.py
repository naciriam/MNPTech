import sqlite3
import os

db_path = "chroma_old.sqlite3"
print("CWD:", os.getcwd())
print("Exists:", os.path.exists(db_path))
print("Size:", os.path.getsize(db_path))

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in c.fetchall()]
print("Tables:", tables)

for table in tables:
    c.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in c.fetchall()]
    c.execute(f"SELECT count(*) FROM {table}")
    count = c.fetchone()[0]
    print(f"Table '{table}': {count} lignes, colonnes: {cols}")

conn.close()
