import sqlite3
import os

SQLITE_PATH = "d:/MicroNanoTech/ai_agent/chroma_old.sqlite3"
KB_DIR = "d:/MicroNanoTech/kb_files"

conn = sqlite3.connect(SQLITE_PATH)
c = conn.cursor()

# 1. Lister les tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in c.fetchall()]
print("Tables:", tables)

# 2. Inspecter chaque table
for table in tables:
    c.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in c.fetchall()]
    c.execute(f"SELECT count(*) FROM {table}")
    count = c.fetchone()[0]
    print(f"\nTable '{table}': {count} lignes, colonnes: {cols}")

conn.close()
