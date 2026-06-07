"""
Recuperation forensique des fichiers Markdown depuis l'ancienne base ChromaDB.
Lit la table 'embedding_fulltext_search_content' qui stocke le texte original.
"""
import sqlite3
import os
import re
from collections import defaultdict

DB_PATH = "D:/chroma_test.sqlite3"
KB_DIR = r"D:\MicroNanoTech\kb_files"

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # La table 'embedding_fulltext_search_content' contient le texte
    # Colonnes typiques: id, c0 (document), c1 (metadata)
    print("Lecture du contenu textuel...")
    try:
        c.execute("SELECT * FROM embedding_fulltext_search_content LIMIT 2")
        sample = c.fetchall()
        print(f"Colonnes/sample: {sample[0] if sample else 'vide'}")
    except Exception as e:
        print(f"Erreur: {e}")

    # Essayons aussi embedding_metadata qui contient les chemins source
    print("\nLecture des metadonnees (chemins source)...")
    try:
        c.execute("PRAGMA table_info(embedding_metadata)")
        cols = [row[1] for row in c.fetchall()]
        print(f"Colonnes embedding_metadata: {cols}")
        c.execute("SELECT * FROM embedding_metadata LIMIT 5")
        for row in c.fetchall():
            print(f"  {row}")
    except Exception as e:
        print(f"Erreur: {e}")

    # Lire les embeddings avec leur texte
    print("\nLecture de la table embeddings...")
    try:
        c.execute("PRAGMA table_info(embeddings)")
        cols = [row[1] for row in c.fetchall()]
        print(f"Colonnes embeddings: {cols}")
        c.execute("SELECT * FROM embeddings LIMIT 2")
        for row in c.fetchall():
            print(f"  {str(row)[:200]}")
    except Exception as e:
        print(f"Erreur: {e}")

    conn.close()

if __name__ == "__main__":
    main()
