"""
Script de restauration forensique des fichiers Markdown depuis l'ancienne base ChromaDB.
Il lit le fichier chroma_old.sqlite3 et reconstitue les fichiers .md originaux.
"""
import sqlite3
import os
import re
from collections import defaultdict

SQLITE_PATH = r"d:\MicroNanoTech\ai_agent\chroma_old.sqlite3"
KB_DIR = r"d:\MicroNanoTech\kb_files"

def main():
    if not os.path.exists(SQLITE_PATH):
        print(f"Erreur : fichier introuvable -> {SQLITE_PATH}")
        return

    file_size = os.path.getsize(SQLITE_PATH)
    print(f"Fichier SQLite trouve : {file_size} octets")

    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()

    # Lister les tables disponibles dans cette base ChromaDB
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables trouvees : {tables}")

    # Chercher la table qui contient les documents textuels
    # ChromaDB utilise generalement 'embedding_fulltext_search_data' ou 'embeddings'
    text_data = defaultdict(list)

    if 'embedding_fulltext_search_data' in tables:
        print("Lecture depuis embedding_fulltext_search_data...")
        cursor.execute("SELECT rowid, c0, c1 FROM embedding_fulltext_search_data")
        rows = cursor.fetchall()
        print(f"{len(rows)} lignes trouvees.")
        for rowid, doc, meta in rows:
            if doc:
                text_data[rowid].append((doc, meta))

    elif 'embeddings' in tables:
        print("Lecture depuis embeddings...")
        # Essayons de voir les colonnes disponibles
        cursor.execute("PRAGMA table_info(embeddings)")
        cols = [c[1] for c in cursor.fetchall()]
        print(f"Colonnes : {cols}")

    # Chercher dans toutes les tables la colonne 'document' ou 'string_value'
    for table in tables:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [c[1] for c in cursor.fetchall()]
            if 'string_value' in cols:
                print(f"Table '{table}' contient 'string_value' - Colonnes: {cols}")
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample = cursor.fetchall()
                for row in sample:
                    print(f"  Sample: {str(row)[:200]}")
        except Exception as e:
            print(f"  Erreur sur {table}: {e}")

    conn.close()

    if text_data:
        print(f"\n{len(text_data)} chunks trouves. Reconstruction des fichiers...")
        reconstruct_files(text_data)
    else:
        print("\nAucune donnee recuperee avec la methode standard.")
        print("Lancement de l'analyse approfondie...")
        deep_analysis()

def deep_analysis():
    """Analyse approfondie de toutes les tables pour trouver le texte."""
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            rows = cursor.fetchall()
            if rows:
                print(f"\n=== TABLE: {table} ===")
                for row in rows:
                    for cell in row:
                        if isinstance(cell, str) and len(cell) > 50:
                            print(f"  Texte long trouve: {cell[:300]}...")
        except Exception as e:
            pass

    conn.close()

def reconstruct_files(text_data):
    """Regroupe les chunks par fichier source et ecrit les .md."""
    # Grouper par nom de fichier source
    files_content = defaultdict(list)

    for rowid, chunks in text_data.items():
        for doc, meta in chunks:
            # Extraire le nom du fichier depuis les metadonnees
            filename = extract_filename(meta, doc)
            files_content[filename].append(doc)

    print(f"\n{len(files_content)} fichiers uniques identifies.")
    restored = 0
    for filename, chunks in files_content.items():
        if filename == "unknown":
            continue
        filepath = os.path.join(KB_DIR, filename)
        # Ne restaurer que les fichiers vides
        if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
            full_content = "\n\n".join(chunks)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)
            restored += 1
            print(f"  Restaure: {filename} ({len(chunks)} chunks)")

    print(f"\nTotal restaure : {restored} fichiers.")

def extract_filename(meta, doc):
    """Extrait le nom du fichier source depuis les metadonnees ou le texte."""
    if meta:
        # Les metadonnees ChromaDB sont souvent du JSON ou une chaine avec 'source'
        match = re.search(r'KB-\d+[^"\']+\.md', str(meta))
        if match:
            return os.path.basename(match.group(0))
    # Chercher dans le texte lui-meme
    match = re.search(r'KB-\d+[^"\']+\.md', str(doc))
    if match:
        return os.path.basename(match.group(0))
    return "unknown"

if __name__ == "__main__":
    main()
