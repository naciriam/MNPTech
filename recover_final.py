"""
Restauration forensique complete des fichiers Markdown depuis l'ancienne ChromaDB.
Regroupe les chunks par fichier source et reconstruit les .md originaux.
"""
import sqlite3
import os
from collections import defaultdict

DB_PATH = "D:/chroma_test.sqlite3"
KB_DIR = r"D:\MicroNanoTech\kb_files"

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Lire tous les chunks avec leur source
    print("Extraction des donnees depuis la base ChromaDB originale...")
    c.execute("""
        SELECT m1.id, m1.string_value as source, m2.string_value as document
        FROM embedding_metadata m1
        JOIN embedding_metadata m2 ON m1.id = m2.id
        WHERE m1.key = 'source'
        AND m2.key = 'chroma:document'
        ORDER BY m1.id
    """)
    rows = c.fetchall()
    conn.close()

    print(f"{len(rows)} chunks recuperes depuis la base.")

    # Regrouper les chunks par fichier source
    files_chunks = defaultdict(list)
    for chunk_id, source_path, document in rows:
        # Extraire le nom de fichier propre
        filename = os.path.basename(source_path)
        files_chunks[filename].append((chunk_id, document))

    print(f"{len(files_chunks)} fichiers uniques identifies.")

    restored = 0
    skipped = 0
    for filename, chunks in files_chunks.items():
        filepath = os.path.join(KB_DIR, filename)

        # Verifier si le fichier existe et est vide
        if not os.path.exists(filepath):
            print(f"  [SKIP] Fichier introuvable sur disque: {filename}")
            skipped += 1
            continue

        if os.path.getsize(filepath) > 0:
            # Deja rempli (par la restauration v2 ou original)
            skipped += 1
            continue

        # Trier les chunks par leur ID pour respecter l'ordre original
        chunks_sorted = [doc for _, doc in sorted(chunks, key=lambda x: x[0])]
        full_content = "\n\n".join(chunks_sorted)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

        restored += 1
        print(f"  [OK] {filename} ({len(chunks)} chunks, {len(full_content)} chars)")

    print(f"\nRestauration terminee !")
    print(f"  Fichiers restaures : {restored}")
    print(f"  Fichiers ignores   : {skipped}")

    # Compter les vides restants
    empty = sum(1 for f in os.listdir(KB_DIR) if f.endswith(".md") and os.path.getsize(os.path.join(KB_DIR, f)) == 0)
    print(f"  Fichiers vides restants : {empty}")

if __name__ == "__main__":
    main()
