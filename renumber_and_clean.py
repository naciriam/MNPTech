import os
import re

KB_DIR = r"d:\MicroNanoTech\kb_files"

def main():
    print("Début du nettoyage et de la renumérotation...")
    
    # 1. Lister tous les fichiers Markdown
    all_files = [f for f in os.listdir(KB_DIR) if f.endswith('.md')]
    
    seen_topics = set()
    files_to_process = []
    
    # 2. Identifier et supprimer les doublons
    for filename in all_files:
        # Extraire le nom du sujet après "KB-XXX_"
        if "_" in filename:
            topic_slug = filename.split("_", 1)[1].replace(".md", "")
        else:
            topic_slug = filename.replace(".md", "")
            
        topic_lower = topic_slug.lower()
        filepath = os.path.join(KB_DIR, filename)
        
        if topic_lower in seen_topics:
            print(f"Doublon trouvé et supprimé : {filename}")
            os.remove(filepath)
        else:
            seen_topics.add(topic_lower)
            # Extraire l'ancien numéro pour le tri
            try:
                old_num = int(re.search(r'\d+', filename.split('_')[0]).group())
            except (IndexError, AttributeError, ValueError):
                old_num = 999999
            
            files_to_process.append((old_num, topic_slug, filename))
            
    # Trier par ancien numéro pour conserver un ordre logique
    files_to_process.sort(key=lambda x: x[0])
    
    # 3. Renuméroter et mettre à jour le contenu
    print(f"\nRenumérotation de {len(files_to_process)} fichiers...")
    
    # On utilise un préfixe temporaire pour éviter les conflits de noms pendant le renommage
    temp_files = []
    for index, (old_num, topic_slug, old_filename) in enumerate(files_to_process, 1):
        old_filepath = os.path.join(KB_DIR, old_filename)
        temp_filename = f"temp_KB-{index}_{topic_slug}.md"
        temp_filepath = os.path.join(KB_DIR, temp_filename)
        
        # Mettre à jour le titre H1 à l'intérieur du fichier
        with open(old_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if lines and lines[0].startswith("# KB-"):
            # Remplacer le titre H1
            clean_title = topic_slug.replace('_', ' ')
            lines[0] = f"# KB-{index} — {clean_title}\n"
            
        # Écrire dans le fichier temporaire
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        # Supprimer l'ancien fichier s'il a un nom différent de temp
        if old_filepath != temp_filepath:
            os.remove(old_filepath)
            
        temp_files.append((temp_filepath, f"KB-{index}_{topic_slug}.md"))

    # 4. Enlever le préfixe 'temp_' pour avoir les noms finaux
    for temp_filepath, final_filename in temp_files:
        final_filepath = os.path.join(KB_DIR, final_filename)
        os.rename(temp_filepath, final_filepath)
        
    print(f"Opération terminée. {len(files_to_process)} fichiers ont été conservés et renumérotés de KB-1 à KB-{len(files_to_process)}.")
    
if __name__ == "__main__":
    main()
