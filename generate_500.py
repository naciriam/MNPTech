import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

KB_DIR = r"d:\MicroNanoTech\kb_files"
INPUT_JSON = r"d:\MicroNanoTech\ai_agent\500_materials.json"

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
current_model_index = 0

def get_start_index():
    max_idx = 0
    for f in os.listdir(KB_DIR):
        if f.endswith('.md') and f.startswith('KB-'):
            try:
                num = int(f.split('_')[0].replace('KB-', ''))
                if num > max_idx:
                    max_idx = num
            except:
                pass
    return max_idx + 1

def main():
    global current_model_index
    
    if not os.path.exists(INPUT_JSON):
        print(f"Fichier {INPUT_JSON} introuvable.")
        return
        
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        materials = json.load(f)
        
    start_index = get_start_index()
    total = len(materials)
    print(f"Début de la génération de {total} fichiers (Index de {start_index} à {start_index+total-1})...")

    prompt_template = PromptTemplate.from_template(
        "Vous êtes un ingénieur expert en science des matériaux.\n"
        "Votre mission est de rédiger une fiche experte complète (format Markdown) sur le matériau suivant : {material_name}\n\n"
        "Consignes strictes :\n"
        "- Le fichier doit faire entre 150 et 350 lignes.\n"
        "- Langue : Français de niveau expert académique/industriel.\n"
        "- Structure attendue : Titre principal H1 : '# KB-{index} — {material_clean}', suivi de chapitres H2 (Propriétés, Fabrication, Utilisations, Normes), de listes et d'exemples précis.\n"
        "- Ne produisez QUE le code Markdown."
    )

    for i, material in enumerate(materials):
        current_index = start_index + i
        material_clean = str(material).replace('_', ' ')
        filename_slug = material_clean.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # S'assurer que le nom de fichier est valide sur Windows
        filename_slug = "".join(c for c in filename_slug if c.isalnum() or c in (' ', '_', '-')).rstrip()
        
        filename = f"KB-{current_index}_{filename_slug}.md"
        filepath = os.path.join(KB_DIR, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"[{i+1}/{total}] {filename} existe déjà. Ignoré.")
            continue
            
        success = False
        attempts = 0
        while not success and attempts < 5:
            model_name = MODELS[current_model_index]
            print(f"[{i+1}/{total}] Génération de {filename} avec {model_name}...")
            llm = ChatGroq(model_name=model_name, temperature=0.2, max_tokens=3000)
            chain = prompt_template | llm
            
            try:
                response = chain.invoke({
                    "material_name": material_clean,
                    "index": current_index,
                    "material_clean": material_clean
                })
                content = response.content
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"   Succès ({len(content.splitlines())} lignes)")
                success = True
                time.sleep(12) # Respect des limites de l'API (Groq)
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                if "RateLimitError" in error_msg or "429" in error_msg:
                    print(f"   [!] Limite API atteinte pour {model_name}.")
                    if current_model_index < len(MODELS) - 1:
                        current_model_index += 1
                        print(f"   [!] Fallback activé : Passage au modèle {MODELS[current_model_index]}...")
                    else:
                        print(f"   [!] Tous les modèles sont épuisés. Pause de 60s (Tentative {attempts}/5)...")
                        time.sleep(60)
                else:
                    print(f"   Erreur inattendue : {e}. Nouvelle tentative dans 5s...")
                    time.sleep(5)
                    
        if not success:
            print(f"   Échec définitif pour {filename}.")

    print("\nGénération massive terminée !")
    print("Relancez indexer.py pour intégrer ces nouveaux fichiers.")

if __name__ == "__main__":
    main()
