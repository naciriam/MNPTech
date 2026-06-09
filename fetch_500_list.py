import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

KB_DIR = r"d:\MicroNanoTech\kb_files"
OUTPUT_JSON = r"d:\MicroNanoTech\ai_agent\500_materials.json"

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
current_model_index = 0

def get_existing_materials():
    existing = set()
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".md"):
            if "_" in filename:
                name = filename.split("_", 1)[1].replace(".md", "").replace("_", " ").lower()
            else:
                name = filename.replace(".md", "").replace("_", " ").lower()
            existing.add(name)
    return existing

def main():
    global current_model_index
    existing_materials = get_existing_materials()
    print(f"{len(existing_materials)} matériaux existants trouvés.")
    
    prompt = PromptTemplate.from_template(
        "Génère une liste JSON stricte contenant 80 matériaux industriels, chimiques ou techniques ultra-spécifiques.\n"
        "Varie au maximum : superalliages, céramiques avancées, polymères, composites, nanomatériaux, semi-conducteurs, biomatériaux.\n"
        "Renvoie UNIQUEMENT le tableau JSON sous la forme : [\"Materiau 1\", \"Materiau 2\"]. Aucun texte avant ni après."
    )
    
    new_materials = set()
    
    while len(new_materials) < 500:
        model_name = MODELS[current_model_index]
        print(f"Demande à l'IA ({model_name})... (Actuellement : {len(new_materials)}/500)")
        llm = ChatGroq(model_name=model_name, temperature=0.7)
        chain = prompt | llm
        
        try:
            response = chain.invoke({})
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            materials_list = json.loads(content)
            
            added_this_round = 0
            for mat in materials_list:
                mat_clean = str(mat).strip()
                mat_lower = mat_clean.lower()
                
                if mat_lower not in existing_materials and mat_lower not in [m.lower() for m in new_materials]:
                    new_materials.add(mat_clean)
                    added_this_round += 1
                    if len(new_materials) >= 500:
                        break
                        
            print(f"   => {added_this_round} matériaux nouveaux ajoutés.")
            time.sleep(12)
            
        except Exception as e:
            error_msg = str(e)
            if "RateLimitError" in error_msg or "429" in error_msg:
                print(f"   [!] Limite API atteinte pour {model_name}.")
                if current_model_index < len(MODELS) - 1:
                    current_model_index += 1
                    print(f"   [!] Fallback activé : Passage au modèle {MODELS[current_model_index]}...")
                else:
                    print(f"   [!] Tous les modèles sont épuisés. Pause de 60s...")
                    time.sleep(60)
            else:
                print(f"   Erreur inattendue : {e}")
                time.sleep(10)
            
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(list(new_materials), f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccès ! {len(new_materials)} matériaux sauvegardés dans {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
