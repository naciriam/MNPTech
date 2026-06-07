import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Charger la clé API Groq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Erreur : GROQ_API_KEY introuvable dans .env")
    exit(1)

# Dossier des fichiers
KB_DIR = r"d:\MicroNanoTech\kb_files"

# Initialiser le modèle Groq
# On utilise la température à 0.2 pour du contenu très technique et factuel
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, max_tokens=3000)

prompt_template = PromptTemplate.from_template(
    "Vous êtes un ingénieur expert en science des matériaux et micro/nano poudres.\n"
    "Votre mission est de rédiger le contenu complet du fichier Markdown suivant : {filename}\n\n"
    "Consignes strictes :\n"
    "- Le fichier doit faire entre 150 et 350 lignes.\n"
    "- Langue : Français de niveau expert académique/industriel.\n"
    "- Structure attendue : Titre principal H1 (le nom du fichier), suivi de chapitres H2, H3, de listes, d'exemples de matériaux, et de normes.\n"
    "- Ne produisez QUE le code Markdown, pas d'introduction du type 'Voici le fichier'."
)

chain = prompt_template | llm

def get_empty_files(directory):
    empty_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            if os.path.getsize(filepath) == 0:
                empty_files.append(filename)
    return empty_files

def main():
    empty_files = get_empty_files(KB_DIR)
    total = len(empty_files)
    print(f"{total} fichiers vides detectes. Debut de la generation...")

    for idx, filename in enumerate(empty_files, 1):
        filepath = os.path.join(KB_DIR, filename)
        print(f"[{idx}/{total}] Generation de {filename}...")
        
        success = False
        attempts = 0
        while not success and attempts < 5:
            try:
                # Appeler l'API Groq
                response = chain.invoke({"filename": filename})
                content = response.content
                
                # Sauvegarder dans le fichier
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"   Succes ({len(content.splitlines())} lignes ecrites)")
                success = True
                
                # Pause pour respecter la limite de l'offre gratuite Groq (ex: 14400 tokens / minute)
                # Un fichier = ~1500 tokens. On attend 10 secondes entre chaque requête (soit ~6 req/min = 9000 TPM)
                time.sleep(10)
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                if "RateLimitError" in error_msg or "429" in error_msg:
                    print(f"   Limite API atteinte. Pause de 30 secondes (Tentative {attempts}/5)...")
                    time.sleep(30)
                else:
                    print(f"   Erreur inattendue : {e}. Nouvelle tentative dans 5s...")
                    time.sleep(5)
        
        if not success:
            print(f"   Echec definitif pour {filename}. Passage au suivant.")

    print("\nGeneration terminee !")
    print("N'oubliez pas de relancer 'python indexer.py' pour mettre a jour la base de donnees vectorielle.")

if __name__ == "__main__":
    main()
