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

KB_DIR = r"d:\MicroNanoTech\kb_files"
START_INDEX = 370

MATERIALS_TOP100 = [
    # Métaux
    "Acier_Inoxydable_304", "Acier_Inoxydable_316L", "Acier_Inoxydable_430", "Aluminium_6061", "Aluminium_7075",
    "Acier_Carbone_1020", "Acier_Carbone_1045", "Fonte_Grise", "Fonte_Ductile", "Laiton_CuZn39Pb3",
    "Bronze_Phosphore", "Bronze_Aluminium", "Cuivre_Electrolytique_C11000", "Titane_Grade_2", "Titane_Grade_5_Ti6Al4V",
    "Nickel_200", "Inconel_625", "Inconel_718", "Monel_400", "Hastelloy_C276",
    "Zinc_Zamak_3", "Magnesium_AZ31B", "Plomb", "Etain", "Tungstene",
    "Molybdene", "Tantale", "Niobium", "Zirconium", "Argent_925", "Or_18K", "Platine_950",
    # Polymères
    "Polyethylene_Basse_Densite_PEBD", "Polyethylene_Haute_Densite_PEHD", "Polypropylene_PP", "Polychlorure_de_Vinyle_PVC", "Polystyrene_PS",
    "Polyethylene_Terephtalate_PET", "Polyurethane_PU", "Polycarbonate_PC", "Acrylonitrile_Butadiene_Styrene_ABS", "Polyamide_Nylon_6",
    "Polyamide_Nylon_66", "Polytetrafluoroethylene_PTFE", "Polymethacrylate_de_Methyle_PMMA", "Polyoxymethylene_POM", "Polyetherethercetone_PEEK",
    "Polyimide_PI", "Silicone", "Caoutchouc_Naturel", "Caoutchouc_Butyle", "Neoprene",
    "EPDM", "Resine_Epoxyde", "Resine_Phenolique", "Resine_Polyester", "Acide_Polylactique_PLA",
    # Céramiques et Verres
    "Alumine_Al2O3", "Zircone_ZrO2", "Carbure_Silicium_SiC", "Nitrure_Silicium_Si3N4", "Carbure_Tungstene_WC",
    "Carbure_Bore_B4C", "Nitrure_Bore_BN", "Steatite", "Cordierite", "Porcelaine",
    "Verre_Sodo_Calcique", "Verre_Borosilicate", "Verre_de_Silice", "Verre_Trempe", "Vitroceramique",
    # Construction et Minéraux
    "Beton_Arme", "Beton_Precontraint", "Ciment_Portland", "Platre", "Chaux",
    "Brique_Terre_Cuite", "Asphalte", "Granit", "Marbre", "Ardoise", "Gres",
    # Composites
    "PRFC_Fibre_de_Carbone", "PRFV_Fibre_de_Verre", "Kevlar_Aramide", "Beton_Fibre",
    "Contreplaque", "MDF", "OSB", "Panneau_de_Particules",
    # Matériaux Naturels
    "Bois_de_Chene", "Bois_de_Pin", "Bois_de_Balsa", "Bambou", "Papier_Kraft",
    "Carton_Ondule", "Coton", "Laine", "Cuir", "Liege",
    # Semiconducteurs et Avancés
    "Silicium_Monocristallin", "Arseniure_Gallium_GaAs", "Nitrure_Gallium_GaN", "Oxyde_Indium_Etain_ITO", "Graphene"
]

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, max_tokens=3000)

prompt_template = PromptTemplate.from_template(
    "Vous êtes un ingénieur expert en science des matériaux.\n"
    "Votre mission est de rédiger une fiche experte complète (format Markdown) sur le matériau suivant : {material_name}\n\n"
    "Consignes strictes :\n"
    "- Le fichier doit faire entre 150 et 350 lignes.\n"
    "- Langue : Français de niveau expert académique/industriel.\n"
    "- Structure attendue : Titre principal H1 : '# KB-{index} — {material_clean}', suivi de chapitres H2 (Propriétés, Fabrication, Utilisations, Normes), de listes et d'exemples précis.\n"
    "- Ne produisez QUE le code Markdown."
)

chain = prompt_template | llm

def main():
    total = len(MATERIALS_TOP100)
    print(f"Début de la génération de {total} fichiers (Index de {START_INDEX} à {START_INDEX+total-1})...")

    for i, material in enumerate(MATERIALS_TOP100):
        current_index = START_INDEX + i
        filename = f"KB-{current_index}_{material}.md"
        filepath = os.path.join(KB_DIR, filename)
        material_clean = material.replace('_', ' ')
        
        # Si le fichier a déjà été généré (en cas de reprise), on le saute
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"[{i+1}/{total}] {filename} existe déjà. Ignoré.")
            continue
            
        print(f"[{i+1}/{total}] Génération de {filename}...")
        
        success = False
        attempts = 0
        while not success and attempts < 5:
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
                
                # Pause pour éviter le RateLimit de Groq
                time.sleep(12)
                
            except Exception as e:
                attempts += 1
                if "RateLimitError" in str(e) or "429" in str(e):
                    print(f"   Limite API atteinte. Pause de 30s (Tentative {attempts}/5)...")
                    time.sleep(30)
                else:
                    print(f"   Erreur : {e}. Nouvelle tentative dans 5s...")
                    time.sleep(5)
                    
        if not success:
            print(f"   Échec définitif pour {filename}.")

    print("\nGénération Top 100 terminée !")
    print("Relancez indexer.py pour intégrer ces 100 fichiers.")

if __name__ == "__main__":
    main()
