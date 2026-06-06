import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration des chemins (Chemins relatifs pour compatibilité Cloud/Linux)
KB_DIR = "../kb_files"
CHROMA_DB_DIR = "./chroma_db"

def main():
    print("🤖 Démarrage de l'indexation de la Base de Connaissances...")
    
    # 1. Chargement des 350 fichiers Markdown
    print(f"📂 Lecture des fichiers dans : {KB_DIR}")
    # On utilise TextLoader car les fichiers sont en UTF-8 standard
    loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    docs = loader.load()
    print(f"✅ {len(docs)} fichiers chargés avec succès.")

    # 2. Découpage du texte en blocs (Chunks)
    # Les chunks permettent à l'IA de lire des morceaux digestes pour trouver les réponses.
    print("✂️ Découpage des documents en chunks...")
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = markdown_splitter.split_documents(docs)
    print(f"✅ {len(splits)} chunks générés.")

    # 3. Création des Embeddings (Vectorisation)
    # Nous utilisons un modèle open-source léger et très performant (all-MiniLM-L6-v2) 
    # qui va tourner localement sur le processeur pour transformer le texte en vecteurs.
    print("🧠 Chargement du modèle de vectorisation local (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Stockage dans la base vectorielle ChromaDB
    print(f"💾 Sauvegarde dans la base vectorielle : {CHROMA_DB_DIR}")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    
    print("🎉 Indexation terminée avec succès ! La base est prête pour l'Agent IA.")

if __name__ == "__main__":
    main()
