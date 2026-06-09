import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter

# Configuration de l'API Key Pinecone (Fournie par l'utilisateur)
os.environ["PINECONE_API_KEY"] = "pcsk_6L5jmA_FGqqxiv3ej7GJarXbCRUEdz5cFGEcVEumGS5VQrs5mJryMPvjfGduLZXMPifNsr"
index_name = "mnptech-kb"

print("Initialisation de Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if index_name not in pc.list_indexes().names():
    print(f"Création de l'index {index_name} (384 dimensions)...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Attendre que l'index soit prêt
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

print("Chargement du modèle d'Embeddings FastEmbed...")
embeddings = FastEmbedEmbeddings()

print("Préparation de la lecture des documents Markdown locaux...")
kb_dir = "./kb_files"
all_splits = []
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

if os.path.exists(kb_dir):
    files = [f for f in os.listdir(kb_dir) if f.endswith('.md')]
    print(f"{len(files)} fichiers Markdown trouvés.")
    
    for idx, filename in enumerate(files):
        filepath = os.path.join(kb_dir, filename)
        try:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            if idx % 100 == 0:
                print(f"Lecture locale : {idx}/{len(files)} fiches traitées...")
        except Exception as e:
            print(f"Erreur de lecture sur {filename}: {e}")

if all_splits:
    print(f"\n=> Lancement de l'upload de {len(all_splits)} segments vectoriels vers Pinecone Cloud !")
    print("Cela peut prendre 1 à 2 minutes...")
    
    vectorstore = PineconeVectorStore.from_documents(
        all_splits,
        embeddings,
        index_name=index_name
    )
    print("Migration terminée avec succès ! Le Cloud Pinecone est prêt.")
else:
    print("Aucun fichier à migrer trouvé dans le dossier kb_files.")
