import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Charger les variables d'environnement (Clé API Groq)
load_dotenv()

# Configuration (Chemins relatifs pour Render/Linux)
CHROMA_DB_DIR = "./chroma_db"
# Utilisation du modèle Llama 3 70B via Groq
LLM_MODEL = "llama3-70b-8192" 

st.set_page_config(page_title="Agent Expert Nano/Micro Poudres (Groq Cloud)", page_icon="🔬", layout="wide")

@st.cache_resource
def load_rag_system():
    # 1. Charger les embeddings ultra-légers
    embeddings = FastEmbedEmbeddings()
    
    # 2. Connecter la base de données vectorielle locale
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # On récupère les 4 meilleurs passages
    
    # 3. Connecter le LLM Cloud Ultra-Rapide (via Groq API)
    # L'API Key est automatiquement récupérée depuis le fichier .env
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.2)
    
    # 4. Créer le Prompt RAG expert
    system_prompt = (
        "Vous êtes un ingénieur expert en science des matériaux, spécialisé dans les micro et nano poudres.\n"
        "Répondez de manière professionnelle en français, en utilisant UNIQUEMENT le contexte fourni ci-dessous, issu de votre base de connaissances interne.\n"
        "Si vous ne trouvez pas la réponse dans le contexte, dites simplement que l'information n'est pas dans la base de connaissances.\n\n"
        "Contexte :\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Assembler la chaîne de génération
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

st.title("🔬 Agent Expert : Nano & Micro Poudres")
st.markdown("Propulsé par **Llama-3-70B** via le Cloud ultra-rapide **Groq**.")

# Vérification de l'API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("Erreur : La clé GROQ_API_KEY est introuvable. Vérifiez le fichier .env.")
    st.stop()

# Initialisation
with st.spinner("Chargement de la base de données locale..."):
    try:
        rag_chain = load_rag_system()
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données. Avez-vous exécuté indexer.py ?\n\nErreur détaillée : {e}")
        st.stop()

# Historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie
if question := st.chat_input("Posez votre question scientifique ici..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse par le modèle Groq..."):
            try:
                # Exécution de la chaîne RAG via Groq
                response = rag_chain.invoke({"input": question})
                answer = response["answer"]
                
                # Formatage des sources
                sources = []
                for doc in response["context"]:
                    source_name = doc.metadata.get('source', 'Document Inconnu')
                    # Extraire le nom de fichier propre du chemin absolu
                    clean_name = source_name.split('\\')[-1]
                    if clean_name not in sources:
                        sources.append(clean_name)
                
                # Affichage de la réponse et des sources
                st.markdown(answer)
                if sources:
                    st.info(f"📚 **Sources utilisées :** {', '.join(sources)}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Erreur lors de la communication avec l'API Groq.\n\nErreur: {e}")
