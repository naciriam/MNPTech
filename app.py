import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
    
    # 3. Connecter le LLM Cloud Ultra-Rapide (via Groq API)
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
    
    # 5. Assembler la chaîne avec LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

st.title("🔬 Agent Expert : Nano & Micro Poudres")
st.markdown("Propulsé par **Llama-3-70B** via le Cloud ultra-rapide **Groq**.")

# Vérification de l'API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("Erreur : La clé GROQ_API_KEY est introuvable. Vérifiez le fichier .env.")
    st.stop()

# Initialisation
with st.spinner("Chargement de la base de données locale..."):
    try:
        rag_chain, retriever = load_rag_system()
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
                # Exécution manuelle pour récupérer les documents (sources)
                docs = retriever.invoke(question)
                
                # Exécution de la chaîne LCEL pour obtenir la réponse
                answer = rag_chain.invoke(question)
                
                # Formatage des sources
                sources = []
                for doc in docs:
                    source_name = doc.metadata.get('source', 'Document Inconnu')
                    clean_name = source_name.split('\\')[-1].split('/')[-1] # Compatibilité Windows/Linux
                    if clean_name not in sources:
                        sources.append(clean_name)
                
                # Affichage de la réponse et des sources
                st.markdown(answer)
                if sources:
                    st.info(f"📚 **Sources utilisées :** {', '.join(sources)}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Erreur lors de la communication avec l'API Groq.\n\nErreur: {e}")
