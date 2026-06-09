import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter

# Charger les variables d'environnement (Clé API Groq)
load_dotenv()

# Configuration (Chemins relatifs pour Render/Linux)
CHROMA_DB_DIR = "./chroma_db"
# Utilisation du modèle Llama 3.3 70B via Groq
LLM_MODEL = "llama-3.3-70b-versatile" 

st.set_page_config(page_title="Agent Expert Nano/Micro Poudres", page_icon="🔬", layout="wide")

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
        "Si le contexte ne contient pas l'information pertinente pour répondre à la question, NE DITES RIEN ET RÉPONDEZ STRICTEMENT PAR LE MOT EXACT : INFO_MISSING\n\n"
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
st.markdown(f"**V0.1** — Propulsé par **{LLM_MODEL}** via le Cloud ultra-rapide **Groq**.")

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
                
                if answer.strip() == "INFO_MISSING":
                    st.warning("⚠️ Information introuvable dans la base locale. Recherche en ligne et apprentissage en cours...")
                    
                    # 1. Recherche Web
                    with st.spinner("Recherche sur le Web via DuckDuckGo..."):
                        search = DuckDuckGoSearchRun()
                        search_results = search.invoke(question)
                    
                    if not search_results or search_results.strip() == "":
                        answer = "Je n'ai malheureusement trouvé aucune information à ce sujet, ni dans ma base, ni sur le web."
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.stop()
                    
                    # 2. Répondre à l'utilisateur
                    with st.spinner("Analyse des résultats Web et rédaction de la réponse..."):
                        web_prompt = ChatPromptTemplate.from_messages([
                            ("system", "Vous êtes un ingénieur expert. Répondez à la question de manière concise et professionnelle en utilisant les informations Web suivantes :\n{context}"),
                            ("human", "{input}")
                        ])
                        llm = ChatGroq(model_name=LLM_MODEL, temperature=0.2)
                        web_chain = web_prompt | llm | StrOutputParser()
                        answer = web_chain.invoke({"context": search_results, "input": question})
                    
                    st.markdown(answer)
                    st.info("🌐 **Sources utilisées :** Recherche Web en temps réel")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 3. Génération de la fiche experte pour auto-apprentissage (MEME PROMPT QU'AVANT)
                    with st.spinner("Auto-apprentissage : Création et mémorisation de la nouvelle fiche experte (150-350 lignes)..."):
                        kb_dir = "./kb_files" if os.path.exists("./kb_files") else "../kb_files"
                        
                        max_idx = 0
                        if os.path.exists(kb_dir):
                            for f in os.listdir(kb_dir):
                                if f.endswith('.md') and f.startswith('KB-'):
                                    try:
                                        num = int(f.split('_')[0].replace('KB-', ''))
                                        if num > max_idx:
                                            max_idx = num
                                    except:
                                        pass
                        next_index = max_idx + 1
                        
                        sujet_clean = question.replace('?', '').replace('!', '').replace(':', '').strip().capitalize()
                        
                        fiche_prompt = ChatPromptTemplate.from_messages([
                            ("system", (
                                "Vous êtes un ingénieur expert en science des matériaux.\n"
                                "Votre mission est de rédiger une fiche experte complète (format Markdown) sur le matériau ou le sujet suivant : {sujet_clean}\n\n"
                                "Pour rédiger cette fiche, appuyez-vous sur vos connaissances d'expert ET sur les informations suivantes issues du Web :\n{context}\n\n"
                                "Consignes strictes :\n"
                                "- Le fichier doit faire entre 150 et 350 lignes.\n"
                                "- Langue : Français de niveau expert académique/industriel.\n"
                                "- Structure attendue : Titre principal H1 : '# KB-{index} — {sujet_clean}', suivi de chapitres H2 (Propriétés, Fabrication, Utilisations, Normes), de listes et d'exemples précis.\n"
                                "- Ne produisez QUE le code Markdown."
                            ))
                        ])
                        
                        fiche_chain = fiche_prompt | llm | StrOutputParser()
                        markdown_content = fiche_chain.invoke({
                            "sujet_clean": sujet_clean,
                            "context": search_results,
                            "index": next_index
                        })
                        
                        # Sauvegarde physique
                        filename_slug = "".join(c for c in sujet_clean if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')[:40]
                        new_filepath = os.path.join(kb_dir, f"KB-{next_index}_{filename_slug}.md")
                        
                        with open(new_filepath, "w", encoding="utf-8") as f:
                            f.write(markdown_content)
                            
                        # Injection dans ChromaDB (Vectorisation dynamique)
                        try:
                            loader = TextLoader(new_filepath, encoding='utf-8')
                            new_docs = loader.load()
                            text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
                            new_splits = text_splitter.split_documents(new_docs)
                            
                            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=FastEmbedEmbeddings())
                            vectorstore.add_documents(new_splits)
                            
                            st.success(f"🧠 J'ai créé et mémorisé de manière permanente la nouvelle fiche experte : KB-{next_index}_{filename_slug}.md")
                        except Exception as e:
                            st.error(f"Erreur lors de l'indexation de la nouvelle fiche : {e}")

                else:
                    # Formatage normal des sources (déduplication avec préservation de l'ordre)
                    raw_sources = [doc.metadata.get('source', 'Document Inconnu') for doc in docs]
                    clean_sources = [src.replace('\\', '/').split('/')[-1] for src in raw_sources]
                    sources = list(dict.fromkeys(clean_sources))
                    
                    # Affichage de la réponse et des sources
                    st.markdown(answer)
                    if sources:
                        st.info(f"📚 **Sources utilisées :** {', '.join(sources)}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Erreur lors de la communication avec l'API Groq.\n\nErreur: {e}")
