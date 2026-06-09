import os
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter

# Charger les variables d'environnement (Clé API Groq)
load_dotenv()

# Configuration (Chemins relatifs pour Render/Linux)
# Utilisation du modèle Llama 3.3 70B via Groq
LLM_MODEL = "llama-3.3-70b-versatile" 

st.set_page_config(page_title="Agent Expert Nano/Micro Poudres", page_icon="🔬", layout="wide")

@st.cache_resource
def load_rag_system():
    # 1. Charger les embeddings ultra-légers
    embeddings = FastEmbedEmbeddings()
    
    # 2. Connecter la base de données vectorielle Cloud (Pinecone)
    index_name = "mnptech-kb"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
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
    
    return rag_chain, retriever, vectorstore

st.title("🔬 Agent Expert : Nano & Micro Poudres")
st.markdown(f"**V0.1** — Propulsé par **{LLM_MODEL}** via le Cloud ultra-rapide **Groq**.")

# Vérification des clés API
if not os.getenv("GROQ_API_KEY"):
    st.error("Erreur : La clé GROQ_API_KEY est introuvable. Vérifiez le fichier .env.")
    st.stop()

if not os.getenv("PINECONE_API_KEY"):
    st.error("Erreur : La clé PINECONE_API_KEY est introuvable. Ajoutez-la dans le fichier .env ou dans les réglages de Render.")
    st.stop()

# Initialisation
with st.spinner("Chargement de la base de données locale..."):
    try:
        rag_chain, retriever, vectorstore = load_rag_system()
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
                    st.warning("⚠️ Information introuvable dans la base locale. Génération experte et apprentissage en cours...")
                    
                    # 1. Réponse directe via le LLM
                    with st.spinner("Analyse et rédaction de la réponse par l'IA experte..."):
                        llm = ChatGroq(model_name=LLM_MODEL, temperature=0.2)
                        direct_prompt = ChatPromptTemplate.from_messages([
                            ("system", "Vous êtes un ingénieur expert en science des matériaux. Répondez à la question de manière concise et professionnelle en vous basant sur vos vastes connaissances scientifiques internes."),
                            ("human", "{input}")
                        ])
                        direct_chain = direct_prompt | llm | StrOutputParser()
                        answer = direct_chain.invoke({"input": question})
                    
                    st.markdown(answer)
                    st.info("🧠 **Sources utilisées :** Connaissances internes du modèle (Apprentissage auto)")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 2. Déterminer si c'est un nouveau sujet ou l'approfondissement d'un sujet existant
                    with st.spinner("Analyse du contexte pour l'auto-apprentissage..."):
                        # on récupère les noms de fichiers des sources trouvées pour cette question
                        raw_sources = [doc.metadata.get('source', '') for doc in docs]
                        clean_sources = list(dict.fromkeys([src.replace('\\', '/').split('/')[-1] for src in raw_sources if src]))
                        
                        router_prompt = ChatPromptTemplate.from_messages([
                            ("system", (
                                "Vous êtes un système de classification documentaire.\n"
                                "Voici une question d'un utilisateur : {question}\n"
                                "Voici les fichiers de référence trouvés dans la base de données : {sources}\n"
                                "La question de l'utilisateur est-elle un approfondissement d'un des fichiers listés ci-dessus, ou bien aborde-t-elle un sujet totalement nouveau ?\n"
                                "- Si c'est un approfondissement d'un fichier pertinent, répondez STRICTEMENT par le nom exact du fichier concerné (ex: KB-145_Silicium.md).\n"
                                "- Si c'est un nouveau sujet, ou si la liste des fichiers est vide/hors sujet, répondez STRICTEMENT par le mot exact : NEW_SUBJECT\n"
                                "Ne dites absolument rien d'autre."
                            ))
                        ])
                        router_chain = router_prompt | llm | StrOutputParser()
                        route = router_chain.invoke({"question": question, "sources": ", ".join(clean_sources) if clean_sources else "Aucune source pertinente."})
                        route = route.strip()

                    kb_dir = "./kb_files"
                    os.makedirs(kb_dir, exist_ok=True)
                    
                    if route == "NEW_SUBJECT" or not route.startswith("KB-"):
                        # --- NOUVEAU FICHIER ---
                        with st.spinner("Auto-apprentissage : Création d'une NOUVELLE fiche experte (très détaillée)..."):
                            max_idx = 0
                            for f in os.listdir(kb_dir):
                                if f.endswith('.md') and f.startswith('KB-'):
                                    try:
                                        num = int(f.split('_')[0].replace('KB-', ''))
                                        if num > max_idx:
                                            max_idx = num
                                    except:
                                        pass
                            
                            if max_idx == 0:
                                max_idx = 1000
                                
                            next_index = max_idx + 1
                            sujet_clean = question.replace('?', '').replace('!', '').replace(':', '').strip().capitalize()
                            
                            fiche_prompt = ChatPromptTemplate.from_messages([
                                ("system", (
                                    "Vous êtes un ingénieur expert en science des matériaux.\n"
                                    "Votre mission est de rédiger une fiche experte complète (format Markdown) sur le matériau ou le sujet suivant : {sujet_clean}\n\n"
                                    "Consignes strictes :\n"
                                    "- Le fichier doit être très long, encyclopédique et approfondi (150 à 350 lignes minimum).\n"
                                    "- Ne soyez surtout pas bref. Rédigez plusieurs paragraphes très détaillés pour chaque section.\n"
                                    "- Langue : Français de niveau expert académique/industriel.\n"
                                    "- Structure attendue : Titre principal H1 : '# KB-{index} — {sujet_clean}', suivi de chapitres H2 (Propriétés, Fabrication, Utilisations, Normes), de listes et d'exemples précis.\n"
                                    "- Ne produisez QUE le code Markdown."
                                ))
                            ])
                            
                            fiche_chain = fiche_prompt | llm | StrOutputParser()
                            new_content = fiche_chain.invoke({"sujet_clean": sujet_clean, "index": next_index})
                            
                            filename_slug = "".join(c for c in sujet_clean if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')[:40]
                            filepath = os.path.join(kb_dir, f"KB-{next_index}_{filename_slug}.md")
                            
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(new_content)
                                
                            st.success(f"🧠 J'ai créé et mémorisé une NOUVELLE fiche experte : KB-{next_index}_{filename_slug}.md")
                    else:
                        # --- COMPLETION DE FICHIER EXISTANT ---
                        with st.spinner(f"Auto-apprentissage : Complétion approfondie de la fiche existante {route}..."):
                            filepath = os.path.join(kb_dir, route)
                            
                            # On génère un chapitre supplémentaire très long
                            fiche_prompt = ChatPromptTemplate.from_messages([
                                ("system", (
                                    "Vous êtes un ingénieur expert en science des matériaux.\n"
                                    "Vous devez rédiger un nouveau chapitre ultra-détaillé en Markdown pour compléter une fiche existante sur le sujet.\n"
                                    "La question spécifique à développer est : {question}\n\n"
                                    "Consignes strictes :\n"
                                    "- Le contenu doit être extrêmement détaillé et encyclopédique (50 à 150 lignes minimum).\n"
                                    "- Ne soyez pas bref. Rédigez des paragraphes longs, techniques et exhaustifs.\n"
                                    "- Langue : Français de niveau expert.\n"
                                    "- Commencez par un titre H2 (##) correspondant à la thématique abordée.\n"
                                    "- Ne produisez QUE le code Markdown du nouveau chapitre (pas de H1, pas d'introduction générale)."
                                ))
                            ])
                            
                            fiche_chain = fiche_prompt | llm | StrOutputParser()
                            new_content = "\n\n" + fiche_chain.invoke({"question": question})
                            
                            if os.path.exists(filepath):
                                with open(filepath, "a", encoding="utf-8") as f:
                                    f.write(new_content)
                            else:
                                # Fallback au cas où le fichier n'est pas trouvé physiquement (ex: sur Render sans le repo kb_files complet)
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(f"# Fiche reconstituée : {route}\n" + new_content)
                                    
                            st.success(f"🧠 J'ai approfondi mes connaissances en ajoutant un long chapitre à la fiche : {route}")
                            
                    # Injection dans ChromaDB (Vectorisation dynamique de la NOUVELLE portion uniquement)
                    try:
                        from langchain_core.documents import Document
                        # On charge le nouveau contenu directement en mémoire pour ne vectoriser QUE le nouveau texte (gain mémoire et temps)
                        new_docs = [Document(page_content=new_content, metadata={"source": filepath})]
                        text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
                        new_splits = text_splitter.split_documents(new_docs)
                        
                        vectorstore.add_documents(new_splits)
                    except Exception as e:
                        st.error(f"Erreur lors de l'indexation de la nouvelle information : {e}")

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
