import streamlit as st
import os
import pypdf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG PDF Document Assistant",
    page_icon="📚",
    layout="wide"
)

# File size limits (1 MB)
MAX_FILE_SIZE_MB = 1
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def process_pdf(file_obj, filename, api_key):
    """Extract text from uploaded PDF, chunk it, and create in-memory vector store."""
    reader = pypdf.PdfReader(file_obj)
    docs = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": idx + 1}
            ))
            
    if not docs:
        raise ValueError("No se pudo extraer texto del PDF. Verifica que contenga texto legible y no consista únicamente de imágenes escaneadas.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 3}), len(chunks)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # 1. API Key Input - Always completely empty by default, never prefilled from .env or secrets
    api_key_input = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="Pega aquí tu API key...",
        help="Introduce tu propia API Key de Google Gemini para esta sesión. No se guardará en ningún archivo."
    )
    api_key = api_key_input.strip() if api_key_input else None
    
    if not api_key:
        st.info("💡 ¿No tienes una key? Consíguela gratis en [Google AI Studio](https://aistudio.google.com/app/apikey).")
        
    # Model Selector
    model_choice = st.selectbox(
        "🤖 Modelo LLM",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-1.5-flash es rápido y cuenta con cuota gratuita en Google AI Studio."
    )
    
    st.divider()
    
    # 2. PDF Uploader (Max 1 MB)
    st.header("📄 Subir Documento")
    uploaded_file = st.file_uploader(
        "Sube un PDF (máx. 1 MB)",
        type=["pdf"],
        help="El archivo no debe superar 1 MB de tamaño."
    )
    
    valid_upload = False
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ El archivo excede el límite permitido de {MAX_FILE_SIZE_MB} MB ({file_size_mb:.2f} MB). Sube un archivo más pequeño.")
            valid_upload = False
        else:
            st.caption(f"📏 Tamaño: {file_size_mb:.2f} MB / {MAX_FILE_SIZE_MB} MB")
            valid_upload = True
    else:
        # If user cleared the upload, reset document session
        if "current_file_sig" in st.session_state:
            st.session_state.pop("retriever", None)
            st.session_state.pop("current_file_sig", None)
            st.session_state.pop("current_doc_name", None)
            st.session_state.pop("rag_messages", None)
            
    st.divider()
    
    # Action button to clear conversation
    if st.button("🗑️ Limpiar historial de chat", use_container_width=True):
        st.session_state.rag_messages = []
        st.rerun()

# --- MAIN PAGE HEADER ---
st.title("📚 RAG PDF Document Assistant")
st.caption(f"Haz preguntas sobre tus documentos PDF usando LangChain y Google Gemini ({model_choice})")

# --- GUARD CONDITION 1: API KEY REQUIRED ---
if not api_key:
    st.warning("⚠️ **Ingresa tu Google Gemini API Key en el menú lateral para comenzar.**")
    st.markdown(
        """
        Para interactuar con el asistente y procesar tus documentos:
        1. Obtén tu API Key gratuita en [Google AI Studio](https://aistudio.google.com/app/apikey).
        2. Pégala en el campo **Google Gemini API Key** de la barra lateral izquierda.
        3. Sube un documento PDF de hasta **1 MB**.
        """
    )
    st.stop()

# --- VECTOR STORE / RETRIEVER INITIALIZATION ---
retriever = None

if uploaded_file is not None and valid_upload:
    file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
    # Re-index only when a new file is uploaded
    if st.session_state.get("current_file_sig") != file_sig:
        with st.spinner(f"Vectorizando y analizando '{uploaded_file.name}' con Gemini..."):
            try:
                retriever, num_chunks = process_pdf(uploaded_file, uploaded_file.name, api_key)
                st.session_state["retriever"] = retriever
                st.session_state["current_file_sig"] = file_sig
                st.session_state["current_doc_name"] = uploaded_file.name
                st.session_state.rag_messages = [
                    {
                        "role": "assistant",
                        "content": f"¡Hola! He procesado exitosamente el documento **{uploaded_file.name}** ({num_chunks} fragmentos indexados). ¿Qué te gustaría saber sobre él?",
                        "sources": []
                    }
                ]
                st.sidebar.success(f"✅ Documento indexado ({num_chunks} chunks)")
            except Exception as e:
                st.error(f"❌ Error al procesar el PDF: {str(e)}")
                st.stop()
    else:
        retriever = st.session_state.get("retriever")
else:
    # Fallback to local persisted database if present and no file uploaded
    if "retriever" not in st.session_state and os.path.exists("chroma_db") and os.listdir("chroma_db"):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            st.session_state["retriever"] = retriever
            st.session_state["current_doc_name"] = "Base local pre-indexada ('chroma_db')"
        except Exception as e:
            st.sidebar.warning(f"No se pudo cargar la base local: {e}")
    else:
        retriever = st.session_state.get("retriever")

# --- GUARD CONDITION 2: NO DOCUMENT AVAILABLE ---
if retriever is None:
    st.info("📂 **Por favor sube un archivo PDF (hasta 1 MB) en la barra lateral para empezar a chatear.**")
    st.stop()

# Display active document indicator
active_doc = st.session_state.get("current_doc_name", "Documento activo")
st.markdown(f"📄 **Documento activo:** `{active_doc}`")

# --- LCEL PIPELINE SETUP ---
llm = ChatGoogleGenerativeAI(
    model=model_choice,
    google_api_key=api_key,
    temperature=0.3
)

system_prompt = (
    "You are an expert assistant specialized in analyzing technical documents.\n"
    "Answer the user's question using ONLY the provided context below. If you do not know "
    "the answer or if it's not present in the context, say that you cannot find it in the "
    "provided documents. Answer in the same language as the user's question. Be precise and structured.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)
generation_chain = prompt | llm | StrOutputParser()

# --- CHAT HISTORY INITIALIZATION ---
if "rag_messages" not in st.session_state or not st.session_state.rag_messages:
    st.session_state.rag_messages = [
        {
            "role": "assistant",
            "content": f"¡Hola! Tengo listo el documento '{active_doc}'. ¡Hazme cualquier pregunta sobre él!",
            "sources": []
        }
    ]

# Render previous chat messages
for msg in st.session_state.rag_messages:
    custom_avatar = "🧠" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=custom_avatar):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Ver fuentes del documento"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Fragmento {idx+1}:**\n{src}")

# --- CHAT INPUT & EXECUTION ---
if user_input := st.chat_input("Escribe tu pregunta sobre el documento..."):
    # Render user query
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    st.session_state.rag_messages.append({"role": "user", "content": user_input, "sources": []})
    
    with st.spinner("Buscando en el documento y generando respuesta..."):
        try:
            # 1. Retrieve relevant chunks
            raw_docs = retriever.invoke(user_input)
            source_texts = [doc.page_content for doc in raw_docs]
            formatted_context = "\n\n".join(source_texts)
            
            # 2. Generate answer
            ai_response = generation_chain.invoke({
                "context": formatted_context,
                "question": user_input
            })
            
            # Render assistant response
            with st.chat_message("assistant", avatar="🧠"):
                st.write(ai_response)
                if source_texts:
                    with st.expander("📄 Ver fuentes del documento"):
                        for idx, src in enumerate(source_texts):
                            st.markdown(f"**Fragmento {idx+1}:**\n{src}")
                            
            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": ai_response,
                "sources": source_texts
            })
        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "402" in error_msg:
                st.error("⚠️ **Cuota de API agotada:** Tu API key ha agotado sus créditos o cuota disponible. Por favor ingresa otra API key en la barra lateral.")
            elif "API_KEY_INVALID" in error_msg or "403" in error_msg or "INVALID_ARGUMENT" in error_msg:
                st.error("⚠️ **Error de API Key:** La API key proporcionada parece ser inválida. Por favor verifícala en la barra lateral.")
            else:
                st.error(f"⚠️ Ocurrió un error al procesar tu solicitud: {error_msg}")
