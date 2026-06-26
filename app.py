import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (API Key)
load_dotenv()

st.title("📚 RAG PDF Document Assistant")
st.caption("Ask questions about your uploaded documents using LangChain and Gemini 3.5 Flash")

# Initialize the exact same embedding model used during data ingestion
@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Load the persisted directory from disk
    vector_store = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks

# Load the document retriever
retriever = load_vector_store()

# Initialize the Gemini 3.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

# Build the system prompt explicitly telling the LLM to use the retrieved context
system_prompt = (
    "You are an expert assistant specialized in analyzing technical documents.\n"
    "Answer the user's question using ONLY the provided context below. If you do not know "
    "the answer or if it's not present in the context, say that you cannot find it in the "
    "provided documents. Be precise and structured.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

# Helper function to format retrieved documents into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🚀 MODERN LCEL ORCHESTRATION CHAIN (Replacing create_retrieval_chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Initialize simple chat logs interface if not present
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {"role": "assistant", "content": "Hello! I have fully processed your documents. Ask me anything about them!"}
    ]

# Display past UI chat messages
for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture user input
if user_input := st.chat_input("Ask a question about your PDF..."):
    # Display user input on the fly
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.rag_messages.append({"role": "user", "content": user_input})
    
    # Process queries using the automated LCEL RAG pipeline
    with st.spinner("Analyzing documents..."):
        ai_response = rag_chain.invoke(user_input)
    
    # Render clean AI response
    with st.chat_message("assistant"):
        st.write(ai_response)
    st.session_state.rag_messages.append({"role": "assistant", "content": ai_response})
