import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (API Key)
load_dotenv()

def ingest_documents():
    print("⏳ Loading PDFs from 'docs/' directory...")
    # Load all PDFs inside the docs folder
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found in the 'docs/' folder. Please add at least one PDF.")
        return

    print(f"✅ Loaded {len(documents)} document pages.")

    # Split text into chunks using modern character recursive splitting
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    # 🧠 FIX: Using the fully supported native Gemini embedding model
    print("🧠 Generating vector embeddings using gemini-embedding-001...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Save vectors to local ChromaDB
    print("💾 Saving vectors to local ChromaDB ('chroma_db/')...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("🎉 Ingestion complete! 'chroma_db' is ready to use.")

if __name__ == "__main__":
    ingest_documents()
