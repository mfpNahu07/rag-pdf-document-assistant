# 📚 RAG PDF Document Assistant

A local **Retrieval-Augmented Generation (RAG)** chatbot that lets you have intelligent conversations with your PDF documents — powered by **Google Gemini** and built with **LangChain** + **Streamlit**.

Drop your PDFs in a folder, run the ingestion script once, and start asking questions. The assistant will answer **only from what's in your documents**, citing the exact source chunks it used.

---

## ✨ Features

- 🗂️ **Multi-document support** — Load and query multiple PDFs simultaneously
- 🔍 **Semantic search** — Uses Google's `gemini-embedding-001` to find the most relevant chunks
- 🧠 **Grounded answers** — The LLM is instructed to answer strictly from retrieved context, not from its general knowledge
- 📄 **Source transparency** — Every answer includes an expandable panel showing the exact document chunks used
- 💬 **Chat interface** — Persistent conversation history within the session via Streamlit
- 🗃️ **Local vector store** — Embeddings are persisted locally using ChromaDB (no cloud database required)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Embeddings** | Google `gemini-embedding-001` |
| **Orchestration** | [LangChain](https://www.langchain.com/) (LCEL pipeline) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (local persistence) |
| **PDF Loader** | `PyPDFDirectoryLoader` from `langchain-community` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mfpNahu07/rag-pdf-document-assistant.git
cd rag-pdf-document-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```



### 5. Add your PDFs

Place one or more PDF files inside the `docs/` folder:

```
docs/
├── my_document.pdf
├── another_report.pdf
└── ...
```

### 6. Ingest the documents

Run the ingestion script to chunk and embed your PDFs into the local ChromaDB vector store:

```bash
python ingest.py
```

You should see output like:

```
⏳ Loading PDFs from 'docs/' directory...
✅ Loaded 42 document pages.
✂️ Splitting text into chunks...
✅ Created 118 text chunks.
🧠 Generating vector embeddings using gemini-embedding-001...
💾 Saving vectors to local ChromaDB ('chroma_db/')...
🎉 Ingestion complete! 'chroma_db' is ready to use.
```

> **Note:** You only need to run `ingest.py` again when you add or change documents in `docs/`.

### 7. Launch the app

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 📁 Project Structure

```
rag-pdf-document-assistant/
├── app.py              # Streamlit chat UI + RAG pipeline
├── ingest.py           # PDF loading, chunking, and embedding script
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── .gitignore
├── docs/               # Place your PDF files here
└── chroma_db/          # Auto-generated local vector store
```

---

## ⚙️ How It Works

```
User Question
      │
      ▼
 ChromaDB Retriever
 (semantic search, top-3 chunks)
      │
      ▼
 Prompt Template
 (system prompt + context + question)
      │
      ▼
 Gemini 2.5 Flash (LLM)
      │
      ▼
 Grounded Answer + Source Chunks
```

1. **Ingestion** — PDFs are loaded, split into 1000-character chunks (200-character overlap), and stored as vector embeddings in ChromaDB.
2. **Retrieval** — At query time, the top 3 semantically similar chunks are retrieved.
3. **Generation** — The LLM receives the retrieved context and is instructed to answer **only** from it.
4. **Display** — The answer and source chunks are rendered in the Streamlit chat UI.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
