# arXiv Research RAG Pipeline

A lightweight toolset for fetching papers from arXiv, downloading their PDFs, extracting text, and generating chunked datasets for Retrieval-Augmented Generation (RAG) applications.

This repository provides twin implementations in **JavaScript (Node.js/Bun)** and **Python** to fit into any stack.

---

## Project Structure

```
├── arxiv_downloader.js   # Node.js script to query arXiv and download PDFs
├── arxiv_downloader.py   # Python script to query arXiv and download PDFs
├── pdf_chunker.js        # Node.js script to extract text and chunk PDFs
├── pdf_chunker.py        # Python script to extract text and chunk PDFs
├── package.json          # Node.js dependencies
├── bun.lock              # Bun lockfile
├── .gitignore            # Git ignore rules
└── data/                 # Generated outputs (papers, metadata, and chunks)
```

---

## Getting Started

### 1. JavaScript / Bun Pipeline
The Node.js implementation uses standard `fetch` (native in Node.js 18+ or Bun) and `pdf-parse` for text extraction.

**Installation:**
```bash
npm install
# or
bun install
```

**Running the scripts:**
1. Fetch and download PDFs (default query: "Retrieval Augmented Generation"):
   ```bash
   node arxiv_downloader.js
   ```
2. Chunk the downloaded PDFs:
   ```bash
   node pdf_chunker.js
   ```

---

### 2. Python Pipeline
The Python implementation uses the `arxiv` client library, `pypdf` for text extraction, **Chroma DB** for the vector store, and the **Gemini API** for embeddings and RAG answers.

**Installation:**
1. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install arxiv pypdf
   ```
2. Configure environment variables:
   * Create a `.env` file (copied from `.env.example`).
   * Set your `GEMINI_API_KEY` in `.env`.

**Running the scripts:**
1. **Download Phase**: Fetch papers and download PDFs:
   ```bash
   python arxiv_downloader.py
   ```
2. **Chunking Phase**: Parse PDFs and create text chunks:
   ```bash
   python pdf_chunker.py
   ```
3. **Database Population**: Generate embeddings via Gemini and store them in Chroma DB:
   ```bash
   python populate_db.py
   ```
4. **RAG Querying**: Search database and answer questions using Gemini LLM:
   * **Single Query Mode**:
     ```bash
     python query_rag.py --query "What is Retrieval Augmented Generation?"
     ```
   * **Interactive Chat Mode**:
     ```bash
     python query_rag.py
     ```

---

## Workflow Details

1. **Download Phase**: The downloaders query the arXiv API, fetch matching papers, save their metadata to `data/papers_metadata.json`, and download the PDFs to `data/papers/`.
2. **Chunking Phase**: The chunkers read the downloaded PDFs, clean hyphenations and newlines, and apply a **sliding window chunking algorithm** (default: 500-word chunks with 50-word overlap). The enriched chunks (including metadata) are outputted to `data/paper_chunks.json`.

---

## Change Log & Execution History
- [2026-08-19] Updated .gitignore to exclude data/chroma_db/, paper_chunks.json, and papers_metadata.json from Git.
- [2026-08-19] Updated LLM model default to gemini-2.5-flash because gemini-1.5-flash is no longer supported on the current API version/region.
- [2026-08-19] Added retry logic with exponential backoff and reduced default batch size to 10 in populate_db.py to prevent 429 quota exhaustion errors.
- [2026-08-19] Switched default embedding model to gemini-embedding-001 in config and code because text-embedding-004 is deprecated/unsupported.



- [2026-08-19] Fixed a key mismatch bug in pdf_chunker.py to support both camelCase and snake_case metadata keys.

- [2026-08-16] Added requirements.txt, populate_db.py, and query_rag.py to build and run the Python Chroma DB + Gemini RAG system.
- [2026-08-16] Created the initial README.md explaining project structure, installation, usage, and pipelines.

