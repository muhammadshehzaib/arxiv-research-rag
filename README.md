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
The Python implementation uses the `arxiv` client library and `pypdf` for text extraction.

**Installation:**
```bash
pip install arxiv pypdf
```

**Running the scripts:**
1. Fetch and download PDFs:
   ```bash
   python arxiv_downloader.py
   ```
2. Chunk the downloaded PDFs:
   ```bash
   python pdf_chunker.py
   ```

---

## Workflow Details

1. **Download Phase**: The downloaders query the arXiv API, fetch matching papers, save their metadata to `data/papers_metadata.json`, and download the PDFs to `data/papers/`.
2. **Chunking Phase**: The chunkers read the downloaded PDFs, clean hyphenations and newlines, and apply a **sliding window chunking algorithm** (default: 500-word chunks with 50-word overlap). The enriched chunks (including metadata) are outputted to `data/paper_chunks.json`.

---

## Change Log & Execution History
- [2026-08-16] Created the initial README.md explaining project structure, installation, usage, and pipelines.
