# arXiv Research RAG Pipeline

A lightweight toolset for fetching papers from arXiv, downloading their PDFs, extracting text, and generating chunked datasets for Retrieval-Augmented Generation (RAG) applications.

This repository provides twin implementations in **JavaScript (Node.js/Bun)** and **Python** to fit into any stack.

---

## Project Structure

```
├── rag_system/            # Python app, RAG logic, MCP server, and shared modules
├── scripts/               # Python download, indexing, evaluation, and migration commands
├── tests/                 # Python tests
├── arxiv_downloader.js    # Node.js paper downloader
├── pdf_chunker.js         # Node.js PDF chunker
├── package.json           # Node.js dependencies
├── requirements.txt       # Python dependencies
└── data/                  # Generated outputs (papers, metadata, and chunks)
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
Run these commands from the project root.
1. **Download Phase**: Fetch papers and download PDFs:
   ```bash
   python -m scripts.arxiv_downloader
   ```
2. **Chunking Phase**: Parse PDFs and create text chunks:
   ```bash
   python -m scripts.pdf_chunker
   ```
3. **Database Population**: Generate embeddings via Gemini and store them in Chroma DB:
   ```bash
   python -m scripts.populate_db
   ```
4. **RAG Querying**: Search database and answer questions using Gemini LLM:
   * **Single Query Mode**:
     ```bash
     python -m rag_system.query_rag --query "What is Retrieval Augmented Generation?"
     ```
   * **Interactive Chat Mode**:
     ```bash
     python -m rag_system.query_rag
     ```

---

### 3. Model Context Protocol (MCP) Integration
This repository includes a native **Model Context Protocol (MCP)** implementation:
* **Academic MCP Server (`rag_system/academic_mcp_server.py`)**: Run with `python -m rag_system.academic_mcp_server` from the project root. It exposes tools (`search_arxiv`, `search_semantic_scholar`, `fetch_paper_by_id`, `search_local_rag`, `ingest_live_paper_to_rag`) over stdio for external AI IDEs (Antigravity, Claude Desktop, Cursor).
* **Live Paper MCP Fallback (`rag_system/mcp_client.py`)**: Automatically connects to the MCP server for eligible research queries when local retrieval falls below its relevance cutoff or returns no matching candidates, fetching live papers and generating grounded answers. Live papers are shown as unscored.

Start the web app from the project root with `python -m rag_system.app`.

Source `relevance_score` values are retrieval ranking signals on a 0–1 scale, **not probabilities that an answer is correct**. Cohere and FlashRank bounded scores are preserved; Sentence Transformers is explicitly asked for logits and receives one sigmoid transformation. The API no longer returns synthetic `confidence` or `distance` fields; unscored live sources have null `relevance_score`, `rerank_score`, and `rrf_score`. Existing semantic cache entries from the old scoring scheme are ignored.

`REFUSAL_THRESHOLD` remains the global relevance cutoff (default `0.70`). Optional `REFUSAL_THRESHOLD_COHERE`, `REFUSAL_THRESHOLD_SENTENCE_TRANSFORMERS`, and `REFUSAL_THRESHOLD_FLASHRANK` override it for each provider. These defaults are heuristics: rerun `python -m scripts.calibrate_threshold` on representative questions after changing provider, model, or scoring version. Old calibration results used the previous scoring scheme and should not be reused. See the [Sentence Transformers activation documentation](https://sbert.net/docs/package_reference/cross_encoder/model.html) and [Cohere threshold guidance](https://docs.cohere.com/v1/docs/reranking-best-practices).

---

### 4. Run the web app locally with Docker

From PowerShell in the project folder, run:

```powershell
cd D:\Ragsystem
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8000` in your browser once `docker compose logs -f web` shows `Application startup complete`. The container reads `.env` and uses the existing `data/` folder, including its Chroma database. Set a real `GEMINI_API_KEY` in `.env`; keep `EMBEDDING_PROVIDER` consistent with the existing database. On the first start, Docker downloads the local embedding and reranking models before serving requests; their cache persists in a Docker volume. Local Docker queries allow up to 90 seconds overall, including up to 45 seconds for Gemini. Docker uses hybrid retrieval: Chroma dense search and SQLite full-text search, combined with reciprocal rank fusion and reranking. It loads paper and citation graph data without the large precomputed coauthor map to reduce memory use. Set `RAG_SPARSE_ONLY: "1"` in `compose.yaml` if keyword-only retrieval is needed to reduce dense-search memory use.

To inspect startup or query errors, run `docker compose logs -f web`. Press Ctrl+C to leave the log view. Stop the app with `docker compose down` (the database in `data/` remains on your computer).

---

## Workflow Details

1. **Download Phase**: The downloaders query the arXiv API, fetch matching papers, save their metadata to `data/papers_metadata.json`, and download the PDFs to `data/papers/`.
2. **Chunking Phase**: The chunkers read the downloaded PDFs, clean hyphenations and newlines, and apply a **sliding window chunking algorithm** (default: 500-word chunks with 50-word overlap). The enriched chunks (including metadata) are outputted to `data/paper_chunks.json`.

---

## Change Log & Execution History

- [2026-10-18] Added cache TTL, bounded LRU retention, corpus/configuration invalidation, process/thread locking, atomic persistence, and concurrent-writer regression tests.
- [2026-10-18] Isolated conversational answers from the shared semantic cache: history-bearing requests always resolve context and never read or write shared cached answers.
- [2026-10-18] Fixed the packaged FTS builder import and made vector/keyword indexing failures propagate to a nonzero ingestion exit status so failed batches can be retried.
- [2026-10-18] Fixed the evaluation button's packaged evaluator import and verified its HTTP success and failure responses without external API calls.
- [2026-10-17] Corrected provider-specific relevance scoring and refusal thresholds, removed fabricated live-source confidence, updated source labels and calibration, invalidated legacy scoring caches, and added regression tests.
- [2026-10-17] Fixed empty local retrieval to attempt permitted MCP live fallback before returning no results, removed the unreachable fallback branch, and added regression coverage.
- [2026-10-12] Enabled Chroma dense retrieval alongside SQLite keyword search in Docker, restoring hybrid retrieval with reciprocal rank fusion and reranking.
- [2026-10-11] Converted NumPy values in streamed chat results to JSON-safe types so completed queries reach the browser instead of ending with a network error.
- [2026-10-11] Added a lower-memory Docker query mode using SQLite full-text retrieval and reranking after Windows reported low virtual memory during dense queries.
- [2026-10-11] Reduced Docker graph memory use by loading papers and citations without the large precomputed coauthor map and calculating collaborators on demand.
- [2026-10-11] Raised local Docker query and Gemini time limits and enabled container restart after Docker engine interruptions.
- [2026-10-11] Added Docker Compose and a lean web image that warms local models before serving and persists their cache alongside the existing `.env` and `data/` directory.
- [2026-10-11] Skipped Gemini metadata parsing for queries without date, paper ID, or page filters, reducing ordinary research query latency and timeout risk.
- [2026-10-11] Moved Python application modules into `rag_system/` and command scripts into `scripts/`, updating imports, subprocess calls, tests, and usage instructions.
- [2026-10-10] Added live, plain-language search progress in the chat UI, with backend status events that distinguish saved-paper search from live website search.

- [2026-10-10] Added fast conversational routing, research-only live fallback, a health endpoint, bounded web queries and external calls, and per-stage RAG timing logs.
- [2026-09-16] Implemented background Auto-Ingestion for live arXiv papers retrieved via MCP fallback, automatically saving full PDFs, embeddings, and FTS5 indices into Chroma DB.
- [2026-09-16] Integrated Model Context Protocol (MCP) Server & Client to dynamically retrieve live papers from arXiv and Semantic Scholar whenever local Chroma DB retrieval falls below the confidence threshold.
- [2026-09-10] Replaced in-memory rank_bm25 linear scan with an on-disk SQLite FTS5 inverted index (data/bm25_fts.db) for sub-5ms BM25 retrieval, instant CLI startup, and ~20MB RAM usage across 224,735 chunks.
- [2026-09-08] Eliminated redundant duplicate embedding API calls per query by computing query_vector once upfront and reusing it across semantic cache lookup, Chroma DB dense retrieval, and semantic cache store.
- [2026-09-07] Refactored duplicate utility functions into centralized utils.py module (date_to_int, clean_text, tokenize_text, setup_windows_encoding) adhering to the DRY principle.
- [2026-09-07] Fixed asymmetric post-filtering bug by adding numeric published_int metadata and enabling true database pre-filtering for published_after in Chroma DB and BM25.
- [2026-09-06] Relocated start_time and semantic cache evaluation to the entry point of query_rag.py, preventing unneeded LLM filter extraction calls on cache hits and ensuring honest latency metrics.
- [2026-09-05] Fixed fail-open vulnerability in query_rag.py by enforcing fail-closed refusal when no reranker is active instead of generating mock scores.
- [2026-09-04] Implemented On-Device Academic Knowledge Graph & Graph RAG with paper-to-paper citations, author collaboration networks, and multi-hop traversal.
- [2026-09-03] Implemented The Refusal Ladder (Multi-Tier Confidence Guardrail) with Sigmoid Cross-Encoder scoring to abort generation on out-of-domain queries and eliminate hallucinations.
- [2026-09-01] Added Multi-Turn Conversational Memory with Query Rephrasing to automatically contextualize pronoun-heavy follow-up questions.
- [2026-09-01] Implemented on-device Semantic Caching with vectorized cosine similarity lookup to achieve sub-30ms response times on repeated/paraphrased queries.
- [2026-08-31] Implemented Metadata Auto-Filtering (Query-to-Filter Translator) in query_rag.py using Gemini in JSON mode.
- [2026-08-29] Integrated local Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) using sentence-transformers in query_rag.py.
- [2026-08-27] Implemented the RAG Triad Quantitative Evaluation Framework (Faithfulness, Answer Relevance, Context Recall & Precision) with a command-line runner and a web dashboard.
- [2026-08-26] Refactored list parsing in frontend helper.js to bundle entire list blocks, fixing number sequence resets (1, 1, 1 -> 1, 2, 3) and line breaks.
- [2026-08-26] Implemented Hybrid Retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF) using rank_bm25.
- [2026-08-25] Updated .gitignore to exclude data/chroma_db_backup/ directory from Git tracking.
- [2026-08-25] Configured standard output stream encoding to UTF-8 on Windows in populate_db.py to prevent console encoding failures.
- [2026-08-25] Increased arXiv search max_results to 300 in batch_ingestion.py to resolve query exhaustion and discover new papers.
- [2026-08-24] Optimized pdf_chunker.py metadata query to skip malformed old-format paper IDs, enforce 3s sleep to avoid HTTP 429, and auto-generate fallback metadata from filenames.
- [2026-08-23] Optimized pdf_chunker.py and populate_db.py to cache and skip re-processing chunk data and re-calculating existing database embeddings.
- [2026-08-23] Scaled target papers count to 3,000 and expanded predefined query list in batch_ingestion.py.
- [2026-08-22] Added individual-fallback mechanism to pdf_chunker.py for fetching metadata to handle invalid/old-format arXiv ID failures.
- [2026-08-22] Fixed UnicodeEncodeError crash in batch_ingestion.py when piping child output containing emoji characters.
- [2026-08-22] Created batch_ingestion.py control script to automate loop downloading and indexing papers in batches of 50 up to 1,500.
- [2026-08-22] Downloaded 43 new research papers on Graph/Agentic RAG and indexed all 4,373 chunks into Chroma DB using local open-source embeddings.
- [2026-08-22] Linked frontend sidebar stats component with backend stats API to dynamically fetch and display actual paper and chunk counts.
- [2026-08-22] Added /api/stats endpoint to app.py to expose the actual dynamic paper and chunk counts from Chroma DB.
- [2026-08-22] Patched clean_text in pdf_chunker.py to strip unpaired surrogate characters that crash Rust-based tokenizers.
- [2026-08-22] Updated query_rag.py to support querying Chroma using local embedding function.
- [2026-08-22] Updated populate_db.py to support local embeddings using Chroma's built-in ONNX embedding function.
- [2026-08-22] Enhanced pdf_chunker.py to dynamically query arXiv API for missing paper metadata and chunk all downloaded files.
- [2026-08-22] Configured local open-source embedding provider (ONNX MiniLM) in the environment configuration.
- [2026-08-21] Wrapped PDF downloader in a try-except block in arxiv_downloader.py to robustly handle transient errors and missing URLs.
- [2026-08-21] Modified paper selection dropdown to display only the paper title without prefixing the paper ID.
- [2026-08-21] Refactored monolithic frontend into modular ES Modules and native reusable Web Components (RagSidebar, RagChat, and RagSources).
- [2026-08-20] Created a premium web user interface (FastAPI + HTML/CSS/JS) for interactive chat and metadata filtering in app.py.
- [2026-08-20] Configured global SSL bypass context in arxiv_downloader.py to prevent local environment certificate verification errors.
- [2026-08-20] Updated PDF downloading method in arxiv_downloader.py to use urllib since result.download_pdf was removed in arxiv python library v4.0.0+.
- [2026-08-20] Increased downloader max results default from 10 to 50 in arxiv_downloader.py.
- [2026-08-20] Added metadata filtering support (paper ID, publication date range, and minimum page count) to query_rag.py.
- [2026-08-19] Updated .gitignore to exclude data/chroma_db/, paper_chunks.json, and papers_metadata.json from Git.
- [2026-08-19] Updated LLM model default to gemini-2.5-flash because gemini-1.5-flash is no longer supported on the current API version/region.
- [2026-08-19] Added retry logic with exponential backoff and reduced default batch size to 10 in populate_db.py to prevent 429 quota exhaustion errors.
- [2026-08-19] Switched default embedding model to gemini-embedding-001 in config and code because text-embedding-004 is deprecated/unsupported.



- [2026-08-19] Fixed a key mismatch bug in pdf_chunker.py to support both camelCase and snake_case metadata keys.

- [2026-08-16] Added requirements.txt, populate_db.py, and query_rag.py to build and run the Python Chroma DB + Gemini RAG system.
- [2026-08-16] Created the initial README.md explaining project structure, installation, usage, and pipelines.

- [2026-09-12] Fixed pdf_chunker.py: all_parents registry was built in memory but never written to disk; added save of paper_parents.json so parent text is stored once and children reference it by parent_id only.
- [2026-09-12] Fixed query_rag.py: removed parent_text from SQLite SELECT and sparse candidate metadata; added get_parent_registry() lazy loader that reads paper_parents.json once and resolves parent text by parent_id at query time, with legacy fallback for old Chroma rows that still have parent_text embedded.
