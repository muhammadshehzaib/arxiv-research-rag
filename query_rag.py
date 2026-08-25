import os
import sys
import argparse
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
from rank_bm25 import BM25Okapi

# Ensure stdout/stderr use UTF-8 encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Global cache for lazy loading in CLI
_bm25_global = None
_bm25_chunks_global = None

def tokenize_text(text):
    """
    Standard alphanumeric lowercasing tokenizer for BM25.
    """
    text = text.lower()
    return re.findall(r'[a-z0-9]+', text)

def init_bm25(chunks_path=os.path.join("data", "paper_chunks.json")):
    """
    Loads chunks and initializes BM25 index.
    """
    if not os.path.exists(chunks_path):
        print(f"⚠️ BM25 warning: Chunks dataset not found at {chunks_path}.")
        return None, None
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            return None, None
        
        # Tokenize each chunk text
        tokenized_corpus = [tokenize_text(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, chunks
    except Exception as e:
        print(f"❌ Error building BM25 index: {e}")
        return None, None

def get_bm25_index():
    global _bm25_global, _bm25_chunks_global
    if _bm25_global is None:
        bm25, chunks = init_bm25()
        _bm25_global = bm25
        _bm25_chunks_global = chunks
    return _bm25_global, _bm25_chunks_global

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "arxiv_papers")

def init_services():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "❌ GEMINI_API_KEY is not set or is still the default placeholder in your .env file.\n"
            "Please add your actual Gemini API key to .env before running this script."
        )
    # Configure Gemini SDK
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize Chroma client
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"❌ Chroma DB not found at {CHROMA_PATH}.\n"
            "Please run populate_db.py to create the vector database first."
        )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    if EMBEDDING_PROVIDER == "local":
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
    else:
        ef = None
        
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        return collection
    except Exception:
        raise ValueError(
            f"❌ Collection '{COLLECTION_NAME}' does not exist in Chroma DB.\n"
            "Please run populate_db.py to populate it."
        )

def get_query_embedding(query_text):
    """
    Generate embedding for the query using Gemini API.
    Uses 'retrieval_query' task type as recommended for search queries.
    """
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query_text,
            task_type="retrieval_query"
        )
        return response['embedding']
    except Exception as e:
        print(f"❌ Failed to generate embedding for query: {e}")
        raise e

def query_rag(collection, query_text, num_results=3, paper_id=None, published_after=None, min_pages=None, bm25=None, bm25_chunks=None):
    # 1. Fetch BM25 index if not passed
    if bm25 is None or bm25_chunks is None:
        bm25, bm25_chunks = get_bm25_index()

    # 2. Construct filter (where clause) for Chroma DB
    where_clauses = []
    if paper_id:
        where_clauses.append({"paper_id": {"$eq": paper_id}})
    if min_pages is not None:
        where_clauses.append({"total_pages": {"$gte": int(min_pages)}})
        
    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}
        
    # Retrieve more candidates (e.g. 20) for blending
    fetch_results = max(num_results * 5, 20)
        
    # --- DENSE RETRIEVAL (Chroma) ---
    dense_candidates = []
    try:
        if EMBEDDING_PROVIDER == "local":
            results = collection.query(
                query_texts=[query_text],
                n_results=fetch_results,
                where=where
            )
        else:
            query_vector = get_query_embedding(query_text)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_results,
                where=where
            )
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for doc, meta, dist, cid in zip(docs, metadatas, distances, ids):
                # Chroma only allows numeric filters, so handle published_after date filtering here
                if published_after:
                    pub_date = meta.get("published", "")
                    if pub_date < published_after:
                        continue
                dense_candidates.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
    except Exception as e:
        print(f"⚠️ Dense search error: {e}")

    # --- SPARSE RETRIEVAL (BM25) ---
    sparse_candidates = []
    if bm25 is not None and bm25_chunks is not None:
        try:
            tokenized_query = tokenize_text(query_text)
            scores = bm25.get_scores(tokenized_query)
            
            # Pair scores with chunk objects and apply metadata filters
            scored_chunks = []
            for idx, (chunk, score) in enumerate(zip(bm25_chunks, scores)):
                if score <= 0.0:  # Skip chunks with zero keyword matches
                    continue
                # Apply metadata filters
                if paper_id and chunk.get("paper_id") != paper_id:
                    continue
                if min_pages is not None and int(chunk.get("total_pages", 0)) < int(min_pages):
                    continue
                if published_after:
                    pub_date = chunk.get("published", "")
                    if pub_date < published_after:
                        continue
                scored_chunks.append((chunk, score, idx))
                
            # Sort by BM25 score descending
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            # Take top candidates
            sparse_candidates = scored_chunks[:fetch_results]
        except Exception as e:
            print(f"⚠️ Sparse search error: {e}")

    # Print diagnostic information to trace dense vs sparse retrieval candidates
    print(f"ℹ️ Dense search retrieved {len(dense_candidates)} candidate(s). Top 3 IDs: {[c['chunk_id'] for c in dense_candidates[:3]]}")
    print(f"ℹ️ Sparse search retrieved {len(sparse_candidates)} candidate(s). Top 3 IDs: {[c[0]['chunk_id'] for c in sparse_candidates[:3]]}")

    # --- BLENDING AND RERANKING (Reciprocal Rank Fusion) ---
    if not dense_candidates and not sparse_candidates:
        return "No relevant papers found matching your query or filters.", []

    # Map chunk_ids to their ranks (1-indexed) in their respective result sets
    dense_ranks = {c["chunk_id"]: i + 1 for i, c in enumerate(dense_candidates)}
    sparse_ranks = {c[0]["chunk_id"]: i + 1 for i, c in enumerate(sparse_candidates)}
    
    all_candidate_ids = set(dense_ranks.keys()).union(sparse_ranks.keys())
    
    # Build a lookup for text and structured metadata
    chunk_lookup = {}
    for c in dense_candidates:
        chunk_lookup[c["chunk_id"]] = {
            "text": c["text"],
            "metadata": c["metadata"],
            "dense_dist": c["distance"]
        }
    for chunk, score, idx in sparse_candidates:
        cid = chunk["chunk_id"]
        if cid not in chunk_lookup:
            authors_str = ", ".join(chunk["authors"]) if isinstance(chunk["authors"], list) else str(chunk.get("authors", ""))
            meta = {
                "paper_id": chunk.get("paper_id", chunk.get("paperId", "")),
                "title": chunk.get("title", ""),
                "authors": authors_str,
                "published": chunk.get("published", ""),
                "pdf_url": chunk.get("pdf_url", chunk.get("pdfUrl", "")),
                "total_pages": int(chunk.get("total_pages", 0)),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "word_count": int(chunk.get("word_count", 0))
            }
            chunk_lookup[cid] = {
                "text": chunk["text"],
                "metadata": meta,
                "dense_dist": 1.0  # Max distance (0% similarity placeholder)
            }
            
    # Calculate RRF scores
    k = 60
    rrf_scores = []
    for cid in all_candidate_ids:
        dense_rank = dense_ranks.get(cid)
        sparse_rank = sparse_ranks.get(cid)
        
        dense_term = 1.0 / (k + dense_rank) if dense_rank is not None else 0.0
        sparse_term = 1.0 / (k + sparse_rank) if sparse_rank is not None else 0.0
        
        score_rrf = dense_term + sparse_term
        rrf_scores.append((cid, score_rrf))
        
    # Sort candidates by RRF score descending
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top results
    final_candidates = rrf_scores[:num_results]
    
    # Max possible RRF score is for rank 1 in dense and rank 1 in sparse
    max_possible_rrf = (1.0 / (k + 1)) + (1.0 / (k + 1)) if (dense_candidates and sparse_candidates) else (1.0 / (k + 1))
    if max_possible_rrf == 0:
        max_possible_rrf = 1.0
        
    # Format context & capture sources
    context_blocks = []
    sources = []
    
    for i, (cid, rrf_score) in enumerate(final_candidates):
        item = chunk_lookup[cid]
        doc = item["text"]
        meta = item["metadata"]
        
        title = meta.get("title", "Unknown Title")
        authors = meta.get("authors", "Unknown Authors")
        pdf_url = meta.get("pdf_url", "")
        paper_id = meta.get("paper_id", "")
        chunk_idx = meta.get("chunk_index", 0)
        
        # Calculate a normalized distance for frontend compatibility (1 - normalized_rrf)
        normalized_rrf = min(1.0, rrf_score / max_possible_rrf)
        compatible_dist = 1.0 - normalized_rrf
        
        block = (
            f"Source [{i+1}]: {title} (ID: {paper_id}, Chunk: {chunk_idx})\n"
            f"Authors: {authors}\n"
            f"Text content:\n{doc}\n"
        )
        context_blocks.append(block)
        
        sources.append({
            "index": i + 1,
            "title": title,
            "authors": authors,
            "pdf_url": pdf_url,
            "paper_id": paper_id,
            "distance": compatible_dist,
            "rrf_score": rrf_score
        })
        
    context = "\n---\n".join(context_blocks)
    
    # Construct prompt
    prompt = f"""You are a helpful and precise research assistant specializing in scientific literature.
Answer the user's question using ONLY the provided search results from arXiv research papers.

Requirements:
1. Ground your answer strictly on the provided Context. Do not make up facts or use external training knowledge.
2. If the Context does not contain enough information to answer the question, state that clearly (e.g. "Based on the retrieved context, I cannot answer this because...").
3. Be professional, detailed, and structure your answer logically.
4. Cite your sources in the text using [Source 1], [Source 2], etc.

Context:
{context}

Question: {query_text}

Answer:"""

    # Generate Answer via Gemini
    try:
        model = genai.GenerativeModel(model_name=LLM_MODEL)
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"❌ Failed to generate response from Gemini model: {e}", sources

def print_result(query_text, answer, sources):
    print("\n" + "="*80)
    print(f"❓ QUESTION: {query_text}")
    print("="*80)
    print(f"\n💡 RAG ANSWER:\n{answer}\n")
    print("="*80)
    print("📚 RETRIEVED SOURCES:")
    for src in sources:
        print(f"  [{src['index']}] {src['title']}")
        print(f"      Authors: {src['authors']}")
        print(f"      ArXiv URL: {src['pdf_url']}")
        print(f"      Distance Score: {src['distance']:.4f}")
    print("="*80 + "\n")

def interactive_chat(collection, paper_id=None, published_after=None, min_pages=None):
    print("\n✨ Entered Interactive RAG Chat Mode! Type 'exit' or 'quit' to close.")
    active_filters = []
    if paper_id:
        active_filters.append(f"Paper ID: {paper_id}")
    if published_after:
        active_filters.append(f"Published After: {published_after}")
    if min_pages is not None:
        active_filters.append(f"Min Pages: {min_pages}")
    if active_filters:
        print(f"⚙️ Active Filters: {', '.join(active_filters)}")
    print("Ask any question based on your downloaded arXiv papers.\n")
    
    while True:
        try:
            query_text = input("RAG Chat > ").strip()
            if not query_text:
                continue
            if query_text.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            print("🔍 Searching vector database and generating answer...")
            answer, sources = query_rag(
                collection, 
                query_text, 
                paper_id=paper_id, 
                published_after=published_after, 
                min_pages=min_pages
            )
            print_result(query_text, answer, sources)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the arXiv Research RAG System.")
    parser.add_argument("--query", type=str, help="Single query to ask the RAG system. If omitted, starts interactive chat mode.")
    parser.add_argument("--results", type=int, default=3, help="Number of context documents to retrieve (default: 3).")
    parser.add_argument("--paper-id", type=str, default=None, help="Filter results by a specific arXiv Paper ID.")
    parser.add_argument("--published-after", type=str, default=None, help="Filter results by publication date (YYYY-MM-DD or newer).")
    parser.add_argument("--min-pages", type=int, default=None, help="Filter results by minimum page count.")
    args = parser.parse_args()
    
    try:
        collection = init_services()
        
        if args.query:
            print(f"🔍 Processing query: '{args.query}'...")
            answer, sources = query_rag(
                collection, 
                args.query, 
                num_results=args.results,
                paper_id=args.paper_id,
                published_after=args.published_after,
                min_pages=args.min_pages
            )
            print_result(args.query, answer, sources)
        else:
            interactive_chat(
                collection,
                paper_id=args.paper_id,
                published_after=args.published_after,
                min_pages=args.min_pages
            )
            
    except Exception as e:
        print(e)
