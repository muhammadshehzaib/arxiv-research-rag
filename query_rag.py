import os
import sys
import argparse
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
from rank_bm25 import BM25Okapi

# Reranker package imports
try:
    from flashrank import Ranker
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

class Reranker:
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = None
        self.flashrank_client = None
        
        # 1. Try to initialize Cohere Reranker if API key is provided
        if self.cohere_api_key and self.cohere_api_key != "your_cohere_api_key_here":
            if HAS_COHERE:
                try:
                    self.cohere_client = cohere.Client(api_key=self.cohere_api_key)
                    print("🚀 Cohere Rerank client initialized successfully.")
                except Exception as e:
                    print(f"⚠️ Failed to initialize Cohere client: {e}")
            else:
                print("⚠️ Cohere package is missing but COHERE_API_KEY is configured.")
                
        # 2. Try to initialize local FlashRank if Cohere is not initialized
        if not self.cohere_client:
            if HAS_FLASHRANK:
                try:
                    # ms-marco-MiniLM-L-12-v2 is an excellent balanced reranker (34MB)
                    model_name = os.getenv("FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")
                    print(f"🤖 Initializing local FlashRank with model: {model_name}...")
                    self.flashrank_client = Ranker(model_name=model_name)
                    print("✅ FlashRank reranker initialized successfully.")
                except Exception as e:
                    print(f"⚠️ Failed to initialize FlashRank: {e}")
            else:
                print("⚠️ flashrank package is not installed. Local reranking is disabled.")

    def rerank(self, query, passages):
        """
        Reranks a list of passages (dicts with 'id', 'text', 'metadata') relative to the query.
        Returns the list of passages sorted by relevancy score descending, with a 'score' key added.
        """
        if not passages:
            return []
            
        # If Cohere client is active
        if self.cohere_client:
            try:
                doc_texts = [p["text"] for p in passages]
                response = self.cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=doc_texts,
                    top_n=len(passages)
                )
                
                reranked = []
                for res in response.results:
                    idx = res.index
                    passage = passages[idx]
                    passage["score"] = res.relevance_score
                    reranked.append(passage)
                return reranked
            except Exception as e:
                print(f"⚠️ Cohere reranking failed: {e}. Falling back to default/local ordering.")
                
        # If FlashRank client is active
        if self.flashrank_client:
            try:
                flash_passages = []
                for p in passages:
                    flash_passages.append({
                        "id": p["id"],
                        "text": p["text"],
                        "metadata": p.get("metadata", {}),
                        "rrf_score": p.get("rrf_score", 0.0)
                    })
                
                results = self.flashrank_client.rerank(query=query, passages=flash_passages)
                
                reranked = []
                for res in results:
                    reranked.append({
                        "id": res["id"],
                        "text": res["text"],
                        "metadata": res["metadata"],
                        "rrf_score": res["rrf_score"],
                        "score": res["score"]
                    })
                return reranked
            except Exception as e:
                print(f"⚠️ FlashRank reranking failed: {e}. Falling back to default/local ordering.")
                
        # Graceful fallback: just add a mock score based on index/RRF and return
        print("⚠️ No active reranker provider. Returning results in RRF order.")
        for idx, p in enumerate(passages):
            p["score"] = 1.0 / (idx + 1)
        return passages

# Global lazy-loaded reranker
_reranker_instance = None
def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


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
        
    # Retrieve more candidates (e.g. 30) for blending and cross-encoder reranking
    fetch_results = max(num_results * 6, 30)
        
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

    # --- BLENDING (Reciprocal Rank Fusion) ---
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
                "word_count": int(chunk.get("word_count", 0)),
                "parent_id": chunk.get("parent_id", ""),
                "parent_text": chunk.get("parent_text", "")
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
    
    # Take top candidates for cross-encoder reranking
    top_k_for_rerank = 30
    rrf_candidates = rrf_scores[:top_k_for_rerank]
    
    # Construct passages list for the Reranker
    passages = []
    for cid, rrf_score in rrf_candidates:
        item = chunk_lookup[cid]
        passages.append({
            "id": cid,
            "text": item["text"],
            "metadata": item["metadata"],
            "rrf_score": rrf_score
        })
        
    # --- CROSS-ENCODER RERANKING ---
    reranker = get_reranker()
    reranked_passages = reranker.rerank(query_text, passages)
    
    # --- PARENT CONTEXT RETRIEVAL & DE-DUPLICATION ---
    # We iterate through the sorted reranked child chunks and resolve unique parent contexts
    # until we collect up to num_results unique parent contexts.
    unique_parents = []
    seen_parent_ids = set()
    for p in reranked_passages:
        pid = p["metadata"].get("parent_id")
        p_text = p["metadata"].get("parent_text")
        
        if pid and p_text:
            # Parent-child chunk found
            if pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                unique_parents.append({
                    "chunk_id": p["id"],
                    "text": p_text,  # Return parent text as context
                    "metadata": p["metadata"],
                    "score": p.get("score", 0.0),
                    "rrf_score": p.get("rrf_score", 0.0)
                })
        else:
            # Fallback if no parent context is present
            cid = p["id"]
            if cid not in seen_parent_ids:
                seen_parent_ids.add(cid)
                unique_parents.append({
                    "chunk_id": cid,
                    "text": p["text"],
                    "metadata": p["metadata"],
                    "score": p.get("score", 0.0),
                    "rrf_score": p.get("rrf_score", 0.0)
                })
                
        if len(unique_parents) == num_results:
            break
            
    print(f"ℹ️ Reranking finished. Selected {len(unique_parents)} unique parent contexts from {len(reranked_passages)} reranked candidates.")
    
    # Format context & capture sources
    context_blocks = []
    sources = []
    
    for i, item in enumerate(unique_parents):
        doc = item["text"]
        meta = item["metadata"]
        
        title = meta.get("title", "Unknown Title")
        authors = meta.get("authors", "Unknown Authors")
        pdf_url = meta.get("pdf_url", "")
        paper_id = meta.get("paper_id", "")
        chunk_idx = meta.get("chunk_index", 0)
        
        # Convert reranker score to compatible distance metric for UI (distance = 1 - score)
        score = item["score"]
        if score < 0.0 or score > 1.0:
            score = max(0.0, min(1.0, score))
        compatible_dist = 1.0 - score
        
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
            "rrf_score": item["rrf_score"],
            "rerank_score": score,
            "text": doc
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
