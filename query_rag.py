import os
import argparse
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
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

def query_rag(collection, query_text, num_results=3):
    # 1. Embed query
    query_vector = get_query_embedding(query_text)
    
    # 2. Query Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=num_results
    )
    
    # Check if we got any results
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant papers found in the database.", []
    
    # 3. Format context & capture sources
    context_blocks = []
    sources = []
    
    # Chroma returns a list of lists since it supports multiple query texts
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        title = meta.get("title", "Unknown Title")
        authors = meta.get("authors", "Unknown Authors")
        pdf_url = meta.get("pdf_url", "")
        paper_id = meta.get("paper_id", "")
        chunk_idx = meta.get("chunk_index", 0)
        
        # Build block for prompt context
        block = (
            f"Source [{i+1}]: {title} (ID: {paper_id}, Chunk: {chunk_idx})\n"
            f"Authors: {authors}\n"
            f"Text content:\n{doc}\n"
        )
        context_blocks.append(block)
        
        # Save source details for UI reference
        sources.append({
            "index": i + 1,
            "title": title,
            "authors": authors,
            "pdf_url": pdf_url,
            "paper_id": paper_id,
            "distance": dist
        })
        
    context = "\n---\n".join(context_blocks)
    
    # 4. Construct prompt
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

    # 5. Generate Answer via Gemini
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

def interactive_chat(collection):
    print("\n✨ Entered Interactive RAG Chat Mode! Type 'exit' or 'quit' to close.")
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
            answer, sources = query_rag(collection, query_text)
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
    args = parser.parse_args()
    
    try:
        collection = init_services()
        
        if args.query:
            print(f"🔍 Processing query: '{args.query}'...")
            answer, sources = query_rag(collection, args.query, num_results=args.results)
            print_result(args.query, answer, sources)
        else:
            interactive_chat(collection)
            
    except Exception as e:
        print(e)
