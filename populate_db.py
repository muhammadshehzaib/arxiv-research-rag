import os
import json
import time
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "arxiv_papers") # Defaulting or using env collection name

def init_gemini():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "❌ GEMINI_API_KEY is not set or is still the default placeholder in your .env file.\n"
            "Please add your actual Gemini API key to .env before running this script."
        )
    genai.configure(api_key=GEMINI_API_KEY)

def load_chunks():
    chunks_path = os.path.join("data", "paper_chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"❌ Chunks dataset not found at {chunks_path}.\n"
            "Please make sure you have run the PDF chunker first (e.g., pdf_chunker.py)."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_texts(texts, model=EMBEDDING_MODEL):
    """
    Generates embeddings for a list of texts using the Gemini API.
    """
    try:
        response = genai.embed_content(
            model=model,
            content=texts,
            task_type="retrieval_document"
        )
        # response['embedding'] contains list of embedding vectors
        return response['embedding']
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise e

def populate_database():
    print("🚀 Initializing RAG Database Population...")
    
    # 1. Initialize Gemini
    init_gemini()
    
    # 2. Load Chunks
    chunks = load_chunks()
    total_chunks = len(chunks)
    print(f"📚 Loaded {total_chunks} text chunks from paper_chunks.json")

    # 3. Setup Chroma Client
    print(f"📁 Connecting to Chroma DB at: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Get or create collection
    # Note: Chroma will run without local embeddings function because we supply precomputed embeddings
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "arXiv Research Papers Chunks"}
    )
    
    print(f"📦 Populating collection '{COLLECTION_NAME}'...")

    # 4. Generate Embeddings and Upsert in Batches
    batch_size = 50
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        
        # Prepare data for Chroma
        ids = [chunk["chunk_id"] for chunk in batch]
        documents = [chunk["text"] for chunk in batch]
        
        # Process metadata (Chroma requires primitive types only: str, int, float, bool)
        metadatas = []
        for chunk in batch:
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
            metadatas.append(meta)
        
        print(f"🌐 Generating embeddings for batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        
        # Generate embeddings via Gemini API
        try:
            embeddings = embed_texts(documents)
            
            # Upsert into Chroma
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"   ✅ Stored batch successfully.")
        except Exception as e:
            print(f"   ⚠️ Failed to process batch starting at index {i}. Error: {e}")
            print("   Skipping this batch...")
        
        # Avoid hitting API rate limits too quickly
        time.sleep(1)

    duration = time.time() - start_time
    print(f"\n🎉 Database population complete in {duration:.2f} seconds!")
    print(f"👉 Total items stored in '{COLLECTION_NAME}': {collection.count()}")

if __name__ == "__main__":
    try:
        populate_database()
    except Exception as err:
        print(err)
