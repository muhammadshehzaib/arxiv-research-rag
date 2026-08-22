import os
import ssl
import sys
import time
import subprocess
import json
import arxiv
from urllib.request import urlretrieve

# Bypass SSL context verification issues (common in Windows setups)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Predefined query list to scan arXiv for a wide selection of relevant papers
QUERIES = [
    "Retrieval-Augmented Generation",
    "Graph RAG",
    "Agentic RAG",
    "Vector Search RAG",
    "Hybrid Search RAG",
    "Dense Passage Retrieval RAG",
    "RAG Evaluation",
    "Self-RAG",
    "Multi-Agent RAG",
    "Knowledge Graph RAG",
    "RAG Hallucinations",
    "Document Retrieval LLM",
    "Query Expansion RAG",
    "Semantic Search LLM",
    "Long Context LLM RAG",
    "Multimodal RAG",
    "Agentic search LLM",
    "RAG optimization",
    "LLM vector database",
    "Context retrieval LLM",
    "Graph neural network RAG",
    "RAG QA system",
    "Medical QA RAG",
    "Financial RAG",
    "Legal RAG LLM",
    "RAG prompt engineering",
    "Tabular RAG",
    "Hierarchical RAG",
    "Dynamic RAG",
    "Adaptive RAG"
]

TARGET_PAPERS = 1500
BATCH_SIZE = 50
DATA_DIR = os.path.join(".", "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")

def count_downloaded_papers():
    if not os.path.exists(PAPERS_DIR):
        return 0
    return len([f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")])

def get_existing_ids():
    if not os.path.exists(PAPERS_DIR):
        return set()
    existing_ids = set()
    for filename in os.listdir(PAPERS_DIR):
        if filename.endswith(".pdf"):
            paper_id = filename.split("_")[0]
            existing_ids.add(paper_id)
    return existing_ids

def run_subprocess(script_name):
    print(f"--- Running {script_name} ---")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Run the script and stream output
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"Error: {script_name} failed with return code {rc}")
        return False
    print(f"--- Finished {script_name} successfully ---\n")
    return True

def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    client = arxiv.Client()
    
    current_query_index = 0
    
    while True:
        current_count = count_downloaded_papers()
        print(f"=== Current Status: {current_count} / {TARGET_PAPERS} papers downloaded ===")
        
        if current_count >= TARGET_PAPERS:
            print(f"Target of {TARGET_PAPERS} papers reached. Exiting.")
            break
            
        remaining = TARGET_PAPERS - current_count
        batch_limit = min(BATCH_SIZE, remaining)
        print(f"Need to download {batch_limit} more papers in this batch.")
        
        existing_ids = get_existing_ids()
        downloaded_in_batch = 0
        
        # We loop through queries until we fulfill the batch limit
        while downloaded_in_batch < batch_limit:
            if current_query_index >= len(QUERIES):
                print("Warning: Exhausted query list. Resetting index to query again.")
                current_query_index = 0
                
            query = QUERIES[current_query_index]
            print(f"Searching arXiv for query: '{query}' (Index: {current_query_index})")
            
            search = arxiv.Search(
                query=query,
                max_results=100, # fetch more results to find unique papers
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            try:
                results = list(client.results(search))
                print(f"Found {len(results)} search results.")
            except Exception as e:
                print(f"Error searching arXiv: {e}")
                time.sleep(10)
                continue
                
            for result in results:
                paper_id = result.entry_id.split("/")[-1]
                if paper_id in existing_ids:
                    continue
                    
                title = result.title
                safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in title).strip()
                filename = f"{paper_id}_{safe_title[:40]}.pdf"
                filepath = os.path.join(PAPERS_DIR, filename)
                
                print(f"Downloading [{downloaded_in_batch + 1}/{batch_limit}]: '{title[:50]}' (ID: {paper_id})")
                try:
                    urlretrieve(result.pdf_url, filepath)
                    existing_ids.add(paper_id)
                    downloaded_in_batch += 1
                    
                    if downloaded_in_batch >= batch_limit:
                        break
                except Exception as ex:
                    print(f"Failed to download PDF {paper_id}: {ex}")
                    
                # Respect arXiv rate limit policy (about 3 seconds between downloads)
                time.sleep(3)
                
            current_query_index += 1
            
        print(f"Batch download complete. Downloaded {downloaded_in_batch} new papers.")
        print("Starting PDF Chunking & Metadata update...")
        
        if not run_subprocess("pdf_chunker.py"):
            print("Failed at chunking stage. Retrying batch loop.")
            time.sleep(10)
            continue
            
        print("Starting Database population...")
        if not run_subprocess("populate_db.py"):
            print("Failed at database population stage. Retrying batch loop.")
            time.sleep(10)
            continue
            
        print(f"Successfully finished batch! Currently indexed papers count: {count_downloaded_papers()}\n")
        print("Sleeping 15 seconds to let the system cool down...")
        time.sleep(15)

if __name__ == "__main__":
    main()
