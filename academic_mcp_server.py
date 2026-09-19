import os
import sys
import json
import ssl
import time
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Bypass SSL issues if present on Windows
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import arxiv
from mcp.server.mcpserver import MCPServer

# Initialize MCP Server
mcp = MCPServer("academic-research-mcp")


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search live academic research papers from arXiv by query or keywords.
    Returns metadata including title, authors, publication date, abstract, and PDF link.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for res in client.results(search):
            paper_id = res.entry_id.split("/")[-1]
            papers.append({
                "paper_id": paper_id,
                "title": res.title.replace("\n", " ").strip(),
                "authors": [a.name for a in res.authors],
                "published": res.published.strftime("%Y-%m-%d"),
                "summary": res.summary.replace("\n", " ").strip(),
                "pdf_url": res.pdf_url,
                "categories": res.categories,
                "source": "arXiv (Live)"
            })
        
        if not papers:
            return json.dumps({"status": "no_results", "query": query, "papers": []}, indent=2)
            
        return json.dumps({"status": "success", "count": len(papers), "papers": papers}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"arXiv search failed: {str(e)}"}, indent=2)


@mcp.tool()
def search_semantic_scholar(query: str, limit: int = 5) -> str:
    """
    Search academic papers from Semantic Scholar Academic Graph.
    Includes title, authors, year, citation count, abstract, and open-access links.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {
        "User-Agent": "ArxivResearchRAG/1.0 (https://github.com/muhammadshehzaib/arxiv-research-rag)"
    }
    if api_key:
        headers["x-api-key"] = api_key

    encoded_query = urllib.parse.quote(query)
    fields = "paperId,title,authors,abstract,year,citationCount,openAccessPdf,url"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={limit}&fields={fields}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            raw_papers = data.get("data", [])
            papers = []
            for p in raw_papers:
                authors = [a.get("name", "") for a in p.get("authors", [])]
                pdf_info = p.get("openAccessPdf") or {}
                pdf_url = pdf_info.get("url", "")
                papers.append({
                    "paper_id": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "authors": authors,
                    "published": str(p.get("year", "Unknown")),
                    "citation_count": p.get("citationCount", 0),
                    "summary": p.get("abstract") or "No abstract available.",
                    "pdf_url": pdf_url,
                    "url": p.get("url", ""),
                    "source": "Semantic Scholar (Live)"
                })
            return json.dumps({"status": "success", "count": len(papers), "papers": papers}, indent=2)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return json.dumps({
                "status": "rate_limited",
                "message": "Semantic Scholar public rate limit reached. Consider adding SEMANTIC_SCHOLAR_API_KEY to .env or using search_arxiv.",
                "papers": []
            }, indent=2)
        return json.dumps({"status": "error", "message": f"Semantic Scholar HTTP {e.code}: {e.reason}"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Semantic Scholar search failed: {str(e)}"}, indent=2)


@mcp.tool()
def fetch_paper_by_id(paper_id: str) -> str:
    """
    Fetch exact metadata and abstract for an arXiv paper given its ID (e.g. '2305.14314').
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id.strip()])
        results = list(client.results(search))
        if not results:
            return json.dumps({"status": "not_found", "paper_id": paper_id}, indent=2)
        res = results[0]
        paper = {
            "paper_id": res.entry_id.split("/")[-1],
            "title": res.title.replace("\n", " ").strip(),
            "authors": [a.name for a in res.authors],
            "published": res.published.strftime("%Y-%m-%d"),
            "summary": res.summary.replace("\n", " ").strip(),
            "pdf_url": res.pdf_url,
            "categories": res.categories,
            "source": "arXiv (Direct ID Lookup)"
        }
        return json.dumps({"status": "success", "paper": paper}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Paper lookup failed: {str(e)}"}, indent=2)


@mcp.tool()
def search_local_rag(query: str, num_results: int = 3) -> str:
    """
    Query the local hybrid RAG (Chroma DB embeddings + SQLite FTS5 BM25) over indexed research papers.
    """
    try:
        from query_rag import init_services, query_rag
        collection = init_services()
        answer, sources = query_rag(collection, query, num_results=num_results)
        return json.dumps({
            "status": "success",
            "answer": answer,
            "sources": sources
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Local RAG search failed: {str(e)}"}, indent=2)


@mcp.tool()
def ingest_live_paper_to_rag(paper_id: str) -> str:
    """
    Download a live arXiv paper PDF, extract & chunk its text, compute embeddings,
    and index it into local Chroma DB vector store and SQLite FTS5 BM25 index.
    """
    try:
        from utils import date_to_int
        import sqlite3

        paper_id = paper_id.strip()
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        if not results:
            return json.dumps({"status": "not_found", "message": f"Paper {paper_id} not found on arXiv."}, indent=2)
        
        result = results[0]
        pdf_dir = os.path.join("data", "papers")
        os.makedirs(pdf_dir, exist_ok=True)
        
        safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in result.title).strip()
        filename = f"{paper_id}_{safe_title[:40]}.pdf"
        filepath = os.path.join(pdf_dir, filename)
        
        if not os.path.exists(filepath):
            from urllib.request import urlretrieve
            urlretrieve(result.pdf_url, filepath)
            
        # Parse PDF using pypdf
        import pypdf
        reader = pypdf.PdfReader(filepath)
        full_text = ""
        total_pages = len(reader.pages)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
                
        if not full_text.strip():
            return json.dumps({"status": "empty_pdf", "message": "Could not extract text from PDF."}, indent=2)

        # Chunk using 500 word sliding window with 50 word overlap
        words = [w for w in full_text.split() if w]
        chunks = []
        chunk_size = 500
        overlap = 50
        i = 0
        idx = 0
        pub_date = result.published.strftime("%Y-%m-%d")
        pub_int = date_to_int(pub_date)
        authors_str = ", ".join([a.name for a in result.authors])

        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk_content = " ".join(words[i:end])
            chunks.append({
                "chunk_id": f"{paper_id}_c{idx}",
                "chunk_index": idx,
                "text": chunk_content,
                "word_count": len(words[i:end]),
                "paper_id": paper_id,
                "title": result.title,
                "authors": authors_str,
                "published": pub_date,
                "published_int": pub_int,
                "pdf_url": result.pdf_url,
                "total_pages": total_pages,
                "parent_id": f"{paper_id}_p0"
            })
            idx += 1
            i += (chunk_size - overlap)
            if i >= len(words):
                break

        # 1. Add chunks into Chroma collection
        from query_rag import init_services
        collection = init_services()

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "paper_id": c["paper_id"],
            "title": c["title"],
            "authors": c["authors"],
            "published": c["published"],
            "published_int": c["published_int"],
            "pdf_url": c["pdf_url"],
            "chunk_index": c["chunk_index"],
            "total_pages": c["total_pages"],
            "word_count": c["word_count"],
            "parent_id": c["parent_id"]
        } for c in chunks]

        # Generate embeddings if using Gemini embedding provider
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini")
        embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        
        if embedding_provider != "local":
            import google.generativeai as genai
            batch_size = 10
            embeddings = []
            for b_idx in range(0, len(documents), batch_size):
                b_docs = documents[b_idx : b_idx + batch_size]
                resp = genai.embed_content(
                    model=embedding_model,
                    content=b_docs,
                    task_type="retrieval_document"
                )
                embeddings.extend(resp["embedding"])
                
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        # 2. Insert into SQLite FTS5 BM25 index if database exists
        fts_db_path = os.path.join("data", "bm25_fts.db")
        if os.path.exists(fts_db_path):
            try:
                con = sqlite3.connect(fts_db_path)
                cur = con.cursor()
                rows = [
                    (
                        c["chunk_id"], c["paper_id"], c["title"], c["authors"],
                        c["published"], c["published_int"], c["total_pages"],
                        c["chunk_index"], c["word_count"], c["parent_id"],
                        c["pdf_url"], c["text"]
                    )
                    for c in chunks
                ]
                cur.executemany("""
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, paper_id, title, authors, published, published_int,
                        total_pages, chunk_index, word_count, parent_id, pdf_url, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                con.commit()
                con.close()
            except Exception as fts_err:
                print(f"⚠️ Warning: Could not update SQLite FTS5 for paper {paper_id}: {fts_err}")

        # 3. Update data/papers_metadata.json
        meta_path = os.path.join("data", "papers_metadata.json")
        try:
            papers_meta = []
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    papers_meta = json.load(f)
            
            # Check if already present
            if not any(p.get("paper_id") == paper_id for p in papers_meta):
                papers_meta.append({
                    "paper_id": paper_id,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": pub_date,
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories
                })
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(papers_meta, f, indent=2)
        except Exception as meta_err:
            print(f"⚠️ Warning: Could not update papers_metadata.json: {meta_err}")

        return json.dumps({
            "status": "success",
            "message": f"Successfully downloaded, chunked, and embedded '{result.title}' into Chroma DB ({len(chunks)} chunks).",
            "paper_id": paper_id,
            "chunks_count": len(chunks)
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Ingestion failed: {str(e)}"}, indent=2)


if __name__ == "__main__":
    # Runs the MCP server with stdio transport by default
    mcp.run()
