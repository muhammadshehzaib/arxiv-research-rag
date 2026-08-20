import arxiv
import os
import json
import ssl
from datetime import datetime

# Bypass SSL certificate verification issues (common in local Python setups)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def fetch_arxiv_papers(query="Retrieval Augmented Generation", max_results=10, download_pdfs=False):
    """
    Fetches paper metadata from arXiv based on a search query.
    Optionally downloads PDFs into a local folder.
    """
    print(f"🔍 Searching arXiv for: '{query}' (Max results: {max_results})...\n")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers_metadata = []
    
    # Create directory for PDFs if download is requested
    pdf_dir = os.path.join(".", "data", "papers")
    if download_pdfs:
        os.makedirs(pdf_dir, exist_ok=True)

    for i, result in enumerate(client.results(search), 1):
        metadata = {
            "paper_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "categories": result.categories
        }
        papers_metadata.append(metadata)

        print(f"[{i}] Title: {metadata['title']}")
        print(f"    Authors: {', '.join(metadata['authors'][:3])}{' et al.' if len(metadata['authors']) > 3 else ''}")
        print(f"    Published: {metadata['published']}")
        print(f"    PDF URL: {metadata['pdf_url']}")
        
        if download_pdfs:
            # Clean title for filename
            safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in metadata['title']).strip()
            filename = f"{metadata['paper_id']}_{safe_title[:40]}.pdf"
            filepath = os.path.join(pdf_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"    📥 Downloading PDF to: {filepath}")
                from urllib.request import urlretrieve
                urlretrieve(metadata['pdf_url'], filepath)
            else:
                print(f"    ✅ PDF already downloaded.")
        
        print("-" * 75)

    # Save metadata summary as JSON for future chunking / embedding pipeline
    os.makedirs(os.path.join(".", "data"), exist_ok=True)
    json_path = os.path.join(".", "data", "papers_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, indent=2)

    print(f"\n✨ Successfully fetched metadata for {len(papers_metadata)} papers.")
    print(f"📁 Metadata saved to: {json_path}")
    if download_pdfs:
        print(f"📁 PDFs downloaded to: {pdf_dir}")

if __name__ == "__main__":
    # Fetch 50 papers related to RAG and save metadata + PDFs
    fetch_arxiv_papers(query="Retrieval Augmented Generation", max_results=50, download_pdfs=True)
