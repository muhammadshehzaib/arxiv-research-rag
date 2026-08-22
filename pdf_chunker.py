import os
import json
import re
import time
import arxiv

"""
Python PDF Text Extractor & Chunker for RAG
- Reads downloaded arXiv PDFs
- Cleans and normalizes text
- Chunks text into sliding window (~500 words with 50 word overlap)
- Attaches rich metadata to each chunk for Qdrant storage
"""

CHUNK_SIZE = 500  # Words per chunk
OVERLAP = 50      # Overlap between chunks

def clean_text(text):
    # Remove surrogate characters (U+D800 to U+DFFF) which break tokenizers in Rust-based python packages
    text = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
    # Remove hyphenated linebreaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Replace newlines with spaces
    text = re.sub(r'\r?\n|\r', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = [w for w in text.split(' ') if w]
    chunks = []
    if not words:
        return chunks

    i = 0
    idx = 0
    while i < len(words):
        end = min(i + chunk_size, len(words))
        chunk_words = words[i:end]
        chunk_content = ' '.join(chunk_words)

        chunks.append({
            "chunk_index": idx,
            "text": chunk_content,
            "word_count": len(chunk_words),
            "start_word": i,
            "end_word": end
        })

        idx += 1
        i += (chunk_size - overlap)
        if i >= len(words) or (len(words) - i < overlap and len(chunks) > 0):
            break

    return chunks

def fetch_missing_metadata(paper_ids):
    """
    Fetches metadata for a list of arXiv paper IDs.
    """
    if not paper_ids:
        return []
    
    print(f"📡 Fetching metadata for {len(paper_ids)} missing papers from arXiv...")
    client = arxiv.Client()
    
    batch_size = 50
    fetched_papers = []
    
    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i:i+batch_size]
        try:
            search = arxiv.Search(id_list=batch_ids)
            results = list(client.results(search))
            for result in results:
                metadata = {
                    "paper_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories
                }
                fetched_papers.append(metadata)
            print(f"   ✅ Fetched {len(fetched_papers)} papers successfully so far.")
        except Exception as e:
            print(f"   ❌ Error fetching batch starting with {batch_ids[0]}: {e}")
            print("   🔄 Falling back to fetching papers individually in this batch...")
            for paper_id in batch_ids:
                try:
                    search = arxiv.Search(id_list=[paper_id])
                    results = list(client.results(search))
                    if results:
                        result = results[0]
                        metadata = {
                            "paper_id": result.entry_id.split("/")[-1],
                            "title": result.title,
                            "authors": [author.name for author in result.authors],
                            "published": result.published.strftime("%Y-%m-%d"),
                            "summary": result.summary,
                            "pdf_url": result.pdf_url,
                            "categories": result.categories
                        }
                        fetched_papers.append(metadata)
                        print(f"     ✅ Successfully fetched metadata for {paper_id}")
                        time.sleep(1)
                except Exception as ex:
                    print(f"     ❌ Failed to fetch metadata for paper {paper_id}: {ex}")
            
    return fetched_papers

def main():
    data_dir = os.path.join(".", "data")
    papers_dir = os.path.join(data_dir, "papers")
    metadata_path = os.path.join(data_dir, "papers_metadata.json")

    # Load current metadata if it exists
    papers_metadata = []
    existing_ids = set()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                papers_metadata = json.load(f)
                for paper in papers_metadata:
                    pid = paper.get('paperId') or paper.get('paper_id')
                    if pid:
                        existing_ids.add(pid)
        except Exception as e:
            print(f"⚠️ Error reading existing metadata file: {e}")

    # Scan downloaded PDF files
    if not os.path.exists(papers_dir):
        print(f"❌ Papers directory not found at {papers_dir}!")
        return

    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    print(f"📁 Found {len(pdf_files)} PDF files in download folder.")

    id_to_filename = {}
    downloaded_ids = set()
    for filename in pdf_files:
        paper_id = filename.split("_")[0]
        id_to_filename[paper_id] = filename
        downloaded_ids.add(paper_id)

    # Check for missing metadata
    missing_ids = list(downloaded_ids - existing_ids)
    if missing_ids:
        print(f"🔍 {len(missing_ids)} papers are downloaded but missing from metadata.json")
        fetched = fetch_missing_metadata(missing_ids)
        for paper in fetched:
            pid = paper.get('paper_id')
            if pid and pid not in existing_ids:
                papers_metadata.append(paper)
                existing_ids.add(pid)
        
        # Save complete metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, indent=2)
        print(f"💾 Updated metadata saved to: {metadata_path}")

    # Filter metadata to keep only the downloaded papers
    active_metadata = [p for p in papers_metadata if (p.get('paperId') or p.get('paper_id')) in downloaded_ids]
    print(f"📚 Processing {len(active_metadata)} downloaded papers for chunking...\n")

    # Try pypdf or pdfplumber
    try:
        from pypdf import PdfReader
        has_pypdf = True
    except ImportError:
        has_pypdf = False

    all_chunks = []
    processed_count = 0

    for paper in active_metadata:
        paper_id = paper.get('paperId') or paper.get('paper_id')
        pdf_url = paper.get('pdfUrl') or paper.get('pdf_url')
        
        # Get filename directly from our scanned map
        filename = id_to_filename.get(paper_id)
        if not filename:
            # Fallback
            safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in paper['title']).strip()[:40]
            filename = f"{paper_id.replace('/', '_')}_{safe_title}.pdf"
        
        pdf_path = os.path.join(papers_dir, filename)

        if not os.path.exists(pdf_path):
            print(f"⚠️ PDF missing: {pdf_path}. Skipping.")
            continue

        raw_text = ""
        total_pages = 0

        if has_pypdf:
            try:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        raw_text += txt + " "
            except Exception as e:
                print(f"❌ Error reading PDF {pdf_path}: {e}")
                continue

        if not raw_text.strip():
            print(f"⚠️ Could not extract text from {filename} (pypdf missing or empty).")
            continue

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, CHUNK_SIZE, OVERLAP)

        print(f"📄 Paper: '{paper['title'][:50]}...'")
        print(f"   └─ Pages: {total_pages} | Words: {len(cleaned.split())} | Chunks: {len(chunks)}")

        for c in chunks:
            enriched = {
                "chunk_id": f"{paper_id}_c{c['chunk_index']}",
                "paper_id": paper_id,
                "title": paper['title'],
                "authors": paper['authors'],
                "published": paper['published'],
                "pdf_url": pdf_url,
                "total_pages": total_pages,
                **c
            }
            all_chunks.append(enriched)
        processed_count += 1

    out_path = os.path.join(data_dir, "paper_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n🎉 Successfully chunked {processed_count} papers into {len(all_chunks)} chunks!")
    print(f"💾 Output saved to: {out_path}")


if __name__ == "__main__":
    main()
