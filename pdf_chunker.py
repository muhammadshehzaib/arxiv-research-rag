import os
import json
import re

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

def main():
    data_dir = os.path.join(".", "data")
    papers_dir = os.path.join(data_dir, "papers")
    metadata_path = os.path.join(data_dir, "papers_metadata.json")

    if not os.path.exists(metadata_path):
        print("❌ Metadata file not found. Run arxiv_downloader first!")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📚 Found {len(papers_metadata)} papers. Extracting & chunking text...\n")

    # Try pypdf or pdfplumber if available, otherwise pdf-parse
    try:
        from pypdf import PdfReader
        has_pypdf = True
    except ImportError:
        has_pypdf = False

    all_chunks = []
    processed_count = 0

    for paper in papers_metadata:
        paper_id = paper.get('paperId') or paper.get('paper_id')
        pdf_url = paper.get('pdfUrl') or paper.get('pdf_url')
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
