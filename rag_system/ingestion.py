"""One parent/child format for batch PDFs and papers fetched live."""
import json
import hashlib
import re
from pathlib import Path
from filelock import FileLock

from rag_system.storage import atomic_write_json
from rag_system.utils import clean_text, date_to_int

PARENT_CHUNK_SIZE, PARENT_OVERLAP = 800, 100
CHILD_CHUNK_SIZE, CHILD_OVERLAP = 150, 20


def build_paper_chunks(text, paper, total_pages):
    from rag_system.pdf_extraction import EXTRACTION_VERSION
    paper_id = paper.get("paper_id") or paper["paperId"]
    common = dict(paper_id=paper_id, title=paper.get("title", ""), authors=paper.get("authors", []),
                  published=paper.get("published", ""), published_int=date_to_int(paper.get("published", "")),
                  pdf_url=paper.get("pdf_url") or paper.get("pdfUrl", ""), total_pages=total_pages)
    pages = text if isinstance(text, list) else [{"text": clean_text(text), "page_number": None}]
    unreadable = [str(page.get("page_number")) for page in pages if not page["text"].strip()]
    if unreadable:
        common["extraction_warning"] = "Unreadable pages " + ", ".join(unreadable) + "; paper coverage is incomplete."
    children, parents = [], {}
    for page in pages:
        content = page["text"]
        words = list(re.finditer(r"\S+", content))
        page_meta = dict(common)
        if page.get("page_number") is not None:
            page_meta.update(page_start=page["page_number"], page_end=page["page_number"], extraction_version=EXTRACTION_VERSION)
        if page.get("warnings"):
            page_meta["extraction_warning"] = " ".join(filter(None, [
                common.get("extraction_warning", ""), *page["warnings"],
            ]))
        for start in range(0, len(words), PARENT_CHUNK_SIZE - PARENT_OVERLAP):
            parent_index = len(parents)
            parent_words = words[start:start + PARENT_CHUNK_SIZE]
            parent_text = content[parent_words[0].start():parent_words[-1].end()]
            parent_id = f"{paper_id}_p{parent_index}"
            parents[parent_id] = dict(page_meta, parent_id=parent_id, parent_index=parent_index,
                                     text=parent_text, word_count=len(parent_words))
            for child_start in range(0, len(parent_words), CHILD_CHUNK_SIZE - CHILD_OVERLAP):
                child_words = parent_words[child_start:child_start + CHILD_CHUNK_SIZE]
                index = len(children)
                child_text = content[child_words[0].start():child_words[-1].end()]
                children.append(dict(page_meta, chunk_id=f"{paper_id}_c{index}", parent_id=parent_id,
                                     parent_index=parent_index, chunk_index=index,
                                     text=child_text, word_count=len(child_words)))
    return children, parents


def chunk_metadata(chunk):
    keys = ("paper_id", "title", "published", "published_int", "pdf_url", "total_pages",
            "chunk_index", "word_count", "parent_id", "parent_index", "page_start", "page_end",
            "extraction_version", "extraction_warning")
    meta = {key: chunk[key] for key in keys if chunk.get(key) is not None}
    authors = chunk.get("authors", [])
    meta["authors"] = ", ".join(authors) if isinstance(authors, list) else str(authors)
    return meta


def paper_record_path(paper_id, data_dir="data"):
    return Path(data_dir) / "paper_records" / (hashlib.sha256(paper_id.encode()).hexdigest() + ".json")


def staged_papers(data_dir="data"):
    for path in sorted((Path(data_dir) / "paper_records").glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def load_chunk_records(chunks_path="data/paper_chunks.json"):
    path = Path(chunks_path)
    staged = list(staged_papers(path.parent))
    replaced = {record["paper"]["paper_id"] for record in staged}
    chunks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    chunks = [chunk for chunk in chunks if chunk.get("paper_id") not in replaced]
    for record in staged:
        chunks.extend(record["children"])
    return chunks


def persist_paper_records(paper, children, parents, data_dir="data", pages=None):
    """Atomically stage one paper; never rewrite the multi-GB legacy chunk file."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    paper = dict(paper, paper_id=paper.get("paper_id") or paper["paperId"])
    if pages:
        paper["extraction_summary"] = dict(
            pages=len(pages), unreadable_pages=[p["page_number"] for p in pages if not p["text"].strip()],
            visually_transcribed_pages=[p["page_number"] for p in pages if p.get("visual_text")],
        )
    with FileLock(str(root / "corpus.lock"), timeout=60):
        atomic_write_json(paper_record_path(paper["paper_id"], root),
                          dict(paper=paper, children=children, parents=parents, pages=pages or []))
        path = root / "papers_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        metadata = [value for value in metadata if (value.get("paper_id") or value.get("paperId")) != paper["paper_id"]]
        metadata.append(paper)
        atomic_write_json(path, metadata)


def merge_paper_metadata(papers, data_dir="data"):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "corpus.lock"), timeout=60):
        path = root / "papers_metadata.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        merged = {p.get("paper_id") or p.get("paperId"): p for p in existing}
        for paper in papers:
            key = paper.get("paper_id") or paper.get("paperId")
            merged[key] = dict(merged.get(key, {}), **paper)
        atomic_write_json(path, list(merged.values()))


def index_paper(collection, paper, children, parents, data_dir="data", pages=None):
    from scripts.populate_db import embed_texts, EMBEDDING_PROVIDER
    from scripts.build_fts_index import sync_paper_fts
    persist_paper_records(paper, children, parents, data_dir, pages=pages)
    for start in range(0, len(children), 1000 if EMBEDDING_PROVIDER == "local" else 10):
        batch = children[start:start + (1000 if EMBEDDING_PROVIDER == "local" else 10)]
        values = dict(ids=[c["chunk_id"] for c in batch], documents=[c["text"] for c in batch],
                      metadatas=[chunk_metadata(c) for c in batch])
        if EMBEDDING_PROVIDER != "local":
            values["embeddings"] = embed_texts(values["documents"])
        collection.upsert(**values)
    existing = collection.get(where={"paper_id": paper["paper_id"]}, include=[])["ids"]
    obsolete = list(set(existing) - {c["chunk_id"] for c in children})
    if obsolete:
        collection.delete(ids=obsolete)
    sync_paper_fts(children, parents, str(Path(data_dir) / "bm25_fts.db"))
