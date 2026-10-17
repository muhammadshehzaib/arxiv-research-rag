"""One parent/child format for batch PDFs and papers fetched live."""
import json
from pathlib import Path
from filelock import FileLock

from rag_system.storage import atomic_write_json
from rag_system.utils import clean_text, date_to_int

PARENT_CHUNK_SIZE, PARENT_OVERLAP = 800, 100
CHILD_CHUNK_SIZE, CHILD_OVERLAP = 150, 20


def build_paper_chunks(text, paper, total_pages):
    words = clean_text(text).split()
    paper_id = paper.get("paper_id") or paper["paperId"]
    common = dict(paper_id=paper_id, title=paper.get("title", ""), authors=paper.get("authors", []),
                  published=paper.get("published", ""), published_int=date_to_int(paper.get("published", "")),
                  pdf_url=paper.get("pdf_url") or paper.get("pdfUrl", ""), total_pages=total_pages)
    children, parents = [], {}
    for parent_index, start in enumerate(range(0, len(words), PARENT_CHUNK_SIZE - PARENT_OVERLAP)):
        parent_words = words[start:start + PARENT_CHUNK_SIZE]
        parent_id = f"{paper_id}_p{parent_index}"
        parents[parent_id] = dict(common, parent_id=parent_id, parent_index=parent_index,
                                 text=" ".join(parent_words), word_count=len(parent_words))
        for child_start in range(0, len(parent_words), CHILD_CHUNK_SIZE - CHILD_OVERLAP):
            child_words = parent_words[child_start:child_start + CHILD_CHUNK_SIZE]
            index = len(children)
            children.append(dict(common, chunk_id=f"{paper_id}_c{index}", parent_id=parent_id,
                                 parent_index=parent_index, chunk_index=index,
                                 text=" ".join(child_words), word_count=len(child_words)))
    return children, parents


def chunk_metadata(chunk):
    keys = ("paper_id", "title", "published", "published_int", "pdf_url", "total_pages",
            "chunk_index", "word_count", "parent_id", "parent_index")
    meta = {key: chunk[key] for key in keys if chunk.get(key) is not None}
    authors = chunk.get("authors", [])
    meta["authors"] = ", ".join(authors) if isinstance(authors, list) else str(authors)
    return meta


def persist_paper_records(paper, children, parents, data_dir="data"):
    """Idempotent per-paper replacement; retries repair partial ingestion stages."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    paper_id = paper.get("paper_id") or paper["paperId"]
    def read(name, default):
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    with FileLock(str(root / "corpus.lock"), timeout=60):
        registry = {key: value for key, value in read("paper_parents.json", {}).items()
                    if value.get("paper_id") != paper_id}
        registry.update(parents)
        chunks = [value for value in read("paper_chunks.json", []) if value.get("paper_id") != paper_id]
        chunks.extend(children)
        metadata = [value for value in read("papers_metadata.json", [])
                    if (value.get("paper_id") or value.get("paperId")) != paper_id]
        metadata.append(paper)
        atomic_write_json(root / "paper_parents.json", registry)
        atomic_write_json(root / "paper_chunks.json", chunks)
        atomic_write_json(root / "papers_metadata.json", metadata)


def index_paper(collection, paper, children, parents, data_dir="data"):
    from scripts.populate_db import embed_texts, EMBEDDING_PROVIDER
    from scripts.build_fts_index import sync_paper_fts
    persist_paper_records(paper, children, parents, data_dir)
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
