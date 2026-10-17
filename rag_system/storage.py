"""Small durable file primitives shared by ingestion and caches."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def corpus_version():
    """Fingerprint corpus files and answer-affecting configuration without reading secrets."""
    root = Path(os.getenv("RAG_DATA_DIR", "data"))
    chroma = Path(os.getenv("CHROMA_PATH", str(root / "chroma_db"))) / "chroma.sqlite3"
    paths = [root / name for name in (
        "paper_chunks.json", "paper_parents.json", "papers_metadata.json", "bm25_fts.db", "bm25_fts.db-wal",
    )] + [chroma, Path(str(chroma) + "-wal")]
    signature = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append((str(path.resolve()), None))
    signature += [(key, os.getenv(key)) for key in (
        "QDRANT_COLLECTION_NAME", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "LLM_MODEL",
        "FLASHRANK_MODEL", "REFUSAL_THRESHOLD", "REFUSAL_THRESHOLD_COHERE",
        "REFUSAL_THRESHOLD_FLASHRANK", "REFUSAL_THRESHOLD_SENTENCE_TRANSFORMERS",
    )]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()
