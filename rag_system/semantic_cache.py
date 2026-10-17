"""Bounded, versioned semantic answer cache with transactional file updates."""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from filelock import FileLock

from rag_system.relevance import SCORING_VERSION
from rag_system.storage import atomic_write_json, corpus_version
from rag_system.utils import to_jsonable

load_dotenv()
CACHE_FILE_PATH = os.path.join("data", "semantic_cache.json")
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")


class SemanticCache:
    def __init__(self, cache_file=CACHE_FILE_PATH, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                 ttl_seconds=None, max_entries=None, version_fn=None, clock=None):
        self.cache_file = str(cache_file)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = float(ttl_seconds if ttl_seconds is not None else os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"))
        self.max_entries = int(max_entries if max_entries is not None else os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500"))
        if self.ttl_seconds <= 0 or self.max_entries < 1:
            raise ValueError("Cache TTL and maximum entries must be positive.")
        self.version_fn = version_fn or corpus_version
        self.clock = clock or time.time
        self.entries = []
        self._embedding_function = None
        self._thread_lock = threading.RLock()
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = FileLock(self.cache_file + ".lock", timeout=5)
        with self._thread_lock, self._file_lock:
            self._load_cache()

    def _get_embedding_function(self):
        """Lazy-loads the embedding function for generating query vectors."""
        if self._embedding_function is not None:
            return self._embedding_function
            
        if EMBEDDING_PROVIDER == "local":
            try:
                from chromadb.utils import embedding_functions
                self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    self._embedding_function = lambda texts: model.encode(texts).tolist()
                except Exception as e:
                    print(f"⚠️ Could not load local embedding for cache: {e}")
        else:
            try:
                import google.generativeai as genai
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
                    def gemini_embed(texts):
                        res = genai.embed_content(
                            model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
                            content=texts[0] if isinstance(texts, list) else texts,
                            task_type="retrieval_query"
                        )
                        return [res['embedding']] if isinstance(texts, list) else res['embedding']
                    self._embedding_function = gemini_embed
            except Exception as e:
                print(f"⚠️ Could not load Gemini embedding for cache: {e}")
                
        return self._embedding_function

    def get_embedding(self, text):
        """Generates an embedding vector for a given text query."""
        ef = self._get_embedding_function()
        if ef is None:
            return None
        try:
            if callable(ef):
                vec = ef([text] if isinstance(text, str) else text)
                if isinstance(vec, list) and len(vec) > 0:
                    return np.array(vec[0], dtype=np.float32)
                return np.array(vec, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Error computing query embedding for cache: {e}")
        return None

    def _load_cache(self):
        # Caller holds both locks. Reload before every transaction to merge writers.
        self.entries = []
        try:
            raw = json.loads(Path(self.cache_file).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        version, now = self.version_fn(), self.clock()
        for item in raw if isinstance(raw, list) else []:
            try:
                if (item.get("scoring_version") != SCORING_VERSION
                        or item.get("corpus_version") != version
                        or not 0 <= now - item["created_at_epoch"] < self.ttl_seconds):
                    continue
                entry = dict(item, embedding=np.asarray(item["embedding"], dtype=np.float32))
                if entry["embedding"].ndim != 1 or not np.isfinite(entry["embedding"]).all():
                    continue
                self.entries.append(entry)
            except (KeyError, TypeError, ValueError):
                continue
        self.entries.sort(key=lambda item: item.get("last_access", item["created_at_epoch"]))
        self.entries = self.entries[-self.max_entries:]

    def _save_cache(self):
        atomic_write_json(self.cache_file, to_jsonable(self.entries))

    def lookup(self, query_text, query_embedding=None, filters=None):
        miss = (False, None, [], 0.0, None)
        with self._thread_lock, self._file_lock:
            self._load_cache()
            if not self.entries:
                self._save_cache()
                return miss
        vector = query_embedding if query_embedding is not None else self.get_embedding(query_text)
        if vector is None:
            return miss
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            return miss
        with self._thread_lock, self._file_lock:
            self._load_cache()
            best, best_score = None, -1.0
            for entry in self.entries:
                other = entry["embedding"]
                if entry.get("filters", {}) != (filters or {}) or other.shape != vector.shape:
                    continue
                norm = np.linalg.norm(other) * np.linalg.norm(vector)
                if norm == 0:
                    continue
                score = float(np.dot(vector, other) / norm)
                if score > best_score:
                    best, best_score = entry, score
            if best is not None and best_score >= self.similarity_threshold:
                best["hit_count"] = best.get("hit_count", 0) + 1
                best["last_access"] = self.clock()
                self._save_cache()
                return True, best["answer"], best["sources"], best_score, best["query_text"]
            self._save_cache()  # Persist pruning of expired/obsolete entries.
            return False, None, [], max(0.0, best_score), None

    def store(self, query_text, answer, sources, query_embedding=None, filters=None, corpus_revision=None):
        version = corpus_revision if corpus_revision is not None else self.version_fn()
        vector = query_embedding if query_embedding is not None else self.get_embedding(query_text)
        if vector is None:
            return False
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            return False
        with self._thread_lock, self._file_lock:
            self._load_cache()
            if version != self.version_fn():
                return False
            now = self.clock()
            self.entries = [entry for entry in self.entries if not (
                entry["query_text"] == query_text and entry.get("filters", {}) == (filters or {})
            )]
            self.entries.append({
                "scoring_version": SCORING_VERSION, "corpus_version": version,
                "query_text": query_text, "embedding": vector, "answer": answer,
                "sources": sources, "filters": filters or {}, "created_at_epoch": now,
                "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "last_access": now, "hit_count": 0,
            })
            self.entries = self.entries[-self.max_entries:]
            self._save_cache()
        return True

    def clear(self):
        with self._thread_lock, self._file_lock:
            self.entries = []
            self._save_cache()

    def stats(self):
        with self._thread_lock, self._file_lock:
            self._load_cache()
            return {"total_cached_queries": len(self.entries),
                    "total_hits": sum(entry.get("hit_count", 0) for entry in self.entries),
                    "threshold": self.similarity_threshold, "ttl_seconds": self.ttl_seconds,
                    "max_entries": self.max_entries, "cache_file": self.cache_file}


_cache_instance = None
_cache_lock = threading.Lock()


def get_semantic_cache():
    global _cache_instance
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = SemanticCache()
        return _cache_instance
