import os
import json
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE_PATH = os.path.join("data", "semantic_cache.json")
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")

class SemanticCache:
    """
    On-device Semantic Cache for RAG queries.
    Matches queries by vector cosine similarity to return pre-computed answers in sub-50ms.
    """
    def __init__(self, cache_file=CACHE_FILE_PATH, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.entries = []
        self._embedding_function = None
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
        """Loads cached queries and vectors from disk."""
        if not os.path.exists(self.cache_file):
            self.entries = []
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.entries = []
            for item in raw_data:
                self.entries.append({
                    "query_text": item["query_text"],
                    "embedding": np.array(item["embedding"], dtype=np.float32),
                    "answer": item["answer"],
                    "sources": item.get("sources", []),
                    "filters": item.get("filters", {}),
                    "created_at": item.get("created_at", ""),
                    "hit_count": item.get("hit_count", 0)
                })
        except Exception as e:
            print(f"⚠️ Error loading semantic cache from {self.cache_file}: {e}")
            self.entries = []

    def _save_cache(self):
        """Persists the cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            serializable = []
            for item in self.entries:
                serializable.append({
                    "query_text": item["query_text"],
                    "embedding": item["embedding"].tolist(),
                    "answer": item["answer"],
                    "sources": item["sources"],
                    "filters": item.get("filters", {}),
                    "created_at": item.get("created_at", ""),
                    "hit_count": item.get("hit_count", 0)
                })
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving semantic cache to {self.cache_file}: {e}")

    def lookup(self, query_text, query_embedding=None, filters=None):
        """
        Looks up a query in the semantic cache.
        Returns:
            (is_hit: bool, answer: str, sources: list, score: float, matched_query: str)
        """
        if not self.entries:
            return False, None, [], 0.0, None

        if query_embedding is None:
            query_embedding = self.get_embedding(query_text)
            
        if query_embedding is None:
            return False, None, [], 0.0, None

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return False, None, [], 0.0, None

        best_score = -1.0
        best_entry = None
        current_filters = filters or {}

        # Vectorized similarity comparison across all matching filter candidates
        for entry in self.entries:
            # Check filter match (queries with different date/paper filters must not collide)
            entry_filters = entry.get("filters", {})
            if current_filters != entry_filters:
                continue

            e_vec = entry["embedding"]
            e_norm = np.linalg.norm(e_vec)
            if e_norm == 0:
                continue

            # Cosine similarity: (A . B) / (||A|| * ||B||)
            sim = float(np.dot(q_vec, e_vec) / (q_norm * e_norm))
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            best_entry["hit_count"] += 1
            self._save_cache()
            return True, best_entry["answer"], best_entry["sources"], best_score, best_entry["query_text"]

        return False, None, [], max(best_score, 0.0), None

    def store(self, query_text, answer, sources, query_embedding=None, filters=None):
        """Stores a new query-response pair into the semantic cache."""
        if query_embedding is None:
            query_embedding = self.get_embedding(query_text)

        if query_embedding is None:
            return False

        q_vec = np.array(query_embedding, dtype=np.float32)
        entry = {
            "query_text": query_text,
            "embedding": q_vec,
            "answer": answer,
            "sources": sources,
            "filters": filters or {},
            "created_at": datetime.now().isoformat(),
            "hit_count": 0
        }
        self.entries.append(entry)
        self._save_cache()
        return True

    def clear(self):
        """Clears all cached entries."""
        self.entries = []
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass

    def stats(self):
        """Returns cache statistics."""
        total_hits = sum(e.get("hit_count", 0) for e in self.entries)
        return {
            "total_cached_queries": len(self.entries),
            "total_hits": total_hits,
            "threshold": self.similarity_threshold,
            "cache_file": self.cache_file
        }

# Global singleton instance
_cache_instance = None
def get_semantic_cache():
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
