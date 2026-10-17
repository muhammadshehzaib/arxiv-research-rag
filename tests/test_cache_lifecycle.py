import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rag_system.semantic_cache import SemanticCache
from rag_system.storage import atomic_write_json


class CacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = str(Path(self.folder) / "cache.json")
        self.now = 1000.0
        self.version = "corpus-a"

    def cache(self, **kwargs):
        return SemanticCache(self.path, version_fn=lambda: self.version,
                             clock=lambda: self.now, **kwargs)

    def test_ttl_and_corpus_changes_invalidate_answers(self):
        cache = self.cache(ttl_seconds=10)
        cache.store("q", "a", [], query_embedding=[1, 0])
        self.assertTrue(cache.lookup("q", query_embedding=[1, 0])[0])
        self.now += 10
        self.assertFalse(cache.lookup("q", query_embedding=[1, 0])[0])
        cache.store("q", "a", [], query_embedding=[1, 0])
        self.version = "corpus-b"
        self.assertFalse(cache.lookup("q", query_embedding=[1, 0])[0])
        self.assertFalse(cache.store("q", "stale", [], query_embedding=[1, 0], corpus_revision="corpus-a"))

    def test_lru_cap_and_duplicate_replacement(self):
        cache = self.cache(max_entries=2)
        cache.store("a", "a", [], query_embedding=[1, 0])
        self.now += 1
        cache.store("b", "b", [], query_embedding=[0, 1])
        self.now += 1
        cache.lookup("a", query_embedding=[1, 0])
        self.now += 1
        cache.store("c", "c", [], query_embedding=[-1, 0])
        self.assertEqual({e["query_text"] for e in cache.entries}, {"a", "c"})
        cache.store("c", "updated", [], query_embedding=[-1, 0])
        self.assertEqual(cache.stats()["total_cached_queries"], 2)

    def test_concurrent_instances_merge_instead_of_losing_writes(self):
        caches = [self.cache() for _ in range(8)]
        def write(i):
            caches[i % 8].store(str(i), "answer", [], query_embedding=[1, 0])
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(32)))
        self.assertEqual(self.cache().stats()["total_cached_queries"], 32)
        self.assertEqual(len(json.loads(Path(self.path).read_text())), 32)

    def test_atomic_failure_preserves_previous_file(self):
        atomic_write_json(self.path, {"old": True})
        with patch("rag_system.storage.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                atomic_write_json(self.path, {"new": True})
        self.assertEqual(json.loads(Path(self.path).read_text()), {"old": True})
        self.assertEqual(list(Path(self.folder).glob("*.tmp")), [])

    def test_dimension_change_and_invalid_vectors_cannot_hit(self):
        cache = self.cache()
        cache.store("q", "answer", [], query_embedding=[1, 0])
        self.assertFalse(cache.lookup("q", query_embedding=[1, 0, 0])[0])
        self.assertFalse(cache.store("q", "bad", [], query_embedding=[float("nan"), 0]))
