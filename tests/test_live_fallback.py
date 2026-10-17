import unittest
from unittest.mock import Mock, patch

from rag_system import query_rag


class EmptyLocalSearchTests(unittest.TestCase):
    def setUp(self):
        cache = Mock()
        cache.lookup.return_value = (False, None, [], 0.0, None)
        self.enterContext(patch.object(query_rag, "get_semantic_cache", return_value=cache))
        self.enterContext(patch.object(query_rag, "EMBEDDING_PROVIDER", "local"))
        self.enterContext(patch.object(query_rag, "RAG_SPARSE_ONLY", False))
        self.enterContext(patch.object(query_rag.os.path, "exists", return_value=True))
        self.enterContext(patch.object(query_rag, "search_fts", return_value=[]))
        self.enterContext(patch.object(
            query_rag, "get_reranker", side_effect=AssertionError("No local candidates to rerank")
        ))
        self.collection = Mock()
        self.collection.query.return_value = {"documents": [[]]}
        self.fallback = self.enterContext(patch.object(query_rag, "fallback_to_live_mcp"))

    def test_empty_local_search_returns_live_answer(self):
        live_sources = [{"title": "Live research paper", "is_live_mcp": True}]
        self.fallback.return_value = ("Answer from live research", live_sources)
        progress = Mock()

        result = query_rag.query_rag(
            self.collection, "Find papers about quantum computing", progress=progress
        )

        self.assertEqual(result, ("Answer from live research", live_sources))
        self.collection.query.assert_called_once()
        self.fallback.assert_called_once()
        self.assertEqual(self.fallback.call_args.args[:4], (
            "Find papers about quantum computing",
            "Find papers about quantum computing",
            3,
            None,
        ))
        self.assertEqual(self.fallback.call_args.kwargs, {
            "reason": "No Matching Local Documents", "progress": progress,
        })

    def test_empty_local_search_respects_disabled_fallback(self):
        answer, sources = query_rag.query_rag(
            self.collection, "Find papers about quantum computing", enable_mcp_fallback=False
        )

        self.fallback.assert_not_called()
        self.assertEqual(answer, "No relevant papers found matching your query or filters.")
        self.assertEqual(sources, [])

    def test_empty_local_search_respects_research_gate(self):
        answer, sources = query_rag.query_rag(self.collection, "What is your favorite food?")

        self.fallback.assert_not_called()
        self.assertEqual(answer, "No relevant papers found matching your query or filters.")
        self.assertEqual(sources, [])

    def test_empty_live_search_returns_no_results(self):
        self.fallback.return_value = (None, None)

        answer, sources = query_rag.query_rag(
            self.collection, "Find papers about quantum computing"
        )

        self.fallback.assert_called_once()
        self.assertEqual(answer, "No relevant papers found matching your query or filters.")
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
