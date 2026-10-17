import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag_system import query_rag


class ConversationCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = Mock()
        self.cache.lookup.return_value = (True, "Answer about a different paper", [], 1.0, "its limitations")
        self.enterContext(patch.object(query_rag, "get_semantic_cache", return_value=self.cache))
        self.enterContext(patch.object(query_rag, "EMBEDDING_PROVIDER", "local"))
        self.enterContext(patch.object(query_rag, "RAG_SPARSE_ONLY", False))
        self.enterContext(patch.object(query_rag.os.path, "exists", return_value=True))
        self.enterContext(patch.object(query_rag, "search_fts", return_value=[]))
        self.collection = Mock()
        self.collection.query.return_value = {"documents": [[]]}

    def test_follow_up_cannot_hit_shared_cache_even_if_rewrite_fails(self):
        for rewritten in ("What are its limitations?", "Limitations of Paper B"):
            with self.subTest(rewritten=rewritten), patch.object(
                query_rag, "rephrase_query_with_history", return_value=rewritten
            ) as rewrite:
                answer, _ = query_rag.query_rag(
                    self.collection, "What are its limitations?",
                    chat_history=[{"role": "user", "content": "Explain Paper B"}],
                    enable_mcp_fallback=False,
                )
                rewrite.assert_called_once()
                self.assertNotIn("different paper", answer)
                self.cache.lookup.assert_not_called()

    def test_standalone_query_still_uses_cache(self):
        answer, _ = query_rag.query_rag(self.collection, "Explain quantum computing")
        self.assertEqual(answer, "Answer about a different paper")
        self.cache.lookup.assert_called_once()
        self.collection.query.assert_not_called()

    def test_generated_conversation_answer_is_not_shared(self):
        self.collection.query.return_value = {
            "documents": [["Paper B limitations"]], "metadatas": [[{"title": "B"}]],
            "distances": [[0.1]], "ids": [["b"]],
        }
        reranker = SimpleNamespace(is_active=True, rerank=lambda q, p: [dict(
            p[0], score=0.99, relevance_score=0.99, reranker_provider="flashrank"
        )])
        with patch.object(query_rag, "rephrase_query_with_history", return_value="Paper B limitations"), patch.object(
            query_rag, "get_reranker", return_value=reranker
        ), patch.object(query_rag, "get_parent_registry", return_value={}), patch.object(
            query_rag, "get_knowledge_graph", return_value=SimpleNamespace(is_loaded=False)
        ), patch.object(query_rag.genai, "GenerativeModel") as model:
            model.return_value.generate_content.return_value.text = "Paper B answer"
            answer, _ = query_rag.query_rag(
                self.collection, "What are its limitations?",
                chat_history=[{"role": "user", "content": "Explain Paper B"}],
                enable_mcp_fallback=False,
            )
        self.assertEqual(answer, "Paper B answer")
        self.cache.lookup.assert_not_called()
        self.cache.store.assert_not_called()
