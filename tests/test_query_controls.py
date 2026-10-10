import asyncio
import time
import unittest
from unittest.mock import patch

from rag_system import app, query_rag
from rag_system.mcp_client import AcademicMCPClient
from rag_system.query_policy import allows_live_research, quick_reply


class QueryControlsTests(unittest.TestCase):
    def test_greeting_skips_rag_work(self):
        with patch.object(query_rag, "get_semantic_cache", side_effect=AssertionError("cache used")):
            answer, sources = query_rag.query_rag(None, "are you working?")
        self.assertIn("/api/health", answer)
        self.assertEqual(sources, [])

    def test_live_gate(self):
        self.assertIsNotNone(quick_reply("Hello!"))
        self.assertFalse(allows_live_research("are you working?"))
        self.assertFalse(allows_live_research("are you an AI?"))
        self.assertFalse(allows_live_research("what is your favorite food?"))
        self.assertTrue(allows_live_research("Explain Vision Transformers"))
        self.assertTrue(allows_live_research("Can you find papers about RAG?"))
        self.assertTrue(allows_live_research("Find papers about quantum computing"))

    def test_filter_extraction_only_for_filter_requests(self):
        self.assertFalse(query_rag.needs_metadata_extraction("what is rag?"))
        self.assertTrue(query_rag.needs_metadata_extraction("RAG papers after 2025"))
        self.assertTrue(query_rag.needs_metadata_extraction("papers with at least 10 pages"))
        self.assertTrue(query_rag.needs_metadata_extraction("paper 2501.18365v1"))

    def test_api_greeting_without_database(self):
        with patch.object(app, "collection", None), patch.object(app, "init_services", side_effect=AssertionError("DB used")):
            response = asyncio.run(app.post_query(app.QueryRequest(query="are you working?")))
        self.assertEqual(response["sources"], [])
        self.assertIn("/api/health", response["answer"])

    def test_mcp_timeout_never_starts_unbounded_fallback(self):
        client = AcademicMCPClient()

        async def slow_tool(*args):
            await asyncio.sleep(0.1)

        with patch.object(client, "_call_tool_stdio", side_effect=slow_tool), patch.object(
            client, "_call_tool_direct", side_effect=AssertionError("direct call used")
        ):
            with self.assertRaises(TimeoutError):
                client.call_tool("search_arxiv", {"query": "test"}, timeout=0.01)

    def test_api_returns_clear_timeout(self):
        def slow_query(*args, **kwargs):
            time.sleep(0.1)
            return "late", []

        async def request():
            with patch.object(app, "collection", object()), patch.object(app, "QUERY_TIMEOUT_SECONDS", 0.02), patch.object(
                app, "query_rag", side_effect=slow_query
            ):
                with self.assertRaises(app.HTTPException) as caught:
                    await app.post_query(app.QueryRequest(query="papers about quantum computing"))
                self.assertEqual(caught.exception.status_code, 504)
                self.assertIn("too long", caught.exception.detail)
                await asyncio.sleep(0.11)

        asyncio.run(request())


if __name__ == "__main__":
    unittest.main()
