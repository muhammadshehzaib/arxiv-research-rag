import asyncio
import json
import unittest
from unittest.mock import patch

from rag_system import app


class QueryProgressTests(unittest.TestCase):
    def test_stream_reports_real_live_search_stage(self):
        def fake_query(*args, progress=None, **kwargs):
            progress("live_search")
            progress("writing_live_answer")
            return "Answer from a live paper", [{"title": "Example"}]

        async def consume():
            with patch.object(app, "collection", object()), patch.object(app, "query_rag", side_effect=fake_query):
                response = await app.stream_query(app.QueryRequest(query="papers about quantum computing"))
                chunks = [chunk async for chunk in response.body_iterator]
                return response, [json.loads(chunk) for chunk in chunks]

        response, events = asyncio.run(consume())
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertEqual([event.get("stage") for event in events[:-1]], [
            "local_search", "live_search", "writing_live_answer"
        ])
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["sources"][0]["title"], "Example")

    def test_greeting_stream_finishes_without_search(self):
        async def consume():
            with patch.object(app, "query_rag", side_effect=AssertionError("search used")):
                response = await app.stream_query(app.QueryRequest(query="hello"))
                return [json.loads(chunk) async for chunk in response.body_iterator]

        events = asyncio.run(consume())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "result")


if __name__ == "__main__":
    unittest.main()
