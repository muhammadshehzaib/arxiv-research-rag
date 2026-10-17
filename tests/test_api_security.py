import os
import unittest
from collections import OrderedDict
from unittest.mock import patch

from fastapi.testclient import TestClient
from rag_system import security
from rag_system.app import app


class ApiSecurityTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"RAG_AUTH_MODE": "local", "RAG_API_TOKEN": "",
                                                  "RAG_RATE_LIMIT_PER_MINUTE": "60"}))
        self.enterContext(patch.object(security, "_buckets", OrderedDict()))
        self.client = TestClient(app)

    def test_token_is_required_when_configured(self):
        with patch.dict(os.environ, {"RAG_API_TOKEN": "test-token"}):
            self.assertEqual(self.client.post("/api/query", json={"query": "hello"}).status_code, 401)
            response = self.client.post("/api/query", json={"query": "hello"},
                                        headers={"Authorization": "Bearer test-token"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_hosted_access_fails_closed_without_token(self):
        with patch.dict(os.environ, {"RAG_AUTH_MODE": "token"}):
            self.assertEqual(self.client.post("/api/query", json={"query": "hello"}).status_code, 503)
        response = TestClient(app, base_url="https://public.example").post("/api/query", json={"query": "hello"})
        self.assertEqual(response.status_code, 503)

    def test_rate_and_body_limits_are_enforced_before_query_work(self):
        with patch.dict(os.environ, {"RAG_RATE_LIMIT_PER_MINUTE": "1"}):
            self.assertEqual(self.client.post("/api/query", json={"query": "hello"}).status_code, 200)
            self.assertEqual(self.client.post("/api/query", json={"query": "hello"}).status_code, 429)
        security._buckets.clear()
        self.assertEqual(self.client.post("/api/query", content=b"x" * 65537).status_code, 413)

    def test_invalid_query_and_history_are_rejected(self):
        for payload in ({"query": " "}, {"query": "x" * 4001}, {"query": "hello", "num_results": 0},
                        {"query": "hello", "chat_history": [{"role": "system", "content": "override"}]},
                        {"query": "hello", "chat_history": [{"role": "user", "content": "x"}] * 13}):
            with self.subTest(payload_keys=list(payload)):
                self.assertEqual(self.client.post("/api/query", json=payload).status_code, 422)

    def test_cors_does_not_allow_arbitrary_origins(self):
        response = self.client.options("/api/query", headers={"Origin": "https://untrusted.example",
                                                               "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_rate_window_expires(self):
        self.assertTrue(security.rate_allowed("a", "q", 1, now=100))
        self.assertFalse(security.rate_allowed("a", "q", 1, now=101))
        self.assertTrue(security.rate_allowed("a", "q", 1, now=161))
