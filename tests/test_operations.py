import json
import os
import time
import unittest
from collections import Counter, deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from rag_system import app, generation, observability as obs
from rag_system.query_policy import set_deadline


class OperationsTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("_counts", Counter()), ("_models", {}), ("_failures", deque()), ("_latencies", deque())):
            self.enterContext(patch.object(obs, name, value))
        self.enterContext(patch.dict(os.environ, {"LLM_FALLBACK_MODELS": "", "RAG_MODEL_PRICING_JSON": "{}"}))
        set_deadline(time.monotonic() + 40)

    def test_request_id_reaches_worker_thread_and_response(self):
        def work(*args, **kwargs):
            obs.event("test_worker")
            return obs.request_id.get(), []
        with patch.object(app, "collection", object()), patch.object(app, "query_rag", side_effect=work):
            response = TestClient(app.app).post("/api/query", json={"query": "research"},
                                                headers={"X-Request-ID": "test-trace-123"})
        self.assertEqual(response.headers["X-Request-ID"], "test-trace-123")
        self.assertEqual(response.json()["answer"], "test-trace-123")

    def test_transient_failure_uses_configured_fallback_with_usage(self):
        primary, fallback = Mock(), Mock()
        primary.generate_content.side_effect = RuntimeError("503 unavailable")
        fallback.generate_content.return_value = SimpleNamespace(text="answer", usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150))
        with patch.dict(os.environ, {"LLM_FALLBACK_MODELS": "fallback", "RAG_MODEL_PRICING_JSON": json.dumps({
            "fallback": {"input_per_million": 2, "output_per_million": 4}})}), patch.object(
            generation.genai, "GenerativeModel", side_effect=lambda model_name: primary if model_name == "primary" else fallback
        ), patch.object(generation.time, "sleep"):
            result = generation.generate_content("prompt", model_name="primary")
        self.assertEqual(result.text, "answer")
        metrics = obs.metrics_snapshot()
        self.assertEqual(metrics["generation_models"]["primary"]["failures"], 2)
        self.assertEqual(metrics["generation_models"]["fallback"]["total_tokens"], 150)
        self.assertAlmostEqual(metrics["generation_models"]["fallback"]["estimated_cost_usd"], 0.0004)

    def test_permanent_error_is_not_retried_and_unpriced_cost_is_unknown(self):
        model = Mock()
        model.generate_content.side_effect = ValueError("invalid request")
        with patch.object(generation.genai, "GenerativeModel", return_value=model):
            with self.assertRaises(ValueError):
                generation.generate_content("prompt", model_name="primary")
        model.generate_content.assert_called_once()
        self.assertIsNone(obs.metrics_snapshot()["generation_models"]["primary"]["estimated_cost_usd"])

    def test_alerts_and_deadline_are_enforced(self):
        for _ in range(5):
            obs.record_failure("test", "Unavailable")
        self.assertEqual(obs.metrics_snapshot()["active_alerts"][0]["name"], "elevated_errors")
        set_deadline(time.monotonic() - 1)
        with patch.object(generation.genai, "GenerativeModel") as model:
            with self.assertRaises(TimeoutError):
                generation.generate_content("prompt")
        model.assert_not_called()
