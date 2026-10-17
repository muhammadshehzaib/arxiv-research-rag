import io
import json
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from rag_system import query_rag
from rag_system.mcp_client import AcademicMCPClient
from rag_system.relevance import (
    SCORING_VERSION, get_relevance_threshold, normalize_relevance_score,
)
from rag_system.semantic_cache import SemanticCache
from scripts import calibrate_threshold


def make_reranker(provider, score):
    """Use the real adapters without downloading models or contacting a provider."""
    reranker = query_rag.Reranker.__new__(query_rag.Reranker)
    reranker.provider = provider
    reranker.is_active = True
    reranker.cohere_client = None
    reranker.sentence_transformers_client = None
    reranker.flashrank_client = None
    if provider == "cohere":
        reranker.cohere_client = Mock()
        reranker.cohere_client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=score)]
        )
    elif provider == "sentence-transformers":
        def predict(pairs, activation_fct):
            return [activation_fct(score)]
        reranker.sentence_transformers_client = SimpleNamespace(predict=predict)
    elif provider == "flashrank":
        reranker.flashrank_client = Mock()
        reranker.flashrank_client.rerank.return_value = [{
            "id": "paper_c0", "text": "Quantum computing research",
            "meta": {"metadata": {"paper_id": "paper", "title": "Paper"}},
            "score": score,
        }]
    return reranker


class RelevanceScoreTests(unittest.TestCase):
    def test_bounded_provider_scores_are_not_sigmoided(self):
        for provider in ("cohere", "flashrank"):
            for score in (0.0, 0.2, 0.9, 1.0):
                with self.subTest(provider=provider, score=score):
                    self.assertEqual(normalize_relevance_score(score, provider), score)

    def test_only_cross_encoder_logits_receive_sigmoid(self):
        self.assertAlmostEqual(normalize_relevance_score(0.2, "sentence-transformers"),
                               1.0 / (1.0 + math.exp(-0.2)))
        self.assertEqual(normalize_relevance_score(-1000, "sentence-transformers"), 0.0)
        self.assertEqual(normalize_relevance_score(1000, "sentence-transformers"), 1.0)

    def test_invalid_scores_fail_closed(self):
        for provider in ("cohere", "flashrank", "sentence-transformers", "unknown"):
            for score in (None, "invalid", float("nan"), float("inf"), -float("inf")):
                with self.subTest(provider=provider, score=score):
                    self.assertEqual(normalize_relevance_score(score, provider), 0.0)
        for provider in ("cohere", "flashrank"):
            self.assertEqual(normalize_relevance_score(-0.1, provider), 0.0)
            self.assertEqual(normalize_relevance_score(1.1, provider), 0.0)
        self.assertEqual(normalize_relevance_score(0.9, "unknown"), 0.0)

    def test_provider_threshold_overrides_global(self):
        with patch.dict(os.environ, {
            "REFUSAL_THRESHOLD": "0.6", "REFUSAL_THRESHOLD_COHERE": "0.3",
        }, clear=True):
            self.assertEqual(get_relevance_threshold("cohere"), 0.3)
            self.assertEqual(get_relevance_threshold("flashrank"), 0.6)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_relevance_threshold("flashrank"), 0.7)

    def test_invalid_thresholds_are_rejected(self):
        for value in ("0", "-1", "1.1", "nan", "inf", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "REFUSAL_THRESHOLD_FLASHRANK": value,
            }, clear=True):
                with self.assertRaises(ValueError):
                    get_relevance_threshold("flashrank")

    def test_adapters_keep_raw_scores_and_record_relevance(self):
        for provider in ("cohere", "flashrank", "sentence-transformers"):
            with self.subTest(provider=provider):
                reranker = make_reranker(provider, 0.2)
                result = reranker.rerank("research", [{
                    "id": "paper_c0", "text": "Research", "metadata": {},
                }])[0]
                self.assertEqual(result["score"], 0.2)
                self.assertEqual(result["reranker_provider"], provider)
                self.assertEqual(result["relevance_score"], normalize_relevance_score(0.2, provider))

    def test_cross_encoder_supports_new_activation_parameter(self):
        reranker = make_reranker("sentence-transformers", 0.2)
        def predict(pairs, activation_fn):
            self.assertEqual(activation_fn(0.2), 0.2)
            return [0.2]
        reranker.sentence_transformers_client.predict = predict
        result = reranker.rerank("query", [{"id": "a", "text": "text"}])
        self.assertEqual(result[0]["score"], 0.2)
        self.assertAlmostEqual(result[0]["relevance_score"], 0.549833997)

    def test_provider_failure_has_no_fabricated_score(self):
        reranker = make_reranker("cohere", 0.9)
        reranker.cohere_client.rerank.side_effect = RuntimeError("Unavailable")
        result = reranker.rerank("query", [{"id": "a", "text": "text"}])[0]
        self.assertIsNone(result["score"])
        self.assertIsNone(result["reranker_provider"])
        self.assertEqual(result["relevance_score"], 0.0)

    def test_live_sources_are_unscored_in_json_and_cli(self):
        _, sources = AcademicMCPClient().format_live_context([{
            "paper_id": "paper", "title": "Live paper", "summary": "Research abstract",
        }])
        source = sources[0]
        for field in ("relevance_score", "rerank_score", "rrf_score", "reranker_provider"):
            self.assertIsNone(source[field])
        self.assertNotIn("confidence", source)
        self.assertNotIn("distance", source)
        self.assertIsNone(json.loads(json.dumps(source))["relevance_score"])
        with redirect_stdout(io.StringIO()) as output:
            query_rag.print_result("question", "answer", sources)
        self.assertIn("Retrieval relevance: unscored", output.getvalue())


class RefusalScoringTests(unittest.TestCase):
    def test_actual_provider_scores_control_generation(self):
        cases = (
            ("cohere", 0.2, 0.3, False),
            ("cohere", 0.8, 0.8, True),
            ("flashrank", 0.9, 0.8, True),
            ("flashrank", 0.1, 0.3, False),
            ("sentence-transformers", 0.2, 0.6, False),
            ("sentence-transformers", 2.0, 0.8, True),
            ("flashrank", float("nan"), 0.3, False),
        )
        for provider, score, threshold, should_generate in cases:
            with self.subTest(provider=provider, score=score, threshold=threshold):
                self._check_pipeline(provider, score, threshold, should_generate)

    def _check_pipeline(self, provider, score, threshold, should_generate):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "REFUSAL_THRESHOLD": str(threshold),
            }, clear=True))
            cache = Mock()
            cache.lookup.return_value = (False, None, [], 0.0, None)
            model = Mock()
            model.generate_content.return_value = SimpleNamespace(text="Grounded answer")
            collection = Mock()
            collection.query.return_value = {
                "documents": [["Quantum computing research"]],
                "metadatas": [[{"paper_id": "paper", "title": "Paper"}]],
                "distances": [[0.2]], "ids": [["paper_c0"]],
            }
            for name, value in (
                ("get_semantic_cache", Mock(return_value=cache)),
                ("get_reranker", Mock(return_value=make_reranker(provider, score))),
                ("get_parent_registry", Mock(return_value={})),
                ("get_knowledge_graph", Mock(return_value=SimpleNamespace(is_loaded=False))),
                ("get_corpus_size", Mock(return_value=10)),
                ("search_fts", Mock(return_value=[])),
                ("EMBEDDING_PROVIDER", "local"), ("RAG_SPARSE_ONLY", False),
            ):
                stack.enter_context(patch.object(query_rag, name, value))
            stack.enter_context(patch.object(query_rag.os.path, "exists", return_value=True))
            stack.enter_context(patch.object(query_rag.genai, "GenerativeModel", return_value=model))
            answer, sources = query_rag.query_rag(
                collection, "Explain quantum computing research", enable_mcp_fallback=False,
            )
            self.assertEqual(model.generate_content.called, should_generate)
            self.assertEqual(cache.store.called, should_generate)
            self.assertEqual(sources[0]["relevance_score"], normalize_relevance_score(score, provider))
            self.assertNotIn("confidence", sources[0])
            self.assertNotIn("distance", sources[0])
            # Invalid provider outputs must not produce NaN/Infinity in API JSON.
            json.dumps(sources, allow_nan=False)
            if should_generate:
                self.assertEqual(answer, "Grounded answer")
                self.assertNotIn("Confidence:", model.generate_content.call_args.args[0])
            else:
                self.assertIn("Retrieval relevance score:", answer)
                self.assertNotIn("%", answer)


class ScoringCacheAndCalibrationTests(unittest.TestCase):
    def test_legacy_scoring_cache_is_ignored_and_new_scores_survive_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.json"
            path.write_text(json.dumps([{
                "query_text": "question", "embedding": [1.0, 0.0], "answer": "Old answer",
                "sources": [{"confidence": 0.95}],
            }]), encoding="utf-8")
            cache = SemanticCache(str(path))
            self.assertEqual(cache.entries, [])
            sources = [{"relevance_score": 0.9, "reranker_provider": "flashrank"}]
            cache.store("question", "New answer", sources, query_embedding=[1.0, 0.0])
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored[0]["scoring_version"], SCORING_VERSION)
            reloaded = SemanticCache(str(path))
            hit, answer, returned_sources, _, _ = reloaded.lookup("question", query_embedding=[1.0, 0.0])
            self.assertTrue(hit)
            self.assertEqual(answer, "New answer")
            self.assertEqual(returned_sources, sources)

    def test_calibration_uses_runtime_normalization(self):
        with patch.object(calibrate_threshold, "IN_SCOPE_QUESTIONS", ["In scope"]), patch.object(
            calibrate_threshold, "OUT_SCOPE_QUESTIONS", ["Out of scope"]
        ), patch.object(calibrate_threshold, "get_top_reranker_score", side_effect=[0.2, None]):
            in_scores, out_scores = calibrate_threshold.collect_scores(
                None, None, None, SimpleNamespace(provider="flashrank")
            )
        self.assertEqual(in_scores[0]["relevance_score"], 0.2)
        self.assertEqual(out_scores[0]["relevance_score"], 0.0)
        results = calibrate_threshold.sweep_thresholds(in_scores, out_scores)
        self.assertEqual(results[0]["threshold"], 0.01)
        self.assertEqual(results[-1]["threshold"], 1.0)
        self.assertEqual(results[0]["accuracy"], 1.0)

    def test_calibration_updates_only_active_provider_threshold(self):
        contents = "REFUSAL_THRESHOLD=0.70\nREFUSAL_THRESHOLD_COHERE=0.40\n"
        with patch.object(calibrate_threshold.os.path, "exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=contents)
        ) as opened:
            calibrate_threshold.update_env_threshold(0.25, "flashrank")
        written = "".join(opened().writelines.call_args.args[0])
        self.assertIn("REFUSAL_THRESHOLD=0.70\n", written)
        self.assertIn("REFUSAL_THRESHOLD_COHERE=0.40\n", written)
        self.assertIn("REFUSAL_THRESHOLD_FLASHRANK=0.25\n", written)


if __name__ == "__main__":
    unittest.main()
