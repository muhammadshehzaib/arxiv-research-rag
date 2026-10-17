import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import evaluate_rag as ev
from rag_system.app import verified_evaluation


class EvaluationIntegrityTests(unittest.TestCase):
    def test_empty_context_does_not_get_full_credit(self):
        self.assertEqual(ev.calculate_context_recall(None, "truth", []), 0)
        self.assertEqual(ev.calculate_faithfulness(None, "answer", []), 0)

    def test_invalid_judge_outputs_and_service_errors_are_not_scores(self):
        model = Mock()
        model.generate_content.return_value.text = "MAYBE YES"
        with self.assertRaises(ev.EvaluationScoringError):
            ev.supported(model, "context", "statement")
        model.generate_content.side_effect = RuntimeError("judge offline")
        with self.assertRaises(RuntimeError):
            ev.calculate_context_recall(model, "truth", ["context"])
        with self.assertRaises(Exception):
            ev.parse_json_array('{"statements": "wrong schema"}')

    def run_case(self, response, scorer_error=False):
        self.enterContext(patch.object(ev, "GEMINI_API_KEY", "test-placeholder-not-a-secret"))
        self.enterContext(patch.object(ev, "load_questions", return_value=[{
            "question": "question", "ground_truth": "truth",
        }]))
        self.enterContext(patch.object(ev, "init_services", return_value=Mock()))
        self.enterContext(patch.object(ev, "get_bm25_index", return_value=(None, None)))
        self.enterContext(patch.object(ev, "save_results"))
        self.enterContext(patch.object(ev.genai, "GenerativeModel", return_value=Mock()))
        query = self.enterContext(patch.object(ev, "query_rag", return_value=response))
        for metric in ("calculate_faithfulness", "calculate_answer_relevance", "calculate_context_precision", "calculate_context_recall"):
            self.enterContext(patch.object(ev, metric, **(
                {"side_effect": RuntimeError("offline")} if scorer_error else {"return_value": 1.0}
            )))
        return ev.run_evaluation(), query

    def test_generation_errors_are_zero_and_mark_run_incomplete(self):
        result, query = self.run_case(("Failed to generate response: quota exceeded", []))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["details"][0]["case_score"], 0)
        self.assertIsNone(result["average_score"])
        self.assertFalse(query.call_args.kwargs["use_cache"])
        self.assertFalse(query.call_args.kwargs["enable_mcp_fallback"])

    def test_scoring_errors_are_null_and_excluded_from_aggregates(self):
        result, _ = self.run_case(("An answer", [{"text": "Evidence"}]), scorer_error=True)
        self.assertIsNone(result["faithfulness"])
        self.assertIsNone(result["average_score"])
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(len(result["details"][0]["errors"]), 4)

    def test_legacy_evaluations_cannot_be_mistaken_for_valid_baselines(self):
        result = verified_evaluation({"average_score": 0.5, "details": [{"faithfulness": 0.5}]})
        self.assertEqual(result["status"], "legacy_unverified")
        self.assertIsNone(result["average_score"])
        self.assertIsNone(result["details"][0]["faithfulness"])
