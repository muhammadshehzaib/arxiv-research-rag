import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from rag_system.app import app


class EvaluationEndpointTests(unittest.TestCase):
    def test_button_uses_packaged_evaluator(self):
        with patch("scripts.evaluate_rag.run_evaluation", return_value={"average_score": 0.8}) as run:
            response = TestClient(app).post("/api/eval/run")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["average_score"], 0.8)
        run.assert_called_once_with()

    def test_failed_initialization_is_an_error(self):
        with patch("scripts.evaluate_rag.run_evaluation", return_value=None):
            response = TestClient(app).post("/api/eval/run")
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
