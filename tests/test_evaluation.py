import unittest

from rag_evaluation_lab import (
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    detect_hallucinations,
    evaluate_examples,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = calculate_retrieval_metrics(
            ["doc-2", "doc-1", "doc-3"],
            ["doc-1", "doc-3"],
            k=2,
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertGreater(metrics["ndcg"], 0.0)

    def test_empty_retrieval(self):
        metrics = calculate_retrieval_metrics([], ["doc-1"])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)
        self.assertEqual(metrics["ndcg"], 0.0)


class AnswerMetricsTests(unittest.TestCase):
    def test_exact_match_is_normalized(self):
        metrics = calculate_answer_metrics("Paris, France!", "paris france")
        self.assertTrue(metrics["exact_match"])
        self.assertEqual(metrics["f1"], 1.0)

    def test_unrelated_answer_has_zero_f1(self):
        metrics = calculate_answer_metrics("Berlin", "Paris")
        self.assertFalse(metrics["exact_match"])
        self.assertEqual(metrics["f1"], 0.0)


class GroundingTests(unittest.TestCase):
    def test_supported_claim(self):
        result = detect_hallucinations(
            "Paris is the capital of France.",
            ["Paris is the capital and largest city of France."],
        )
        self.assertEqual(result["unsupported_claims"], [])
        self.assertEqual(result["hallucination_score"], 0.0)

    def test_unsupported_claim(self):
        result = detect_hallucinations(
            "Mars is made of cheese.",
            ["Paris is the capital of France."],
        )
        self.assertEqual(result["unsupported_claims"], ["Mars is made of cheese"])
        self.assertEqual(result["hallucination_score"], 1.0)


class AggregateTests(unittest.TestCase):
    def test_evaluate_examples(self):
        result = evaluate_examples([{
            "retrieved_ids": ["doc-1"],
            "relevant_ids": ["doc-1"],
            "prediction": "Paris",
            "reference": "Paris",
            "contexts": ["Paris"],
        }])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["retrieval"]["recall"], 1.0)
        self.assertEqual(result["answer"]["exact_match"], 1.0)

    def test_empty_evaluation(self):
        result = evaluate_examples([])
        self.assertEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
