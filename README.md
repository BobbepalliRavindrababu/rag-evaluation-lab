# RAG Evaluation Lab

A hands-on, dependency-free lab for measuring RAG quality across three stages:

- **Retrieval:** Precision@k, Recall@k, MRR, and NDCG
- **Answer quality:** exact match and token-level F1
- **Grounding:** a lexical hallucination heuristic that identifies answer claims weakly supported by retrieved context

The default metrics are intentionally small and transparent. Replace or supplement
them with embedding, NLI, or LLM-based judges as your dataset and production
requirements grow.

## Quick start

Requires Python 3.9+.

```bash
python -m pip install -e .
```

Create JSONL input with one example per line:

```json
{"retrieved_ids":["doc-2","doc-1"],"relevant_ids":["doc-1","doc-3"],"prediction":"Paris is the capital of France.","reference":"Paris is the capital of France.","contexts":["Paris is the capital and largest city of France."]}
```

Run the evaluator:

```bash
rag-evaluate --k 2 < examples.jsonl
```

Example output:

```json
{
  "answer": {"exact_match": 1.0, "f1": 1.0},
  "count": 1,
  "hallucination": {
    "mean_hallucination_score": 0.0,
    "mean_unsupported_claims": 0.0
  },
  "retrieval": {
    "mrr": 0.5,
    "ndcg": 0.38685280723454163,
    "precision": 0.5,
    "recall": 0.5
  }
}
```

## Python API

```python
from rag_evaluation_lab import (
    calculate_retrieval_metrics,
    calculate_answer_metrics,
    detect_hallucinations,
    evaluate_examples,
)

retrieval = calculate_retrieval_metrics(
    ["doc-2", "doc-1"], ["doc-1", "doc-3"], k=2
)
answers = calculate_answer_metrics("Paris is France's capital", "Paris is the capital of France")
grounding = detect_hallucinations(
    "Paris is the capital of France.",
    ["Paris is the capital and largest city of France."],
)
```

## Evaluation-driven improvement loop

1. Build a labeled query set with relevant document IDs and reference answers.
2. Measure retrieval and answer baselines before changing the pipeline.
3. Change one component at a time (chunking, embeddings, reranking, prompting).
4. Compare aggregate metrics and inspect examples with low grounding or answer scores.
5. Keep passing and failing slices to prevent regressions and guide the next experiment.

The grounding score is a heuristic: it counts a claim unsupported when fewer than
50% of its normalized words occur in any retrieved context. It is useful for
triage, not a substitute for factual verification.