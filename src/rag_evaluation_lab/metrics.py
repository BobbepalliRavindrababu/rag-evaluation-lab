"""Standard overlap-based metrics for a small RAG evaluation pipeline.

The implementation deliberately uses only the Python standard library.  In
production, replace these lexical answer and grounding heuristics with
task-specific LLM judges or embedding models.
"""

import math
import re
from collections import Counter
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def calculate_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: Optional[int] = None,
) -> dict[str, float]:
    """Calculate Precision@k, Recall@k, MRR, and NDCG@k."""
    relevant = set(relevant_ids)
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    hits = sum(doc_id in relevant for doc_id in retrieved_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    recall = hits / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, doc_id in enumerate(retrieved_ids, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


retrieval_metrics = calculate_retrieval_metrics


def calculate_answer_metrics(
    prediction: str,
    reference: str,
) -> Dict[str, Union[float, bool]]:
    """Calculate exact match and token-level F1 for a textual answer."""
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    exact_match = normalized_prediction == normalized_reference

    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())

    if not prediction_tokens and not reference_tokens:
        f1 = 1.0
    elif not overlap:
        f1 = 0.0
    else:
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact_match, "f1": f1}


answer_metrics = calculate_answer_metrics


def detect_hallucinations(answer: str, contexts: Sequence[str]) -> dict[str, object]:
    """Heuristically identify answer sentences unsupported by retrieved context."""
    context_words = set(normalize_text(" ".join(contexts)).split())
    claims = [
        sentence.strip()
        for sentence in re.split(r"[.!?\n]+", answer)
        if sentence.strip()
    ]
    unsupported = []

    for claim in claims:
        claim_words = set(normalize_text(claim).split())
        if not claim_words:
            continue
        overlap = len(claim_words & context_words) / len(claim_words)
        if overlap < 0.5:
            unsupported.append(claim)

    hallucination_score = len(unsupported) / len(claims) if claims else 0.0
    return {
        "unsupported_claims": unsupported,
        "hallucination_score": hallucination_score,
        "supported_ratio": 1.0 - hallucination_score,
    }


def evaluate_examples(
    examples: Sequence[Mapping[str, object]],
    k: Optional[int] = None,
) -> dict[str, object]:
    """Aggregate retrieval, answer, and grounding metrics over examples."""
    retrieval_totals = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    answer_totals = {"exact_match": 0.0, "f1": 0.0}
    hallucination_scores = []
    unsupported_claim_counts = []

    for example in examples:
        retrieval = calculate_retrieval_metrics(
            list(example["retrieved_ids"]),
            list(example["relevant_ids"]),
            k=k,
        )
        for name, value in retrieval.items():
            retrieval_totals[name] += value

        answer = calculate_answer_metrics(
            str(example["prediction"]),
            str(example["reference"]),
        )
        answer_totals["exact_match"] += float(answer["exact_match"])
        answer_totals["f1"] += float(answer["f1"])

        grounding = detect_hallucinations(
            str(example["prediction"]),
            list(example.get("contexts", [])),
        )
        hallucination_scores.append(float(grounding["hallucination_score"]))
        unsupported_claim_counts.append(len(grounding["unsupported_claims"]))

    count = len(examples)
    if not count:
        return {
            "count": 0,
            "retrieval": retrieval_totals,
            "answer": answer_totals,
            "hallucination": {
                "mean_hallucination_score": 0.0,
                "mean_unsupported_claims": 0.0,
            },
        }

    return {
        "count": count,
        "retrieval": {name: value / count for name, value in retrieval_totals.items()},
        "answer": {
            "exact_match": answer_totals["exact_match"] / count,
            "f1": answer_totals["f1"] / count,
        },
        "hallucination": {
            "mean_hallucination_score": sum(hallucination_scores) / count,
            "mean_unsupported_claims": sum(unsupported_claim_counts) / count,
        },
    }
