
import re
import statistics

from retriever import retrieve_documents


TOP_K = 3


def normalize_text(text):
    """Normalize whitespace and Markdown formatting for evidence matching."""
    text = text.lower()
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_question(question_data, top_k=TOP_K):
    question = question_data["question"]
    evidence = question_data["evidence"]

    retrieved_chunks = retrieve_documents(
        question=question,
        top_k=top_k
    )

    normalized_chunks = [
        normalize_text(chunk["content"])
        for chunk in retrieved_chunks
    ]

    normalized_evidence = [
        normalize_text(statement)
        for statement in evidence
    ]

    # Identify which evidence statements were retrieved.
    matched_evidence = [
        statement
        for statement in normalized_evidence
        if any(statement in chunk for chunk in normalized_chunks)
    ]

    # A chunk is relevant if it contains at least one evidence statement.
    relevant_ranks = []

    for rank, chunk in enumerate(normalized_chunks, start=1):
        if any(statement in chunk for statement in normalized_evidence):
            relevant_ranks.append(rank)

    hit_rate = 1.0 if relevant_ranks else 0.0

    precision = (
        len(relevant_ranks) / len(retrieved_chunks)
        if retrieved_chunks else 0.0
    )

    recall = (
        len(matched_evidence) / len(normalized_evidence)
        if normalized_evidence else 0.0
    )

    reciprocal_rank = (
        1.0 / relevant_ranks[0]
        if relevant_ranks else 0.0
    )

    return {
        "question_id": question_data["id"],
        "question": question,
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "matched_evidence": len(matched_evidence),
        "total_evidence": len(normalized_evidence),
        "relevant_ranks": relevant_ranks
    }


def evaluate_dataset(top_k=TOP_K):
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (
        project_root / "data" / "evaluation" / "golden_dataset.json"
    )

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    answerable_questions = [
        question
        for question in dataset["questions"]
        if question["is_answerable"]
    ]

    unanswerable_count = sum(
        not question["is_answerable"]
        for question in dataset["questions"]
    )

    results = []

    print(f"\nEvaluating {len(answerable_questions)} answerable questions")
    print(f"Top K: {top_k}")

    for question_data in answerable_questions:
        result = evaluate_question(question_data, top_k)
        results.append(result)

        print("\n" + "=" * 65)
        print("Question ID:", result["question_id"])
        print("Question:", result["question"])
        print("Hit Rate:", result["hit_rate"])
        print("Precision:", round(result["precision"], 4))
        print("Recall:", round(result["recall"], 4))
        print("MRR:", round(result["mrr"], 4))
        print(
            "Evidence Coverage:",
            f'{result["matched_evidence"]}/{result["total_evidence"]}'
        )
        print("Relevant chunk ranks:", result["relevant_ranks"])

    if not results:
        raise ValueError("No answerable questions found in dataset")

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    print("Evaluated questions:", len(results))
    print("Unanswerable questions excluded:", unanswerable_count)
    print(
        f"Hit Rate @ {top_k}:",
        round(statistics.mean(r["hit_rate"] for r in results), 4)
    )
    print(
        f"Precision @ {top_k}:",
        round(statistics.mean(r["precision"] for r in results), 4)
    )
    print(
        f"Recall @ {top_k}:",
        round(statistics.mean(r["recall"] for r in results), 4)
    )
    print(
        f"MRR @ {top_k}:",
        round(statistics.mean(r["mrr"] for r in results), 4)
    )


if __name__ == "__main__":
    evaluate_dataset()
