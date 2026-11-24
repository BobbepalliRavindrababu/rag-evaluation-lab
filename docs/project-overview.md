# RAG Evaluation Lab — Engineering & Interview Notes

## 1. Project Objective

Build a practical RAG evaluation system that measures the quality of retrieval and generated answers, identifies failure points, and improves the pipeline through controlled experiments.

The objective is not just to build a RAG chatbot. It is to understand whether the system retrieves the right information, generates grounded answers, and performs reliably across different types of questions.

## 2. Knowledge Base

We created a fictional company knowledge base for Nexora Technologies.

Source documents:
- `employee_handbook.md`
- `leave_and_attendance_policy.md`
- `expense_reimbursement_policy.md`

These documents contain company policies such as working hours, leave entitlements, probation, and expense reimbursement.

## 3. Golden Evaluation Dataset

Created `data/evaluation/golden_dataset.json` with 15 questions.

The dataset contains:
- Simple factual questions
- Comparison questions
- Multi-document questions
- Unanswerable questions

Each question includes:
- Unique question ID
- Question
- Expected answer
- Answerability information
- Relevant source documents
- Ground-truth evidence statements

### Why a golden dataset?

A golden dataset gives us a consistent set of questions and expected evidence against which we can evaluate retrieval performance.

Without a fixed evaluation dataset, we cannot reliably determine whether a retrieval improvement actually works.

## 4. Initial RAG Retrieval Pipeline

Current implementation:

1. Load Markdown documents.
2. Split documents into chunks.
3. Generate embeddings.
4. Store embeddings and metadata in ChromaDB.
5. Convert user questions into query embeddings.
6. Retrieve the top-K most similar chunks.

### Technology choices

| Component | Technology |
|---|---|
| Programming language | Python |
| Embedding model | Gemini Embedding 001 |
| Embedding dimensions | 768 |
| Document chunking | RecursiveCharacterTextSplitter |
| Vector database | ChromaDB |
| Initial chunk size | 500 |
| Initial chunk overlap | 50 |
| Initial Top-K | 3 |

Document embeddings use the `RETRIEVAL_DOCUMENT` task type. Query embeddings use `RETRIEVAL_QUERY`.

Embeddings are L2-normalized before storage and retrieval.

## 5. Initial Retrieval Evaluation

Implemented `src/retrieval_evaluator.py`.

The initial evaluator checks whether retrieved chunks contain the expected evidence statements.

### Metrics implemented

**Hit Rate**

Measures whether at least one retrieved chunk contains expected evidence.

**Chunk Precision**

Number of retrieved chunks containing at least one expected evidence statement divided by the number of retrieved chunks.

**Evidence Coverage Recall**

Number of expected evidence statements found in retrieved chunks divided by the total expected evidence statements.

**Mean Reciprocal Rank (MRR)**

Measures how highly the first relevant chunk is ranked.

For a first relevant result at rank 2:

MRR = 1 / 2 = 0.5

Note: Our current precision and recall use different units of relevance: chunks and evidence statements. They are useful initial diagnostic metrics, but should not be presented as a conventional, fully aligned Precision@K and Recall@K pair. We will refine the evaluation methodology as the project develops.

### Initial baseline results

Evaluation covered 14 answerable questions. One unanswerable question was excluded from this initial retrieval evaluation.

| Metric | Initial result |
|---|---:|
| Hit Rate@3 | 100% |
| Chunk Precision@3 | 35.71% |
| Evidence Coverage Recall@3 | 95.71% |
| MRR@3 | 91.67% |

These are provisional baseline results from our first custom evaluator, not final production-quality evaluation metrics.

## 6. Failure Analysis — Q013

Question:

"What is the difference between annual leave and sick leave?"

The question is classified as a comparison question.

Its five expected evidence statements cover:
1. Purpose of annual leave.
2. Annual leave entitlement: 20 days.
3. Sick leave entitlement: 10 days.
4. Annual leave carry-forward: maximum 5 days.
5. Sick leave cannot be carried forward.

Initial Top-K = 3 retrieval:

| Rank | Retrieved information | Assessment |
|---|---|---|
| 1 | Sick leave cannot be converted into annual leave | Low relevance |
| 2 | Sick leave eligibility during probation and public holidays | Low relevance |
| 3 | Sick leave entitlement, purpose and carry-forward | Relevant |

Evidence coverage: 2/5 = 40%.

### Investigation

We tried a more focused annual-leave query:

"What is the annual leave entitlement, purpose, and carry-forward policy?"

This successfully retrieved annual-leave information.

We then increased Top-K to 6 for the original comparison question.

| Rank | Retrieved information |
|---|---|
| 1 | Sick leave conversion restriction |
| 2 | Sick leave probation eligibility and public holidays |
| 3 | Sick leave entitlement and carry-forward |
| 4 | Annual leave carry-forward and March 31 deadline |
| 5 | Annual leave entitlement and purpose |
| 6 | Leave year and annual leave section |

### Finding

The annual-leave information exists in the vector database and can be retrieved.

However, the original comparison question ranks sick-leave chunks above annual-leave chunks.

Increasing Top-K exposes more relevant information, but also retrieves additional context.

This reveals an important RAG engineering trade-off:

- Low Top-K can miss relevant evidence.
- High Top-K can introduce irrelevant context, increase token consumption, and potentially affect answer quality.

The objective is to improve retrieval quality without unnecessarily increasing the context sent to the LLM.

## 7. Retrieval Improvement Roadmap

Planned experiments:

### Experiment 1 — Top-K comparison
Test different Top-K values while keeping other variables unchanged.

### Experiment 2 — Hybrid Search
Combine semantic vector search with keyword-based BM25 retrieval.

### Experiment 3 — Reranking
Retrieve a larger candidate set and reorder candidates according to their relevance to the original question.

### Experiment 4 — Query Expansion
Break complex or comparison questions into focused subqueries and combine the results.

### Experiment 5 — Contextual Compression
Remove irrelevant information from retrieved chunks before passing context to the LLM.

Each experiment will be compared against the initial baseline.

## 8. Interview Preparation — Questions to Understand

1. What is a golden evaluation dataset, and why is it important?
2. What is the difference between retrieval evaluation and generation evaluation?
3. What do Hit Rate, Precision, Recall, and MRR measure?
4. Why can a RAG system retrieve relevant information but still produce poor answers?
5. What is the Top-K parameter?
6. What is the trade-off between retrieval quality and context size?
7. What is hybrid search?
8. How does reranking improve retrieval?
9. What is query expansion?
10. What is contextual compression?

## 9. Engineering Principle

Every retrieval improvement must be measured against the same evaluation dataset.

We should change one major variable at a time, record the results, analyze failures, and avoid assuming that a technique is better simply because it sounds more advanced.

The final objective is a reliable, measurable, and explainable RAG system.
