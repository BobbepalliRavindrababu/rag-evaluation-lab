import argparse
import json
import sys
from typing import Iterable

from .metrics import evaluate_examples


def read_jsonl(stream: Iterable[str]) -> list[dict]:
    examples = []
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and answer quality from JSONL on stdin"
    )
    parser.add_argument("--k", type=int, default=None, help="retrieval cutoff")
    args = parser.parse_args()

    try:
        examples = read_jsonl(sys.stdin)
        result = evaluate_examples(examples, k=args.k)
    except ValueError as error:
        parser.error(str(error))

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
