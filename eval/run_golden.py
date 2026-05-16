"""Evaluation pipeline: run the golden dataset and compute retrieval metrics.

Usage:
    python -m eval.run_golden --golden eval/golden.jsonl --k 5
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from eval.metrics import bootstrap_ci, mrr, recall_at_k
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import hybrid_search


def load_golden(path: Path) -> list[dict]:
    """JSONL: each line is {query: str, relevant_doc_ids: list[int]}."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold-recall", type=float, default=0.0)
    args = parser.parse_args()

    golden = load_golden(args.golden)
    logger.info("loaded {n} golden examples", n=len(golden))

    try:
        bm25 = BM25Index.load()
    except FileNotFoundError:
        bm25 = None
        logger.warning("BM25 index not found — eval will be dense-only")

    recalls = []
    mrrs = []

    for item in golden:
        query = item["query"]
        gt_ids = [str(x) for x in item["relevant_doc_ids"]]

        chunks = hybrid_search(query, bm25=bm25)
        predicted = [str(c["doc_id"]) for c in chunks]

        r = recall_at_k(predicted, gt_ids, k=args.k)
        m = mrr(predicted, gt_ids)
        recalls.append(r)
        mrrs.append(m)

    mean_recall = sum(recalls) / len(recalls)
    mean_mrr = sum(mrrs) / len(mrrs)

    if len(recalls) >= 2:
        lo, hi = bootstrap_ci(recalls, n_samples=1000, ci=0.95, seed=42)
    else:
        lo, hi = mean_recall, mean_recall

    print(f"recall@{args.k}: {mean_recall:.3f}  CI95%: [{lo:.3f}, {hi:.3f}]")
    print(f"MRR        : {mean_mrr:.3f}")

    if mean_recall < args.threshold_recall:
        print(f"FAIL: recall {mean_recall:.3f} below threshold {args.threshold_recall}")
        sys.exit(1)


if __name__ == "__main__":
    main()
