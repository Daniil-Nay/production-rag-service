"""Evaluation metrics: recall@k, MRR, and bootstrap confidence intervals."""

import random


def recall_at_k(predicted: list[str], ground_truth: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    top_k = set(predicted[:k])
    return len(top_k & gt_set) / len(gt_set)


def mrr(predicted: list[str], ground_truth: list[str]) -> float:
    if not predicted or not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    for rank, doc_id in enumerate(predicted, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def bootstrap_ci(
    scores: list[float],
    n_samples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    if len(scores) < 2:
        raise ValueError("need at least 2 scores")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = random.Random(seed) if seed is not None else random.Random()

    means = []
    n = len(scores)
    for _ in range(n_samples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = 1 - ci
    lower_idx = int((alpha / 2) * n_samples)
    upper_idx = max(0, min(n_samples - 1, int((1 - alpha / 2) * n_samples) - 1))
    return means[lower_idx], means[upper_idx]
