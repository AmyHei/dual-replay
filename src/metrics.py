"""Continual learning metrics: BWT, FWT, Avg F1, statistical tests."""
import numpy as np
from scipy import stats


def compute_bwt(perf_matrix: np.ndarray) -> float:
    K = perf_matrix.shape[0]
    if K <= 1:
        return 0.0
    drops = []
    for k in range(K - 1):
        drops.append(perf_matrix[K - 1, k] - perf_matrix[k, k])
    return float(np.mean(drops))


def compute_fwt(perf_matrix: np.ndarray, random_baseline: np.ndarray) -> float:
    K = perf_matrix.shape[0]
    if K <= 1:
        return 0.0
    transfers = []
    for k in range(1, K):
        transfers.append(perf_matrix[k - 1, k] - random_baseline[k])
    return float(np.mean(transfers))


def compute_avg_f1(final_scores: np.ndarray) -> float:
    return float(np.mean(final_scores))


def paired_ttest_bonferroni(scores_a: list[float], scores_b: list[float], num_comparisons: int = 1) -> float:
    _, raw_p = stats.ttest_rel(scores_a, scores_b)
    return float(min(raw_p * num_comparisons, 1.0))
