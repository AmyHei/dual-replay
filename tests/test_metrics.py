"""Tests for continual learning metrics."""
import pytest
import numpy as np


def test_bwt_no_forgetting():
    from src.metrics import compute_bwt
    perf_matrix = np.array([
        [80.0, 0.0, 0.0],
        [80.0, 75.0, 0.0],
        [80.0, 75.0, 70.0],
    ])
    bwt = compute_bwt(perf_matrix)
    assert bwt == 0.0


def test_bwt_with_forgetting():
    from src.metrics import compute_bwt
    perf_matrix = np.array([
        [80.0, 0.0, 0.0],
        [70.0, 75.0, 0.0],
        [65.0, 70.0, 72.0],
    ])
    bwt = compute_bwt(perf_matrix)
    assert bwt == pytest.approx(-10.0)


def test_fwt():
    from src.metrics import compute_fwt
    perf_matrix = np.array([
        [80.0, 30.0, 25.0],
        [70.0, 75.0, 35.0],
        [65.0, 70.0, 72.0],
    ])
    random_baseline = np.array([20.0, 20.0, 20.0])
    fwt = compute_fwt(perf_matrix, random_baseline)
    assert fwt == pytest.approx(12.5)


def test_avg_f1():
    from src.metrics import compute_avg_f1
    final_scores = np.array([65.0, 70.0, 72.0])
    avg = compute_avg_f1(final_scores)
    assert avg == pytest.approx(69.0)


def test_paired_ttest():
    from src.metrics import paired_ttest_bonferroni
    scores_a = [89.1, 88.5, 89.8, 88.9, 89.2]
    scores_b = [84.7, 84.2, 85.1, 84.5, 84.8]
    p_value = paired_ttest_bonferroni(scores_a, scores_b, num_comparisons=8)
    assert p_value < 0.05
