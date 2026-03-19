"""End-to-end integration test: tiny model, 2 domains."""
import pytest
import numpy as np


def test_sequential_runner_end_to_end():
    """Run full sequential training on 2 domains with tiny model."""
    from src.training.runner import SequentialRunner
    from src.methods.sequential_ft import SequentialFT

    domains = [
        {
            "domain_id": 0,
            "train": [{"text": f"domain0 {i}", "label": i % 2} for i in range(20)],
            "test": [{"text": f"d0 test {i}", "label": i % 2} for i in range(10)],
        },
        {
            "domain_id": 1,
            "train": [{"text": f"domain1 {i}", "label": i % 2} for i in range(20)],
            "test": [{"text": f"d1 test {i}", "label": i % 2} for i in range(10)],
        },
    ]

    method = SequentialFT(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=8,
        max_seq_len=32,
    )

    runner = SequentialRunner(method=method, domains=domains)
    results = runner.run()

    assert "perf_matrix" in results
    assert results["perf_matrix"].shape == (2, 2)
    assert "bwt" in results
    assert "avg_f1" in results
