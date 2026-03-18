"""Tests for CLINC150 data loading and 15-domain protocol."""
import pytest


def test_load_clinc150_raw():
    """Should load raw CLINC150 and return train/val/test splits."""
    from src.data.clinc150 import load_clinc150_raw

    splits = load_clinc150_raw()
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    train_intents = set(splits["train"]["intent"])
    assert len(train_intents) >= 150


def test_build_15_domain_protocol():
    """Should split 150 intents into 15 domains of 10 intents each."""
    from src.data.clinc150 import build_15_domain_protocol

    domains = build_15_domain_protocol()
    assert len(domains) == 15
    all_intents = set()
    for domain in domains:
        assert len(domain["intents"]) == 10
        assert len(domain["train"]) > 0
        assert len(domain["test"]) > 0
        all_intents.update(domain["intents"])
    assert len(all_intents) == 150


def test_general_buffer_from_oos():
    """General replay buffer should come from OOS examples."""
    from src.data.clinc150 import get_general_buffer

    buffer = get_general_buffer(max_size=100)
    assert len(buffer) == 100
    assert all(ex["intent"] == "oos" for ex in buffer)


def test_domain_ordering_reproducibility():
    """Same seed should produce same domain ordering."""
    from src.data.domain_sequence import generate_domain_orderings

    ord1 = generate_domain_orderings(num_domains=15, num_orderings=2, seed=42)
    ord2 = generate_domain_orderings(num_domains=15, num_orderings=2, seed=42)
    assert ord1 == ord2
    assert ord1[0] != ord1[1]
