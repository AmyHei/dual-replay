"""Tests for dual-stream replay buffer management."""
import pytest


def test_domain_buffer_add_and_sample():
    from src.replay.buffer import DomainReplayBuffer
    buf = DomainReplayBuffer(max_per_domain=50)
    examples = [{"text": f"example {i}", "intent": i % 5, "domain": 0} for i in range(100)]
    buf.add_domain(domain_id=0, examples=examples)
    sampled = buf.sample(domain_id=0, n=10)
    assert len(sampled) == 10
    assert all(s["domain"] == 0 for s in sampled)


def test_domain_buffer_respects_max_size():
    from src.replay.buffer import DomainReplayBuffer
    buf = DomainReplayBuffer(max_per_domain=50)
    examples = [{"text": f"ex {i}", "intent": 0, "domain": 0} for i in range(200)]
    buf.add_domain(domain_id=0, examples=examples)
    assert buf.size(domain_id=0) <= 50


def test_domain_buffer_sample_all_domains():
    from src.replay.buffer import DomainReplayBuffer
    buf = DomainReplayBuffer(max_per_domain=100)
    for d in range(5):
        examples = [{"text": f"d{d}_ex{i}", "intent": i, "domain": d} for i in range(100)]
        buf.add_domain(domain_id=d, examples=examples)
    sampled = buf.sample_all(total_n=50)
    assert len(sampled) == 50
    domains_seen = set(s["domain"] for s in sampled)
    assert len(domains_seen) >= 3


def test_general_buffer():
    from src.replay.buffer import GeneralReplayBuffer
    buf = GeneralReplayBuffer(max_size=100)
    examples = [{"text": f"general {i}", "intent": "oos"} for i in range(200)]
    buf.fill(examples)
    assert buf.size() <= 100
    sampled = buf.sample(n=20)
    assert len(sampled) == 20


def test_dual_replay_buffer_batch_composition():
    from src.replay.buffer import DualReplayBuffer
    dual = DualReplayBuffer(max_per_domain=100, general_max_size=100)
    for d in range(3):
        examples = [{"text": f"d{d}_{i}", "intent": i, "domain": d} for i in range(100)]
        dual.add_domain(d, examples)
    general = [{"text": f"gen_{i}", "intent": "oos"} for i in range(100)]
    dual.fill_general(general)
    domain_samples, general_samples = dual.sample_replay(domain_n=10, general_n=10)
    assert len(domain_samples) == 10
    assert len(general_samples) == 10
