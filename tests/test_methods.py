"""Tests for continual learning method interface."""
import pytest


def test_base_method_interface():
    from src.methods.base import BaseContinualMethod
    with pytest.raises(TypeError):
        BaseContinualMethod(model_name="bert-base-uncased", num_domains=3)


def test_sequential_ft_trains_and_scores():
    from src.methods.sequential_ft import SequentialFT

    method = SequentialFT(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
    )
    method.setup()

    train_data = [
        {"text": "book a flight to boston", "label": 0},
        {"text": "cancel my reservation", "label": 1},
        {"text": "change my seat", "label": 0},
        {"text": "I need a refund", "label": 1},
    ] * 5

    test_data = [
        {"text": "reserve a flight", "label": 0},
        {"text": "cancel booking", "label": 1},
    ] * 3

    metrics = method.train_domain(domain_id=0, train_data=train_data)
    assert "loss" in metrics

    eval_metrics = method.run_evaluation(test_data)
    assert "f1" in eval_metrics
    assert 0.0 <= eval_metrics["f1"] <= 100.0


def test_dual_replay_trains_and_scores():
    from src.methods.dual_replay import DualReplay

    method = DualReplay(
        model_name="prajjwal1/bert-tiny",
        num_domains=3,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
        adapter_r=8,
        embed_dim=16,
        replay_ratio=0.20,
        domain_replay_fraction=0.10,
        domain_buffer_size=20,
        general_buffer_size=50,
    )
    method.setup()

    general_data = [{"text": f"general example {i}", "label": -1} for i in range(50)]
    method.fill_general_buffer(general_data)

    train0 = [{"text": f"domain0 example {i}", "label": i % 3} for i in range(40)]
    test0 = [{"text": f"domain0 test {i}", "label": i % 3} for i in range(10)]
    method.train_domain(domain_id=0, train_data=train0)

    train1 = [{"text": f"domain1 example {i}", "label": i % 3} for i in range(40)]
    method.train_domain(domain_id=1, train_data=train1)

    eval0 = method.run_evaluation(test0)
    assert "f1" in eval0

    trainable = method.get_trainable_param_count()
    total = sum(p.numel() for p in method.model.parameters())
    assert trainable < total * 0.3


def test_dual_replay_base_frozen():
    from src.methods.dual_replay import DualReplay
    import torch

    method = DualReplay(
        model_name="prajjwal1/bert-tiny",
        num_domains=2,
        learning_rate=5e-4,
        epochs=1,
        batch_size=4,
        max_seq_len=32,
        adapter_r=8,
        embed_dim=16,
        replay_ratio=0.0,
        domain_replay_fraction=0.0,
        domain_buffer_size=10,
        general_buffer_size=10,
    )
    method.setup()

    base_params_before = {
        name: p.clone()
        for name, p in method.model.named_parameters()
        if not p.requires_grad
    }

    train_data = [{"text": f"example {i}", "label": i % 2} for i in range(20)]
    method.train_domain(domain_id=0, train_data=train_data)

    for name, p in method.model.named_parameters():
        if not p.requires_grad and name in base_params_before:
            torch.testing.assert_close(p, base_params_before[name])
