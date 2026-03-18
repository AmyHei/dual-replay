"""Tests for task-conditioned gating mechanism."""
import pytest
import torch


def test_domain_embeddings_shape():
    from src.models.gating import DomainEmbeddings
    emb = DomainEmbeddings(num_domains=15, embed_dim=64)
    e = emb(domain_id=3)
    assert e.shape == (64,)


def test_gating_vector_shape_and_range():
    from src.models.gating import TaskConditionedGating
    gating = TaskConditionedGating(adapter_r=64, embed_dim=64)
    e_k = torch.randn(64)
    g = gating(e_k)
    assert g.shape == (64,)
    assert (g >= 0).all() and (g <= 1).all()


def test_gating_modulates_adapter_output():
    from src.models.gating import TaskConditionedGating
    gating = TaskConditionedGating(adapter_r=64, embed_dim=64)
    e_k = torch.randn(64)
    adapter_out = torch.randn(2, 10, 64)
    g = gating(e_k)
    modulated = adapter_out * g
    assert modulated.shape == adapter_out.shape


def test_soft_mixture_routing():
    from src.models.gating import DomainEmbeddings, soft_mixture_routing
    emb = DomainEmbeddings(num_domains=3, embed_dim=64)
    probs = torch.tensor([0.7, 0.2, 0.1])
    mixed = soft_mixture_routing(emb, probs)
    assert mixed.shape == (64,)
