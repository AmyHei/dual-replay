"""Tests for domain classifier head."""
import pytest
import torch


def test_classifier_output_shape():
    from src.models.domain_classifier import DomainClassifier
    clf = DomainClassifier(hidden_dim=128, num_domains=15)
    h = torch.randn(4, 10, 128)
    logits = clf(h)
    assert logits.shape == (4, 15)


def test_classifier_probabilities_sum_to_one():
    from src.models.domain_classifier import DomainClassifier
    clf = DomainClassifier(hidden_dim=128, num_domains=15)
    h = torch.randn(4, 10, 128)
    probs = clf.predict_probs(h)
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4))
