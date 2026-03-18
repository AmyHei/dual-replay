"""Tests for bottleneck adapter insertion and parameter freezing."""
import pytest
import torch


def test_bottleneck_adapter_shape():
    from src.models.adapters import BottleneckAdapter
    adapter = BottleneckAdapter(d_model=768, bottleneck_dim=64)
    x = torch.randn(2, 10, 768)
    out = adapter(x)
    assert out.shape == x.shape


def test_bottleneck_adapter_residual():
    from src.models.adapters import BottleneckAdapter
    adapter = BottleneckAdapter(d_model=768, bottleneck_dim=64)
    x = torch.randn(2, 10, 768)
    out = adapter(x)
    torch.testing.assert_close(out, x)


def test_adapted_model_freezes_base():
    from src.models.adapters import create_adapted_model
    model = create_adapted_model("prajjwal1/bert-tiny", adapter_r=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < total * 0.2
    assert trainable > 0


def test_adapted_model_forward():
    from src.models.adapters import create_adapted_model
    from transformers import BertTokenizer
    model = create_adapted_model("prajjwal1/bert-tiny", adapter_r=16)
    # bert-tiny uses a standard BERT WordPiece vocab; use BertTokenizer directly
    # to avoid protobuf dependency triggered by AutoTokenizer's fast-tokenizer path.
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.last_hidden_state.shape[0] == 1
