"""Bottleneck adapter module following Houlsby et al. (2019).

Adapter(h) = h + GeLU(h @ W_down) @ W_up
W_up is zero-initialized so adapter starts as identity.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertConfig


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # Gate vector set externally before forward pass (by DualReplayModel)
        self._gate: torch.Tensor | None = None

    def set_gate(self, gate: torch.Tensor | None):
        """Set a (bottleneck_dim,) gate vector for task-conditioned modulation."""
        self._gate = gate

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mid = self.activation(self.down(h))  # (..., bottleneck_dim)
        if self._gate is not None:
            mid = mid * self._gate  # element-wise gating
        return h + self.up(mid)


def _load_model(model_name: str):
    """Load model, falling back to BertModel for checkpoints missing model_type."""
    try:
        return AutoModel.from_pretrained(model_name)
    except ValueError:
        # Some old checkpoints (e.g. prajjwal1/bert-tiny) omit `model_type`
        # in config.json; load them explicitly as BertModel.
        config = BertConfig.from_pretrained(model_name)
        return BertModel.from_pretrained(model_name, config=config)


def create_adapted_model(model_name: str, adapter_r: int = 64):
    base_model = _load_model(model_name)
    d_model = base_model.config.hidden_size
    for param in base_model.parameters():
        param.requires_grad = False

    adapters = nn.ModuleList()
    layers = get_transformer_layers(base_model)

    for i, layer in enumerate(layers):
        adapter = BottleneckAdapter(d_model, adapter_r)
        adapters.append(adapter)
        _hook_adapter_after_ffn(layer, adapter)

    base_model.adapters = adapters
    return base_model


def get_transformer_layers(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError(f"Cannot find transformer layers in {type(model)}")


def _hook_adapter_after_ffn(layer, adapter):
    original_output = layer.output

    class AdaptedOutput(nn.Module):
        def __init__(self, original, adapter):
            super().__init__()
            self.original = original
            self.adapter = adapter

        def forward(self, hidden_states, input_tensor):
            output = self.original(hidden_states, input_tensor)
            return self.adapter(output)

    layer.output = AdaptedOutput(original_output, adapter)
