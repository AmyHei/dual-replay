"""Domain classifier for inference-time routing.
C(x) = softmax(W_c @ MEAN_POOL(h_enc(x)) + b_c)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_domains)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = encoder_hidden_states.mean(dim=1)
        return self.classifier(pooled)

    def predict_probs(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.forward(encoder_hidden_states)
        return F.softmax(logits, dim=-1)
