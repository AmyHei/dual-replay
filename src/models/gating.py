"""Task-conditioned gating and domain embeddings.
g_k = sigmoid(W_g @ e_k + b_g)
"""
import torch
import torch.nn as nn


class DomainEmbeddings(nn.Module):
    def __init__(self, num_domains: int, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(num_domains, embed_dim)

    def forward(self, domain_id: int) -> torch.Tensor:
        idx = torch.tensor(domain_id, dtype=torch.long, device=self.embeddings.weight.device)
        return self.embeddings(idx)

    @property
    def num_domains(self) -> int:
        return self.embeddings.num_embeddings


class TaskConditionedGating(nn.Module):
    def __init__(self, adapter_r: int, embed_dim: int = 64):
        super().__init__()
        self.gate = nn.Linear(embed_dim, adapter_r)

    def forward(self, domain_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(domain_embedding))


def soft_mixture_routing(domain_embeddings: DomainEmbeddings, probs: torch.Tensor) -> torch.Tensor:
    all_embeds = domain_embeddings.embeddings.weight
    return (probs.unsqueeze(-1) * all_embeds).sum(dim=0)
