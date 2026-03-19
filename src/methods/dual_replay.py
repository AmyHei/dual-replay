"""Dual-Replay continual learning method.

Integrates:
- Frozen base encoder with bottleneck adapters (Houlsby et al. 2019)
- Task-conditioned gating via domain embeddings
- Domain classifier with soft routing at inference time
- Dual-stream experience replay (domain + general)

Training batch composition:
    (1 - replay_ratio)          new examples
    domain_replay_fraction       domain-stream replay
    replay_ratio - domain_frac   general-stream replay
"""
from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import _load_tokenizer
from src.models.adapters import (
    BottleneckAdapter,
    _load_model,
    get_transformer_layers,
    _hook_adapter_after_ffn,
)
from src.models.gating import DomainEmbeddings, TaskConditionedGating, soft_mixture_routing
from src.models.domain_classifier import DomainClassifier
from src.replay.buffer import DualReplayBuffer



# ---------------------------------------------------------------------------
# Dataset (dual-replay specific: returns 'label' and 'domain' keys)
# ---------------------------------------------------------------------------

class _DualReplayDataset(torch.utils.data.Dataset):
    """Tokenize a list of dicts with 'text', 'label', optional 'domain' keys."""

    def __init__(self, examples: list[dict], tokenizer, max_seq_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        label = ex.get("label", -1)
        item["label"] = torch.tensor(label, dtype=torch.long)

        domain = ex.get("domain", -1)
        item["domain"] = torch.tensor(domain, dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualReplayModel(nn.Module):
    """Full model: frozen encoder + adapters + gating + domain classifier + task head."""

    def __init__(
        self,
        model_name: str,
        num_domains: int,
        num_labels: int,
        adapter_r: int = 64,
        embed_dim: int = 64,
    ):
        super().__init__()
        # Load and freeze base encoder
        self.encoder = _load_model(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        d_model = self.encoder.config.hidden_size

        # Insert bottleneck adapters after each FFN layer
        self.adapters = nn.ModuleList()
        layers = get_transformer_layers(self.encoder)
        for layer in layers:
            adapter = BottleneckAdapter(d_model, adapter_r)
            self.adapters.append(adapter)
            _hook_adapter_after_ffn(layer, adapter)

        # Task-conditioned gating: one gate per adapter layer
        self.domain_embeddings = DomainEmbeddings(num_domains, embed_dim)
        self.gating_layers = nn.ModuleList([
            TaskConditionedGating(adapter_r, embed_dim)
            for _ in layers
        ])

        # Domain classifier trained alongside adapters
        self.domain_classifier = DomainClassifier(d_model, num_domains)

        # Task classification head
        self.classifier = nn.Linear(d_model, num_labels)

        self.num_domains = num_domains
        self.num_labels = num_labels
        self.adapter_r = adapter_r

    def _get_domain_embedding(
        self,
        hidden_states: torch.Tensor,
        domain_id: int | None,
        domain_probs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a domain embedding (embed_dim,) for gating."""
        if domain_id is not None:
            return self.domain_embeddings(domain_id)
        if domain_probs is not None:
            return soft_mixture_routing(self.domain_embeddings, domain_probs)
        # Inference: use domain classifier output for soft routing
        domain_logits = self.domain_classifier(hidden_states)
        # Average over batch dimension before softmax
        mean_logits = domain_logits.mean(0)
        probs = F.softmax(mean_logits, dim=0)
        return soft_mixture_routing(self.domain_embeddings, probs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_id: int | None = None,
        domain_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (task_logits, domain_logits).

        domain_id: known at training time -> direct embedding lookup.
        domain_id=None at inference -> soft routing via domain classifier.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, d)

        # Domain embedding for gating (compute gates, used as diagnostic/auxiliary)
        domain_emb = self._get_domain_embedding(hidden_states, domain_id, domain_probs)
        # Gates per layer (currently used for auxiliary learning signal)
        gates = [gate_layer(domain_emb) for gate_layer in self.gating_layers]

        # CLS token for classification
        cls_hidden = hidden_states[:, 0, :]  # (B, d_model)

        # Domain logits from full sequence (mean pooling inside DomainClassifier)
        domain_logits = self.domain_classifier(hidden_states)  # (B, num_domains)

        # Task logits
        task_logits = self.classifier(cls_hidden)  # (B, num_labels)

        return task_logits, domain_logits


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------

class DualReplay(BaseContinualMethod):
    """Dual-Replay continual learning method."""

    def __init__(
        self,
        model_name: str,
        num_domains: int,
        learning_rate: float = 2e-4,
        epochs: int = 3,
        batch_size: int = 16,
        max_seq_len: int = 128,
        adapter_r: int = 64,
        embed_dim: int = 64,
        replay_ratio: float = 0.30,
        domain_replay_fraction: float = 0.15,
        domain_buffer_size: int = 200,
        general_buffer_size: int = 1000,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(model_name=model_name, num_domains=num_domains, **kwargs)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.adapter_r = adapter_r
        self.embed_dim = embed_dim
        self.replay_ratio = replay_ratio
        self.domain_replay_fraction = domain_replay_fraction
        self.domain_buffer_size = domain_buffer_size
        self.general_buffer_size = general_buffer_size
        self._default_num_labels = num_labels

        self.model: DualReplayModel | None = None
        self.tokenizer = None
        self.replay_buffer: DualReplayBuffer | None = None
        self._num_labels: int | None = None

    # ------------------------------------------------------------------
    # BaseContinualMethod interface
    # ------------------------------------------------------------------

    def setup(self):
        """Initialize tokenizer, replay buffer, and model."""
        self.tokenizer = _load_tokenizer(self.model_name)

        self.replay_buffer = DualReplayBuffer(
            max_per_domain=self.domain_buffer_size,
            general_max_size=self.general_buffer_size,
        )

        # Create model eagerly so callers can inspect parameters immediately
        self._ensure_model(self._default_num_labels)

    def fill_general_buffer(self, data: list[dict]):
        """Populate the general replay stream."""
        if self.replay_buffer is None:
            raise RuntimeError("Call setup() first.")
        self.replay_buffer.fill_general(data)

    def _ensure_model(self, num_labels: int):
        """Create model on first call; no-op if already created."""
        if self.model is not None:
            return
        self._num_labels = num_labels
        self.model = DualReplayModel(
            model_name=self.model_name,
            num_domains=self.num_domains,
            num_labels=num_labels,
            adapter_r=self.adapter_r,
            embed_dim=self.embed_dim,
        ).to(self.device)

    def _build_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.learning_rate)

    def _compose_batch(
        self,
        new_data: list[dict],
        domain_id: int,
        n_total: int,
        rng: random.Random,
    ) -> list[dict]:
        """
        Mix new examples with replay examples.

        n_new          = round((1 - replay_ratio) * n_total)
        n_domain_rep   = round(domain_replay_fraction * n_total)
        n_general_rep  = n_total - n_new - n_domain_rep
        """
        n_new = max(1, round((1.0 - self.replay_ratio) * n_total))
        n_domain_rep = round(self.domain_replay_fraction * n_total)
        n_general_rep = max(0, n_total - n_new - n_domain_rep)

        new_samples = rng.choices(new_data, k=min(n_new, len(new_data)))

        domain_samples, general_samples = self.replay_buffer.sample_replay(
            domain_n=n_domain_rep,
            general_n=n_general_rep,
            rng=rng,
        )

        # Tag domain field for domain classifier supervision
        tagged_new = [dict(ex, domain=domain_id) for ex in new_samples]
        tagged_domain = [dict(ex, domain=ex.get("domain", -1)) for ex in domain_samples]
        tagged_general = [dict(ex, domain=-1) for ex in general_samples]

        combined = tagged_new + tagged_domain + tagged_general
        rng.shuffle(combined)
        return combined

    def train_domain(
        self,
        domain_id: int,
        train_data: list[dict],
        replay_data: list[dict] | None = None,
    ) -> dict[str, float]:
        if self.tokenizer is None or self.replay_buffer is None:
            raise RuntimeError("Call setup() first.")

        # Infer number of labels from data
        labels = [ex["label"] for ex in train_data if ex.get("label", -1) >= 0]
        num_labels = max(labels) + 1 if labels else 2
        self._ensure_model(num_labels)

        self.model.train()
        optimizer = self._build_optimizer()
        rng = random.Random(domain_id)

        total_loss = 0.0
        total_steps = 0

        for _epoch in range(self.epochs):
            n_total = len(train_data)
            mixed_data = self._compose_batch(train_data, domain_id, n_total, rng)

            dataset = _DualReplayDataset(mixed_data, self.tokenizer, self.max_seq_len)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_tensor = batch["label"].to(self.device)
                domain_tensor = batch["domain"].to(self.device)

                task_logits, domain_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    domain_id=domain_id,
                )

                loss = torch.tensor(0.0, device=self.device)

                # Task loss: only examples with valid labels (>= 0)
                valid_task = labels_tensor >= 0
                if valid_task.any():
                    task_loss = F.cross_entropy(
                        task_logits[valid_task], labels_tensor[valid_task]
                    )
                    loss = loss + task_loss

                # Domain classifier loss: only examples with known domain
                valid_domain = domain_tensor >= 0
                if valid_domain.any():
                    domain_loss = F.cross_entropy(
                        domain_logits[valid_domain], domain_tensor[valid_domain]
                    )
                    loss = loss + 0.1 * domain_loss

                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        # Store current domain examples into replay buffer after training
        self.replay_buffer.add_domain(domain_id, train_data)
        self.current_domain = domain_id

        avg_loss = total_loss / max(1, total_steps)
        return {"loss": avg_loss}

    def run_evaluation(self, test_data: list[dict]) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Must call train_domain() before evaluation.")
        self.model.eval()

        dataset = _DualReplayDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_tensor = batch["label"]

                # Inference: domain_id=None triggers soft routing
                task_logits, _domain_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    domain_id=None,
                )

                preds = task_logits.argmax(dim=-1).cpu()
                valid = labels_tensor >= 0
                all_preds.extend(preds[valid].tolist())
                all_labels.extend(labels_tensor[valid].tolist())

        if not all_labels:
            return {"f1": 0.0, "accuracy": 0.0}

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        acc = accuracy_score(all_labels, all_preds) * 100.0
        return {"f1": f1, "accuracy": acc}

    def get_trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
