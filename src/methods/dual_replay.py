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

import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import _load_tokenizer, build_optimizer_and_scheduler, masked_argmax
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
    """Full model: frozen encoder + adapters + gating + domain classifier + per-domain task heads.

    Per-domain heads: each domain has its own (d_model, labels_per_domain) linear head.
    During training, per-sample head routing ensures each sample's task loss flows only
    through its own domain's head — no cross-head interference.
    At inference, global task logits are a log-probability mixture:
        log p(label|x) = log p(domain=d|x) + log p(label|domain=d, x)
    where the first term is the domain classifier and the second uses head_d.
    """

    def __init__(
        self,
        model_name: str,
        num_domains: int,
        num_labels: int,
        adapter_r: int = 64,
        embed_dim: int = 64,
        unfreeze_top_k: int = 0,
        labels_per_domain: int | None = None,
    ):
        super().__init__()
        # Load and freeze base encoder
        self.encoder = _load_model(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze top-k encoder layers for more capacity (especially useful
        # for small models like bert-tiny where adapters alone are insufficient)
        if unfreeze_top_k > 0:
            layers_list = get_transformer_layers(self.encoder)
            for layer in layers_list[-unfreeze_top_k:]:
                for p in layer.parameters():
                    p.requires_grad = True

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

        # Per-domain task heads: each domain gets its own head, sized to its label count.
        # Assumes uniform labels_per_domain (common in CL benchmarks like CLINC150).
        if labels_per_domain is None:
            assert num_labels % num_domains == 0, (
                f"num_labels ({num_labels}) must be divisible by num_domains ({num_domains}) "
                f"when labels_per_domain is not specified"
            )
            labels_per_domain = num_labels // num_domains
        self.labels_per_domain = labels_per_domain
        self.heads = nn.ModuleList([
            nn.Linear(d_model, labels_per_domain) for _ in range(num_domains)
        ])

        self.num_domains = num_domains
        self.num_labels = num_domains * labels_per_domain
        self.adapter_r = adapter_r

    def _set_adapters_bypass(self, bypass: bool):
        for adapter in self.adapters:
            adapter.set_bypass(bypass)

    def _set_adapter_gates(self, domain_id: int | None, domain_probs: torch.Tensor | None):
        """Compute and set gate vectors on adapters BEFORE the encoder forward pass."""
        if domain_id is not None:
            domain_emb = self.domain_embeddings(domain_id)
        elif domain_probs is not None:
            domain_emb = soft_mixture_routing(self.domain_embeddings, domain_probs)
        else:
            # Will be set to None → adapters run un-gated; domain classifier
            # routing happens in a second pass (see forward).
            for adapter in self.adapters:
                adapter.set_gate(None)
            return

        for adapter, gate_layer in zip(self.adapters, self.gating_layers):
            gate = gate_layer(domain_emb)  # (adapter_r,)
            adapter.set_gate(gate)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_id: int | None = None,
        domain_probs: torch.Tensor | None = None,
        sample_domain_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (task_logits, domain_logits).

        Paper 4.2: domain classifier uses the **frozen encoder's output**, i.e.
        encoder with adapters bypassed. We run two encoder passes: one bypassed
        (for domain classifier, consistent at train and test) and one gated
        (for task heads). This avoids training/inference gate distribution
        mismatch in the routing signal.
        """
        # --- Domain classifier pass: frozen encoder (adapters bypassed) ---
        self._set_adapters_bypass(True)
        with torch.no_grad():
            frozen_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        self._set_adapters_bypass(False)
        # domain_classifier is trainable; rerun it outside no_grad on cached hidden
        domain_logits = self.domain_classifier(frozen_out.last_hidden_state)

        if domain_id is not None or domain_probs is not None:
            # Training (or known-domain eval): single gated encoder pass for task
            self._set_adapter_gates(domain_id, domain_probs)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            cls_hidden = hidden_states[:, 0, :]

            B = cls_hidden.shape[0]
            device = cls_hidden.device
            task_logits = torch.full(
                (B, self.num_labels), -1e4, device=device, dtype=cls_hidden.dtype
            )
            if sample_domain_ids is None:
                sample_domain_ids = torch.full(
                    (B,), domain_id if domain_id is not None else -1,
                    dtype=torch.long, device=device,
                )
            for d_val in torch.unique(sample_domain_ids[sample_domain_ids >= 0]):
                d_int = int(d_val.item())
                sel = sample_domain_ids == d_val
                sub_logits = self.heads[d_int](cls_hidden[sel])
                start = d_int * self.labels_per_domain
                task_logits[sel, start : start + self.labels_per_domain] = sub_logits
            return task_logits, domain_logits
        else:
            # Inference: one gated encoder pass per domain for head output
            all_local_logits = []
            for d in range(self.num_domains):
                self._set_adapter_gates(d, None)
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                h = outputs.last_hidden_state
                cls_h = h[:, 0, :]
                all_local_logits.append(self.heads[d](cls_h))

            stacked = torch.stack(all_local_logits, dim=0)
            B = stacked.shape[1]
            device = stacked.device

            log_domain_probs = F.log_softmax(domain_logits, dim=-1)

            task_logits = torch.full(
                (B, self.num_labels), -1e4, device=device, dtype=stacked.dtype
            )
            for d in range(self.num_domains):
                local_log_probs = F.log_softmax(stacked[d], dim=-1)
                log_weight_d = log_domain_probs[:, d].unsqueeze(-1)
                start = d * self.labels_per_domain
                task_logits[:, start : start + self.labels_per_domain] = (
                    local_log_probs + log_weight_d
                )

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
        learning_rate: float = 2e-5,
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
        self.unfreeze_top_k: int = kwargs.get("unfreeze_top_k", 0)
        self.replay_ratio = replay_ratio
        self.domain_replay_fraction = domain_replay_fraction
        self.domain_buffer_size = domain_buffer_size
        self.general_buffer_size = general_buffer_size
        self._default_num_labels = num_labels
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)
        self.warmup_ratio: float = kwargs.get("warmup_ratio", 0.1)

        self.model: DualReplayModel | None = None
        self.tokenizer = None
        self.replay_buffer: DualReplayBuffer | None = None
        self._num_labels: int | None = None

    # ------------------------------------------------------------------
    # BaseContinualMethod interface
    # ------------------------------------------------------------------

    def setup(self):
        """Initialize tokenizer and replay buffer.

        Model creation is deferred until `train_domain()` so that
        `labels_per_domain` can be inferred from the first domain's actual
        label count (e.g. 10 on CLINC150), not from a generic default.
        """
        self.tokenizer = _load_tokenizer(self.model_name)

        self.replay_buffer = DualReplayBuffer(
            max_per_domain=self.domain_buffer_size,
            general_max_size=self.general_buffer_size,
        )

    def fill_general_buffer(self, data: list[dict]):
        """Populate the general replay stream."""
        if self.replay_buffer is None:
            raise RuntimeError("Call setup() first.")
        self.replay_buffer.fill_general(data)

    def _ensure_model(self, num_labels: int):
        """Create model on first call.

        `num_labels` from the first domain's training data determines
        labels_per_domain (assumes uniform per-domain label count, e.g. 10 on CLINC150).
        All per-domain heads are allocated up front, no resizing needed.
        """
        if self.model is None:
            labels_per_domain = num_labels  # first-domain count = per-domain count
            self._num_labels = self.num_domains * labels_per_domain
            self.model = DualReplayModel(
                model_name=self.model_name,
                num_domains=self.num_domains,
                num_labels=self._num_labels,
                adapter_r=self.adapter_r,
                embed_dim=self.embed_dim,
                unfreeze_top_k=self.unfreeze_top_k,
                labels_per_domain=labels_per_domain,
            ).to(self.device)

    def _build_optimizer(self, num_training_steps: int):
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, self.learning_rate, num_training_steps,
            warmup_ratio=self.warmup_ratio,
        )
        return optimizer, scheduler

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
        rng = random.Random(domain_id)

        # Estimate steps for scheduler: use train_data size as proxy for loader length per epoch
        n_total = len(train_data)
        est_batches_per_epoch = math.ceil(n_total / self.batch_size)
        steps_per_epoch = math.ceil(est_batches_per_epoch / self.gradient_accumulation_steps)
        num_training_steps = steps_per_epoch * self.epochs

        optimizer, scheduler = self._build_optimizer(num_training_steps)

        total_loss = 0.0
        total_steps = 0

        for _epoch in range(self.epochs):
            mixed_data = self._compose_batch(train_data, domain_id, n_total, rng)

            dataset = _DualReplayDataset(mixed_data, self.tokenizer, self.max_seq_len)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )

            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_tensor = batch["label"].to(self.device)
                domain_tensor = batch["domain"].to(self.device)

                task_logits, domain_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    domain_id=domain_id,
                    sample_domain_ids=domain_tensor,
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
                    loss = loss + 1.0 * domain_loss

                if loss.requires_grad:
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()

                    total_loss += loss.item()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        total_steps += 1
                else:
                    total_loss += loss.item()
                    total_steps += 1

        # Store current domain examples into replay buffer after training
        self.replay_buffer.add_domain(domain_id, train_data)
        self.current_domain = domain_id

        avg_loss = total_loss / max(1, total_steps)
        return {"loss": avg_loss}

    def run_evaluation(
        self,
        test_data: list[dict],
        valid_labels=None,
        oracle_domain_id: int | None = None,
    ) -> dict[str, float]:
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
                    domain_id=oracle_domain_id,
                )

                preds = masked_argmax(task_logits, valid_labels).cpu()
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
