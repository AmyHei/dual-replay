"""EWC baseline: Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Uses online EWC (consolidated Fisher) to avoid O(T) memory growth.
"""
import math
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import (
    TextDataset, _load_model_for_classification, _load_tokenizer,
    build_optimizer_and_scheduler, resize_classifier, masked_argmax,
)


class EWC(BaseContinualMethod):
    """Full-model fine-tuning with online EWC regularization.

    Uses online consolidation: a single running Fisher + optimal params,
    updated after each domain. Memory is O(params), not O(params × domains).
    """

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.ewc_lambda: float = kwargs.get("ewc_lambda", 5000.0)
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)
        self.warmup_ratio: float = kwargs.get("warmup_ratio", 0.1)
        self.fisher_samples: int = kwargs.get("ewc_fisher_samples", 500)
        self.fisher_gamma: float = kwargs.get("ewc_fisher_gamma", 0.95)

        self.tokenizer = None
        self.model = None
        self._num_labels: int = 0
        # Online EWC: single consolidated Fisher + optimal params (on CPU)
        self._consolidated_fisher: dict | None = None
        self._consolidated_optpar: dict | None = None
        self._n_tasks_seen: int = 0

    def setup(self):
        self.tokenizer = _load_tokenizer(self.model_name)

    def _ensure_model(self, num_labels: int):
        if self.model is None:
            self._num_labels = num_labels
            self.model = _load_model_for_classification(self.model_name, num_labels)
            self.model.to(self.device)
        elif num_labels > self._num_labels:
            resize_classifier(self.model, self._num_labels, num_labels, self.device)
            self._num_labels = num_labels

    def _compute_fisher(self, data: list) -> dict:
        """Diagonal Fisher on a random subset (saves memory + time)."""
        self.model.eval()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p, device="cpu")

        # Subsample for efficiency
        if len(data) > self.fisher_samples:
            data = random.sample(data, self.fisher_samples)

        dataset = TextDataset(data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.model.zero_grad()
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] = fisher[n] + p.grad.detach().cpu().pow(2)

            total += input_ids.size(0)

        norm = max(total, 1)
        for k in fisher:
            fisher[k] = fisher[k] / norm

        return fisher

    def _consolidate_fisher(self, new_fisher: dict):
        """Online EWC: merge new Fisher into consolidated Fisher with decay."""
        opt_params = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                opt_params[n] = p.detach().clone().cpu()

        if self._consolidated_fisher is None:
            self._consolidated_fisher = new_fisher
            self._consolidated_optpar = opt_params
        else:
            gamma = self.fisher_gamma
            for n in new_fisher:
                if n in self._consolidated_fisher:
                    old_fi = self._consolidated_fisher[n]
                    new_fi = new_fisher[n]
                    if old_fi.shape != new_fi.shape:
                        # Classifier grew: pad old Fisher to new size
                        padded = torch.zeros_like(new_fi)
                        slices = tuple(slice(0, s) for s in old_fi.shape)
                        padded[slices] = old_fi
                        self._consolidated_fisher[n] = gamma * padded + new_fi
                    else:
                        self._consolidated_fisher[n] = gamma * old_fi + new_fi
                else:
                    self._consolidated_fisher[n] = new_fi
            self._consolidated_optpar = opt_params
        self._n_tasks_seen += 1

    def _ewc_penalty(self) -> torch.Tensor:
        """Online EWC penalty against single consolidated Fisher + params."""
        if self._consolidated_fisher is None:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._consolidated_fisher:
                fi = self._consolidated_fisher[n].to(self.device)
                opt = self._consolidated_optpar[n].to(self.device)
                # Handle classifier head resize: only penalize the old portion
                if fi.shape != p.shape:
                    slices = tuple(slice(0, s) for s in fi.shape)
                    penalty = penalty + (fi * (p[slices] - opt).pow(2)).sum()
                else:
                    penalty = penalty + (fi * (p - opt).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * penalty

    def train_domain(
        self,
        domain_id: int,
        train_data: list,
        replay_data: list = None,
    ) -> dict:
        self.current_domain = domain_id
        combined = train_data + (replay_data or [])
        max_label = max(item["label"] for item in combined)
        self._ensure_model(num_labels=max_label + 1)

        dataset = TextDataset(combined, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        steps_per_epoch = math.ceil(len(loader) / self.gradient_accumulation_steps)
        num_training_steps = steps_per_epoch * self.epochs

        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, self.learning_rate, num_training_steps,
            warmup_ratio=self.warmup_ratio,
        )

        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs):
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = (out.loss + self._ewc_penalty()) / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1

        fisher = self._compute_fisher(train_data)
        self._consolidate_fisher(fisher)

        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data: list, valid_labels=None) -> dict:
        if self.model is None:
            raise RuntimeError("Model not yet trained; call train_domain first.")

        dataset = TextDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = masked_argmax(out.logits, valid_labels).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        return {"f1": f1}

    def get_trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
