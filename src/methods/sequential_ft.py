"""Sequential fine-tuning baseline: full-model FT on each domain with no forgetting mitigation."""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import (
    TextDataset, _load_model_for_classification, _load_tokenizer,
    build_optimizer_and_scheduler, resize_classifier, masked_argmax,
    masked_cross_entropy,
)


class SequentialFT(BaseContinualMethod):
    """Full-model sequential fine-tuning -- no forgetting mitigation."""

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)
        self.warmup_ratio: float = kwargs.get("warmup_ratio", 0.1)
        self.unfreeze_top_k: int = kwargs.get("unfreeze_top_k", 0)  # 0 = full FT

        self.tokenizer = None
        self.model: nn.Module | None = None
        self._num_labels: int = 0

    # ------------------------------------------------------------------
    # BaseContinualMethod interface
    # ------------------------------------------------------------------

    def setup(self):
        """Load tokenizer. Model is deferred until first labels are seen."""
        self.tokenizer = _load_tokenizer(self.model_name)

    def train_domain(
        self,
        domain_id: int,
        train_data: list[dict],
        replay_data: list[dict] | None = None,
    ) -> dict[str, float]:
        """Fine-tune on train_data (and optional replay_data) for this domain."""
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

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1

        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data: list[dict], valid_labels: list[int] | None = None) -> dict[str, float]:
        """Compute macro-averaged F1 on test_data with optional label masking."""
        if self.model is None:
            raise RuntimeError("Model not yet trained; call train_domain first.")

        dataset = TextDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                preds = masked_argmax(outputs.logits, valid_labels).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        return {"f1": f1}

    def get_trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, num_labels: int):
        """Create or resize the classification head when the label space grows."""
        if self.model is None:
            self._num_labels = num_labels
            self.model = _load_model_for_classification(self.model_name, num_labels)
            self.model.to(self.device)
            if self.unfreeze_top_k > 0:
                # Freeze all, then unfreeze top-k encoder layers + classifier
                for param in self.model.parameters():
                    param.requires_grad = False
                # Unfreeze classifier
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                # Unfreeze top-k encoder layers
                if hasattr(self.model, 'bert'):
                    layers = self.model.bert.encoder.layer
                    for layer in layers[-self.unfreeze_top_k:]:
                        for param in layer.parameters():
                            param.requires_grad = True
        elif num_labels > self._num_labels:
            resize_classifier(self.model, self._num_labels, num_labels, self.device)
            self._num_labels = num_labels
