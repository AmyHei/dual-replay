"""DER baseline: Dark Experience Replay with LoRA.

Reference: Buzzega et al. 2020.  Stores model logits alongside replay
examples.  During replay adds a distillation term:
  der_loss = MSE(current_logits, stored_logits)
"""
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import (
    TextDataset, _load_model_for_classification, _load_tokenizer,
    build_optimizer_and_scheduler, resize_classifier, masked_argmax,
)
from src.replay.buffer import DomainReplayBuffer


class LogitDataset(Dataset):
    """Like TextDataset but also returns a stored_logits tensor per sample."""

    def __init__(self, data: list, tokenizer, max_seq_len: int, num_labels: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        stored = item.get("_logits")
        if stored is None:
            stored = torch.zeros(self.num_labels)
        else:
            if stored.shape[0] < self.num_labels:
                stored = torch.cat([stored, torch.zeros(self.num_labels - stored.shape[0])])
            stored = stored[: self.num_labels]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
            "stored_logits": stored,
        }


class DER(BaseContinualMethod):
    """Dark Experience Replay with LoRA adapters."""

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.replay_ratio: float = kwargs.get("replay_ratio", 0.20)
        self.domain_buffer_size: int = kwargs.get("domain_buffer_size", 200)
        self.der_alpha: float = kwargs.get("der_alpha", 0.5)
        self.adapter_r: int = kwargs.get("lora_r", 16)
        self.adapter_alpha: int = kwargs.get("lora_alpha", 32)
        self.adapter_drop: float = kwargs.get("lora_dropout", 0.1)
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)
        self.warmup_ratio: float = kwargs.get("warmup_ratio", 0.1)

        self.tokenizer = None
        self.model = None
        self._num_labels: int = 0
        self._replay_buffer = None

    def setup(self):
        self.tokenizer = _load_tokenizer(self.model_name)
        self._replay_buffer = DomainReplayBuffer(max_per_domain=self.domain_buffer_size)

    def _make_lora_config(self):
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.adapter_r,
            lora_alpha=self.adapter_alpha,
            lora_dropout=self.adapter_drop,
            bias="none",
        )

    def _ensure_model(self, num_labels: int):
        if self.model is None:
            self._num_labels = num_labels
            base_model = _load_model_for_classification(self.model_name, num_labels)
            self.model = get_peft_model(base_model, self._make_lora_config())
            self.model.to(self.device)
        elif num_labels > self._num_labels:
            base = self.model.base_model.model
            resize_classifier(base, self._num_labels, num_labels, self.device)
            self._num_labels = num_labels

    def _collect_logits(self, data: list) -> list:
        """Run inference and return logits list (one tensor per sample)."""
        self.model.eval()
        all_logits = []
        dataset = TextDataset(data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.extend(out.logits.cpu().unbind(0))
        return all_logits

    def train_domain(
        self,
        domain_id: int,
        train_data: list,
        replay_data: list = None,
    ) -> dict:
        self.current_domain = domain_id
        combined_new = train_data + (replay_data or [])
        max_label = max(item["label"] for item in combined_new)
        self._ensure_model(num_labels=max(2, max_label + 1))

        rng = random.Random(42 + domain_id)
        replay_n = max(1, int(len(train_data) * self.replay_ratio))
        buffer_samples = self._replay_buffer.sample_all(replay_n, rng=rng)

        all_data = combined_new + buffer_samples

        dataset = LogitDataset(all_data, self.tokenizer, self.max_seq_len, self._num_labels)
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
                s_log = batch["stored_logits"].to(self.device)

                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ce = out.loss

                has_stored = s_log.abs().sum(dim=-1) > 0
                if has_stored.any():
                    dist = F.mse_loss(out.logits[has_stored], s_log[has_stored])
                else:
                    dist = torch.tensor(0.0, device=self.device)

                loss = (ce + self.der_alpha * dist) / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1

        logits_list = self._collect_logits(train_data)
        annotated = []
        for item, lg in zip(train_data, logits_list):
            entry = dict(item)
            entry["_logits"] = lg
            annotated.append(entry)
        self._replay_buffer.add_domain(domain_id, annotated)

        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data, valid_labels=None):
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
