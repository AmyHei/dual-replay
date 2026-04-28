"""LoRAReplayDual: same LoRA backbone as LoRAReplay but with the dual-stream
replay buffer (domain-specific + general) used by DualReplay.

Ablation purpose: isolate the contribution of the dual-buffer mechanism by
removing DualReplay's adapter gating + trainable domain classifier.
"""
import math
import random
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import (
    TextDataset, _load_model_for_classification, _load_tokenizer,
    build_optimizer_and_scheduler, resize_classifier, masked_argmax,
)
from src.replay.buffer import DualReplayBuffer


class LoRAReplayDual(BaseContinualMethod):
    """LoRA + dual-stream replay (domain-specific + general buffers)."""

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.replay_ratio: float = kwargs.get("replay_ratio", 0.20)
        self.domain_replay_fraction: float = kwargs.get("domain_replay_fraction", 0.10)
        self.domain_buffer_size: int = kwargs.get("domain_buffer_size", 200)
        self.general_buffer_size: int = kwargs.get("general_buffer_size", 1000)
        self.adapter_r: int = kwargs.get("lora_r", kwargs.get("adapter_r", 16))
        self.adapter_alpha: int = kwargs.get("lora_alpha", 32)
        self.adapter_dropout: float = kwargs.get("lora_dropout", 0.1)
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)
        self.warmup_ratio: float = kwargs.get("warmup_ratio", 0.1)

        self.tokenizer = None
        self.model = None
        self._num_labels: int = 0
        self.replay_buffer = None

    def setup(self):
        self.tokenizer = _load_tokenizer(self.model_name)
        self.replay_buffer = DualReplayBuffer(
            max_per_domain=self.domain_buffer_size,
            general_max_size=self.general_buffer_size,
        )

    def fill_general_buffer(self, data):
        if self.replay_buffer is None:
            raise RuntimeError("Call setup() first.")
        self.replay_buffer.fill_general(data)

    def _ensure_model(self, num_labels: int):
        if self.model is None:
            self._num_labels = num_labels
            base_model = _load_model_for_classification(self.model_name, num_labels)
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.adapter_r,
                lora_alpha=self.adapter_alpha,
                lora_dropout=self.adapter_dropout,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.to(self.device)
        elif num_labels > self._num_labels:
            base = self.model.base_model.model
            resize_classifier(base, self._num_labels, num_labels, self.device)
            self._num_labels = num_labels

    def train_domain(self, domain_id, train_data, replay_data=None):
        self.current_domain = domain_id
        rng = random.Random(42 + domain_id)
        n_total = len(train_data)
        n_domain_rep = int(self.domain_replay_fraction * n_total)
        n_general_rep = max(0, int(self.replay_ratio * n_total) - n_domain_rep)
        domain_samples, general_samples = self.replay_buffer.sample_replay(
            domain_n=n_domain_rep, general_n=n_general_rep, rng=rng,
        )
        # General samples have label=-1; drop them since PEFT has no
        # use for unlabelled data here.
        labelled_general = [ex for ex in general_samples if ex.get("label", -1) >= 0]
        combined = train_data + domain_samples + labelled_general + (replay_data or [])
        max_label = max(item["label"] for item in combined)
        self._ensure_model(num_labels=max(2, max_label + 1))

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
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item() * self.gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1
        self.replay_buffer.add_domain(domain_id, train_data)
        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data, valid_labels=None):
        if self.model is None:
            raise RuntimeError("Model not yet trained; call train_domain first.")
        dataset = TextDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = masked_argmax(outputs.logits, valid_labels).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        return {"f1": f1}

    def get_trainable_param_count(self):
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
