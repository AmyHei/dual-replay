"""Shared utilities for continual learning methods."""
import math
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_linear_schedule_with_warmup,
)


class TextDataset(Dataset):
    """Tokenize text samples and return tensors for sequence classification."""

    def __init__(self, data: list[dict], tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = [item["text"] for item in data]
        self.labels = [item["label"] for item in data]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _load_model_for_classification(model_name: str, num_labels: int):
    """Load sequence classification model with fallback for checkpoints missing model_type."""
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    except (ValueError, OSError):
        # Some old checkpoints (e.g. prajjwal1/bert-tiny) omit model_type
        # in config.json; load them explicitly as BertForSequenceClassification.
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        return BertForSequenceClassification.from_pretrained(model_name, config=config)


def _load_tokenizer(model_name: str):
    """Load tokenizer with fallback to BertTokenizer for checkpoints missing sentencepiece."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except (ValueError, OSError, ImportError):
        return BertTokenizer.from_pretrained(model_name)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, valid_labels: list[int] | None) -> torch.Tensor:
    """Cross-entropy restricted to valid_labels only (task-incremental training)."""
    if valid_labels is None:
        return torch.nn.functional.cross_entropy(logits, labels)
    # Gather only the columns for valid labels
    indices = torch.tensor(valid_labels, device=logits.device)
    masked_logits = logits[:, indices]
    # Remap labels to 0..len(valid_labels)-1
    label_remap = {v: i for i, v in enumerate(valid_labels)}
    remapped = torch.tensor([label_remap[l.item()] for l in labels], device=labels.device)
    return torch.nn.functional.cross_entropy(masked_logits, remapped)


def masked_argmax(logits: torch.Tensor, valid_labels: list[int] | None) -> torch.Tensor:
    """Argmax over logits, restricted to valid_labels if provided (task-incremental eval)."""
    if valid_labels is None:
        return logits.argmax(dim=-1)
    mask = torch.full(logits.shape, float("-inf"), device=logits.device)
    for label_id in valid_labels:
        if label_id < logits.shape[-1]:
            mask[:, label_id] = 0.0
    return (logits + mask).argmax(dim=-1)


def resize_classifier(model, old_num_labels: int, new_num_labels: int, device=None):
    """Expand the classifier head, preserving existing weights."""
    import torch.nn as nn
    old_weight = model.classifier.weight.data
    old_bias = model.classifier.bias.data
    in_features = old_weight.shape[1]
    model.classifier = nn.Linear(in_features, new_num_labels)
    model.classifier.weight.data[:old_num_labels] = old_weight
    model.classifier.bias.data[:old_num_labels] = old_bias
    model.config.num_labels = new_num_labels
    model.num_labels = new_num_labels
    if device is not None:
        model.to(device)


def build_optimizer_and_scheduler(
    model,
    learning_rate: float,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
):
    """Build AdamW optimizer with linear warmup + decay (HF best practice for BERT)."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler
