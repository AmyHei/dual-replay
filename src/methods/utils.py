"""Shared utilities for continual learning methods."""
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
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
