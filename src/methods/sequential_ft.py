"""Sequential fine-tuning baseline: full-model FT on each domain with no forgetting mitigation."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
)
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod


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
    except Exception:
        return BertTokenizer.from_pretrained(model_name)


class SequentialFT(BaseContinualMethod):
    """Full-model sequential fine-tuning -- no forgetting mitigation."""

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)

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

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data: list[dict]) -> dict[str, float]:
        """Compute macro-averaged F1 on test_data."""
        if self.model is None:
            raise RuntimeError("Model not yet trained; call train_domain first.")

        dataset = TextDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Switch model to inference mode
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
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
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
        if self.model is None or num_labels > self._num_labels:
            self._num_labels = num_labels
            self.model = _load_model_for_classification(self.model_name, num_labels)
            self.model.to(self.device)
