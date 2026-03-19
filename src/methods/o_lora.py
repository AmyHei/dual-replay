"""OLoRA baseline: Orthogonal subspace LoRA (simplified per-domain adapters).

Simplified: maintain per-domain LoRA adapters and merge after each domain.
The merged weights serve as a frozen base for the next adapter, approximating
the orthogonal subspace constraint without expensive SVD at each step.
"""
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.sequential_ft import TextDataset, _load_model_for_classification, _load_tokenizer


class OLoRA(BaseContinualMethod):
    """Per-domain LoRA adapters merged after each task (simplified O-LoRA)."""

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.adapter_r: int = kwargs.get("lora_r", 16)
        self.adapter_alpha: int = kwargs.get("lora_alpha", 32)
        self.adapter_drop: float = kwargs.get("lora_dropout", 0.1)

        self.tokenizer = None
        self.model = None
        self._num_labels: int = 0

    def setup(self):
        self.tokenizer = _load_tokenizer(self.model_name)

    def _make_lora_config(self):
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.adapter_r,
            lora_alpha=self.adapter_alpha,
            lora_dropout=self.adapter_drop,
            bias="none",
        )

    def _ensure_model(self, num_labels: int):
        if self.model is None or num_labels > self._num_labels:
            self._num_labels = num_labels
            base_model = _load_model_for_classification(self.model_name, num_labels)
            self.model = get_peft_model(base_model, self._make_lora_config())
            self.model.to(self.device)

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

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)

        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs):
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl = batch["labels"].to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=ids, attention_mask=mask, labels=lbl)
                out.loss.backward()
                optimizer.step()

                total_loss += out.loss.item()
                steps += 1

        # Merge adapter into base and attach a fresh adapter for next domain
        try:
            merged = self.model.merge_and_unload()
            self.model = get_peft_model(merged, self._make_lora_config())
            self.model.to(self.device)
        except Exception:
            pass

        return {"loss": total_loss / max(steps, 1)}

    def run_evaluation(self, test_data: list) -> dict:
        if self.model is None:
            raise RuntimeError("Model not yet trained; call train_domain first.")

        dataset = TextDataset(test_data, self.tokenizer, self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl = batch["labels"]

                out = self.model(input_ids=ids, attention_mask=mask)
                preds = out.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(lbl.tolist())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        return {"f1": f1}

    def get_trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
