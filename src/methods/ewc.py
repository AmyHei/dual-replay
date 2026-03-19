"""EWC baseline: Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.methods.base import BaseContinualMethod
from src.methods.utils import TextDataset, _load_model_for_classification, _load_tokenizer


class EWC(BaseContinualMethod):
    """Full-model fine-tuning with EWC regularization.

    After each domain, compute diagonal Fisher information and penalize drift:
    ewc_loss = (lambda/2) * sum_i F_i * (theta_i - theta_star_i)^2
    """

    def __init__(self, model_name: str, num_domains: int, **kwargs):
        super().__init__(model_name, num_domains, **kwargs)
        self.learning_rate: float = kwargs.get("learning_rate", 2e-5)
        self.epochs: int = kwargs.get("epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 16)
        self.max_seq_len: int = kwargs.get("max_seq_len", 128)
        self.ewc_lambda: float = kwargs.get("ewc_lambda", 1000.0)

        self.tokenizer = None
        self.model = None
        self._num_labels: int = 0
        self._ewc_tasks: list = []

    def setup(self):
        self.tokenizer = _load_tokenizer(self.model_name)

    def _ensure_model(self, num_labels: int):
        if self.model is None or num_labels > self._num_labels:
            self._num_labels = num_labels
            self.model = _load_model_for_classification(self.model_name, num_labels)
            self.model.to(self.device)

    def _compute_fisher(self, data: list) -> dict:
        """Diagonal Fisher: mean of squared gradients over data."""
        self.model.eval()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)

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
                    fisher[n] = fisher[n] + p.grad.detach().pow(2)

            total += input_ids.size(0)

        norm = max(total, 1)
        for k in fisher:
            fisher[k] = fisher[k] / norm

        return fisher

    def _ewc_penalty(self) -> torch.Tensor:
        """Sum EWC penalties from all consolidated tasks."""
        if not self._ewc_tasks:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for f_dict, opt_dict in self._ewc_tasks:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in f_dict:
                    fi = f_dict[n].to(self.device)
                    opt = opt_dict[n].to(self.device)
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

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss + self._ewc_penalty()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

        fisher = self._compute_fisher(train_data)
        opt_params = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                opt_params[n] = p.detach().clone().cpu()
        self._ewc_tasks.append((fisher, opt_params))

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
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0
        return {"f1": f1}

    def get_trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
