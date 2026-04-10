"""Base interface for all continual learning methods."""
from abc import ABC, abstractmethod
import torch


class BaseContinualMethod(ABC):
    def __init__(self, model_name: str, num_domains: int, **kwargs):
        self.model_name = model_name
        self.num_domains = num_domains
        self.config = kwargs
        self.device = self._get_device()
        self.current_domain = -1

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def setup(self):
        ...

    @abstractmethod
    def train_domain(self, domain_id: int, train_data: list[dict], replay_data: list[dict] | None = None) -> dict[str, float]:
        ...

    @abstractmethod
    def run_evaluation(self, test_data: list[dict], valid_labels: list[int] | None = None) -> dict[str, float]:
        ...

    def get_trainable_param_count(self) -> int:
        return 0
