"""Sequential training runner."""
import numpy as np
from typing import Any
from src.methods.base import BaseContinualMethod
from src.metrics import compute_bwt, compute_avg_f1


def _build_contiguous_label_map(domains: list[dict]) -> dict[int, int]:
    """Build a mapping from scattered global intent IDs to contiguous labels.

    Domain 0's intents → labels 0-9, domain 1's → 10-19, etc.
    This gives each domain its own non-overlapping section of the output space.
    """
    mapping = {}
    next_label = 0
    for domain in domains:
        intents = sorted(set(
            ex.get("intent", ex.get("label", -1))
            for ex in domain["train"]
            if not isinstance(ex.get("intent", ex.get("label", -1)), str)
            and ex.get("intent", ex.get("label", -1)) >= 0
        ))
        for intent_id in intents:
            mapping[intent_id] = next_label
            next_label += 1
    return mapping


def _remap_data(data: list[dict], mapping: dict[int, int]) -> list[dict]:
    """Remap intent IDs to contiguous labels using the provided mapping."""
    out = []
    for ex in data:
        new_ex = dict(ex)
        raw = ex.get("intent", ex.get("label", -1))
        if isinstance(raw, str):
            new_ex["label"] = -1
        else:
            new_ex["label"] = mapping.get(int(raw), -1)
        out.append(new_ex)
    return out


class SequentialRunner:
    """Runs sequential continual learning over a list of domains.

    Each domain dict must have at minimum:
        "domain_id": int
        "train": list[dict]   (examples with "text" and "intent"/"label" keys)
        "test":  list[dict]   (same structure)

    Uses contiguous labels (domain 0 → 0-9, domain 1 → 10-19, etc.) with
    class-incremental evaluation: at test time, the model predicts among ALL
    seen classes (no domain ID known), not just the current domain's labels.
    """

    def __init__(self, method: BaseContinualMethod, domains: list[dict]):
        self.method = method
        self.domains = domains

    def run(self) -> dict[str, Any]:
        K = len(self.domains)
        perf_matrix = np.zeros((K, K))

        # Build contiguous label mapping across all domains
        label_map = _build_contiguous_label_map(self.domains)

        # Remap data to contiguous labels
        prepared_domains: list[dict] = []
        for domain in self.domains:
            train_remapped = _remap_data(domain["train"], label_map)
            test_remapped = _remap_data(domain["test"], label_map)
            prepared_domains.append({
                **domain,
                "train": train_remapped,
                "test": test_remapped,
            })

        self.method.setup()

        for step, domain in enumerate(prepared_domains):
            self.method.train_domain(
                domain_id=domain["domain_id"],
                train_data=domain["train"],
            )
            for j in range(step + 1):
                # Class-incremental eval: no label masking, argmax over all classes
                result = self.method.run_evaluation(prepared_domains[j]["test"])
                perf_matrix[step, j] = result["f1"]

        final_scores = perf_matrix[K - 1, :]
        bwt = compute_bwt(perf_matrix)
        avg_f1 = compute_avg_f1(final_scores)

        return {
            "perf_matrix": perf_matrix,
            "bwt": bwt,
            "avg_f1": avg_f1,
            "final_scores": final_scores,
        }
