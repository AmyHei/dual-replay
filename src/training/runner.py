"""Sequential training runner."""
import numpy as np
from typing import Any
from src.methods.base import BaseContinualMethod
from src.metrics import compute_bwt, compute_avg_f1


def remap_to_local_labels(data: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """Remap global intent/label IDs to local 0-based label IDs.

    CLINC150 examples carry global intent IDs (0-149).  Each method expects
    a per-domain label space starting from 0 (i.e. 0 … N-1 for N intents in
    the domain).  This helper builds the mapping and returns remapped copies
    of the examples, together with the mapping dict for reuse on test data.
    """
    # Support both "intent" (CLINC150 raw) and "label" (already remapped) keys.
    def _get_raw_label(ex: dict) -> int:
        if "label" in ex:
            return ex["label"]
        if "intent" in ex:
            v = ex["intent"]
            # The general buffer uses "oos" string; treat as -1 (skip)
            if isinstance(v, str):
                return -1
            return int(v)
        return -1

    global_ids = sorted(set(_get_raw_label(ex) for ex in data if _get_raw_label(ex) >= 0))
    mapping = {gid: lid for lid, gid in enumerate(global_ids)}

    remapped = []
    for ex in data:
        raw = _get_raw_label(ex)
        new_ex = dict(ex)
        new_ex["label"] = mapping.get(raw, -1)
        # Keep domain field if present
        remapped.append(new_ex)
    return remapped, mapping


class SequentialRunner:
    """Runs sequential continual learning over a list of domains.

    Each domain dict must have at minimum:
        "domain_id": int
        "train": list[dict]   (examples with "text" and "intent"/"label" keys)
        "test":  list[dict]   (same structure)

    The runner handles the global->local label remapping so that methods always
    receive 0-based label spaces.
    """

    def __init__(self, method: BaseContinualMethod, domains: list[dict]):
        self.method = method
        self.domains = domains

    def run(self) -> dict[str, Any]:
        K = len(self.domains)
        perf_matrix = np.zeros((K, K))

        # Precompute local-label versions of train/test splits once.
        # We store the per-domain label mapping so test data uses the same
        # mapping as training data.
        local_domains: list[dict] = []
        for domain in self.domains:
            train_local, mapping = remap_to_local_labels(domain["train"])
            test_local, _ = remap_to_local_labels(domain["test"])
            # Re-apply the *training* mapping to the test set so label IDs are
            # consistent (a test intent not seen in training maps to -1, which
            # the methods handle gracefully via zero_division=0 in f1_score).
            def apply_mapping(examples, m):
                out = []
                for ex in examples:
                    new_ex = dict(ex)
                    raw = ex.get("label", ex.get("intent", -1))
                    if isinstance(raw, str):
                        raw = -1
                    new_ex["label"] = m.get(int(raw), -1) if raw != -1 else -1
                    out.append(new_ex)
                return out

            test_local = apply_mapping(domain["test"], mapping)
            local_domains.append({
                **domain,
                "train": train_local,
                "test": test_local,
            })

        self.method.setup()

        for step, domain in enumerate(local_domains):
            self.method.train_domain(
                domain_id=domain["domain_id"],
                train_data=domain["train"],
            )
            for j in range(step + 1):
                eval_result = self.method.run_evaluation(local_domains[j]["test"])
                perf_matrix[step, j] = eval_result["f1"]

        final_scores = perf_matrix[K - 1, :]
        bwt = compute_bwt(perf_matrix)
        avg_f1 = compute_avg_f1(final_scores)

        return {
            "perf_matrix": perf_matrix,
            "bwt": bwt,
            "avg_f1": avg_f1,
            "final_scores": final_scores,
        }
