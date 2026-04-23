"""Banking77 dataset loading (Casanueva et al., 2020).

77 fine-grained banking intents with no natural domain grouping, so we split
deterministically into 7 sequential tasks of 11 intents each by sorting the
label names alphabetically. This matches the split convention used in recent
CL-for-NLU work that treats Banking77 as a sequential benchmark.
"""
from datasets import load_dataset
import random


def load_banking77_raw():
    return load_dataset("banking77")


def build_7_task_protocol(seed: int = 42) -> list[dict]:
    """7 sequential tasks × 11 intents each, alphabetical label split.

    The partition is deterministic (for reproducibility across papers that
    cite Banking77 CL results); `seed` only controls the order in which the
    7 tasks are presented.
    """
    ds = load_banking77_raw()
    intent_names = ds["train"].features["label"].names
    assert len(intent_names) == 77, f"Expected 77 intents, got {len(intent_names)}"

    # Sort intent names alphabetically, chunk into 7 tasks of 11
    sorted_idx = sorted(range(len(intent_names)), key=lambda i: intent_names[i])
    tasks: list[list[int]] = [sorted_idx[i * 11 : (i + 1) * 11] for i in range(7)]

    rng = random.Random(seed)
    task_order = list(range(7))
    rng.shuffle(task_order)

    domains: list[dict] = []
    for d, t in enumerate(task_order):
        ids = set(tasks[t])
        train = [{"text": ex["text"], "intent": ex["label"]}
                 for ex in ds["train"] if ex["label"] in ids]
        test = [{"text": ex["text"], "intent": ex["label"]}
                for ex in ds["test"] if ex["label"] in ids]
        domains.append({
            "domain_id": d,
            "domain_name": f"bank_task_{t}",
            "intents": sorted(ids),
            "intent_names": [intent_names[i] for i in sorted(ids)],
            "train": train,
            "test": test,
            "validation": [],
        })
    return domains


def get_general_buffer(max_size: int = 1000, seed: int = 42) -> list[dict]:
    """Banking77 has no OOS; return an empty buffer. DualReplay can still run
    with general_buffer empty (replay_ratio × domain_replay_fraction controls
    how much domain-specific vs general replay is used)."""
    return []
