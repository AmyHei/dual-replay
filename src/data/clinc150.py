"""CLINC150 dataset loading and 15-domain protocol construction.

Split strategy: Sort all 150 in-scope intents alphabetically by name,
then partition into 15 consecutive groups of 10.
"""
from datasets import load_dataset
import random


def load_clinc150_raw():
    """Load raw CLINC150 dataset from HuggingFace."""
    ds = load_dataset("clinc_oos", "plus")
    return ds


def _get_intent_names(ds):
    """Get intent ID to name mapping from the dataset."""
    return ds["train"].features["intent"].names


def build_15_domain_protocol(seed=42):
    """Split 150 CLINC150 intents into 15 domains of 10 intents each."""
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)

    oos_idx = intent_names.index("oos") if "oos" in intent_names else -1
    in_scope = [i for i in range(len(intent_names)) if i != oos_idx]
    assert len(in_scope) == 150, f"Expected 150 in-scope intents, got {len(in_scope)}"

    in_scope_sorted = sorted(in_scope, key=lambda i: intent_names[i])

    domains = []
    for d in range(15):
        domain_intents = in_scope_sorted[d * 10 : (d + 1) * 10]
        intent_set = set(domain_intents)

        train_examples = [ex for ex in ds["train"] if ex["intent"] in intent_set]
        test_examples = [ex for ex in ds["test"] if ex["intent"] in intent_set]
        val_examples = [ex for ex in ds["validation"] if ex["intent"] in intent_set]

        domains.append({
            "domain_id": d,
            "intents": domain_intents,
            "intent_names": [intent_names[i] for i in domain_intents],
            "train": train_examples,
            "test": test_examples,
            "validation": val_examples,
        })

    return domains


def get_general_buffer(max_size=1000, seed=42):
    """Get general replay buffer from OOS examples."""
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)
    oos_idx = intent_names.index("oos") if "oos" in intent_names else None

    if oos_idx is None:
        raise ValueError("OOS intent not found in CLINC150")

    oos_examples = [ex for ex in ds["train"] if ex["intent"] == oos_idx]

    rng = random.Random(seed)
    if len(oos_examples) > max_size:
        oos_examples = rng.sample(oos_examples, max_size)

    return [{"text": ex["text"], "intent": "oos"} for ex in oos_examples]
