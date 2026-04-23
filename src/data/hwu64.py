"""HWU64 dataset loading (Liu et al., 2019 NLU evaluation benchmark).

Uses `DeepPavlov/hwu_intent_classification`. Intent labels follow the format
`{scenario}_{action}` (e.g., alarm_query, calendar_remove), so we derive the
scenario grouping by splitting on the first underscore. Yields 18 scenarios
with 1–10 intents each.

Sequential CL protocol: one scenario per step, random ordering by `seed`.
"""
from datasets import load_dataset
import random


def load_hwu64_raw():
    return load_dataset("DeepPavlov/hwu_intent_classification")


def _intent_to_scenario(intent_text: str) -> str:
    return intent_text.split("_", 1)[0]


def build_scenario_protocol(seed: int = 42) -> list[dict]:
    """One HWU64 scenario per domain, ordered by `seed`.

    Returns list of dicts: domain_id, domain_name (scenario),
    intents (global label ids belonging to this scenario),
    intent_names, train, test.

    Train/test examples use `label` (int) as the target and carry `text`
    for compatibility with DualReplay's dataset schema.
    """
    ds = load_hwu64_raw()

    # Build label_id → label_text and scenario → label_ids
    id_to_text: dict[int, str] = {}
    scen_to_ids: dict[str, set[int]] = {}
    for split in ds:
        for ex in ds[split]:
            lid = ex["label"]
            if lid not in id_to_text:
                id_to_text[lid] = ex["label_text"]
                scen = _intent_to_scenario(ex["label_text"])
                scen_to_ids.setdefault(scen, set()).add(lid)

    scenario_order = sorted(scen_to_ids.keys())
    rng = random.Random(seed)
    rng.shuffle(scenario_order)

    domains: list[dict] = []
    for d, scen in enumerate(scenario_order):
        ids = scen_to_ids[scen]
        train = [{"text": ex["text"], "intent": ex["label"]}
                 for ex in ds["train"] if ex["label"] in ids]
        test = [{"text": ex["text"], "intent": ex["label"]}
                for ex in ds["test"] if ex["label"] in ids]
        domains.append({
            "domain_id": d,
            "domain_name": scen,
            "intents": sorted(ids),
            "intent_names": sorted(id_to_text[i] for i in ids),
            "train": train,
            "test": test,
            "validation": [],
        })
    return domains


def get_general_buffer(max_size: int = 1000, seed: int = 42) -> list[dict]:
    """HWU64 lacks an OOS split; sample from the `general` scenario as a proxy
    for the general-knowledge replay stream."""
    ds = load_hwu64_raw()
    general_samples = [
        {"text": ex["text"], "intent": "general"}
        for ex in ds["train"]
        if _intent_to_scenario(ex["label_text"]) == "general"
    ]
    rng = random.Random(seed)
    if len(general_samples) > max_size:
        general_samples = rng.sample(general_samples, max_size)
    return general_samples
