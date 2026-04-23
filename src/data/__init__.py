"""Unified benchmark API for CL experiments.

`build_benchmark(name, seed)` returns `(domains, general_buffer)` where
`domains` is a list of sequential task/domain dicts (each with
train/test/intents), and `general_buffer` is a list of out-of-scope examples
for the general-knowledge replay stream (empty for benchmarks without OOS).

Supported names: "clinc150_10", "hwu64", "banking77".
"""
from . import clinc150, hwu64, banking77


def build_benchmark(name: str, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if name == "clinc150_10":
        domains = clinc150.build_10_domain_protocol(seed=seed)
        general = clinc150.get_general_buffer(max_size=1000, seed=seed)
    elif name == "clinc150_15":
        domains = clinc150.build_15_domain_protocol(seed=seed)
        general = clinc150.get_general_buffer(max_size=1000, seed=seed)
    elif name == "hwu64":
        domains = hwu64.build_scenario_protocol(seed=seed)
        general = hwu64.get_general_buffer(max_size=1000, seed=seed)
    elif name == "banking77":
        domains = banking77.build_7_task_protocol(seed=seed)
        general = banking77.get_general_buffer(max_size=1000, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return domains, general


__all__ = ["build_benchmark", "clinc150", "hwu64", "banking77"]
