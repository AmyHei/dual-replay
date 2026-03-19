"""CLI entry point for running experiments.

Usage:
    uv run python scripts/run_experiment.py --method dual_replay --config debug --seed 42
"""
import argparse
import json
import os
import sys
import yaml

# Ensure the project root is on sys.path so `src` is importable when the
# script is invoked directly (e.g. `uv run python scripts/run_experiment.py`).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.clinc150 import build_15_domain_protocol, get_general_buffer
from src.data.domain_sequence import generate_domain_orderings
from src.training.runner import SequentialRunner


METHOD_MAP = {
    "sequential_ft": ("src.methods.sequential_ft", "SequentialFT"),
    "lora_only": ("src.methods.lora_only", "LoRAOnly"),
    "replay_only": ("src.methods.replay_only", "ReplayOnly"),
    "lora_replay": ("src.methods.lora_replay", "LoRAReplay"),
    "ewc": ("src.methods.ewc", "EWC"),
    "o_lora": ("src.methods.o_lora", "OLoRA"),
    "der": ("src.methods.der", "DER"),
    "dual_replay": ("src.methods.dual_replay", "DualReplay"),
}


def load_config(config_name: str) -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "default.yaml",
    )
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    if config_name not in all_configs:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(all_configs.keys())}")
    return all_configs[config_name]


def get_method(method_name: str, config: dict):
    import importlib

    module_path, class_name = METHOD_MAP[method_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    # Map config keys to constructor kwargs.
    # Some configs use "epochs_per_domain" but methods expect "epochs".
    kwargs = {k: v for k, v in config.items() if k not in ("model_name", "num_domains")}
    if "epochs_per_domain" in kwargs and "epochs" not in kwargs:
        kwargs["epochs"] = kwargs.pop("epochs_per_domain")
    else:
        kwargs.pop("epochs_per_domain", None)

    return cls(
        model_name=config["model_name"],
        num_domains=config["num_domains"],
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="Run a continual learning experiment on CLINC150.")
    parser.add_argument("--method", required=True, choices=list(METHOD_MAP.keys()))
    parser.add_argument("--config", default="debug", help="Config name from configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loading CLINC150 (seed={args.seed})...")
    domains = build_15_domain_protocol(seed=args.seed)

    orderings = generate_domain_orderings(len(domains), num_orderings=1, seed=args.seed)
    ordered_domains = [domains[i] for i in orderings[0]]

    num_domains = config.get("num_domains", len(ordered_domains))
    ordered_domains = ordered_domains[:num_domains]

    print(f"Method: {args.method}  |  Config: {args.config}  |  Domains: {num_domains}")
    method = get_method(args.method, config)

    # For methods that need a general replay buffer (dual_replay)
    if hasattr(method, "fill_general_buffer"):
        print("Filling general replay buffer...")
        general_buffer = get_general_buffer(
            max_size=config.get("general_buffer_size", 1000), seed=args.seed
        )
        # The general buffer uses "intent": "oos" strings; remap to label=-1 so
        # the runner treats them as unlabelled / general-domain examples.
        general_buffer_clean = [
            {"text": ex["text"], "label": -1} for ex in general_buffer
        ]
        method.setup()
        method.fill_general_buffer(general_buffer_clean)
        # runner.run() calls method.setup() again; skip the second call.
        method.setup = lambda: None

    runner = SequentialRunner(method=method, domains=ordered_domains)

    print("Starting sequential training...")
    results = runner.run()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{args.method}_{args.config}_seed{args.seed}.json"
    )
    serializable = {
        "method": args.method,
        "config": args.config,
        "seed": args.seed,
        "avg_f1": results["avg_f1"],
        "bwt": results["bwt"],
        "perf_matrix": results["perf_matrix"].tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"  Avg F1: {results['avg_f1']:.1f}")
    print(f"  BWT:    {results['bwt']:.1f}")


if __name__ == "__main__":
    main()
