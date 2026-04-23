#!/usr/bin/env python3
"""Autoresearch: autonomous hyperparameter search for CL experiments.

Usage:
    uv run python scripts/autoresearch.py --method sequential_ft --time-budget 3600

Runs experiments in a loop, trying different hyperparameter configs,
keeping improvements and discarding regressions.
Inspired by Karpathy's autoresearch/nanochat.
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

_project_root = Path(__file__).resolve().parent.parent
os.chdir(_project_root)

# ---------------------------------------------------------------------------
# Search space per method
# ---------------------------------------------------------------------------

COMMON_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3],
    "warmup_ratio": [0.0, 0.05, 0.1, 0.15, 0.2],
    "epochs_per_domain": [1, 2, 3, 5],
}

METHOD_SPACE = {
    "sequential_ft": {
        **COMMON_SPACE,
    },
    "ewc": {
        **COMMON_SPACE,
        "ewc_lambda": [100, 500, 1000, 2000, 5000, 10000, 20000],
        "ewc_fisher_samples": [50, 100, 200, 500],
    },
    "lora_only": {
        **COMMON_SPACE,
        "adapter_r": [4, 8, 16, 32],
    },
    "replay_only": {
        **COMMON_SPACE,
        "domain_buffer_size": [20, 50, 100, 200],
        "replay_ratio": [0.1, 0.15, 0.2, 0.3, 0.4],
    },
    "lora_replay": {
        **COMMON_SPACE,
        "adapter_r": [4, 8, 16, 32],
        "replay_ratio": [0.1, 0.15, 0.2, 0.3],
        "domain_buffer_size": [20, 50, 100, 200],
    },
    "o_lora": {
        **COMMON_SPACE,
        "adapter_r": [4, 8, 16, 32],
    },
    "der": {
        **COMMON_SPACE,
        "adapter_r": [4, 8, 16, 32],
        "replay_ratio": [0.1, 0.15, 0.2, 0.3],
    },
    "dual_replay": {
        **COMMON_SPACE,
        "adapter_r": [4, 8, 16, 32],
        "replay_ratio": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        "domain_buffer_size": [20, 50, 100, 200],
        "general_buffer_size": [50, 100, 200, 500],
        "domain_replay_fraction": [0.05, 0.1, 0.15, 0.2],
        "unfreeze_top_k": [0, 1, 2],
    },
}


OVERRIDE_FILE = "configs/autoresearch_override.yaml"


def load_config(config_name="autoresearch"):
    with open("configs/default.yaml") as f:
        all_configs = yaml.safe_load(f)
    base = dict(all_configs[config_name])
    # Apply overrides if they exist
    if os.path.exists(OVERRIDE_FILE):
        with open(OVERRIDE_FILE) as f:
            overrides = yaml.safe_load(f) or {}
        base.update(overrides)
    return base


def save_config(config, config_name="autoresearch"):
    """Write overrides to a separate file, never touch default.yaml."""
    # Load the base to compute diff
    with open("configs/default.yaml") as f:
        all_configs = yaml.safe_load(f)
    base = dict(all_configs[config_name])
    overrides = {k: v for k, v in config.items() if config[k] != base.get(k)}
    if overrides:
        with open(OVERRIDE_FILE, "w") as f:
            yaml.dump(overrides, f, default_flow_style=False)
    elif os.path.exists(OVERRIDE_FILE):
        os.remove(OVERRIDE_FILE)


def run_experiment(method, config_name="autoresearch", seed=42, timeout=300):
    """Run one experiment, return (avg_f1, bwt) or None on failure."""
    cmd = [
        "uv", "run", "python", "scripts/run_experiment.py",
        "--method", method,
        "--config", config_name,
        "--seed", str(seed),
        "--output_dir", "results/autoresearch",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        # Parse results from output
        avg_f1 = None
        bwt = None
        for line in output.split("\n"):
            if "Avg F1:" in line:
                avg_f1 = float(line.split(":")[-1].strip())
            if "BWT:" in line:
                bwt = float(line.split(":")[-1].strip())
        if avg_f1 is not None and bwt is not None:
            return avg_f1, bwt, output
        else:
            return None, None, output
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    except Exception as e:
        return None, None, str(e)


def mutate_config(base_config, method, rng):
    """Mutate one random hyperparameter."""
    space = METHOD_SPACE.get(method, COMMON_SPACE)
    param = rng.choice(list(space.keys()))
    value = rng.choice(space[param])
    new_config = copy.deepcopy(base_config)
    new_config[param] = value
    return new_config, param, value


def log_result(tsv_path, commit, method, config_name, avg_f1, bwt, status, desc):
    """Append one row to results.tsv."""
    if not os.path.exists(tsv_path):
        with open(tsv_path, "w") as f:
            f.write("commit\tmethod\tconfig\tavg_f1\tbwt\tstatus\tdescription\n")
    with open(tsv_path, "a") as f:
        f1_str = f"{avg_f1:.1f}" if avg_f1 is not None else "0.0"
        bwt_str = f"{bwt:.1f}" if bwt is not None else "0.0"
        f.write(f"{commit}\t{method}\t{config_name}\t{f1_str}\t{bwt_str}\t{status}\t{desc}\n")


def get_short_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="Method to optimize")
    parser.add_argument("--time-budget", type=int, default=3600,
                        help="Total time budget in seconds (default: 1 hour)")
    parser.add_argument("--config", default="autoresearch",
                        help="Base config to modify")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-timeout", type=int, default=300,
                        help="Max seconds per experiment")
    args = parser.parse_args()

    tsv_path = "results/autoresearch_results.tsv"
    os.makedirs("results/autoresearch", exist_ok=True)

    rng = random.Random(args.seed)
    base_config = load_config(args.config)
    best_f1 = -1.0
    best_config = copy.deepcopy(base_config)

    t_start = time.time()
    experiment_num = 0

    print(f"=== Autoresearch: {args.method} ===")
    print(f"Time budget: {args.time_budget}s")
    print(f"Config: {args.config}")
    print()

    # --- Baseline ---
    print(f"[{experiment_num}] Running baseline...")
    avg_f1, bwt, output = run_experiment(
        args.method, args.config, args.seed, args.experiment_timeout
    )
    commit = get_short_commit()

    if avg_f1 is not None:
        best_f1 = avg_f1
        print(f"  Baseline: F1={avg_f1:.1f} BWT={bwt:.1f}")
        log_result(tsv_path, commit, args.method, args.config,
                   avg_f1, bwt, "keep", "baseline")
    else:
        print(f"  Baseline CRASHED: {output[-200:]}")
        log_result(tsv_path, commit, args.method, args.config,
                   None, None, "crash", "baseline crashed")
        return

    experiment_num += 1

    # --- Experiment loop ---
    while True:
        elapsed = time.time() - t_start
        remaining = args.time_budget - elapsed
        if remaining < args.experiment_timeout:
            print(f"\nTime budget exhausted ({elapsed:.0f}s elapsed).")
            break

        # Mutate config
        new_config, param, value = mutate_config(best_config, args.method, rng)
        desc = f"{param}={value}"
        print(f"\n[{experiment_num}] Trying {desc}... ", end="", flush=True)

        # Save mutated config
        save_config(new_config, args.config)

        # Run
        t_exp_start = time.time()
        avg_f1, bwt, output = run_experiment(
            args.method, args.config, args.seed, args.experiment_timeout
        )
        dt = time.time() - t_exp_start

        if avg_f1 is None:
            print(f"CRASH ({dt:.0f}s)")
            log_result(tsv_path, "-------", args.method, args.config,
                       None, None, "crash", desc)
            # Revert config
            save_config(best_config, args.config)
        elif avg_f1 > best_f1:
            improvement = avg_f1 - best_f1
            print(f"KEEP  F1={avg_f1:.1f} BWT={bwt:.1f} (+{improvement:.1f}) ({dt:.0f}s)")
            best_f1 = avg_f1
            best_config = copy.deepcopy(new_config)
            log_result(tsv_path, "-------", args.method, args.config,
                       avg_f1, bwt, "keep", desc)
        else:
            print(f"DISCARD F1={avg_f1:.1f} vs best={best_f1:.1f} ({dt:.0f}s)")
            log_result(tsv_path, "-------", args.method, args.config,
                       avg_f1, bwt, "discard", desc)
            # Revert config
            save_config(best_config, args.config)

        experiment_num += 1

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Autoresearch complete: {experiment_num} experiments in {time.time()-t_start:.0f}s")
    print(f"Best F1: {best_f1:.1f}")
    print(f"Best config for {args.method}:")
    for k, v in sorted(best_config.items()):
        print(f"  {k}: {v}")
    print(f"\nFull results in {tsv_path}")

    # Save best config back
    save_config(best_config, args.config)


if __name__ == "__main__":
    main()
