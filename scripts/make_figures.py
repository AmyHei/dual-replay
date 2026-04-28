"""Generate paper figures from Phase 4 / 5 result JSONs.

Outputs to paper/figures/:
  fig1_main_comparison.{png,pdf}      — bar chart of mean F1 by method per benchmark
  fig2_hwu64_per_domain.{png,pdf}     — HWU64 per-domain F1 sorted by train size
  fig3_hwu64_forgetting.{png,pdf}     — forgetting trajectory for a head domain
"""
from __future__ import annotations
import json
import os
import sys
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from src.data import build_benchmark  # noqa: E402

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})

METHOD_ORDER = ["sequential_ft", "ewc", "lora_only", "lora_replay",
                "der", "replay_only", "dual_replay"]
METHOD_LABEL = {
    "sequential_ft": "Sequential FT",
    "ewc": "EWC",
    "lora_only": "LoRA-Only",
    "lora_replay": "LoRA+Replay",
    "der": "DER",
    "replay_only": "Replay-Only",
    "dual_replay": "Dual-Replay (ours)",
}
METHOD_COLOR = {
    "sequential_ft": "#9e9e9e",
    "ewc": "#bdbdbd",
    "lora_only": "#cccccc",
    "lora_replay": "#4c78a8",
    "der": "#9b59b6",
    "replay_only": "#2ca02c",
    "dual_replay": "#d62728",
}
BENCHES = [
    ("clinc150_10", "CLINC150 (10 dom., balanced)"),
    ("hwu64", "HWU64 (18 dom., long-tail)"),
    ("banking77", "Banking77 (7 tasks, balanced)"),
]


def load_method_f1s(bench: str, method: str) -> list[float]:
    f1s = []
    for path in sorted(glob.glob(str(RESULTS / bench / f"{method}_{bench}_seed*.json"))):
        try:
            f1s.append(json.load(open(path))["avg_f1"])
        except Exception:
            pass
    return f1s


# ============================================================
# Figure 1: main 3-panel comparison
# ============================================================
def fig1_main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    for ax, (bench, title) in zip(axes, BENCHES):
        means, stds, colors, labels = [], [], [], []
        for m in METHOD_ORDER:
            f1s = load_method_f1s(bench, m)
            if not f1s:
                continue
            means.append(np.mean(f1s))
            stds.append(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
            colors.append(METHOD_COLOR[m])
            labels.append(METHOD_LABEL[m])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
               linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Avg F1 (class-incremental)" if bench == "clinc150_10" else "")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(means) * 1.15 + 5)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.suptitle("Phase 4 main results (BERT-base, n=5 seeds)", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_main_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] saved to {OUT}/fig1_main_comparison.{{png,pdf}}")


# ============================================================
# Figure 2: HWU64 per-domain F1, sorted by train size
# ============================================================
def fig2_hwu64_per_domain():
    doms, _ = build_benchmark("hwu64", seed=42)
    n_int = {d["domain_id"]: len(d["intents"]) for d in doms}
    n_train = {d["domain_id"]: len(d["train"]) for d in doms}
    name = {d["domain_id"]: d["domain_name"] for d in doms}

    def collect(method):
        rows = []
        for path in sorted(glob.glob(str(RESULTS / "hwu64" / f"{method}_hwu64_seed*.json"))):
            try:
                m = np.array(json.load(open(path))["perf_matrix"])
                rows.append(m[-1])  # final-step F1 per domain
            except Exception:
                pass
        return np.stack(rows) if rows else None

    methods_to_plot = [("dual_replay", METHOD_COLOR["dual_replay"]),
                       ("lora_replay", METHOD_COLOR["lora_replay"]),
                       ("der", METHOD_COLOR["der"])]

    sorted_doms = sorted(range(18), key=lambda d: n_train[d])
    x = np.arange(len(sorted_doms))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.0))
    for i, (m, c) in enumerate(methods_to_plot):
        rows = collect(m)
        means = [rows[:, d].mean() for d in sorted_doms]
        stds = [rows[:, d].std(ddof=1) for d in sorted_doms]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=METHOD_LABEL[m],
               color=c, edgecolor="black", linewidth=0.4, capsize=2,
               error_kw={"linewidth": 0.6, "alpha": 0.5})

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name[d]}\n({n_int[d]}-int, n={n_train[d]})" for d in sorted_doms],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_ylabel("Final-step F1 (class-incremental, mean ± std)")
    ax.set_title("HWU64 per-domain final F1 (sorted by train size, smallest left)", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.axvline(3.5, color="black", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.text(3.5, ax.get_ylim()[1] * 0.93, "  tail | head  ",
            ha="center", fontsize=8, color="gray", alpha=0.7)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_hwu64_per_domain.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] saved to {OUT}/fig2_hwu64_per_domain.{{png,pdf}}")


# ============================================================
# Figure 3: HWU64 forgetting trajectory for a head domain
# ============================================================
def fig3_forgetting_trajectory():
    """Track F1 of a single domain (e.g. music or qa) through every training step."""
    doms, _ = build_benchmark("hwu64", seed=42)
    name = {d["domain_id"]: d["domain_name"] for d in doms}
    # Pick a representative head domain. Use 'music' (was trained early in seed=42 ordering, big regression).
    target_name = "music"
    target_idx_in_seed42 = next(d["domain_id"] for d in doms if d["domain_name"] == target_name)

    methods_to_plot = [("dual_replay", METHOD_COLOR["dual_replay"]),
                       ("lora_replay", METHOD_COLOR["lora_replay"]),
                       ("der", METHOD_COLOR["der"])]

    fig, ax = plt.subplots(figsize=(7, 3.6))

    K = 18
    for m, c in methods_to_plot:
        # Average trajectory across seeds where this domain was at the same position.
        # Simpler: just look at seed 42 perf_matrix.
        path = RESULTS / "hwu64" / f"{m}_hwu64_seed42.json"
        if not path.exists():
            continue
        perf = np.array(json.load(open(path))["perf_matrix"])
        # F1 of `music` (target_idx_in_seed42) across training steps k = target_idx .. K-1
        steps = list(range(target_idx_in_seed42, K))
        traj = perf[steps, target_idx_in_seed42]
        ax.plot(steps, traj, marker="o", color=c, label=METHOD_LABEL[m], linewidth=1.5, markersize=5)

    ax.axvline(target_idx_in_seed42, color="black", alpha=0.3, linestyle=":", linewidth=0.8)
    ax.text(target_idx_in_seed42 + 0.1, ax.get_ylim()[1] * 0.05,
            f"trained '{target_name}' here", fontsize=8, color="gray")
    ax.set_xlabel(f"Training step k (after which step's training)")
    ax.set_ylabel(f"F1 on '{target_name}' test set")
    ax.set_title(f"HWU64: '{target_name}' F1 trajectory after training each subsequent domain (seed=42)",
                 fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_hwu64_forgetting.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved to {OUT}/fig3_hwu64_forgetting.{{png,pdf}}")


if __name__ == "__main__":
    fig1_main()
    fig2_hwu64_per_domain()
    fig3_forgetting_trajectory()
    print("done")
