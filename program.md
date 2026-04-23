# autoresearch: Dual-Replay Hyperparameter Optimization

Adapted from Karpathy's autoresearch for continual learning experiments.

## Setup

1. **Agree on a run tag** with user (e.g. `mar20`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `program.md` — this file, your instructions.
   - `configs/default.yaml` — hyperparameters you modify.
   - `src/methods/` — method implementations (read, understand, but modify sparingly).
   - `scripts/run_experiment.py` — the runner, do not modify.
4. **Initialize results.tsv**: create with header row. First run is the baseline.
5. **Confirm and go**.

## What You Optimize

**Goal**: Maximize Avg F1 while minimizing |BWT| (less forgetting) for each method.

**Target methods** (in priority order):
1. `sequential_ft` — simplest baseline, tune first to validate the pipeline
2. `ewc` — EWC lambda, Fisher samples, learning rate
3. `lora_only` — adapter rank, learning rate
4. `replay_only` — replay buffer size, learning rate
5. `lora_replay` — combination of LoRA + replay params
6. `dual_replay` — the main method, most important to get right

**Target numbers** (from the paper, CLINC150 with bert-large, 15 domains):

| Method | Avg F1 | BWT |
|--------|--------|-----|
| Sequential FT | 79.3 | -21.4 |
| EWC | 81.2 | -15.1 |
| LoRA-Only | 83.1 | -12.8 |
| Replay-Only | 84.0 | -10.5 |
| LoRA+Replay | 84.7 | -8.9 |
| O-LoRA | 84.2 | -7.5 |
| DER | 85.3 | -7.8 |
| Dual-Replay | 89.1 | -5.2 |

## Time Budget

- **Proxy config** (`autoresearch`): bert-tiny, 5 domains, ~30-45s per experiment.
- **Validation config** (`debug`): bert-tiny, 3 domains, ~40s — use to cross-check.
- **Full config** (`clinc150`): bert-large, 15 domains — only run when proxy shows a promising config.

Use the proxy for fast iteration. Run clinc150 only for the final "blessed" config.

## Hyperparameter Search Space

These are the knobs you can turn in `configs/default.yaml`:

| Parameter | Range | Notes |
|-----------|-------|-------|
| learning_rate | 1e-5 to 1e-3 | Model-dependent. bert-tiny tolerates higher LR. |
| warmup_ratio | 0.0 to 0.3 | 0.1 is standard for BERT |
| epochs_per_domain | 1 to 10 | More epochs = more training, but slower |
| batch_size | 4 to 64 | With gradient_accumulation_steps |
| gradient_accumulation_steps | 1 to 8 | Effective batch = batch_size * grad_accum |
| ewc_lambda | 100 to 50000 | EWC regularization strength |
| ewc_fisher_samples | 50 to 1000 | Samples for Fisher diagonal estimation |
| adapter_r | 4 to 64 | LoRA/adapter rank |
| replay_ratio | 0.05 to 0.5 | Fraction of replay data mixed in |
| domain_replay_fraction | 0.05 to 0.3 | Fraction of buffer for domain-specific replay |
| domain_buffer_size | 20 to 500 | Per-domain replay buffer size |
| general_buffer_size | 50 to 2000 | General replay buffer size |

## Running an Experiment

```bash
# Proxy (fast, ~30s)
uv run python scripts/run_experiment.py --method METHOD --config autoresearch --seed 42 --output_dir results/autoresearch > run.log 2>&1

# Extract results
grep "Avg F1\|BWT" run.log
```

## Logging

Log to `results.tsv` (tab-separated):

```
commit	method	config	avg_f1	bwt	status	description
```

Example:
```
commit	method	config	avg_f1	bwt	status	description
a1b2c3d	sequential_ft	autoresearch	45.2	-38.1	keep	baseline lr=5e-4
b2c3d4e	sequential_ft	autoresearch	52.7	-35.2	keep	lr=1e-3 warmup=0.1
c3d4e5f	sequential_ft	autoresearch	12.3	-0.5	discard	lr=1e-2 (collapsed)
```

## The Experiment Loop

LOOP FOREVER:

1. Look at `results.tsv` — what has been tried, what worked.
2. Pick a method and a hypothesis (e.g. "higher LR for bert-tiny", "more EWC lambda").
3. Modify `configs/default.yaml` (autoresearch section only).
4. `git commit -m "autoresearch: METHOD try DESCRIPTION"`
5. Run: `uv run python scripts/run_experiment.py --method METHOD --config autoresearch --seed 42 --output_dir results/autoresearch > run.log 2>&1`
6. Read results: `grep "Avg F1\|BWT" run.log`
7. If crashed: `tail -50 run.log`, fix, retry.
8. Log to `results.tsv`.
9. If improved: keep commit, advance branch.
10. If worse: `git reset --soft HEAD~1` to revert config.

**NEVER STOP** until human interrupts. If out of ideas, try:
- Combine top-performing settings from different methods
- Ablate: remove one improvement at a time to find what really matters
- Try extreme values to understand boundaries
- Scale validated proxy settings to clinc150

## Strategy Tips

1. **Start with sequential_ft** — it has no CL-specific params, so tuning LR/warmup/epochs gives you a clean signal.
2. **One variable at a time** — change one param, measure, decide.
3. **Proxy → full transfer**: once proxy config is stable, scale to clinc150 by adjusting LR for model size (bert-tiny LR ~10x higher than bert-large).
4. **The scaling rule**: `lr_large ≈ lr_tiny * (dim_tiny / dim_large)` where dim_tiny=128, dim_large=1024.
5. **Watch for collapse**: F1 < 5% means model collapsed to single-class prediction. Reduce LR.
6. **EWC is memory-hungry**: on Mac, keep fisher_samples low (100-500) and use online consolidation.
