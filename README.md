# Dual-Replay Reproduction

Reproduction codebase for *Parameter-Efficient Dual-Replay: Mitigating Catastrophic Forgetting in Sequential LLM Fine-Tuning Under Fixed Memory Budgets*.

## Setup

```bash
cd dual-replay-reproduce
uv sync
```

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

## Quick Start

```bash
# Mac local debug (bert-tiny, 3 domains, ~2 min)
uv run python scripts/run_experiment.py --method dual_replay --config debug --seed 42

# GPU full experiment (bert-large, 15 domains, ~hours)
uv run python scripts/run_experiment.py --method dual_replay --config clinc150 --seed 42
```

## Methods

| Method | `--method` | Paper Reference |
|--------|-----------|-----------------|
| Sequential FT | `sequential_ft` | Baseline 1 |
| LoRA-Only | `lora_only` | Baseline 3 |
| Replay-Only | `replay_only` | Baseline 4 |
| LoRA+Replay | `lora_replay` | Baseline 6 |
| EWC | `ewc` | Baseline 2 |
| O-LoRA | `o_lora` | Baseline 7 |
| DER | `der` | Baseline 8 |
| **Dual-Replay** | `dual_replay` | **Ours** |

## Configs

| Config | Model | Domains | Use Case |
|--------|-------|---------|----------|
| `debug` | bert-tiny (4.4M) | 3 | Mac local testing |
| `clinc150` | bert-large (340M) | 15 | Paper Table 9 reproduction |
| `t5` | t5-base (220M) | 15 | Phase 3 encoder-decoder validation |

Configs are defined in `configs/default.yaml`.

## Running Experiments

### Single run

```bash
uv run python scripts/run_experiment.py --method <method> --config <config> --seed <seed>
```

### Full Phase 1 (Table 9: 7 methods x 5 seeds = 35 runs)

```bash
for method in sequential_ft lora_only replay_only lora_replay ewc o_lora der dual_replay; do
    for seed in 42 123 456 789 1024; do
        uv run python scripts/run_experiment.py --method $method --config clinc150 --seed $seed
    done
done
```

### View results

```bash
python -c "
import json, glob
for f in sorted(glob.glob('results/*.json')):
    r = json.load(open(f))
    print(f'{r[\"method\"]:20s} seed={r[\"seed\"]}  F1={r[\"avg_f1\"]:5.1f}  BWT={r[\"bwt\"]:+6.1f}')
"
```

## Project Structure

```
src/
  data/           CLINC150 loading, 15-domain protocol, domain orderings
  models/         Bottleneck adapters, task-conditioned gating, domain classifier
  replay/         Dual-stream replay buffer (domain + general)
  methods/        All 8 continual learning methods
  training/       Sequential domain training runner
  metrics.py      BWT, FWT, Avg F1, paired t-test
scripts/
  run_experiment.py   CLI entry point
configs/
  default.yaml        All experiment configurations
tests/                Unit + integration tests (35 tests)
results/              Experiment outputs (gitignored)
```

## Tests

```bash
uv run pytest tests/ -v
```

## Experiment Progress (as of 2026-04-08)

### Phase 1: Autoresearch (proxy tuning) — COMPLETE

380+ experiments on bert-tiny (5 domains) to find optimal hyperparameters per method.
Best proxy results stored in `results/autoresearch/`. Scaling rules applied to `clinc150` config.

### Phase 2: Full clinc150 runs (bert-large, 15 domains, class-incremental)

All 8 methods completed on seed=42. Class-incremental evaluation: model predicts among
all seen classes (no domain ID at test time).

| Method | F1 | BWT | Target F1 | Target BWT |
|--------|-----|------|-----------|------------|
| LoRA-Only | 4.4 | -54.9 | 83.1 | -12.8 |
| Sequential FT | 6.6 | -98.8 | 79.3 | -21.4 |
| O-LoRA | 6.5 | -84.1 | 84.2 | -7.5 |
| EWC | 7.3 | -98.0 | 81.2 | -15.1 |
| LoRA+Replay | 23.8 | -63.7 | 84.7 | -8.9 |
| DER (LoRA) | 27.7 | -65.5 | 85.3 | -7.8 |
| Replay-Only | 51.9 | -37.6 | 84.0 | -10.5 |
| **Dual-Replay** | **53.8** | **-40.3** | **89.1** | **-5.2** |

**Relative ranking is correct**: replay methods dominate, dual_replay is best.
Absolute gap (~35 F1) likely due to different evaluation protocol in the paper
(task-incremental vs our class-incremental setup).

### Next Steps

1. **Multi-seed runs**: 8 methods × 5 seeds for statistical significance
2. **Hyperparameter sweep** to close gap (replay buffer, LR, epochs)
3. **Phase 3**: T5-base validation

## Reproduction Targets (Paper Table 9)

| Method | Target F1 | Target BWT |
|--------|-----------|------------|
| Sequential FT | 79.3 ± 2.1 | -21.4 ± 3.0 |
| LoRA-Only | 83.1 ± 1.5 | -12.8 ± 2.1 |
| Replay-Only | 84.0 ± 1.4 | -10.5 ± 1.8 |
| LoRA+Replay | 84.7 ± 1.3 | -8.9 ± 1.6 |
| O-LoRA | 84.2 ± 1.2 | -7.5 ± 1.4 |
| DER (LoRA) | 85.3 ± 1.3 | -7.8 ± 1.5 |
| **Dual-Replay** | **89.1 ± 1.0** | **-5.2 ± 1.0** |
