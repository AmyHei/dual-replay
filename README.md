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

### Phase 2: Full clinc150 runs (bert-large, 15 domains) — 8/40

| Method | Seeds Done | Current F1 | Target F1 | Target BWT | Status |
|--------|-----------|------------|-----------|------------|--------|
| Sequential FT | 5/5 | 25.8 ± 2.1 | 79.3 ± 2.1 | -21.4 ± 3.0 | **Needs LR fix** |
| EWC | 2/5 | 27.0 ± 0.1 | 81.2 | -15.1 | **Needs LR fix** |
| LoRA-Only | 0/5 | — | 83.1 ± 1.5 | -12.8 ± 2.1 | Not started |
| Replay-Only | 0/5 | — | 84.0 ± 1.4 | -10.5 ± 1.8 | Not started |
| LoRA+Replay | 0/5 | — | 84.7 ± 1.3 | -8.9 ± 1.6 | Not started |
| O-LoRA | 0/5 | — | 84.2 ± 1.2 | -7.5 ± 1.4 | Not started |
| DER (LoRA) | 0/5 | — | 85.3 ± 1.3 | -7.8 ± 1.5 | Not started |
| **Dual-Replay** | **1/5** | **88.3** | **89.1 ± 1.0** | **-5.2 ± 1.0** | **On target** |

**Known issue**: Full-FT baselines (sequential_ft, ewc) get ~25-27 F1 on bert-large, far below targets.
The proxy→full LR scaling (÷8) may be too aggressive for full-FT methods. Dual-Replay (adapter-based) scales correctly.

### Next Steps

1. **[BLOCKER] Fix full-FT baseline LR**: The `1.5e-5` LR for sequential_ft/ewc on bert-large is too low.
   Try `2e-5` to `5e-5` (standard BERT-large fine-tuning range). Run single seed to validate.
2. **Re-run sequential_ft & ewc** with corrected LR (5 seeds each).
3. **Run remaining methods**: lora_only, replay_only, lora_replay, o_lora, der (5 seeds each).
4. **Complete dual_replay**: seeds 123, 456, 789, 1024.
5. **Phase 3**: T5-base validation (after Phase 2 is complete).

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
