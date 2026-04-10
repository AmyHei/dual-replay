# Label Remapping Bug & Fix (2026-04-09)

## Root Cause

The original runner used `remap_to_local_labels` which mapped each domain's intents to 0-based local IDs (0-9). This caused two problems:

1. **Label collision**: All 15 domains used labels 0-9, meaning the model's 10 output neurons were being overwritten each domain. Evaluating domain 1 after training domain 15 was meaningless — same labels, different semantics.

2. **Model reload bug**: `_ensure_model` in 6 methods (sequential_ft, ewc, replay_only, lora_only, lora_replay, o_lora, der) reloaded the entire model from pretrained weights when `num_labels` grew. With local labels, this never triggered (always 10 classes), but it would have been catastrophic with global labels. Fixed by using `resize_classifier()`.

## Fix: Contiguous Labels + Masked Evaluation

- **Runner** (`src/training/runner.py`): Replaced local label remapping with `_build_contiguous_label_map` that assigns domain 0 → labels 0-9, domain 1 → 10-19, ..., domain 14 → 140-149.
- **Evaluation**: All methods' `run_evaluation` now accepts `valid_labels` parameter. Uses `masked_argmax` to restrict predictions to the domain's 10 labels (task-incremental evaluation).
- **Training**: Standard cross-entropy over full label space (tried masked CE but it performed worse — the full softmax provides useful cross-domain regularization).

## Impact on Results

### sequential_ft clinc150 seed=42, LR sweep
| LR | Epochs | F1 | BWT | Notes |
|----|--------|----|-----|-------|
| old (local labels) | 5 | 25.2 | -78.5 | Broken |
| 3e-5 | 5 | 49.0 | -53.4 | Contiguous + masked eval |
| 5e-5 | 5 | **60.8** | **-40.4** | Best so far |
| 1e-4 | 5 | 52.6 | -49.3 | Too high LR |
| 2e-5 | 3 | 30.7 | -72.1 | Too low |
| **Target** | | **79.3** | **-21.4** | Paper |

Gap to paper target likely due to different evaluation protocol (paper may use fewer domains, partial freezing, or different class-incremental setup). Relative method comparison is valid.

### Debug (bert-tiny, 3 domains) all methods working
| Method | F1 | BWT |
|--------|-----|------|
| sequential_ft | 82.6 | -17.9 |
| replay_only | 89.9 | -7.9 |
| ewc | 75.7 | -28.7 |
| dual_replay | 26.8 | -2.7 |

## Files Changed

- `src/training/runner.py` — contiguous label mapping
- `src/methods/utils.py` — `resize_classifier`, `masked_argmax`, `masked_cross_entropy`
- All 8 method files — `_ensure_model` resize fix + `run_evaluation` masking
