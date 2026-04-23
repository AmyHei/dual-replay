#!/bin/bash
# Run all remaining reviewer experiments overnight.
# Usage: caffeinate -s bash scripts/run_all_reviewer_exps.sh 2>&1 | tee results/overnight_log.txt
#
# Estimated time on M1 Pro 32GB: ~10-12 hours total
# caffeinate -s keeps Mac awake on AC power even if you walk away.

set -e
cd "$(dirname "$0")/.."
mkdir -p results

SEEDS=(42 123 456 789 1024)
CONFIG="clinc150"

echo "============================================"
echo "Starting overnight experiment run"
echo "Time: $(date)"
echo "Config: $CONFIG"
echo "Seeds: ${SEEDS[*]}"
echo "============================================"

# -----------------------------------------------------------
# 1. EWC-LoRA baseline (R1 request) — if method exists
# -----------------------------------------------------------
if grep -q "ewc_lora" scripts/run_experiment.py 2>/dev/null; then
  echo ""
  echo "=== [1/5] EWC-LoRA baseline ==="
  for seed in "${SEEDS[@]}"; do
    echo "  Running ewc_lora seed=$seed ..."
    uv run python scripts/run_experiment.py --method ewc_lora --config "$CONFIG" --seed "$seed" --output_dir results
  done
else
  echo ""
  echo "=== [1/5] EWC-LoRA: method not implemented, skipping ==="
fi

# -----------------------------------------------------------
# 2. LoRA+Replay with higher rank r=64 (capacity-matched)
#    Need to add a clinc150_lora64 config or pass override
# -----------------------------------------------------------
echo ""
echo "=== [2/5] All baselines with clinc150 config (5 seeds each) ==="
for method in sequential_ft ewc lora_only replay_only lora_replay o_lora der dual_replay; do
  for seed in "${SEEDS[@]}"; do
    OUTFILE="results/${method}_${CONFIG}_seed${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "  $method seed=$seed already exists, skipping"
    else
      echo "  Running $method seed=$seed ..."
      uv run python scripts/run_experiment.py --method "$method" --config "$CONFIG" --seed "$seed" --output_dir results || echo "  FAILED: $method seed=$seed"
    fi
  done
done

# -----------------------------------------------------------
# 3. Buffer size sensitivity (R2/R3 Q4)
#    Run dual_replay with different buffer sizes
# -----------------------------------------------------------
echo ""
echo "=== [3/5] Buffer size sensitivity ==="
for bufsize in 50 100 500 1000; do
  for seed in "${SEEDS[@]}"; do
    OUTFILE="results/dual_replay_bufsize${bufsize}_seed${seed}.json"
    if [ -f "$OUTFILE" ]; then
      echo "  bufsize=$bufsize seed=$seed already exists, skipping"
    else
      echo "  Running dual_replay bufsize=$bufsize seed=$seed ..."
      # Override domain_buffer_size via env var or config; for now use the standard run
      # and manually vary. This requires the script to accept --domain_buffer_size.
      # Fallback: we'll create a small python wrapper.
      uv run python -c "
import sys, os, json, yaml
sys.path.insert(0, '.')
from scripts.run_experiment import load_config, get_method
from src.data.clinc150 import build_15_domain_protocol, get_general_buffer
from src.data.domain_sequence import generate_domain_orderings
from src.training.runner import SequentialRunner

config = load_config('clinc150')
config['domain_buffer_size'] = $bufsize
config['general_buffer_size'] = $bufsize  # keep 50:50

domains = build_15_domain_protocol(seed=$seed)
orderings = generate_domain_orderings(len(domains), num_orderings=1, seed=$seed)
ordered_domains = [domains[i] for i in orderings[0]][:config['num_domains']]

method = get_method('dual_replay', config)
if hasattr(method, 'fill_general_buffer'):
    gb = get_general_buffer(max_size=config['general_buffer_size'], seed=$seed)
    gb_clean = [{'text': ex['text'], 'label': -1} for ex in gb]
    method.setup()
    method.fill_general_buffer(gb_clean)
    method.setup = lambda: None

runner = SequentialRunner(method=method, domains=ordered_domains)
results = runner.run()

os.makedirs('results', exist_ok=True)
with open('$OUTFILE', 'w') as f:
    json.dump({'method': 'dual_replay', 'bufsize': $bufsize, 'seed': $seed,
               'avg_f1': results['avg_f1'], 'bwt': results['bwt'],
               'perf_matrix': results['perf_matrix'].tolist()}, f, indent=2)
print(f'  bufsize=$bufsize seed=$seed -> F1={results[\"avg_f1\"]:.1f} BWT={results[\"bwt\"]:.1f}')
" || echo "  FAILED: bufsize=$bufsize seed=$seed"
    fi
  done
done

# -----------------------------------------------------------
# 4. Soft vs hard routing analysis (R2/R3 Q3)
#    Post-hoc: re-run dual_replay inference with hard routing
# -----------------------------------------------------------
echo ""
echo "=== [4/5] Soft vs hard routing: post-hoc analysis ==="
echo "  (This requires modifying inference to use argmax instead of soft mixture."
echo "   Will be done as a separate analysis script. Skipping for now.)"

# -----------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "All experiments completed at $(date)"
echo "Results in results/"
echo "============================================"
ls -la results/*.json 2>/dev/null | wc -l | xargs echo "Total result files:"
