#!/bin/bash
# Run ewc, lora_replay, der on clinc150 with latest autoresearch-tuned configs.
# 3 methods x 5 seeds = 15 runs. Sequentially one by one.

set -e
cd "$(dirname "$0")/.."

SEEDS="42 123 456 789 1024"
METHODS="ewc lora_replay der"
LOG="results/three_methods.log"

echo "=== Start: $(date) ===" | tee -a "$LOG"

for method in $METHODS; do
    for seed in $SEEDS; do
        outfile="results/${method}_clinc150_seed${seed}.json"
        if [ -f "$outfile" ]; then
            echo "SKIP $method seed=$seed (exists)" | tee -a "$LOG"
            continue
        fi
        echo "=== $(date '+%H:%M:%S') Running $method seed=$seed ===" | tee -a "$LOG"
        uv run python scripts/run_experiment.py \
            --method "$method" --config clinc150 --seed "$seed" 2>&1 | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    done
done

echo "=== Done: $(date) ===" | tee -a "$LOG"
