#!/bin/bash
# Compare dual_replay vs replay_only on clinc150 (bert-large, 15 domains)
# Uses latest autoresearch best configs (Apr-16 run) scaled for bert-large.
# 5 seeds x 2 methods = 10 runs.

set -e
cd "$(dirname "$0")/.."

SEEDS="42 123 456 789 1024"
METHODS="replay_only dual_replay"
LOG="results/dual_vs_single.log"

echo "=== Start: $(date) ===" | tee -a "$LOG"

for method in $METHODS; do
    for seed in $SEEDS; do
        if [ "$method" = "dual_replay" ]; then
            outfile="results/clinc150/dual_replay_clinc150_seed${seed}.json"
            extra="--output_dir results/clinc150"
        else
            outfile="results/${method}_clinc150_seed${seed}.json"
            extra=""
        fi

        if [ -f "$outfile" ]; then
            echo "SKIP $method seed=$seed (exists)" | tee -a "$LOG"
            continue
        fi

        echo "=== $(date '+%H:%M:%S') Running $method seed=$seed ===" | tee -a "$LOG"
        uv run python scripts/run_experiment.py \
            --method "$method" --config clinc150 --seed "$seed" $extra 2>&1 | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    done
done

echo "=== Done: $(date) ===" | tee -a "$LOG"
