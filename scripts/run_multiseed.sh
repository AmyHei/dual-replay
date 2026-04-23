#!/bin/bash
# Run all 8 methods x 5 seeds for statistical significance.
# Methods are sorted by expected speed (fastest first).
# Results saved to results/<method>_clinc150_seed<N>.json
# For dual_replay, results go to results/clinc150/

set -e

SEEDS="42 123 456 789 1024"
METHODS="sequential_ft ewc lora_only o_lora replay_only lora_replay der dual_replay"

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
            echo "SKIP $method seed=$seed (already exists: $outfile)"
            continue
        fi

        echo "=== Running $method seed=$seed ==="
        uv run python scripts/run_experiment.py --method "$method" --config clinc150 --seed "$seed" $extra 2>&1
        echo ""
    done
done

echo "=== All runs complete ==="
