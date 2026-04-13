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
        uv run python scripts/run_experiment.py --method "$method" --config clinc150 --seed "$seed" $extra 2>&1 | tail -3
        echo ""
    done
done

echo "=== All runs complete ==="
echo ""
echo "Summary:"
uv run python3 -c "
import json, glob, numpy as np
from collections import defaultdict
results = defaultdict(list)
for f in sorted(glob.glob('results/*clinc150*.json') + glob.glob('results/clinc150/*clinc150*.json')):
    r = json.load(open(f))
    results[r['method']].append((r['avg_f1'], r['bwt']))
for method in ['sequential_ft','ewc','lora_only','o_lora','replay_only','lora_replay','der','dual_replay']:
    if method in results:
        f1s = [x[0] for x in results[method]]
        bwts = [x[1] for x in results[method]]
        n = len(f1s)
        print(f'{method:20s}  n={n}  F1={np.mean(f1s):5.1f}±{np.std(f1s):4.1f}  BWT={np.mean(bwts):+6.1f}±{np.std(bwts):4.1f}')
"
