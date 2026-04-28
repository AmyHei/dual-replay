#!/bin/bash
# Phase 4: bert-base × 3 benchmarks × 5 seeds × 8 methods (class-incremental).
# Resumable: skips runs whose JSON output already exists.
# Writes per-run JSON to results/<benchmark>/<method>_<config>_seed<N>.json
# and a flat log to results/phase4.log

set -e
cd "$(dirname "$0")/.."

BENCHMARKS="${BENCHMARKS:-clinc150_10 hwu64 banking77}"
SEEDS="${SEEDS:-42 123 456 789 1024}"
METHODS="${METHODS:-sequential_ft ewc lora_only o_lora replay_only lora_replay der dual_replay}"
LOG="results/phase4.log"

mkdir -p results
echo "=== Phase 4 start: $(date) ===" | tee -a "$LOG"
echo "benchmarks=$BENCHMARKS" | tee -a "$LOG"
echo "seeds=$SEEDS" | tee -a "$LOG"
echo "methods=$METHODS" | tee -a "$LOG"

for bench in $BENCHMARKS; do
    outdir="results/$bench"
    mkdir -p "$outdir"
    for method in $METHODS; do
        for seed in $SEEDS; do
            outfile="$outdir/${method}_${bench}_seed${seed}.json"
            if [ -f "$outfile" ]; then
                echo "SKIP $bench $method seed=$seed (exists)" | tee -a "$LOG"
                continue
            fi
            echo "=== $(date '+%F %T') $bench $method seed=$seed ===" | tee -a "$LOG"
            t0=$(date +%s)
            uv run python scripts/run_experiment.py \
                --method "$method" --config "$bench" --seed "$seed" \
                --output_dir "$outdir" 2>&1 | tee -a "$LOG"
            t1=$(date +%s)
            echo "elapsed: $((t1 - t0))s" | tee -a "$LOG"
            echo "" | tee -a "$LOG"
        done
    done
done

echo "=== Phase 4 done: $(date) ===" | tee -a "$LOG"
