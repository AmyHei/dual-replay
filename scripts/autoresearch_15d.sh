#!/bin/bash
# Autoresearch on 15-domain proxy (bert-tiny, task-incremental eval)
# Goal: find hyperparams that match paper targets, then scale to clinc150

set -e
OUTDIR="results/autoresearch_15d"
mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/results.tsv"

# Initialize results TSV
if [ ! -f "$RESULTS" ]; then
    echo -e "method\tlr\tepochs\twarmup\treplay_ratio\tbuffer_size\tadapter_r\textra\tavg_f1\tbwt" > "$RESULTS"
fi

run_exp() {
    local method="$1"
    local desc="$2"
    shift 2

    # Build override args from remaining params
    local result
    result=$(uv run python scripts/run_experiment.py --method "$method" --config autoresearch_15d --seed 42 --output_dir "$OUTDIR" "$@" 2>&1 | grep -E "Avg F1|BWT")

    local f1=$(echo "$result" | grep "Avg F1" | awk '{print $NF}')
    local bwt=$(echo "$result" | grep "BWT" | awk '{print $NF}')

    echo -e "$method\t$desc\t$f1\t$bwt"
    echo -e "$method\t$desc\t$f1\t$bwt" >> "$RESULTS"
}

echo "========================================="
echo "Phase 1: sequential_ft LR sweep"
echo "========================================="

# Baseline already done: lr=5e-4, epochs=3 → F1=23.3
echo -e "sequential_ft\tlr=5e-4,ep=3 (baseline)\t23.3\t-77.3" >> "$RESULTS"

for lr in 1e-3 2e-3 5e-3 1e-4 5e-5; do
    echo "--- sequential_ft lr=$lr ---"
    # Temporarily modify config
    uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['learning_rate'] = float('$lr')
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
    run_exp sequential_ft "lr=$lr,ep=3"
done

# Reset LR to best and sweep epochs
echo ""
echo "========================================="
echo "Phase 2: sequential_ft epochs sweep (with best LR)"
echo "========================================="

# Find best LR from results
BEST_LR=$(uv run python3 -c "
import csv
best_f1, best_lr = 0, '5e-4'
with open('$RESULTS') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] == 'sequential_ft' and len(row) >= 4:
            try:
                f1 = float(row[2])
                if f1 > best_f1:
                    best_f1 = f1
                    best_lr = row[1].split(',')[0].replace('lr=','')
            except: pass
print(best_lr)
")
echo "Best LR so far: $BEST_LR"

uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['learning_rate'] = float('$BEST_LR')
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

for ep in 1 2 5 8 10; do
    echo "--- sequential_ft epochs=$ep ---"
    uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['epochs_per_domain'] = $ep
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
    run_exp sequential_ft "lr=$BEST_LR,ep=$ep"
done

echo ""
echo "========================================="
echo "Phase 3: replay methods with best base config"
echo "========================================="

# Reset epochs to best
BEST_EP=$(uv run python3 -c "
import csv
best_f1, best_ep = 0, '3'
with open('$RESULTS') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] == 'sequential_ft' and len(row) >= 4:
            try:
                f1 = float(row[2])
                if f1 > best_f1:
                    best_f1 = f1
                    desc = row[1]
                    for part in desc.split(','):
                        if 'ep=' in part:
                            best_ep = part.replace('ep=','')
            except: pass
print(best_ep)
")
echo "Best config: LR=$BEST_LR, epochs=$BEST_EP"

uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['learning_rate'] = float('$BEST_LR')
cfg['autoresearch_15d']['epochs_per_domain'] = int('$BEST_EP')
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

# Sweep replay_ratio for replay_only
for rr in 0.10 0.20 0.30 0.40 0.50; do
    echo "--- replay_only replay_ratio=$rr ---"
    uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['replay_ratio'] = float('$rr')
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
    run_exp replay_only "rr=$rr"
done

# Sweep replay + buffer for dual_replay
for rr in 0.20 0.30 0.40; do
    for buf in 50 100 200; do
        echo "--- dual_replay rr=$rr buf=$buf ---"
        uv run python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['autoresearch_15d']['replay_ratio'] = float('$rr')
cfg['autoresearch_15d']['domain_buffer_size'] = int('$buf')
with open('configs/default.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
        run_exp dual_replay "rr=$rr,buf=$buf"
    done
done

echo ""
echo "========================================="
echo "DONE. Results in $RESULTS"
echo "========================================="
cat "$RESULTS"
