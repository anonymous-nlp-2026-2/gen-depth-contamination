#!/bin/bash
# exp-026: arXiv Graduated vs Binary Filtering
# 3 strategies on 3 GPUs (cuda:0-2), 5 seeds each
# Uses CUDA_VISIBLE_DEVICES to isolate each process to a single GPU

set -e
cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/bin/activate

SEEDS="42 123 456 789 1024"
SCRIPT="scripts/exp026_arxiv_filtering.py"

run_strategy() {
    local strategy=$1
    local gpu_id=$2
    echo "[$(date '+%H:%M:%S')] Starting strategy=$strategy on GPU $gpu_id"
    for seed in $SEEDS; do
        echo "[$(date '+%H:%M:%S')] Running $strategy seed=$seed on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python $SCRIPT --strategy $strategy --seed $seed --device cuda:0
    done
    echo "[$(date '+%H:%M:%S')] Finished strategy=$strategy on GPU $gpu_id"
}

run_strategy no_filter 0 &
PID_NO_FILTER=$!

run_strategy binary 1 &
PID_BINARY=$!

run_strategy graduated_a09 2 &
PID_GRADUATED=$!

echo "Launched 3 strategies:"
echo "  no_filter     GPU 0  PID=$PID_NO_FILTER"
echo "  binary        GPU 1  PID=$PID_BINARY"
echo "  graduated_a09 GPU 2  PID=$PID_GRADUATED"

wait $PID_NO_FILTER
STATUS_NF=$?
echo "[$(date '+%H:%M:%S')] no_filter exited with $STATUS_NF"

wait $PID_BINARY
STATUS_BIN=$?
echo "[$(date '+%H:%M:%S')] binary exited with $STATUS_BIN"

wait $PID_GRADUATED
STATUS_GRAD=$?
echo "[$(date '+%H:%M:%S')] graduated_a09 exited with $STATUS_GRAD"

echo "All done. Exit codes: no_filter=$STATUS_NF binary=$STATUS_BIN graduated=$STATUS_GRAD"
