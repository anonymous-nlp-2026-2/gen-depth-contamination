#!/bin/bash
# exp-009: Launch dose-response experiments on 3 GPUs in parallel
# cuda:1 = 10%, cuda:2 = 30%, cuda:3 = 50%

set -e
cd /root/autodl-tmp/gen-depth-contamination

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting exp-009 dose-response experiments..."

# Launch 3 ratios in parallel
python scripts/exp009_dose_response.py --ratio 0.1 --device cuda:1 --num_seeds 5 \
    > "$LOG_DIR/exp_009_dose_response_10.log" 2>&1 &
PID1=$!

python scripts/exp009_dose_response.py --ratio 0.3 --device cuda:2 --num_seeds 5 \
    > "$LOG_DIR/exp_009_dose_response_30.log" 2>&1 &
PID2=$!

python scripts/exp009_dose_response.py --ratio 0.5 --device cuda:3 --num_seeds 5 \
    > "$LOG_DIR/exp_009_dose_response_50.log" 2>&1 &
PID3=$!

echo "PIDs: 10%=$PID1, 30%=$PID2, 50%=$PID3"
echo "Logs: $LOG_DIR/exp_009_dose_response_{10,30,50}.log"

wait $PID1 $PID2 $PID3
echo "[$(date)] All experiments complete!"
