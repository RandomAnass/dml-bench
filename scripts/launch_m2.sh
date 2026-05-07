#!/bin/bash
# ============================================================================
# M2 — Extrapolation split launcher
# ============================================================================
# Splits 360 cells across two GPUs in shard mode. Each shard runs in its own
# nohup process with structured logging, matching launch_full_rerun.sh style.
#
# Usage:
#   chmod +x scripts/launch_m2.sh
#   nohup scripts/launch_m2.sh > logs/m2_master.log 2>&1 &
#
# Or smoke test only (foreground, 4 cells):
#   scripts/launch_m2.sh smoke
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate dml-bench-env

# Thread caps (matches scripts/launch_full_rerun.sh and scripts/rerun_vanilla_autodiff.py).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p logs results/extrapolation_M2

if [[ "$1" == "smoke" ]]; then
    echo "[m2] smoke test (4 cells, foreground)"
    CUDA_VISIBLE_DEVICES=0 python experiments/extrapolation/run_m2.py --smoke
    exit $?
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/m2_${TS}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "M2 START: $(date)"
echo "Git: $(git rev-parse HEAD 2>/dev/null || echo 'no git')"
echo "Logs: $LOG_DIR"
echo "============================================"

# Shard 0 → GPU 0; shard 1 → GPU 1. Each shard processes ~half of 360 cells.
CUDA_VISIBLE_DEVICES=0 nohup python experiments/extrapolation/run_m2.py \
    --shard 0 --n-shards 2 \
    > "$LOG_DIR/shard0_gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 nohup python experiments/extrapolation/run_m2.py \
    --shard 1 --n-shards 2 \
    > "$LOG_DIR/shard1_gpu1.log" 2>&1 &
PID1=$!

echo "Shard 0 (GPU 0): PID=$PID0  log=$LOG_DIR/shard0_gpu0.log"
echo "Shard 1 (GPU 1): PID=$PID1  log=$LOG_DIR/shard1_gpu1.log"

# Wait for both — || true so set -e doesn't kill the script if one shard
# returns non-zero, and the second shard's status still gets logged.
wait "$PID0" || true
RC0=$?
wait "$PID1" || true
RC1=$?
echo "============================================"
echo "M2 DONE: $(date)  shard0_rc=$RC0  shard1_rc=$RC1"
echo "============================================"
