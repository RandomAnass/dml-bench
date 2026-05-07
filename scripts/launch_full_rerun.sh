#!/bin/bash
# ============================================================================
# FULL BENCHMARK RE-RUN
# ============================================================================
# Purpose: Re-run entire DML-Bench with corrected trainer.py (autodiff eval)
# Date: 2026-04-06
# Reason: FIX-5 corrected vanilla gradient evaluation from zeros to autodiff
#         (matching original Huge & Savine 2020). All results need fresh numbers.
#
# Usage:
#   chmod +x scripts/launch_full_rerun.sh
#   nohup scripts/launch_full_rerun.sh > logs/full_rerun_master.log 2>&1 &
#
# Hardware: 2x NVIDIA RTX A6000 (49 GB each), ~48 CPUs
# Estimated time: 60-80 hours total with 2 GPUs
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Record environment
echo "============================================"
echo "FULL RE-RUN STARTED: $(date)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
echo "Python: $(conda run -n dml-bench-env python3 --version 2>&1)"
echo "PyTorch: $(conda run -n dml-bench-env python3 -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA: $(conda run -n dml-bench-env python3 -c 'import torch; print(torch.version.cuda)' 2>&1)"
echo "============================================"

# Thread controls
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# ============================================================================
# Step 1: Archive old results
# ============================================================================
echo ""
echo "Step 1: Archiving old results..."
BACKUP_DIR="results/_pre_autodiff_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for dir in tier1_benchmark tier2_benchmark tier3_benchmark tier4_benchmark \
           unified_comparison spy_options spy_options_temporal spy_options_purged_cv \
           warmup_experiments compute_matched_controls realworld \
           lrm_comparison gradient_noise_sweep gradnorm_fix \
           lambda_j_ablation higher_order_dml learned_derivatives gnn_md17; do
    if [ -d "results/$dir" ]; then
        cp -r "results/$dir" "$BACKUP_DIR/"
        echo "  Backed up: $dir"
    fi
done
echo "Backup complete: $BACKUP_DIR"

# Clear existing results for fresh run
for dir in tier1_benchmark tier2_benchmark tier3_benchmark tier4_benchmark; do
    rm -f results/$dir/*.json
    echo "  Cleared: results/$dir/"
done

# ============================================================================
# Step 2: Run Tier 1 + 2 on GPU 0, Tier 3 + 4 on GPU 1
# ============================================================================
echo ""
echo "Step 2: Launching tier benchmark runs..."
mkdir -p logs

# GPU 0: Tier 1 + Tier 2
echo "  Launching Tier 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 \
    conda run -n dml-bench-env python3 run_full_benchmark.py --tier 1 --gpu 0 \
    > logs/full_rerun_tier1.log 2>&1

echo "  Tier 1 complete. Launching Tier 2 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 \
    conda run -n dml-bench-env python3 run_full_benchmark.py --tier 2 --gpu 0 \
    > logs/full_rerun_tier2.log 2>&1 &
TIER2_PID=$!

# GPU 1: Tier 3 + Tier 4
echo "  Launching Tier 3 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 taskset -c 8-15 \
    conda run -n dml-bench-env python3 run_full_benchmark.py --tier 3 --gpu 1 \
    > logs/full_rerun_tier3.log 2>&1 &
TIER3_PID=$!

echo "  Tier 2 PID: $TIER2_PID, Tier 3 PID: $TIER3_PID"
echo "  Waiting for Tier 2 and Tier 3..."
wait $TIER2_PID
echo "  Tier 2 complete."
wait $TIER3_PID
echo "  Tier 3 complete."

echo "  Launching Tier 4 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 \
    conda run -n dml-bench-env python3 run_full_benchmark.py --tier 4 --gpu 0 --save-logs \
    > logs/full_rerun_tier4.log 2>&1
echo "  Tier 4 complete."

# ============================================================================
# Step 3: Run unified comparison, SPY, ablations
# ============================================================================
echo ""
echo "Step 3: Unified comparison, SPY, ablations..."

# These need their own scripts — they're not part of run_full_benchmark.py
# Unified comparison (5 datasets x 11 methods x 10 seeds)
echo "  Launching unified comparison..."
# Clear old unified results
rm -f results/unified_comparison/multi_seed/*.json

CUDA_VISIBLE_DEVICES=0 taskset -c 0-11 \
    conda run -n dml-bench-env python3 experiments/unified_comparison/run_unified_experiment.py \
    > logs/full_rerun_unified.log 2>&1 &
UNIFIED_PID=$!

# SPY temporal split
echo "  Launching SPY experiments..."
rm -f results/spy_options_temporal/*.json

CUDA_VISIBLE_DEVICES=1 taskset -c 12-19 \
    conda run -n dml-bench-env python3 experiments/real_data_spy/run_spy_experiment.py --split-mode temporal \
    > logs/full_rerun_spy.log 2>&1 &
SPY_PID=$!

echo "  Unified PID: $UNIFIED_PID, SPY PID: $SPY_PID"
wait $UNIFIED_PID
echo "  Unified comparison complete."
wait $SPY_PID
echo "  SPY complete."

# ============================================================================
# Step 4: Verification
# ============================================================================
echo ""
echo "Step 4: Verification..."
echo "  Running win-rate computation..."
conda run -n dml-bench-env python3 scripts/compute_win_rates.py > logs/full_rerun_winrates.log 2>&1
cat logs/full_rerun_winrates.log

echo "  Running tests..."
conda run -n dml-bench-env python3 -m pytest tests/ -q > logs/full_rerun_tests.log 2>&1
tail -3 logs/full_rerun_tests.log

echo ""
echo "============================================"
echo "FULL RE-RUN COMPLETE: $(date)"
echo "============================================"
