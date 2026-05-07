#!/bin/bash
# ============================================================================
# POST-TIER EXPERIMENT RUNS
# ============================================================================
# This script runs AFTER the 4 tier benchmarks complete.
# It handles: unified comparison, SPY, all ablations, revision experiments.
#
# Usage: Called automatically by launch_continuation.sh, or manually:
#   GPU=0 nohup scripts/launch_post_tier_runs.sh gpu0 > logs/post_tier_gpu0.log 2>&1 &
#   GPU=1 nohup scripts/launch_post_tier_runs.sh gpu1 > logs/post_tier_gpu1.log 2>&1 &
# ============================================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PARTITION="${1:-gpu0}"  # gpu0 or gpu1

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "============================================"
echo "POST-TIER RUNS ($PARTITION): $(date)"
echo "============================================"

if [ "$PARTITION" = "gpu0" ]; then
    export CUDA_VISIBLE_DEVICES=0
    CORES="0-11"

    # --- GPU 0: Unified comparison + compute-matched + lambda_j + noise sweep ---

    echo ""
    echo "=== Unified Comparison (GPU 0) ==="
    echo "Clearing old unified results..."
    rm -f results/unified_comparison/multi_seed/*.json
    echo "Running 11 methods × 5 datasets × 10 seeds = 550 experiments..."
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/unified_comparison/run_unified_experiment.py --mode multi_seed --gpu 0
    echo "Unified comparison COMPLETE: $(date)"

    echo ""
    echo "=== Compute-Matched Controls (GPU 0) ==="
    rm -f results/compute_matched_controls/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/hs_comparison/run_compute_matched_controls.py
    echo "Compute-matched controls COMPLETE: $(date)"

    echo ""
    echo "=== Lambda_j Ablation (GPU 0) ==="
    rm -f results/lambda_j_ablation/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/hs_comparison/run_lambda_j_ablation.py
    echo "Lambda_j ablation COMPLETE: $(date)"

    echo ""
    echo "=== Gradient Noise Sweep (GPU 0) ==="
    rm -f results/gradient_noise_sweep/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/gradient_noise_sweep.py
    echo "Noise sweep COMPLETE: $(date)"

    echo ""
    echo "=== Revision Experiments: Warmup Fixed-Lambda Phase 2 (GPU 0) ==="
    taskset -c $CORES conda run -n dml-bench-env python3 \
        scripts/run_revision_experiments.py --run 1
    echo "Warmup fixed-lambda COMPLETE: $(date)"

    echo ""
    echo "=== Revision Experiments: Early Stopping Ablation (GPU 0) ==="
    taskset -c $CORES conda run -n dml-bench-env python3 \
        scripts/run_revision_experiments.py --run 3
    echo "Early stopping ablation COMPLETE: $(date)"

elif [ "$PARTITION" = "gpu1" ]; then
    export CUDA_VISIBLE_DEVICES=1
    CORES="12-23"

    # --- GPU 1: SPY + realworld + LRM + warmup + higher-order ---

    echo ""
    echo "=== SPY Temporal Split (GPU 1) ==="
    rm -f results/spy_options_temporal/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/real_data_spy/run_spy_experiment.py --split-mode temporal
    echo "SPY temporal COMPLETE: $(date)"

    echo ""
    echo "=== SPY Purged CV (GPU 1) ==="
    rm -f results/spy_options_purged_cv/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/real_data_spy/run_spy_purged_cv.py
    echo "SPY purged CV COMPLETE: $(date)"

    echo ""
    echo "=== Warmup Experiments (GPU 1) ==="
    rm -f results/warmup_experiments/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/proposed_method/run_warmup_experiment.py
    echo "Warmup experiments COMPLETE: $(date)"

    echo ""
    echo "=== LRM Comparison (GPU 1) ==="
    rm -f results/lrm_comparison/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        experiments/lrm_comparison/run_lrm_vs_adaptive.py
    echo "LRM comparison COMPLETE: $(date)"

    echo ""
    echo "=== Real-world: rMD17 + Basket + Yield (GPU 1) ==="
    rm -f results/realworld/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 \
        run_realworld_experiments.py
    echo "Real-world experiments COMPLETE: $(date)"

    echo ""
    echo "=== Revision Experiments: Autodiff Vanilla Re-eval (GPU 1) ==="
    taskset -c $CORES conda run -n dml-bench-env python3 \
        scripts/run_revision_experiments.py --run 2
    echo "Autodiff vanilla re-eval COMPLETE: $(date)"

fi

echo ""
echo "============================================"
echo "POST-TIER RUNS ($PARTITION) ALL COMPLETE: $(date)"
echo "============================================"
