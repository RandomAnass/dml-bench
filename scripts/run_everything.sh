#!/bin/bash
# Master script: runs EVERYTHING sequentially on one GPU. 
# Launch TWO instances, one per GPU.
# Usage:
#   nohup scripts/run_everything.sh 0 > logs/everything_gpu0.log 2>&1 &
#   nohup scripts/run_everything.sh 1 > logs/everything_gpu1.log 2>&1 &
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU=${1:?Usage: run_everything.sh GPU_ID}
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if [ "$GPU" = "0" ]; then
    CORES="0-11"
    echo "=== GPU 0 STARTED: $(date) ==="

    echo "--- Tier 3 (fresh) ---"
    taskset -c $CORES conda run -n dml-bench-env python3 run_full_benchmark.py --tier 3 --gpu 0
    echo "TIER 3 DONE: $(date)"

    echo "--- Tier 4 ---"
    taskset -c $CORES conda run -n dml-bench-env python3 run_full_benchmark.py --tier 4 --gpu 0 --save-logs
    echo "TIER 4 DONE: $(date)"

    echo "--- Unified Comparison (multi_seed) ---"
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/unified_comparison/run_unified_experiment.py --mode multi_seed --gpu 0
    echo "UNIFIED DONE: $(date)"

    echo "--- Compute-Matched Controls ---"
    rm -f results/compute_matched_controls/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/hs_comparison/run_compute_matched_controls.py
    echo "CONTROLS DONE: $(date)"

    echo "--- Lambda_j Ablation ---"
    rm -f results/lambda_j_ablation/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/hs_comparison/run_lambda_j_ablation.py
    echo "LAMBDA_J DONE: $(date)"

    echo "--- Noise Sweep ---"
    rm -f results/gradient_noise_sweep/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/gradient_noise_sweep.py
    echo "NOISE DONE: $(date)"

    echo "--- Revision: Warmup Fixed-Lambda ---"
    taskset -c $CORES conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 1
    echo "WARMUP_FIXED DONE: $(date)"

    echo "--- Revision: Early Stopping Ablation ---"
    taskset -c $CORES conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 3
    echo "ES_ABLATION DONE: $(date)"

    echo "=== GPU 0 ALL COMPLETE: $(date) ==="

elif [ "$GPU" = "1" ]; then
    CORES="12-23"
    echo "=== GPU 1 STARTED: $(date) ==="

    echo "--- SPY Temporal ---"
    rm -f results/spy_options_temporal/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/real_data_spy/run_spy_experiment.py --split-mode temporal
    echo "SPY_TEMPORAL DONE: $(date)"

    echo "--- SPY Purged CV ---"
    rm -f results/spy_options_purged_cv/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/real_data_spy/run_spy_purged_cv.py
    echo "SPY_CV DONE: $(date)"

    echo "--- Warmup Experiments ---"
    rm -f results/warmup_experiments/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/proposed_method/run_warmup_experiment.py
    echo "WARMUP DONE: $(date)"

    echo "--- LRM Comparison ---"
    rm -f results/lrm_comparison/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 experiments/lrm_comparison/run_lrm_vs_adaptive.py
    echo "LRM DONE: $(date)"

    echo "--- Real-world (rMD17, basket, yield) ---"
    rm -f results/realworld/*.json
    taskset -c $CORES conda run -n dml-bench-env python3 run_realworld_experiments.py
    echo "REALWORLD DONE: $(date)"

    echo "--- Revision: Autodiff Vanilla Re-eval ---"
    taskset -c $CORES conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 2
    echo "AUTODIFF DONE: $(date)"

    echo "=== GPU 1 ALL COMPLETE: $(date) ==="
fi
