#!/bin/bash
# GPU 0: Run tier3 in background, independent jobs in foreground simultaneously
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
CORES="0-11"

echo "=== GPU 0 PARALLEL START: $(date) ==="

# --- TIER 3 runs in background (long job, uses GPU) ---
echo "Starting Tier 3 in background..."
taskset -c 0-5 conda run -n dml-bench-env python3 run_full_benchmark.py --tier 3 --gpu 0 &
T3_PID=$!
echo "Tier 3 PID: $T3_PID"

# --- Wait 30s for tier3 to start, then run CPU-heavy independent jobs ---
sleep 30

echo "--- Unified Comparison (multi_seed) ---"
rm -f results/unified_comparison/multi_seed/*.json
taskset -c 6-11 conda run -n dml-bench-env python3 experiments/unified_comparison/run_unified_experiment.py --mode multi_seed --gpu 0
echo "UNIFIED DONE: $(date)"

echo "--- Compute-Matched Controls ---"
rm -f results/compute_matched_controls/*.json
taskset -c 6-11 conda run -n dml-bench-env python3 experiments/hs_comparison/run_compute_matched_controls.py
echo "CONTROLS DONE: $(date)"

echo "--- Lambda_j Ablation ---"
rm -f results/lambda_j_ablation/*.json
taskset -c 6-11 conda run -n dml-bench-env python3 experiments/hs_comparison/run_lambda_j_ablation.py
echo "LAMBDA_J DONE: $(date)"

echo "--- Noise Sweep ---"
rm -f results/gradient_noise_sweep/*.json
taskset -c 6-11 conda run -n dml-bench-env python3 experiments/gradient_noise_sweep.py
echo "NOISE DONE: $(date)"

echo "--- Revision: Warmup Fixed-Lambda ---"
taskset -c 6-11 conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 1
echo "WARMUP_FIXED DONE: $(date)"

echo "--- Revision: Early Stopping Ablation ---"
taskset -c 6-11 conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 3
echo "ES_ABLATION DONE: $(date)"

# --- Now wait for Tier 3 to finish ---
echo "Independent jobs done. Waiting for Tier 3 (PID $T3_PID)..."
wait $T3_PID
echo "TIER 3 DONE: $(date)"

# --- Tier 4 (depends on tier 3 being done for sequencing, but not data) ---
echo "--- Tier 4 ---"
taskset -c $CORES conda run -n dml-bench-env python3 run_full_benchmark.py --tier 4 --gpu 0 --save-logs
echo "TIER 4 DONE: $(date)"

echo "=== GPU 0 ALL COMPLETE: $(date) ==="
