#!/usr/bin/env bash
# Validation chain for GPU 1: BS reproduction → ε-sweep → CG variance check.
# All three sequential since they share GPU 1 and are individually small.
# CPU pinned via taskset; OMP/MKL threads capped to avoid CPU thrashing.

set -e

CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"

cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

GPU="${1:-1}"

echo "============================================================"
echo " VALIDATION CHAIN on GPU $GPU"
echo " Started: $(date)"
echo "============================================================"

echo ""
echo ">>> Run #2: BS barrier reproduction (vs G&K v2 Table 2 row n=2)"
python experiments/heston_barrier_4way/run_bs_reproduction.py --gpu "$GPU" --resume

echo ""
echo ">>> Run #3: ε-sweep on fuzzy-Heston-barrier"
python experiments/heston_barrier_4way/run_eps_sweep.py --gpu "$GPU" --resume

echo ""
echo ">>> Run #4: CG multi-step LRM variance scaling check"
python experiments/heston_barrier_4way/run_cg_variance.py --gpu "$GPU" --resume

echo ""
echo "============================================================"
echo " VALIDATION CHAIN complete: $(date)"
echo "============================================================"
