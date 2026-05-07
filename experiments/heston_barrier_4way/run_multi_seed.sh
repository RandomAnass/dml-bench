#!/usr/bin/env bash
# Heston barrier multi-seed (HEADLINE) on GPU 0.
# 5 seeds × 12 methods = 60 runs. ~2-2.5h.

set -e

CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"

cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

GPU="${1:-0}"

echo "============================================================"
echo " HESTON BARRIER MULTI-SEED (HEADLINE) on GPU $GPU"
echo " Started: $(date)"
echo "============================================================"

python experiments/heston_barrier_4way/run_multi_seed.py --gpu "$GPU" --resume

echo ""
echo "============================================================"
echo " MULTI-SEED RUN complete: $(date)"
echo "============================================================"
