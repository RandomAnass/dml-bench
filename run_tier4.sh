#!/bin/bash
# ============================================================================
# Tier 4 Launch Script — Statistical Power (10 seeds)
#
# Runs 5 extra seeds for key configurations + captures training logs.
# Splits across 2 GPUs. Safe to run alongside existing Tier 3.
#
# Estimated runtime: ~4-6h per GPU (depending on Tier 3 load)
#
# Usage:
#   chmod +x run_tier4.sh
#   ./run_tier4.sh            # Runs both GPUs in separate tmux sessions
#   ./run_tier4.sh gpu0       # Only GPU 0
#   ./run_tier4.sh gpu1       # Only GPU 1
# ============================================================================

set -e

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/anaconda3)}"
ENV_NAME="dml-bench-env"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

activate_env() {
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
    conda activate "$ENV_NAME"
    cd "$WORKDIR"
}

# GPU 0: vanilla + dml_fixed (less compute, same GPU as Tier 3 group 0)
run_gpu0() {
    activate_env
    echo "=== TIER 4 — GPU 0: vanilla + dml_fixed ==="
    echo "Started: $(date)"
    python run_full_benchmark.py --tier 4 --gpu 0 --nn-only --resume \
        --save-logs --methods vanilla dml_fixed
    echo "=== GPU 0 DONE: $(date) ==="
}

# GPU 1: gradnorm + relobralo
run_gpu1() {
    activate_env
    echo "=== TIER 4 — GPU 1: gradnorm + relobralo ==="
    echo "Started: $(date)"
    python run_full_benchmark.py --tier 4 --gpu 1 --nn-only --resume \
        --save-logs --methods dml_gradnorm dml_relobralo
    echo "=== GPU 1 DONE: $(date) ==="
}

# Baselines (CPU-bound, separate session)
run_baselines() {
    activate_env
    echo "=== TIER 4 — Baselines ==="
    echo "Started: $(date)"
    python run_full_benchmark.py --tier 4 --baselines-only --resume
    echo "=== Baselines DONE: $(date) ==="
}

case "${1:-both}" in
    gpu0)
        run_gpu0
        ;;
    gpu1)
        run_gpu1
        ;;
    baselines)
        run_baselines
        ;;
    both)
        # Launch in tmux sessions (safe alongside existing Tier 3 sessions)
        echo "Launching Tier 4 in tmux sessions..."
        
        # Wait for Tier 3 to finish on each GPU before starting
        echo "NOTE: If Tier 3 is still running, Tier 4 will share GPU memory."
        echo "      Consider waiting for Tier 3 to finish first."
        echo ""
        
        tmux new-session -d -s dml_tier4_gpu0 "bash -c 'source $0 gpu0; exec bash'"
        tmux new-session -d -s dml_tier4_gpu1 "bash -c 'source $0 gpu1; exec bash'"
        tmux new-session -d -s dml_tier4_bl   "bash -c 'source $0 baselines; exec bash'"
        
        echo "Started tmux sessions:"
        echo "  dml_tier4_gpu0  — vanilla + dml_fixed on GPU 0"
        echo "  dml_tier4_gpu1  — gradnorm + relobralo on GPU 1"
        echo "  dml_tier4_bl    — baselines on CPU"
        echo ""
        echo "Monitor: tmux attach -t dml_tier4_gpu0"
        echo "List:    tmux ls"
        ;;
    *)
        echo "Usage: $0 [gpu0|gpu1|baselines|both]"
        exit 1
        ;;
esac
