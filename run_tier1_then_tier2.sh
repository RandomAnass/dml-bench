#!/usr/bin/env bash
# ===========================================================================
# Auto-chain: Tier 1 → Tier 2 → Tier 3
# Runs all tiers sequentially. Each tier is COMPLEMENTARY (no overlap).
# Uses --resume so crashes are recoverable.
#
# Usage (in tmux):
#   tmux new-session -d -s dml_full_nn \
#     'cd /path/to/o_df_ml && bash run_tier1_then_tier2.sh nn 0 2>&1 | tee results/chain_nn.log; exec bash -i'
#
#   tmux new-session -d -s dml_full_bl \
#     'cd /path/to/o_df_ml && bash run_tier1_then_tier2.sh baselines 2>&1 | tee results/chain_bl.log; exec bash -i'
#
# Arguments:
#   $1 = "nn" | "baselines" | "all"
#   $2 = GPU id (only used for nn/all mode)
# ===========================================================================
set -e

MODE="${1:-all}"
GPU="${2:-0}"

# ---- Portable environment setup ----
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/anaconda3)}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"

cd "$(dirname "$0")"

echo "============================================================"
echo " AUTO-CHAIN: Tier 1 → Tier 2 → Tier 3"
echo " Mode: $MODE | GPU: $GPU"
echo " Started: $(date)"
echo "============================================================"

# ---- Build common args ----
if [ "$MODE" = "nn" ]; then
    COMMON_ARGS="--nn-only --gpu $GPU --resume"
elif [ "$MODE" = "baselines" ]; then
    COMMON_ARGS="--baselines-only --resume"
else
    COMMON_ARGS="--gpu $GPU --resume"
fi

run_tier() {
    local TIER=$1
    local EXTRA_ARGS="${2:-}"
    
    echo ""
    echo ">>>>>>>>>> STARTING TIER $TIER — $(date) <<<<<<<<<<"
    echo ""
    
    python run_full_benchmark.py --tier $TIER $COMMON_ARGS $EXTRA_ARGS
    
    local EXIT_CODE=$?
    
    echo ""
    echo ">>>>>>>>>> TIER $TIER FINISHED — exit=$EXIT_CODE — $(date) <<<<<<<<<<"
    echo ""
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  Tier $TIER had errors (exit=$EXIT_CODE), continuing to next tier..."
    fi
    
    return $EXIT_CODE
}

# ============================================================
# TIER 1 — Minimum Publishable
# ============================================================
run_tier 1
T1_EXIT=$?

# ============================================================
# TIER 2 — Finance/Quant + Noise + Step
# ============================================================
TIER2_EXTRA=""
if [ "$MODE" = "nn" ] || [ "$MODE" = "all" ]; then
    TIER2_EXTRA="--hedging"
fi
run_tier 2 "$TIER2_EXTRA"
T2_EXIT=$?

# ============================================================
# TIER 3 — Fill remaining gaps
# ============================================================
run_tier 3
T3_EXIT=$?

# ---- Summary ----
echo "============================================================"
echo " AUTO-CHAIN COMPLETE"
echo " Tier 1 exit: $T1_EXIT"
echo " Tier 2 exit: $T2_EXIT"
echo " Tier 3 exit: $T3_EXIT"
echo " Finished: $(date)"
echo "============================================================"

TOTAL_ERRORS=$((T1_EXIT + T2_EXIT + T3_EXIT))
if [ $TOTAL_ERRORS -eq 0 ]; then
    echo "🟢 ALL TIERS PASSED"
else
    echo "🟡 Some tiers had errors (total non-zero exits: $TOTAL_ERRORS)"
fi

exit $TOTAL_ERRORS
