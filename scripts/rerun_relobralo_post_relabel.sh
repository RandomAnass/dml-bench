#!/usr/bin/env bash
# Rerun dml_relobralo (faithful Bischof-Kraus 2022 Eq.11) and dml_softmax_balance
# (simplified) on the configs that currently have only one of the two on disk
# after the 2026-04-26 legacy relabel. Both runners support --methods and
# --resume, so we just specify --methods and let the runner skip configs that
# already have a result JSON.
#
# Critical scope (paper tables): tier 3 + unified comparison.
# Optional scope: tier 1, tier 2 (appendix). Toggle below.
#
# Parallel: 2 A6000 GPUs, taskset CPU pinning, OMP/MKL capped at 4 per worker.
#
# Usage:
#   chmod +x scripts/rerun_relobralo_post_relabel.sh
#   nohup scripts/rerun_relobralo_post_relabel.sh > logs/rerun_relobralo_master.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG_DIR=logs
mkdir -p "$LOG_DIR"
PY=python
GIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)

MASTER_LOG="$LOG_DIR/rerun_relobralo_master.log"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "============================================"
log "RELOBRALO RERUN START: $(date -u)"
log "git HEAD: $GIT"
log "Repo: $(pwd)"
log "============================================"

# Thread budget per worker; ~32 CPU peak on 48-core host.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Toggle: do tier 1 and tier 2 ReLoBRaLo too? They are appendix-relevant only.
INCLUDE_TIER1=1
INCLUDE_TIER2=1

# ----------------------------------------------------------------------------
# GPU 0: Tier 3 — both methods (1302 relobralo + 138 softmax_balance still missing)
# ----------------------------------------------------------------------------
(
    LOG="$LOG_DIR/rerun_relobralo_tier3.log"
    log "[Tier3 GPU0] start → $LOG"
    taskset -c 0-11 "$PY" run_full_benchmark.py \
        --tier 3 --gpu 0 --resume \
        --methods dml_relobralo dml_softmax_balance \
        >> "$LOG" 2>&1
    log "[Tier3 GPU0] done"
) &
TIER3_PID=$!
log "Tier 3 bucket PID $TIER3_PID"

# ----------------------------------------------------------------------------
# GPU 1: Unified comparison multi_seed — relobralo only (softmax was relabeled)
# ----------------------------------------------------------------------------
(
    LOG="$LOG_DIR/rerun_relobralo_unified.log"
    log "[Unified GPU1] start → $LOG"
    taskset -c 24-29 "$PY" experiments/unified_comparison/run_unified_experiment.py \
        --mode multi_seed --gpu 1 --resume \
        --methods dml_relobralo \
        >> "$LOG" 2>&1
    log "[Unified GPU1] done"
) &
UNIFIED_PID=$!
log "Unified bucket PID $UNIFIED_PID"

# Optional: Tier 1 (180 relobralo) — light, share GPU 1 with unified.
if [ "$INCLUDE_TIER1" = "1" ]; then
    sleep 30   # let unified allocate first
    (
        LOG="$LOG_DIR/rerun_relobralo_tier1.log"
        log "[Tier1 GPU1] start → $LOG"
        taskset -c 30-35 "$PY" run_full_benchmark.py \
            --tier 1 --gpu 1 --resume \
            --methods dml_relobralo \
            >> "$LOG" 2>&1
        log "[Tier1 GPU1] done"
    ) &
    TIER1_PID=$!
    log "Tier 1 bucket PID $TIER1_PID"
fi

# Optional: Tier 2 (1050 relobralo) — share GPU 0 with tier 3.
if [ "$INCLUDE_TIER2" = "1" ]; then
    sleep 60
    (
        LOG="$LOG_DIR/rerun_relobralo_tier2.log"
        log "[Tier2 GPU0] start → $LOG"
        taskset -c 12-17 "$PY" run_full_benchmark.py \
            --tier 2 --gpu 0 --resume \
            --methods dml_relobralo \
            >> "$LOG" 2>&1
        log "[Tier2 GPU0] done"
    ) &
    TIER2_PID=$!
    log "Tier 2 bucket PID $TIER2_PID"
fi

log "============================================"
log "All buckets launched."
log "============================================"

# Wait for all
wait
log "============================================"
log "RELOBRALO RERUN DONE: $(date -u)"
log "============================================"
