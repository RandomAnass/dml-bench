#!/bin/bash
# ============================================================================
# Re-run of I-H5 contaminated result sets.
#
# Incident: commit 967bc5e0 (2026-04-16) silently disabled GradNorm's weight
# updates via .detach() in the gradnorm-side weighted_loss. Fix landed at
# cff4862a (2026-04-23). This script re-runs every result set that was
# generated inside the broken window (2026-04-16 18:58 → 2026-04-23).
#
# See EVIDENCE/INCIDENT_I-H5_GRADNORM_SILENT_BREAK.md for full scope.
#
# Parallelism: 2× A6000, ~48 CPU cores available.
#   - GPU 0: PaiNN (heavy, needs full GPU) + basket + burgers + P6/P7/P9
#   - GPU 1: MLP + GATv2 + J-H2 ablation
# CPU pinning via `taskset -c` keeps each launcher's numpy/torch threads
# inside its own core block so workers don't thrash.
#
# Usage:
#   chmod +x scripts/rerun_i_h5_contamination.sh
#   nohup scripts/rerun_i_h5_contamination.sh > logs/rerun_i_h5_master.log 2>&1 &
#   echo "master PID: $!"
#
# Per-worker logs are written to logs/rerun_i_h5_<bucket>.log.
# Check progress:
#   tail -F logs/rerun_i_h5_*.log
#   ls results/molecular_painn/*dml_gradnorm*.json 2>/dev/null | wc -l
# ============================================================================
set -u   # undefined variables fail; do NOT use -e (we want each launcher to
         # run independently — a failure in one bucket must not abort others)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/rerun_i_h5_master.log"
GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

PY="python"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "============================================"
log "I-H5 RERUN START: $(date -u)"
log "git HEAD: $GIT_HASH"
log "Python: $($PY --version)"
log "Torch: $($PY -c 'import torch; print(torch.__version__)' 2>&1)"
log "Host: $(hostname)"
log "============================================"

# Thread budget (per-process). 4 threads/proc × 8 procs peak = 32 ≤ 48 cores.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# ----------------------------------------------------------------------------
# GPU 1 (cores 24-35): Molecular MLP + GATv2 (dml_gradnorm + dml_warmup only)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1
    LOG="$LOG_DIR/rerun_i_h5_molecular_mlp_gatv2.log"
    log "[MLP+GATv2 on GPU1] start → $LOG"

    # Delete contaminated dml_gradnorm + dml_warmup files to force rerun
    find results/molecular_mlp -name '*dml_gradnorm*.json' \
        -newermt '2026-04-16 18:58' -delete
    find results/molecular_mlp -name '*dml_warmup*.json' \
        -newermt '2026-04-16 18:58' -delete
    find results/gnn_md17 -name '*dml_gradnorm*.json' \
        -newermt '2026-04-16 18:58' -delete
    find results/gnn_md17 -name '*dml_warmup*.json' \
        -newermt '2026-04-16 18:58' -delete

    # MLP dml_gradnorm + dml_warmup, all molecules × all splits
    taskset -c 24-29 "$PY" experiments/molecular/run_mlp_molecular.py \
        --gpu 1 \
        --methods dml_gradnorm dml_warmup \
        --resume \
        >> "$LOG" 2>&1
    log "[MLP on GPU1] done"

    # GATv2 dml_gradnorm + dml_warmup
    # gnn_md17.py has no --methods flag — it runs all 7 methods internally and
    # uses --resume to skip existing JSONs. We already deleted only the
    # contaminated (dml_gradnorm, dml_warmup) files above, so --resume will
    # rerun exactly those while skipping the untouched vanilla / dml_fixed /
    # dml_fixed_half / dml_softmax_balance / dml_relobralo results.
    #
    # CRITICAL: gnn_md17.py's DEFAULT_MOLECULES is ["ethanol","aspirin"] only
    # (a smoke-test default), NOT all 10. We MUST pass --molecules explicitly,
    # otherwise 8 of 10 molecules get silently skipped. The first run of this
    # script (2026-04-24) hit exactly this bug — fixed here.
    taskset -c 24-29 "$PY" experiments/gnn_md17.py \
        --gpu 1 \
        --molecules aspirin azobenzene benzene ethanol malonaldehyde \
                    naphthalene paracetamol salicylic toluene uracil \
        --resume \
        >> "$LOG" 2>&1 || log "[GATv2] gnn_md17.py invocation — inspect $LOG"
    log "[GATv2 on GPU1] done"
) &
GPU1_MOL_PID=$!
log "GPU1 molecular-MLP+GATv2 bucket PID $GPU1_MOL_PID"

# ----------------------------------------------------------------------------
# GPU 0 (cores 0-11): PaiNN (dml_gradnorm + dml_warmup only)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=0
    LOG="$LOG_DIR/rerun_i_h5_molecular_painn.log"
    log "[PaiNN on GPU0] start → $LOG"

    find results/molecular_painn -name '*dml_gradnorm*.json' \
        -newermt '2026-04-16 18:58' -delete
    find results/molecular_painn -name '*dml_warmup*.json' \
        -newermt '2026-04-16 18:58' -delete

    taskset -c 0-11 "$PY" experiments/molecular/run_painn.py \
        --gpu 0 \
        --methods dml_gradnorm dml_warmup \
        --batch_size 128 \
        --resume \
        >> "$LOG" 2>&1
    log "[PaiNN on GPU0] done"
) &
PAINN_PID=$!
log "GPU0 PaiNN bucket PID $PAINN_PID"

# ----------------------------------------------------------------------------
# Wait 60s to let GPU launchers allocate; then run CPU/light-GPU buckets.
# ----------------------------------------------------------------------------
sleep 60

# ----------------------------------------------------------------------------
# GPU 1 (cores 36-41): Basket + Burgers + J-H2 ablation (small, fast)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1
    LOG="$LOG_DIR/rerun_i_h5_basket_burgers.log"
    log "[Basket + Burgers on GPU1] start → $LOG"

    # Delete contaminated basket dml_gradnorm + dml_warmup
    find results/basket_bachelier -name '*dml_gradnorm*.json' \
        -newermt '2026-04-16 18:58' -delete
    find results/basket_bachelier -name '*dml_warmup*.json' \
        -newermt '2026-04-16 18:58' -delete

    # Basket — only the 2 contaminated methods
    taskset -c 36-41 "$PY" scripts/run_basket_bachelier.py \
        --gpu 1 \
        --methods dml_gradnorm dml_warmup \
        --resume \
        >> "$LOG" 2>&1
    log "[Basket on GPU1] done"

    # Burgers — rewrites single JSON; delete first so it regenerates cleanly
    rm -f results/burgers_1d_results.json
    taskset -c 36-41 "$PY" scripts/run_burgers_experiment.py \
        >> "$LOG" 2>&1
    log "[Burgers on GPU1] done"

    # J-H2 ablation — the original script wasn't saved; skip auto-rerun.
    # Directory is flagged in EVIDENCE/known_issues.md for manual re-run
    # (it's a corroboration ablation, not in main paper tables).
    log "[J-H2 ablation] skipped — manual rerun needed (see incident doc)"
) &
BASKET_PID=$!
log "GPU1 basket+burgers bucket PID $BASKET_PID"

# ----------------------------------------------------------------------------
# GPU 0 (cores 12-17): P6 corruption (750 runs, small MLPs)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=0
    LOG="$LOG_DIR/rerun_i_h5_p6.log"
    log "[P6 corruption on GPU0] start → $LOG"

    taskset -c 12-17 "$PY" scripts/p6_corruption_run.py \
        --gpu 0 --resume \
        >> "$LOG" 2>&1
    log "[P6 on GPU0] done"
) &
P6_PID=$!
log "GPU0 P6 bucket PID $P6_PID"

# ----------------------------------------------------------------------------
# GPU 0 (cores 18-23): P9 dim-norm sweep (75 runs)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=0
    LOG="$LOG_DIR/rerun_i_h5_p9.log"
    log "[P9 dim-norm on GPU0] start → $LOG"

    taskset -c 18-23 "$PY" scripts/p9_dimnorm_run.py \
        --gpu 0 --resume \
        >> "$LOG" 2>&1
    log "[P9 on GPU0] done"
) &
P9_PID=$!
log "GPU0 P9 bucket PID $P9_PID"

# ----------------------------------------------------------------------------
# GPU 1 (cores 42-47): P7 fuzzy 2D sweep (105 runs, uses fixed eps_mult wiring)
# ----------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1
    LOG="$LOG_DIR/rerun_i_h5_p7.log"
    log "[P7 fuzzy on GPU1] start → $LOG"

    taskset -c 42-47 "$PY" scripts/p7_fuzzy_2d_run.py \
        --gpu 1 --resume \
        >> "$LOG" 2>&1
    log "[P7 on GPU1] done"
) &
P7_PID=$!
log "GPU1 P7 bucket PID $P7_PID"

log "============================================"
log "All buckets launched. PIDs:"
log "  GPU1 MLP+GATv2: $GPU1_MOL_PID"
log "  GPU0 PaiNN:      $PAINN_PID"
log "  GPU1 basket+burg: $BASKET_PID"
log "  GPU0 P6:         $P6_PID"
log "  GPU0 P9:         $P9_PID"
log "  GPU1 P7:         $P7_PID"
log "============================================"

# Wait for all
wait "$GPU1_MOL_PID" "$PAINN_PID" "$BASKET_PID" "$P6_PID" "$P9_PID" "$P7_PID"

log "============================================"
log "ALL BUCKETS DONE: $(date -u)"
log "============================================"

# Final check: run correctness regression tests on current code
log "Running correctness regression tests..."
"$PY" -m pytest tests/test_balancing_correctness.py -q \
    >> "$LOG_DIR/rerun_i_h5_tests.log" 2>&1 \
    && log "regression tests PASS" \
    || log "regression tests FAIL — investigate $LOG_DIR/rerun_i_h5_tests.log"

log "Rerun complete. Master log: $MASTER_LOG"
