#!/usr/bin/env bash
# Queue:
#   1. SPY temporal Option A  — dml_fixed_half (resume; fills ~20 cells)
#   2. SPY temporal Option C  — dml_fixed_half (resume; fills ~20 cells)
#   3. SPY purged-CV Option A — dml_fixed_half (resume; fills ~50–56 cells)
#   4. SPY purged-CV Option C — dml_fixed_half (resume; fills ~50–55 cells)
#   5. PaiNN salicylic on rMD17 (5 splits × 6 methods = 30 cells)
#
# All five jobs land in their existing result directories so previous results
# stay untouched (resume logic skips files that already exist on disk).
#
# Triggers when current SPY purged-CV processes (PIDs supplied by env vars
# WAIT_PID_BS / WAIT_PID_SVI) terminate. Logs to logs/queue_dml_fixed_half_*.log.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOGDIR=logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOGDIR/queue_dml_fixed_half_and_salicylic_${STAMP}.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MAIN_LOG"; }

WAIT_PID_BS=${WAIT_PID_BS:-1079635}
WAIT_PID_SVI=${WAIT_PID_SVI:-1112618}

log "Queue script started; waiting for PIDs $WAIT_PID_BS (purged-CV bs_price) and $WAIT_PID_SVI (purged-CV svi) to terminate."
while kill -0 "$WAIT_PID_BS" 2>/dev/null || kill -0 "$WAIT_PID_SVI" 2>/dev/null; do
    sleep 60
done
log "Both SPY purged-CV processes have exited. Beginning queued launches on GPU 1."

PYBIN=python

run_step() {
    local name=$1; shift
    local step_log="$LOGDIR/queue_${name}_${STAMP}.log"
    log "=== START $name ==="
    log "log file: $step_log"
    log "command: $*"
    "$@" >"$step_log" 2>&1
    local rc=$?
    log "=== END   $name (rc=$rc) ==="
    return $rc
}

# 1. SPY temporal Option A
run_step spy_temporal_optionA \
    "$PYBIN" experiments/real_data_spy/run_spy_experiment.py \
        --target-mode bs_price --resume --gpu 1

# 2. SPY temporal Option C
run_step spy_temporal_optionC \
    "$PYBIN" experiments/real_data_spy/run_spy_experiment.py \
        --target-mode svi --resume --gpu 1

# 3. SPY purged-CV Option A
run_step spy_purged_cv_optionA \
    "$PYBIN" experiments/real_data_spy/run_spy_purged_cv.py \
        --target-mode bs_price --resume --gpu 1

# 4. SPY purged-CV Option C
run_step spy_purged_cv_optionC \
    "$PYBIN" experiments/real_data_spy/run_spy_purged_cv.py \
        --target-mode svi --resume --gpu 1

# 5. PaiNN salicylic — 5 splits × 6 methods (matches existing 9-mol corpus)
run_step painn_salicylic \
    "$PYBIN" experiments/molecular/run_painn.py \
        --molecules salicylic \
        --methods vanilla dml_fixed dml_fixed_half dml_gradnorm dml_warmup native_EF \
        --gpu 1 --resume

log "Queue script finished."
