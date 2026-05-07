#!/usr/bin/env bash
# ERA5 pipeline chain: pilot bare → pilot state → go/no-go → Stage 2 download → preprocess → Stage 2 bare + state.
# Bails on any failure with explicit phase log. Each phase appends to logs/era5_chain.log.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=logs/era5_chain.log
PY=python

phase() { echo "[$(date -Iseconds)] CHAIN $1" | tee -a "$LOG"; }
fail()  { echo "[$(date -Iseconds)] CHAIN FAIL $1" | tee -a "$LOG"; exit 1; }

# Phase 1: wait for bare pilot to fill (max 90 min of waiting)
phase "phase=1 wait_bare_pilot expect=15"
WAIT=0
while [ "$(ls results/era5/bare/*.json 2>/dev/null | wc -l)" -lt 15 ]; do
  sleep 60; WAIT=$((WAIT+60))
  if [ "$WAIT" -gt 5400 ]; then fail "bare_pilot timed out after 90 min"; fi
done
phase "phase=1 done bare_pilot=$(ls results/era5/bare/*.json | wc -l)"

# Phase 2: bare go/no-go
phase "phase=2 go_nogo bare"
if ! "$PY" scripts/era5_go_nogo.py results/era5/bare 2>&1 | tee -a "$LOG"; then
  fail "bare go/no-go FAILED"
fi

# Phase 3: launch state pilot (foreground, blocks until done)
phase "phase=3 launch state_pilot"
"$PY" scripts/run_era5.py \
  --cache-path data/era5/pilot/era5_pilot_cache.npz \
  --regime state --methods vanilla dml_fixed dml_fixed_half --archs 4x256 \
  --n-seeds 5 --gpus 1 --workers-per-gpu 4 \
  --cores-per-worker 2 --start-core 24 \
  --n-epochs 200 --n-points-per-epoch 100000 2>&1 | tee -a logs/era5_stage1_state.nohup
N_STATE=$(ls results/era5/state/*.json 2>/dev/null | wc -l)
phase "phase=3 done state_pilot=$N_STATE"
if [ "$N_STATE" -lt 15 ]; then fail "state_pilot incomplete: $N_STATE/15"; fi

# Phase 4: state go/no-go
phase "phase=4 go_nogo state"
if ! "$PY" scripts/era5_go_nogo.py results/era5/state 2>&1 | tee -a "$LOG"; then
  fail "state go/no-go FAILED"
fi

# Phase 5: download Stage 2 multi-year (2014-2020 except 2019 which we already have)
phase "phase=5 download Stage2 multiyear"
mkdir -p data/era5/full_1deg
cp -n data/era5/pilot/z500_12z_2019_1p0deg.nc data/era5/full_1deg/ 2>/dev/null || true
"$PY" scripts/download_era5.py \
  --years 2014 2015 2016 2017 2018 2020 --resolution 1.0 \
  --out-dir data/era5/full_1deg 2>&1 | tee -a "$LOG"

# Phase 6: preprocess multi-year
phase "phase=6 preprocess Stage2"
"$PY" scripts/preprocess_era5.py \
  --in-glob "data/era5/full_1deg/z500_12z_*_1p0deg.nc" \
  --out data/era5/full_1deg/era5_full_cache.npz \
  --eof-K 16 --val-frac 0.15 --test-frac 0.15 --embargo-days 30 2>&1 | tee -a "$LOG"

# Phase 7: Stage 2 bare on GPU 1 (4 workers, cores 24-31)
phase "phase=7 launch Stage2 bare on GPU1"
"$PY" scripts/run_era5.py \
  --cache-path data/era5/full_1deg/era5_full_cache.npz \
  --regime bare --methods vanilla dml_fixed dml_fixed_half --archs 4x256 6x512 \
  --n-seeds 5 --gpus 1 --workers-per-gpu 4 \
  --cores-per-worker 2 --start-core 24 \
  --n-epochs 200 --n-points-per-epoch 100000 2>&1 | tee -a logs/era5_stage2_bare.nohup

# Phase 8: Stage 2 state on GPU 1 (sequential after bare; could split GPUs but keeps GPU 0 free for #197)
phase "phase=8 launch Stage2 state on GPU1"
"$PY" scripts/run_era5.py \
  --cache-path data/era5/full_1deg/era5_full_cache.npz \
  --regime state --methods vanilla dml_fixed dml_fixed_half --archs 4x256 6x512 \
  --n-seeds 5 --gpus 1 --workers-per-gpu 4 \
  --cores-per-worker 2 --start-core 24 \
  --n-epochs 200 --n-points-per-epoch 100000 2>&1 | tee -a logs/era5_stage2_state.nohup

phase "phase=DONE Stage2 bare=$(ls results/era5_stage2/bare 2>/dev/null | wc -l) state=$(ls results/era5_stage2/state 2>/dev/null | wc -l)"
