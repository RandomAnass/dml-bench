#!/usr/bin/env bash
# ERA5 Stage 2 chain (v2): preprocess multi-year → run bare → run state.
# Run after pilot bare+state pass go/no-go. Outputs to results/era5/{bare,state}.
# (Stage 2 cells overwrite pilot cells, since they share the same JSON keys
# era5_4x256_*; that's a deliberate replacement: pilot was single-year, Stage 2
# is multi-year canonical. We rename pilot output dirs out of the way first.)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=logs/era5_chain_v2.log
PY=python

phase() { echo "[$(date -Iseconds)] CHAIN2 $1" | tee -a "$LOG"; }
fail()  { echo "[$(date -Iseconds)] CHAIN2 FAIL $1" | tee -a "$LOG"; exit 1; }

# Preserve pilot JSONs under a separate dir name
phase "phase=0 archive_pilot"
mkdir -p results/era5_pilot_2019
[ -d results/era5/bare ] && cp -n results/era5/bare/*.json results/era5_pilot_2019/ 2>/dev/null || true
[ -d results/era5/state ] && cp -n results/era5/state/*.json results/era5_pilot_2019/ 2>/dev/null || true

# Phase 1: preprocess multi-year
phase "phase=1 preprocess multi-year (2014-2020)"
"$PY" scripts/preprocess_era5.py \
  --in-glob "data/era5/full_1deg/z500_12z_*_1p0deg.nc" \
  --out data/era5/full_1deg/era5_full_cache.npz \
  --eof-K 16 --val-frac 0.15 --test-frac 0.15 --embargo-days 30 2>&1 | tee -a "$LOG"
[ -f data/era5/full_1deg/era5_full_cache.npz ] || fail "preprocess produced no cache"

# Phase 2: clear pilot JSONs from results/era5/{bare,state} so Stage 2 starts fresh
phase "phase=2 clear results/era5/{bare,state} for Stage 2"
rm -f results/era5/bare/*.json
rm -f results/era5/state/*.json

# Phase 3: Stage 2 bare on GPU 1 — 2 archs × 3 methods × 5 seeds = 30 cells
phase "phase=3 Stage2 bare grid=30 (4x256+6x512 × 3 methods × 5 seeds)"
"$PY" scripts/run_era5.py \
  --cache-path data/era5/full_1deg/era5_full_cache.npz \
  --regime bare --methods vanilla dml_fixed dml_fixed_half --archs 4x256 6x512 \
  --n-seeds 5 --gpus 1 --workers-per-gpu 4 \
  --cores-per-worker 2 --start-core 24 \
  --n-epochs 200 --n-points-per-epoch 100000 2>&1 | tee -a logs/era5_stage2_bare.nohup
N_BARE=$(ls results/era5/bare/*.json 2>/dev/null | wc -l)
phase "phase=3 done bare=$N_BARE/30"
[ "$N_BARE" -ge 30 ] || fail "stage2 bare incomplete: $N_BARE/30"

# Phase 4: Stage 2 state on GPU 1 — 30 cells
phase "phase=4 Stage2 state grid=30"
"$PY" scripts/run_era5.py \
  --cache-path data/era5/full_1deg/era5_full_cache.npz \
  --regime state --methods vanilla dml_fixed dml_fixed_half --archs 4x256 6x512 \
  --n-seeds 5 --gpus 1 --workers-per-gpu 4 \
  --cores-per-worker 2 --start-core 24 \
  --n-epochs 200 --n-points-per-epoch 100000 2>&1 | tee -a logs/era5_stage2_state.nohup
N_STATE=$(ls results/era5/state/*.json 2>/dev/null | wc -l)
phase "phase=4 done state=$N_STATE/30"

phase "phase=DONE Stage2 bare=$N_BARE state=$N_STATE"
