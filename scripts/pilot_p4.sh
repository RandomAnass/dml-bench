#!/bin/bash
# Phase 4 pilot: aspirin × seed 42 × 100 epochs × 6 representative configs.
# Sequential on GPU 1 (Tier 3+4 still on GPU 0 — must not disturb).
# Logs to logs/pilot_p4_<timestamp>.log

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs results/pilot_p4
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/pilot_p4_${TS}.log"

echo "=== Phase 4 Pilot — aspirin × seed 42 × 100 epochs ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "GPU 1 (Tier 3+4 untouched on GPU 0)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- MLP × {vanilla, dml_fixed, dml_warmup} ---
for METHOD in vanilla dml_fixed dml_warmup; do
    echo "=== MLP $METHOD aspirin s42 100 epochs ===" | tee -a "$LOG"
    conda run -n dml-bench-env python3 experiments/molecular/run_mlp_molecular.py \
        --gpu 1 --molecules aspirin --seeds 42 --methods "$METHOD" \
        --n_train 1000 --n_val 1000 --n_test 1000 --n_epochs 100 \
        --results_dir results/pilot_p4 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# --- GATv2 × {vanilla, dml_fixed} (use existing gnn_md17.py) ---
for METHOD in vanilla dml_fixed; do
    echo "=== GATv2 $METHOD aspirin s42 100 epochs ===" | tee -a "$LOG"
    conda run -n dml-bench-env python3 experiments/gnn_md17.py \
        --gpu 1 --molecules aspirin --seeds 42 \
        2>&1 | tee -a "$LOG" || true   # gnn_md17.py runs all METHODS list per call
    # NOTE: gnn_md17.py's METHODS now has all 5; for pilot we'd want only 1.
    # Use the smoke script as a single-method runner instead.
    break
done

# Replace the GATv2 block with the smoke script run (cleaner per-method runs).
echo "=== GATv2 vanilla+dml_fixed via smoke script (100 epochs) ===" | tee -a "$LOG"
conda run -n dml-bench-env python3 -c "
import sys, json, time
sys.path.insert(0, '.')
import os; os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
torch.set_num_threads(4)
from experiments.gnn_md17 import GATv2EnergyModel, load_rmd17_graphs, train_gnn_md17, set_deterministic, HPARAMS
from datetime import datetime
from pathlib import Path
RD = Path('results/pilot_p4'); RD.mkdir(parents=True, exist_ok=True)
for method in ['vanilla','dml_fixed']:
    print(f'=== GATv2 {method} aspirin s42 100ep ===')
    set_deterministic(42)
    train_data, val_data, test_data, meta = load_rmd17_graphs('aspirin', n_train=1000, n_val=1000, n_test=1000, seed=42, r_cut=HPARAMS['r_cut'])
    set_deterministic(42)
    model = GATv2EnergyModel(hidden_dim=HPARAMS['hidden_dim'], n_heads=HPARAMS['n_heads'], n_layers=HPARAMS['n_layers'], n_rbf=HPARAMS['n_rbf'], r_cut=HPARAMS['r_cut'], max_z=HPARAMS['max_z'])
    t0 = time.time()
    metrics = train_gnn_md17(model=model, train_data=train_data, val_data=val_data, test_data=test_data, method=method, n_epochs=100, batch_size=HPARAMS['batch_size'], lr=HPARAMS['lr'], weight_decay=HPARAMS['weight_decay'], patience=HPARAMS['patience'], min_lr=HPARAMS['min_lr'], lambda_force=HPARAMS['lambda_force'], device='cuda:0')
    elapsed = time.time() - t0
    out = {'key': f'gatv2_pilot_aspirin_s42_{method}', 'method': method, 'model': 'GATv2', 'molecule': 'aspirin', 'seed': 42, 'n_epochs_actual': metrics['n_epochs_actual'], 'best_epoch': metrics['best_epoch'], 'test_value_mse': metrics['test_energy_mse'], 'test_grad_mse': metrics['test_force_mse'], 'test_energy_mae_mev': metrics['test_energy_mae_mev'], 'test_force_mae_mev': metrics['test_force_mae_mev'], 'time_s': round(elapsed,2), 'timestamp': datetime.now().isoformat()}
    p = RD / f'{out[\"key\"]}.json'
    with open(p,'w') as f: json.dump(out, f, indent=2, default=str)
    print(f'  Saved {p} elapsed={elapsed:.1f}s E_MAE={metrics[\"test_energy_mae_mev\"]:.1f}meV F_MAE={metrics[\"test_force_mae_mev\"]:.1f}meV/A')
" 2>&1 | tee -a "$LOG"

# --- PaiNN × native_EF (100 epochs) ---
echo "=== PaiNN native_EF aspirin s42 100 epochs ===" | tee -a "$LOG"
conda run -n dml-bench-env python3 -c "
import sys, json
sys.path.insert(0, '.')
from experiments.molecular.run_painn import train_one, HPARAMS_CANONICAL
from pathlib import Path
RD = Path('results/pilot_p4'); RD.mkdir(parents=True, exist_ok=True)
hp = dict(HPARAMS_CANONICAL); hp['n_epochs']=100; hp['num_train']=1000; hp['num_val']=1000
result = train_one('aspirin', 'native_EF', 42, hp, smoke=False, gpu=1)
result['key'] = 'painn_pilot_aspirin_s42_native_EF'
p = RD / 'painn_pilot_aspirin_s42_native_EF.json'
with open(p,'w') as f: json.dump(result, f, indent=2, default=str)
print(f'  Saved {p} time={result[\"time_s\"]}s E_MAE={result[\"test_energy_mae_mev\"]:.1f}meV F_MAE={result[\"test_force_mae_mev\"]:.1f}meV/A')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Pilot complete: $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Summary
echo "=== SUMMARY ===" | tee -a "$LOG"
for f in results/pilot_p4/*.json; do
    python3 -c "
import json,sys
d=json.load(open('$f'))
key = d.get('key','?')
mol = d.get('molecule','?')
method = d.get('method','?')
e = d.get('test_energy_mae_mev', d.get('test_energy_mae_approx_mev', float('nan')))
f_ = d.get('test_force_mae_mev', d.get('test_force_mae_approx_mev', float('nan')))
t = d.get('time_s', 0)
print(f'  {key:50s}  E_MAE={e:7.1f}meV  F_MAE={f_:7.1f}meV/A  t={t:6.1f}s')
" 2>/dev/null | tee -a "$LOG"
done

echo "Log: $LOG"
