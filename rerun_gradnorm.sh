#!/bin/bash
# Rerun GradNorm experiments after bug fixes (Bugs 1, 2, 3)
# Strategy:
#   - Delete broken GradNorm results
#   - Rerun with --resume --methods dml_gradnorm --nn-only
#   - Other methods (vanilla, dml_fixed, dml_relobralo, baselines) are untouched
#
# Usage: Run in tmux on GPU 1 while Tier 2/3 runs on GPU 0
#   tmux new -s dml_gradnorm_rerun
#   bash rerun_gradnorm.sh 2>&1 | tee results/gradnorm_rerun.log

set -e

# ---- Portable environment setup ----
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/anaconda3)}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"
cd "$(dirname "$0")"  # project root

echo "=========================================="
echo "  GradNorm Rerun — After Bug Fixes"
echo "  Bugs fixed:"
echo "    1. Gradient contamination (model.zero_grad)"
echo "    2. shared_params (last hidden, not output)"
echo "    3. loss_fn eval mode during validation"
echo "=========================================="

# ---- TIER 1 ----
echo ""
echo "[$(date)] TIER 1: Deleting broken GradNorm results..."
TIER1_COUNT=$(find results/tier1_benchmark -name "*gradnorm*" -type f | wc -l)
echo "  Found $TIER1_COUNT broken GradNorm files"
find results/tier1_benchmark -name "*gradnorm*" -type f -delete
echo "  Deleted."

echo "[$(date)] TIER 1: Rerunning GradNorm (180 experiments, ~1.5h)..."
python run_full_benchmark.py --tier 1 --gpu 1 --resume --methods dml_gradnorm --nn-only

echo "[$(date)] TIER 1: GradNorm rerun complete."

# ---- TIER 2 ----
# Wait for Tier 2 to finish (check if dml_full_nn tmux is still running tier 2)
echo ""
echo "[$(date)] TIER 2: Waiting for main Tier 2 run to complete..."
while true; do
    # Check if any tier2 .tmp files exist (atomic write in progress)
    TMP_COUNT=$(find results/tier2_benchmark -name "*.tmp" -type f 2>/dev/null | wc -l)
    # Check if the tier2 summary.json exists (written at end of run)
    if [ -f "results/tier2_benchmark/summary.json" ] && [ "$TMP_COUNT" -eq 0 ]; then
        echo "  Tier 2 main run appears complete (summary.json exists, no .tmp files)"
        break
    fi
    echo "  [$(date)] Still waiting... ($TMP_COUNT .tmp files remain)"
    sleep 120
done

echo "[$(date)] TIER 2: Deleting broken GradNorm results..."
TIER2_COUNT=$(find results/tier2_benchmark -name "*gradnorm*" -type f | wc -l)
echo "  Found $TIER2_COUNT broken GradNorm files"
find results/tier2_benchmark -name "*gradnorm*" -type f -delete
echo "  Deleted."

echo "[$(date)] TIER 2: Rerunning GradNorm (~1050 experiments, ~8h)..."
python run_full_benchmark.py --tier 2 --gpu 1 --resume --methods dml_gradnorm --nn-only

echo "[$(date)] TIER 2: GradNorm rerun complete."

# ---- UPDATE SUMMARIES ----
echo ""
echo "[$(date)] Regenerating summaries for Tier 1 and Tier 2..."
python -c "
import json, numpy as np
from pathlib import Path

for tier in [1, 2]:
    results_dir = Path(f'results/tier{tier}_benchmark')
    all_results = {}
    for f in results_dir.glob('*.json'):
        if f.name == 'summary.json': continue
        with open(f) as fh:
            d = json.load(fh)
            all_results[f.stem] = d
    
    # Rebuild summary
    method_groups = {}
    for key, r in all_results.items():
        m = r['method']
        if m not in method_groups:
            method_groups[m] = {'val': [], 'grad': [], 'times': []}
        method_groups[m]['val'].append(r['test_value_mse'])
        method_groups[m]['grad'].append(r['test_grad_mse'])
        method_groups[m]['times'].append(r.get('time_s', 0))
    
    summary = {
        'n_experiments': len(all_results),
        'method_summary': {},
        'note': 'Updated after GradNorm bug fix rerun',
    }
    for m, d in method_groups.items():
        summary['method_summary'][m] = {
            'mean_value_mse': float(np.mean(d['val'])),
            'std_value_mse': float(np.std(d['val'])),
            'mean_grad_mse': float(np.mean(d['grad'])),
            'std_grad_mse': float(np.std(d['grad'])),
            'mean_time_s': float(np.mean(d['times'])),
            'count': len(d['val']),
        }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f'Tier {tier}: {len(all_results)} total results')
    for m in sorted(summary['method_summary'].keys()):
        s = summary['method_summary'][m]
        print(f'  {m:<25} val_mse={s[\"mean_value_mse\"]:>12.4f}  grad_mse={s[\"mean_grad_mse\"]:>12.4f}  n={s[\"count\"]}')
"

echo ""
echo "[$(date)] ALL DONE — GradNorm rerun complete for Tiers 1 & 2."
echo "Tier 3 will use the fixed code automatically (new Python process)."
