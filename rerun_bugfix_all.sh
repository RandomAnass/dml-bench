#!/bin/bash
# ==========================================================================
#  Comprehensive Bugfix Rerun — GradNorm + ReLoBRaLo (Tiers 1, 2, 3)
# ==========================================================================
# Bugs fixed (see AUDIT_REPORT.md):
#   Bug A: GradNorm shared_params selected output layer instead of last hidden
#   Bug B: GradNorm gradients contaminated model params (missing zero_grad)
#   Bug 2: loss_fn.eval() not called during validation — corrupts adaptive
#          weights/history. GradNorm catastrophically, ReLoBRaLo subtly
#          (50% validation contamination in loss_history for n<=1024)
#
# Strategy:
#   1. Delete broken results for both methods, rerun with --resume
#   2. Other methods (vanilla, dml_fixed, baselines) are untouched
#   3. Wait for Tier 2 main run to finish before Tier 2 rerun
#   4. Tier 3 uses fixed code automatically (launched as new process)
#
# Usage:
#   tmux new -s dml_bugfix_rerun
#   bash rerun_bugfix_all.sh 2>&1 | tee results/bugfix_rerun.log
# ==========================================================================

set -e

# ---- Portable environment setup ----
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/anaconda3)}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"
cd "$(dirname "$0")"  # project root

GPU=1  # GPU to use for reruns (GPU 0 is running Tier 2 main)
METHODS="dml_gradnorm dml_relobralo"

echo "=========================================================="
echo "  Comprehensive Bugfix Rerun"
echo "  Methods:  ${METHODS}"
echo "  GPU:      ${GPU}"
echo "  Started:  $(date)"
echo "  Bugs:     A (shared_params), B (zero_grad), 2 (eval mode)"
echo "=========================================================="

# ==========================================================================
# TIER 1 RERUN
# ==========================================================================
echo ""
echo "############################################################"
echo "  TIER 1: GradNorm + ReLoBRaLo Rerun"
echo "############################################################"

# Delete broken GradNorm results
GCOUNT=$(find results/tier1_benchmark -name "*gradnorm*" -type f 2>/dev/null | wc -l)
echo "[$(date)] Deleting $GCOUNT broken GradNorm files..."
find results/tier1_benchmark -name "*gradnorm*" -type f -delete 2>/dev/null || true

# Delete contaminated ReLoBRaLo results
RCOUNT=$(find results/tier1_benchmark -name "*relobralo*" -type f 2>/dev/null | wc -l)
echo "[$(date)] Deleting $RCOUNT contaminated ReLoBRaLo files..."
find results/tier1_benchmark -name "*relobralo*" -type f -delete 2>/dev/null || true

echo "[$(date)] TIER 1: Rerunning ${METHODS} (360 experiments, ~2.5h)..."
python run_full_benchmark.py \
    --tier 1 \
    --gpu ${GPU} \
    --resume \
    --methods ${METHODS} \
    --nn-only

echo ""
echo "[$(date)] ✅ TIER 1 RERUN COMPLETE"
echo ""

# Quick sanity check on Tier 1 results
python -c "
import json, numpy as np
from pathlib import Path

results_dir = Path('results/tier1_benchmark')
for method in ['dml_gradnorm', 'dml_relobralo']:
    files = list(results_dir.glob(f'*{method}*'))
    if not files:
        print(f'  ⚠️  {method}: No results found!')
        continue
    vals = [json.load(open(f))['test_value_mse'] for f in files]
    epochs = [json.load(open(f))['best_epoch'] for f in files]
    print(f'  {method}: n={len(files)}, mean_mse={np.mean(vals):.4f}, mean_epoch={np.mean(epochs):.0f}')
    bad = sum(1 for v in vals if v > 1e6)
    if bad > 0:
        print(f'    ⚠️  {bad} results with MSE > 1M — something may still be wrong!')
    early = sum(1 for e in epochs if e < 5)
    if early > len(epochs)*0.5:
        print(f'    ⚠️  {early}/{len(epochs)} stuck at early epochs — check convergence')
"

# ==========================================================================
# WAIT FOR TIER 2 MAIN RUN (on GPU 0)
# ==========================================================================
echo ""
echo "############################################################"
echo "  Waiting for Tier 2 main run to complete on GPU 0..."
echo "############################################################"

# The main Tier 2 run writes summary.json at the very end.
# We also check that no .tmp files remain (atomic write pattern).
WAIT_START=$(date +%s)
while true; do
    TMP_COUNT=$(find results/tier2_benchmark -name "*.tmp" -type f 2>/dev/null | wc -l)
    TOTAL=$(find results/tier2_benchmark -name "*.json" ! -name "summary.json" -type f 2>/dev/null | wc -l)
    
    if [ -f "results/tier2_benchmark/summary.json" ] && [ "$TMP_COUNT" -eq 0 ]; then
        WAIT_END=$(date +%s)
        WAIT_MINS=$(( (WAIT_END - WAIT_START) / 60 ))
        echo "  [$(date)] Tier 2 main run complete! (waited ${WAIT_MINS}m)"
        echo "  Total results: ${TOTAL}"
        break
    fi
    
    echo "  [$(date)] Still running... ${TOTAL} results, ${TMP_COUNT} .tmp files"
    sleep 120
done

# ==========================================================================
# TIER 2 RERUN
# ==========================================================================
echo ""
echo "############################################################"
echo "  TIER 2: GradNorm + ReLoBRaLo Rerun"
echo "############################################################"

# Delete broken GradNorm results
GCOUNT=$(find results/tier2_benchmark -name "*gradnorm*" -type f 2>/dev/null | wc -l)
echo "[$(date)] Deleting $GCOUNT broken GradNorm files..."
find results/tier2_benchmark -name "*gradnorm*" -type f -delete 2>/dev/null || true

# Delete contaminated ReLoBRaLo results
RCOUNT=$(find results/tier2_benchmark -name "*relobralo*" -type f 2>/dev/null | wc -l)
echo "[$(date)] Deleting $RCOUNT contaminated ReLoBRaLo files..."
find results/tier2_benchmark -name "*relobralo*" -type f -delete 2>/dev/null || true

echo "[$(date)] TIER 2: Rerunning ${METHODS} (~2100 experiments, ~12h)..."
python run_full_benchmark.py \
    --tier 2 \
    --gpu ${GPU} \
    --resume \
    --methods ${METHODS} \
    --nn-only

echo ""
echo "[$(date)] ✅ TIER 2 RERUN COMPLETE"
echo ""

# ==========================================================================
# REGENERATE SUMMARIES
# ==========================================================================
echo ""
echo "############################################################"
echo "  Regenerating Tier 1 & Tier 2 Summaries"
echo "############################################################"

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
        'note': 'Regenerated after GradNorm + ReLoBRaLo bugfix rerun',
        'rerun_date': '$(date -Iseconds)',
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

    print(f'\\n=== Tier {tier}: {len(all_results)} total results ===')
    for m in sorted(summary['method_summary'].keys()):
        s = summary['method_summary'][m]
        print(f'  {m:<25} val_mse={s[\"mean_value_mse\"]:>12.4f}  grad_mse={s[\"mean_grad_mse\"]:>12.4f}  n={s[\"count\"]}')
"

# ==========================================================================
# FINAL REPORT
# ==========================================================================
echo ""
echo "=========================================================="
echo "  ALL RERUNS COMPLETE"
echo "  Finished: $(date)"
echo "=========================================================="
echo ""
echo "Next steps:"
echo "  - Tier 3 will use fixed code automatically (new Python process)"
echo "  - Review Tier 1 & 2 summaries in results/tierX_benchmark/summary.json"
echo "  - Run statistical tests when ready"
echo ""
