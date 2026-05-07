#!/bin/bash
# ==========================================================================
#  Tier 3 Launch — Split across 2 GPUs for ~47h wall clock
# ==========================================================================
# GPU 0: vanilla + dml_fixed + lambda sweep  (~3200 NN runs)
# GPU 1: dml_gradnorm + dml_relobralo        (~2880 NN runs)
# CPU:   baselines (after Tier 2 BL finishes) (~3405 runs)
#
# Each GPU instance uses --resume for cross-tier dedup.
# Results saved to results/tier3_benchmark/
#
# Usage:
#   # Terminal 1 (GPU 0) — start immediately
#   tmux new -s dml_tier3_gpu0
#   bash run_tier3.sh gpu0 2>&1 | tee results/tier3_gpu0.log
#
#   # Terminal 2 (GPU 1) — start after bugfix rerun finishes
#   tmux new -s dml_tier3_gpu1
#   bash run_tier3.sh gpu1 2>&1 | tee results/tier3_gpu1.log
#
#   # Terminal 3 (CPU) — start after Tier 2 baselines finish
#   tmux new -s dml_tier3_bl
#   bash run_tier3.sh baselines 2>&1 | tee results/tier3_bl.log
# ==========================================================================

set -e

# ---- Portable environment setup ----
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/anaconda3)}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-dml-bench-env}"
cd "$(dirname "$0")"  # project root

MODE="${1:-gpu0}"

case "$MODE" in
    gpu0)
        echo "=========================================================="
        echo "  Tier 3 — GPU 0: vanilla + dml_fixed + lambda sweep"
        echo "  Started: $(date)"
        echo "  Est: ~3200 NN runs, ~47h"
        echo "=========================================================="
        python run_full_benchmark.py \
            --tier 3 \
            --gpu 0 \
            --resume \
            --methods vanilla dml_fixed \
            --nn-only
        ;;
    gpu1)
        echo "=========================================================="
        echo "  Tier 3 — GPU 1: dml_gradnorm + dml_relobralo"
        echo "  Started: $(date)"
        echo "  Est: ~2880 NN runs, ~47h"
        echo "=========================================================="
        
        # Wait for bugfix rerun to finish if still running
        while true; do
            # Check if dml_bugfix_rerun is still actively running experiments
            BUGFIX_RUNNING=$(tmux capture-pane -t dml_bugfix_rerun -p 2>/dev/null | grep -c "ALL RERUNS COMPLETE" || true)
            BUGFIX_SHELL=$(tmux capture-pane -t dml_bugfix_rerun -p 2>/dev/null | tail -1 | grep -c '\$' || true)
            if [ "$BUGFIX_RUNNING" -gt 0 ] || [ "$BUGFIX_SHELL" -gt 0 ]; then
                echo "  Bugfix rerun finished, starting Tier 3 on GPU 1..."
                break
            fi
            # Also check if tmux session doesn't exist
            if ! tmux has-session -t dml_bugfix_rerun 2>/dev/null; then
                echo "  Bugfix rerun session gone, starting Tier 3 on GPU 1..."
                break
            fi
            echo "  [$(date)] Waiting for bugfix rerun to finish..."
            sleep 60
        done
        
        python run_full_benchmark.py \
            --tier 3 \
            --gpu 1 \
            --resume \
            --methods dml_gradnorm dml_relobralo \
            --nn-only
        ;;
    baselines)
        echo "=========================================================="
        echo "  Tier 3 — CPU: baselines (GP, KRR, RF)"
        echo "  Started: $(date)"
        echo "  Est: ~3405 baseline runs, ~44h"
        echo "=========================================================="
        python run_full_benchmark.py \
            --tier 3 \
            --resume \
            --baselines-only
        ;;
    *)
        echo "Usage: bash run_tier3.sh {gpu0|gpu1|baselines}"
        exit 1
        ;;
esac

echo ""
echo "[$(date)] $MODE COMPLETE"

# If this is the last GPU job, regenerate summary
echo "Regenerating Tier 3 summary..."
python -c "
import json, numpy as np
from pathlib import Path

results_dir = Path('results/tier3_benchmark')
all_results = {}
for f in results_dir.glob('*.json'):
    if f.name == 'summary.json': continue
    with open(f) as fh:
        d = json.load(fh)
        all_results[f.stem] = d

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
    'note': 'Tier 3 partial summary (may be updated by other GPU)',
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

print(f'Tier 3: {len(all_results)} results')
for m in sorted(summary['method_summary'].keys()):
    s = summary['method_summary'][m]
    print(f'  {m:<25} val_mse={s[\"mean_value_mse\"]:>12.4f}  grad_mse={s[\"mean_grad_mse\"]:>12.4f}  n={s[\"count\"]}')
"
