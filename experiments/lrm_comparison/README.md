# `experiments/lrm_comparison/` — LRM vs adaptive balancing

LRM (likelihood-ratio method) gradient labels are the
unbiased-but-noisy alternative to pathwise gradients on discontinuous
payoffs. This sub-pillar tests whether adaptive balancing (GradNorm,
ReLoBRaLo) can automatically down-weight noisy LRM labels and beat
the Glasserman & Karmarkar (2025) fixed-50/50 weighting.

## Output corpus

`results/lrm_comparison/` (~680–920 cells across the three sub-experiments).

## Sub-experiments

| Sub-experiment | What it varies | Datasets |
|---|---|---|
| **A. LRM + adaptive vs fixed-λ** | balancer ∈ {fixed-λ, GradNorm, ReLoBRaLo} | digital, barrier, basket-digital |
| **B. LRM variance scaling**       | dimension $d \in \{1, …, 50\}$           | basket-digital |
| **D. Network size sensitivity**   | architecture ∈ {4×20, 4×256}             | digital |

## Hypothesis

GradNorm should automatically down-weight noisy LRM labels,
outperforming G&K's fixed 50-50 weighting. The advantage should grow
with $d$ because LRM variance scales as $\mathcal{O}(d)$ (Glasserman 2004).

## Scripts

| Script | Purpose |
|---|---|
| `run_lrm_vs_adaptive.py` | Sub-experiments A, B, and D combined |
| `run_heston_lrm.py`      | Heston-Euler digital with LRM labels (also referenced by `experiments/heston_barrier_4way/`) |

## Running

```bash
python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0           # all three
python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0 --only digital
python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0 --only dimscaling
python experiments/lrm_comparison/run_heston_lrm.py     --gpu 0
```

Expected runtime: ~2–4 GPU-h on one A6000.

## Aggregator

`evidence/lrm_summary.py` emits the LRM-variance-scaling table behind
`fig:lrm_variance_scaling` and the side-by-side balancer table.
