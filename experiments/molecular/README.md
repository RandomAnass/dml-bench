# `experiments/molecular/` — rMD17 force learning

The molecular-force-field pillar (§5.4). Tests DML on the ten rMD17
small-organic-molecule force-field benchmarks across three backbones:
PaiNN (canonical equivariant), GATv2 (graph attention), and
MLP-pairwise (Cartesian-force chain rule). The headline finding is
that the canonical chemistry-community weighting `native_EF` (E:F =
0.01:0.99) outperforms every DML balancer we tested.

## Output corpus

| Sub-dir | Cells | Producing script | What it shows |
|---|---:|---|---|
| `results/molecular_painn/`            | 350+ | `run_painn.py`               | PaiNN, 10 mol × 5 splits × 7 methods |
| `results/molecular_mlp/`              | 350  | `run_mlp_molecular.py`       | MLP-pairwise, same shape |
| `results/molecular_gatv2/`            | 350  | `../gnn_md17.py`             | GATv2 (legacy filename, post-fix corpus) |
| `results/rmd17_tau_sweep/`            | 15   | `../../scripts/run_rmd17_tau_sweep.py` | PaiNN τ-sensitivity (3 τ × 5 seeds, aspirin) |

## Headline findings

- **Force MAE reduction:** DML cuts force MAE 25–40× over the vanilla
  baseline on rMD17 aspirin (725.3 → 17.9–28.8 meV/Å).
- **Friedman omnibus across all PaiNN molecules:**
  $p = 4.8 \times 10^{-9}$.
- **Cross-arch ranking:** `native_EF` (the canonical SchNetPack
  weighting) ranks 1/6 across PaiNN, GATv2, and MLP-pairwise.
- **τ-sensitivity:** lowering τ from 0.50 to 0.10 recovers 9.6× in
  force-MAE quality on aspirin (713.2 → 74.1 meV/Å). The native_EF
  weighting at τ=0.5 is approximately the vanilla-baseline regression.

## Scripts

| Script | Purpose |
|---|---|
| `run_painn.py`             | PaiNN runner (SchNetPack 2.2.0 programmatic API). 7 methods: vanilla, native_EF, dml_fixed, dml_fixed_half, dml_gradnorm, dml_relobralo, dml_warmup. |
| `run_mlp_molecular.py`     | MLP-pairwise runner using a hand-rolled chain-rule for Cartesian forces. Same 7 methods. |
| `_build_painn_db.py`       | One-shot helper: builds the SchNetPack-format DB from the rMD17 npz files; cached at `data/schnetpack_rmd17/`. |

## Running

```bash
# PaiNN full grid (~150 GPU-h on one A6000):
python experiments/molecular/run_painn.py            --gpu 0

# MLP-pairwise full grid (~60 GPU-h):
python experiments/molecular/run_mlp_molecular.py    --gpu 0

# GATv2 full grid (~80 GPU-h):
python experiments/gnn_md17.py                       --gpu 0

# τ-sensitivity on aspirin (~3 GPU-h):
python scripts/run_rmd17_tau_sweep.py                --gpu 0
```

## Smoke

Each runner accepts `--smoke` to run a single molecule × single
seed. Useful for quick CI:

```bash
python experiments/molecular/run_painn.py --gpu 0 --molecules ethanol --seeds 42 --smoke
```

## Methods

PaiNN-compatible methods (registered in
`dml_benchmark/loss_balancing.py:METHOD_REGISTRY` and translated to
SchNetPack tasks in `run_painn.py`):

- `native_EF` — canonical PaiNN-on-rMD17 setup with E:F weight
  0.01:0.99. Loss is energy MSE + force MSE; force gradients come from
  autograd of the energy. **This is the chemistry-community baseline**;
  named "native" because it does not invoke any of our DML balancers.
- `vanilla`              — energy-only loss, force learned via autograd
- `dml_fixed`            — H&S fixed-λ
- `dml_fixed_half`       — 50/50 weight (disentangle dim-aware from any DML)
- `dml_gradnorm`         — GradNorm balancer
- `dml_relobralo`        — ReLoBRaLo balancer
- `dml_warmup`           — two-phase warmup (energy-only → DML)

## Salicylic-acid filename convention

The rMD17 archive stores the file as `rmd17_salicylic.npz` while ASE
expects `salicylic_acid`. The runner applies a one-line key
translation at load time. The result-JSON filenames preserve the
Figshare convention so the aggregators do not need to know.

## Aggregators

| Aggregator | Output |
|---|---|
| `paper/figures/scripts/fig_rmd17_force_mae.py`     | force-MAE bar plot (`fig:rmd17-force-mae`) |
| `paper/figures/scripts/fig_rmd17_cd_diagram.py`    | CD diagram (`fig:cd-rmd17`) |
| `paper/figures/scripts/fig_rmd17_tau_sensitivity.py` | τ-sweep plot (`fig:rmd17-tau-sensitivity`) |
| `evidence/rmd17_tau_sweep.py`                       | `tab:rmd17-tau-sweep` |

## Tests

`tests/test_rmd17_cross_arch_splits.py` confirms split-index alignment
across the three backbones; needs `data/rmd17/rmd17/`.
