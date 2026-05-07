# `dml_benchmark/` — DML-Bench Python package

The installable Python library that backs every experiment in this
repository. Importing the package gives the user the full DML training
pipeline: function generators with exact gradients, the DML loss with
component tracking, four adaptive balancers, classical baselines, and
the trainer / metrics scaffolding.

## Installation

```bash
pip install -e .
```

(or `make install` from the repository root). The package is namespaced
`dml_benchmark` (version `0.2.0`); the entry point is `import dml_benchmark`.

## Module map

| Module | Purpose | Key public symbols |
|---|---|---|
| `functions.py`        | Six synthetic function families with JAX-autodiff ground-truth gradients | `generate_data`, `FunctionData`, `add_gaussian_noise` |
| `model.py`            | `DmlFeedForward` MLP with `forward_with_greek` autograd interface; `DmlLoss`, `VanillaLoss` | `DmlFeedForward`, `DmlLoss`, `VanillaLoss` |
| `trainer.py`          | Training loop, ES on total validation loss, ReduceLROnPlateau, atomic JSON write | `DmlTrainer`, `train_single_experiment`, `set_deterministic` |
| `loss_balancing.py`   | Adaptive balancers: GradNorm, ReLoBRaLo, softmax-balance, dim-norm-GradNorm | `GradNormDmlLoss`, `DimNormGradNormDmlLoss`, `SoftmaxBalanceDmlLoss`, `ReLoBRaLoDmlLoss` |
| `baselines.py`        | Classical regressors used in §App F | `GPBaseline`, `KRRBaseline`, `RFBaseline` |
| `metrics.py`          | MSE / DML advantage / per-seed aggregation; results manager | `compute_metrics`, `compute_dml_advantage`, `ResultsManager`, `BenchmarkMetrics` |
| `stats.py`            | Wilcoxon, Holm–Bonferroni, bootstrap CI, Cohen's $d$ family, full pairwise reports | `paired_wilcoxon_test`, `bootstrap_ci`, `cohens_d`, `holm_bonferroni`, `full_comparison_report` |
| `visualization.py`    | Matplotlib rcparams + colour map registration for paper-style figures | `setup_matplotlib`, `STYLE` |
| `config.py`           | Experiment-config dataclasses (smoke / quick / full) | `ExperimentConfig`, `SMOKE_CONFIG` |
| `io.py`               | JSON read/iterate helpers for cell results | `load_result_json`, `iter_result_jsons` |
| `era5_dataset.py`     | Dataset loader for the ERA5 Z500 benchmark | `Era5Dataset`, `load_era5_cells` |
| `fuzzy_smoothing.py`  | Fuzzy-mollifier label construction (digital / barrier payoffs); call-spread + butterfly mollifiers; T-norm / T-conorm fuzzy logic | `call_spread`, `butterfly`, `fuzzy_digital_bs`, `fuzzy_barrier_bs`, `calibrate_epsilon` |
| `lrm_labels.py`       | Likelihood-ratio-method gradient labels (BS digital, barrier, basket, Heston Euler, Heston barrier, BEL multi-step) | `lrm_digital_bs`, `lrm_barrier_bs`, `lrm_basket_bachelier`, `lrm_euler_heston`, `bel_barrier_heston`, `lrm_multistep_heston_barrier` |
| `lrm_labels_bs_barrier.py` | LRM specialisation for the BS barrier multi-step n-sweep | `bs_barrier_lrm_labels` |
| `heston_2d_inputs.py` | Heston path-dependent barrier input construction (V_0 + S_0) | `make_heston_2d_dataset` |
| `high_fidelity_references.py` | Cached high-fidelity Monte-Carlo references for finance-family eval | `load_reference`, `precompute_references` |
| `finance/hedging.py`  | BS delta-hedging back-test scaffold (not on the main benchmark path) | `BlackScholesHedging` |

## Minimal API example

```python
from dml_benchmark.functions import generate_data
from dml_benchmark.trainer import train_single_experiment

data = generate_data("poly_trig", n_dim=5, n_samples=1024, seed=42)

result = train_single_experiment(
    x_train=data.x, y_train=data.y, dydx_train=data.dydx,
    x_test=data.x,  y_test=data.y,  dydx_test=data.dydx,
    method="dml_fixed",
    lambda_=1.0,
    n_epochs=200, batch_size=256,
    n_layers=4, hidden_size=256, lr=0.005, activation="softplus",
    seed=42,
)
print(result.test_value_mse, result.test_grad_mse)
```

## How to register a new method

The current code dispatches methods by name string inside the runners.
Adding a new method is a three-step pattern:

1. Subclass `nn.Module` in `loss_balancing.py` matching the
   `(value_loss, grad_loss, step) -> (w_y, w_g)` interface used by
   the existing `GradNormDmlLoss` / `ReLoBRaLoDmlLoss`.
2. Wire it into the runner's method-dispatch (e.g.
   `run_full_benchmark.py:_make_balancing_loss`,
   `experiments/molecular/run_painn.py:_make_balancing_loss`).
3. Pass `--methods <new_method>` to any of the experiment runners.

## Determinism contract

`set_deterministic(seed)` sets `torch.manual_seed`, `numpy.random.seed`,
`torch.cuda.manual_seed_all`, and `torch.backends.cudnn.deterministic`.
`PYTHONHASHSEED=0` is exported by the runners. Re-running a cell with
the same seed on the same hardware revision produces a JSON whose
`value_mse` and `grad_mse` fields are bit-identical (verified within an
RTX A6000 cluster).

## What lives elsewhere

- The experiment grids that *use* this package live under `experiments/`
  and `scripts/`. This package never knows about result directories
  directly; the runners do.
- Aggregator scripts that turn per-cell JSONs into paper tables live
  under `evidence/` (or, in the current draft tree, under
  `papers/neurips_DB/evidence/`).

## Tests

```bash
python -m pytest tests/ -v
```

158 `def test_` functions are defined across 18 files; on a clean
install ~150 collect successfully and the rest auto-skip when the
relevant external dataset is not yet on disk.
