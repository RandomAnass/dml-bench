#!/usr/bin/env python3
"""
MLP molecular runner for DML-Bench Phase 4 (pairwise-distance representation).

Input representation:
  Pairwise interatomic distances {|r_i - r_j|} for all unordered pairs, giving
  d = n_atoms · (n_atoms-1) / 2 features per frame. This is the standard MLP
  baseline for MD17/rMD17 (Chmiela et al. 2017 sGDML: inverse distances;
  Christensen & von Lilienfeld 2020 rMD17; SchNet uses Gaussian-smeared
  distances internally). Flat XYZ is NOT a valid MD17 baseline because it
  breaks rotational/translational invariance.

Derivative labels:
  Forces F_i = -∂E/∂R_i (kcal/mol/Å) from DFT are converted to ∂E/∂d_ij
  via the chain rule:
      ∂E/∂R_{i,α} = Σ_j ∂E/∂d_ij · (R_{i,α} - R_{j,α})/d_ij
  Stacking this over all atoms/dimensions yields a (3N × n_pairs) Jacobian A
  such that -F_flat = A @ g, where g = ∂E/∂d. We solve g = lstsq(A, -F_flat)
  per frame (under-determined for n_pairs > 3N; least-squares residual
  printed as a sanity check at load time).

Protocol:
  Canonical rMD17 Figshare splits 1..5. Each split: n_train=950, n_val=50,
  n_test=1000 (SchNetPack v2.2.0 canonical). Total grid per molecule: 5 splits.

Hyperparameters (literature-backed for small-data force learning on MD17):
  hidden=512, n_layers=4, softplus, n_epochs=1000, batch=128, lr=1e-3.
  See Chmiela et al. 2017/2018 and PaiNN paper Table 5 for precedent.

Methods (7 total):
  vanilla, dml_fixed (H&S), dml_fixed_half (0.5/0.5), dml_gradnorm,
  dml_softmax_balance (simplified ReLoBRaLo), dml_relobralo (faithful Eq.11),
  dml_warmup (two-phase).

Usage:
  python experiments/molecular/run_mlp_molecular.py --gpu 1
  python experiments/molecular/run_mlp_molecular.py --gpu 1 --molecules ethanol
  python experiments/molecular/run_mlp_molecular.py --gpu 1 --resume

Grid size: 10 molecules × 5 canonical splits × 7 methods = 350 runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================================
# CONFIG
# ============================================================================

METHODS = [
    "vanilla",
    "dml_fixed",            # H&S 1/(1+λd) weights
    "dml_fixed_half",       # 0.5/0.5 weights (cross-arch parity)
    "dml_gradnorm",
    "dml_softmax_balance",  # Simplified softmax balancing
    "dml_relobralo",        # Faithful Bischof & Kraus 2022 Eq.11
    "dml_warmup",
]

DEFAULT_MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]
DEFAULT_SPLIT_IDS = [1, 2, 3, 4, 5]
KCAL_TO_MEV = 43.3641

# Hyperparameters aligned with MD17 literature precedent (sGDML, SchNet, PaiNN).
# hidden=512 handles complex PES; lr=1e-3 stable for force-dominated loss at
# small data (n_train=950); 1000 epochs is standard MD17 protocol.
HPARAMS = {
    "n_epochs": 1000,
    "batch_size": 128,
    "lr": 0.001,
    "n_layers": 4,
    "hidden_size": 512,
    "activation": "softplus",
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    "warmup_fraction": 0.5,
}


# ============================================================================
# DATA LOADING — Pairwise distances + chain-rule ∂E/∂d conversion
# ============================================================================

def load_rmd17_pairwise(
    molecule: str,
    data_dir: str = "data/rmd17/rmd17/npz_data",
    splits_dir: str = "data/rmd17/rmd17/splits",
    n_train: int = 950,
    n_val: int = 50,
    n_test: int = 1000,
    split_id: int = 1,
    verbose: bool = True,
    return_test_jacobian: bool = False,
):
    """
    Load rMD17 with pairwise-distance features and chain-rule-converted
    derivative labels.

    Returns:
        (x_train, y_train, dydx_train),
        (x_val,   y_val,   dydx_val),
        (x_test,  y_test,  dydx_test),
        metadata dict
        + if return_test_jacobian: also (A_test, F_test_flat) for Cartesian
          force MSE reconstruction (I-H1)
    Shapes:
        x:    (N, n_pairs)        pairwise distances, units = Å
        y:    (N, 1)              shifted energies (zero-mean), units = kcal/mol
        dydx: (N, 1, n_pairs)     ∂E/∂d_ij via lstsq, units = kcal/mol/Å
    """
    npz_path = Path(data_dir) / f"rmd17_{molecule}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"rMD17 data not found at {npz_path}")

    raw = np.load(str(npz_path))
    coords = raw["coords"].astype(np.float64)         # (N, n_atoms, 3) Å
    energies = raw["energies"].astype(np.float64)     # (N,) kcal/mol
    forces = raw["forces"].astype(np.float64)         # (N, n_atoms, 3) kcal/mol/Å
    nuclear_charges = raw["nuclear_charges"]

    n_frames, n_atoms, _ = coords.shape
    pairs = list(combinations(range(n_atoms), 2))
    n_pairs = len(pairs)

    energy_mean = float(energies.mean())
    energies_shifted = energies - energy_mean

    if verbose:
        print(f"  Loaded {molecule}: {n_frames} frames, {n_atoms} atoms → d={n_pairs} pairwise distances")

    # Canonical Figshare splits
    train_csv = Path(splits_dir) / f"index_train_0{split_id}.csv"
    test_csv = Path(splits_dir) / f"index_test_0{split_id}.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Canonical splits not found at {train_csv} / {test_csv}."
        )
    train_idx_full = np.loadtxt(str(train_csv)).flatten().astype(int)
    test_idx_full = np.loadtxt(str(test_csv)).flatten().astype(int)

    if n_train + n_val > len(train_idx_full):
        raise ValueError(
            f"Requested n_train={n_train} + n_val={n_val} > canonical split "
            f"{split_id} train has {len(train_idx_full)}."
        )
    # Deterministic sub-split via split_id as seed (same across methods/seeds)
    rng = np.random.RandomState(split_id)
    perm = rng.permutation(len(train_idx_full))
    train_idx = train_idx_full[perm[:n_train]]
    val_idx = train_idx_full[perm[n_train:n_train + n_val]]

    if n_test < len(test_idx_full):
        test_rng = np.random.RandomState(split_id + 1000)
        test_perm = test_rng.permutation(len(test_idx_full))
        test_idx = test_idx_full[test_perm[:n_test]]
    else:
        test_idx = test_idx_full

    # Pre-allocate
    def _build(indices):
        N = len(indices)
        x = np.zeros((N, n_pairs), dtype=np.float64)
        dydx = np.zeros((N, 1, n_pairs), dtype=np.float64)
        y = energies_shifted[indices].reshape(N, 1)

        for k, frame_idx in enumerate(indices):
            R = coords[frame_idx]  # (n_atoms, 3)
            F = forces[frame_idx]  # (n_atoms, 3)

            # Jacobian A (3N × n_pairs) such that -F_flat = A @ g
            A = np.zeros((3 * n_atoms, n_pairs), dtype=np.float64)
            for pair_idx, (i, j) in enumerate(pairs):
                diff = R[i] - R[j]
                dist = float(np.sqrt((diff ** 2).sum()))
                x[k, pair_idx] = dist
                unit = diff / (dist + 1e-12)
                A[3 * i:3 * i + 3, pair_idx] = unit
                A[3 * j:3 * j + 3, pair_idx] = -unit

            neg_F_flat = -F.reshape(-1)
            g, _, _, _ = np.linalg.lstsq(A, neg_F_flat, rcond=None)
            dydx[k, 0, :] = g

        return x.astype(np.float32), y.astype(np.float32), dydx.astype(np.float32)

    train = _build(train_idx)
    val = _build(val_idx)
    test = _build(test_idx)

    # I-H1 (2026-04-16): per-frame Jacobians + Cartesian forces for test set,
    # to enable Cartesian-force MSE computation at eval time. n_pairs > 3N
    # means lstsq-derived ∂E/∂d is under-determined (min-norm solution), so
    # pairwise-distance grad MSE is NOT cross-arch comparable. The Cartesian
    # reconstruction F_pred = -A @ dE/dd_pred IS comparable to GATv2/PaiNN
    # because A @ g is uniquely determined.
    test_jacobians = None
    test_forces_flat = None
    if return_test_jacobian:
        test_jacobians = np.zeros((len(test_idx), 3 * n_atoms, n_pairs), dtype=np.float32)
        test_forces_flat = np.zeros((len(test_idx), 3 * n_atoms), dtype=np.float32)
        for k, frame_idx in enumerate(test_idx):
            R = coords[frame_idx]
            F = forces[frame_idx]
            A_k = np.zeros((3 * n_atoms, n_pairs), dtype=np.float64)
            for pair_idx, (i, j) in enumerate(pairs):
                diff = R[i] - R[j]
                dist = float(np.sqrt((diff ** 2).sum()))
                unit = diff / (dist + 1e-12)
                A_k[3 * i:3 * i + 3, pair_idx] = unit
                A_k[3 * j:3 * j + 3, pair_idx] = -unit
            test_jacobians[k] = A_k.astype(np.float32)
            test_forces_flat[k] = F.reshape(-1).astype(np.float32)

    # I-H1: informative load-time diagnostic — rank of A vs n_pairs.
    # For under-determined systems, (n_pairs - rank(A)) > 0 means there's a
    # null space and lstsq returns an arbitrary min-norm vector in it.
    R0 = coords[train_idx[0]]
    F0 = forces[train_idx[0]]
    A0 = np.zeros((3 * n_atoms, n_pairs))
    for pair_idx, (i, j) in enumerate(pairs):
        diff = R0[i] - R0[j]
        dist = np.sqrt((diff ** 2).sum())
        unit = diff / (dist + 1e-12)
        A0[3 * i:3 * i + 3, pair_idx] = unit
        A0[3 * j:3 * j + 3, pair_idx] = -unit
    recon_F = -(A0 @ train[2][0, 0, :].astype(np.float64))
    residual_rmse = float(np.sqrt(((recon_F - F0.reshape(-1)) ** 2).mean()))
    rank_A = int(np.linalg.matrix_rank(A0))
    nullspace_dim = n_pairs - rank_A
    if verbose:
        print(f"  Jacobian A: shape=({3*n_atoms},{n_pairs}), rank={rank_A}, "
              f"nullspace_dim={nullspace_dim}")
        print(f"  Force reconstruction RMSE (A @ g ≈ -F): {residual_rmse:.6e} kcal/mol/Å")
        if nullspace_dim > 0:
            print(f"  NOTE: n_pairs > 3N → ∂E/∂d from lstsq is min-norm (arbitrary "
                  f"in {nullspace_dim}-dim null space). Cartesian-force MSE via "
                  f"F_pred = -A @ dE/dd is the cross-arch-comparable metric.")

    metadata = {
        "molecule": molecule,
        "n_atoms": int(n_atoms),
        "n_pairs": int(n_pairs),
        "input_dim": int(n_pairs),
        "feature_type": "pairwise_distances",
        "derivative_type": "dE_d(pairwise_distance)_via_lstsq",
        "force_reconstruction_rmse_kcal_per_mol_per_angstrom": residual_rmse,
        "jacobian_rank": rank_A,
        "jacobian_nullspace_dim": nullspace_dim,
        "n_frames_total": int(n_frames),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "energy_mean_kcal_per_mol": energy_mean,
        "energy_std_kcal_per_mol": float(energies.std()),
        "nuclear_charges": nuclear_charges.tolist(),
        "data_dir": str(data_dir),
        "splits_dir": str(splits_dir),
        "split_id": int(split_id),
        "split_source": f"canonical_split_{split_id}",
    }
    # I-H1 (2026-04-16): always return 5-tuple. Jacobians may be None.
    test_extras = None
    if return_test_jacobian:
        test_extras = {
            "A_test": test_jacobians,        # (n_test, 3N, n_pairs)
            "F_test_flat": test_forces_flat, # (n_test, 3N)  raw forces, kcal/mol/Å
        }
    return train, val, test, metadata, test_extras


# ============================================================================
# TRAIN ONE (molecule, method, split)
# ============================================================================

def train_one(molecule, method, split_id, hparams):
    """Run one (molecule, method, split_id) configuration."""
    # Late imports so CUDA_VISIBLE_DEVICES takes effect
    import torch
    from dml_benchmark.trainer import train_single_experiment
    from dml_benchmark.model import DmlFeedForward, DataNormalizer
    from experiments.unified_comparison.run_unified_experiment import train_warmup

    # I-H1 (2026-04-16): return_test_jacobian=True for Cartesian-force MSE.
    (x_tr, y_tr, dy_tr), (x_va, y_va, dy_va), (x_te, y_te, dy_te), meta, test_extras = \
        load_rmd17_pairwise(
            molecule, split_id=split_id,
            n_train=950, n_val=50, n_test=1000, verbose=False,
            return_test_jacobian=True,
        )

    # J-M5 (2026-04-16): seed varies with BOTH split_id AND method so that
    # per-method random init/shuffle sequences are distinct. Previously, seed
    # only depended on split_id, so within a split all methods shared the same
    # DataLoader shuffle and the same weight init — hiding method-specific
    # variance in cross-method comparisons.
    method_hash = sum(ord(c) for c in method) & 0xFFFF
    seed = split_id * 101 + 42 + method_hash

    if method == "dml_warmup":
        # Pass explicit val to preserve the 950/50 canonical split.
        result = train_warmup(
            x_train=x_tr, y_train=y_tr, dydx_train=dy_tr,
            x_test=x_te, y_test=y_te, dydx_test=dy_te,
            x_val=x_va, y_val=y_va, dydx_val=dy_va,
            warmup_fraction=hparams.get("warmup_fraction", 0.5),
            seed=seed, pbar=False,
            n_epochs=hparams["n_epochs"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            n_layers=hparams["n_layers"],
            hidden_size=hparams["hidden_size"],
            activation=hparams["activation"],
            max_grad_norm=hparams["max_grad_norm"],
            scheduler_patience=hparams["scheduler_patience"],
            scheduler_factor=hparams["scheduler_factor"],
        )
    else:
        result = train_single_experiment(
            x_train=x_tr, y_train=y_tr, dydx_train=dy_tr,
            x_test=x_te, y_test=y_te, dydx_test=dy_te,
            x_val=x_va, y_val=y_va, dydx_val=dy_va,
            method=method, seed=seed, pbar=False,
            n_epochs=hparams["n_epochs"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            n_layers=hparams["n_layers"],
            hidden_size=hparams["hidden_size"],
            activation=hparams["activation"],
            lambda_=hparams["lambda_"],
            max_grad_norm=hparams["max_grad_norm"],
            scheduler_patience=hparams["scheduler_patience"],
            scheduler_factor=hparams["scheduler_factor"],
        )

    energy_mse = float(result.test_value_mse)
    forces_mse = float(result.test_grad_mse)
    energy_rmse = float(np.sqrt(max(energy_mse, 0.0)))
    deriv_rmse = float(np.sqrt(max(forces_mse, 0.0)))

    # I-H1 (2026-04-16): reconstruct Cartesian force MSE via chain rule.
    #   F_pred = -A @ (dE/dd_pred)  where  A = ∂d/∂R  (3N × n_pairs)
    # A @ g is UNIQUE even though g (min-norm lstsq) is not. So the Cartesian
    # forces predicted by the trained MLP are physically well-defined and
    # cross-arch-comparable with GATv2 / PaiNN.
    test_force_mse_cartesian = float("nan")
    if test_extras is not None and result.best_model_state is not None:
        # Rebuild model + normalizer, load best state, rerun inference.
        model = DmlFeedForward(
            input_dim=x_tr.shape[1], output_dim=1,
            n_layers=hparams["n_layers"],
            hidden_size=hparams["hidden_size"],
            activation=hparams["activation"],
        )
        model.load_state_dict(result.best_model_state)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        normalizer = DataNormalizer()
        normalizer.initialize_with_data(x_tr, y_tr, dy_tr)

        x_te_n = normalizer.normalize_x(x_te)
        x_te_t = torch.tensor(x_te_n, dtype=torch.float32, device=device)
        _, dydx_pred_norm = model.forward_with_greek(x_te_t)
        dydx_pred = normalizer.unscale_dydx(dydx_pred_norm.detach().cpu().numpy())
        # Shape: (n_test, 1, n_pairs). Squeeze out output_dim for F = A @ g:
        g_pred = dydx_pred[:, 0, :]               # (n_test, n_pairs)
        A_test = test_extras["A_test"]            # (n_test, 3N, n_pairs)
        F_true_flat = test_extras["F_test_flat"]  # (n_test, 3N)
        # Reconstruct forces: F_pred = -A @ g (same chain rule used for labels)
        F_pred_flat = -np.einsum("bij,bj->bi", A_test.astype(np.float32), g_pred.astype(np.float32))
        test_force_mse_cartesian = float(np.mean((F_pred_flat - F_true_flat) ** 2))

    return {
        "method": method,
        "model": "MLP_pairwise",
        "dataset": f"md17_{molecule}",
        "molecule": molecule,
        "n_atoms": meta["n_atoms"],
        "input_dim": meta["input_dim"],
        "feature_type": meta["feature_type"],
        "derivative_type": meta["derivative_type"],
        "split_id": split_id,
        "seed": seed,
        "test_value_mse": energy_mse,
        # I-H1 (2026-04-16): use Cartesian force MSE as the primary
        # cross-arch-comparable gradient metric (since ∂E/∂d is under-determined
        # and MSE in d-space is not comparable to GATv2/PaiNN's force MSE).
        "test_grad_mse": test_force_mse_cartesian if not np.isnan(test_force_mse_cartesian) else forces_mse,
        "test_grad_mse_pairwise": forces_mse,        # pairwise-space (legacy metric)
        "test_grad_mse_cartesian": test_force_mse_cartesian,  # cross-arch comparable
        "test_energy_rmse_kcal": energy_rmse,
        "test_energy_rmse_mev": energy_rmse * KCAL_TO_MEV,
        "test_deriv_rmse_pairwise": deriv_rmse,
        "best_epoch": int(result.best_epoch) if result.best_epoch is not None else -1,
        "early_stopped": bool(getattr(result, "early_stopped", False)),
        "time_s": float(result.total_time_s) if hasattr(result, "total_time_s") else 0.0,
        "n_epochs_actual": len(result.training_logs) if getattr(result, "training_logs", None) else 0,
        "metadata": meta,
        "hparams": dict(hparams),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--molecules", nargs="+", default=DEFAULT_MOLECULES)
    parser.add_argument("--split_ids", type=str, default="1,2,3,4,5",
                        help="Comma-sep canonical Figshare split IDs (1..5)")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--n_epochs", type=int, default=HPARAMS["n_epochs"])
    parser.add_argument("--hidden_size", type=int, default=HPARAMS["hidden_size"])
    parser.add_argument("--lr", type=float, default=HPARAMS["lr"])
    parser.add_argument("--batch_size", type=int, default=HPARAMS["batch_size"])
    parser.add_argument("--results_dir", default="results/molecular_mlp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny: 1 molecule × 1 split × 1 method × 10 epochs.")
    args = parser.parse_args()

    split_ids = [int(s.strip()) for s in args.split_ids.split(",")]
    hparams = dict(HPARAMS)
    hparams["n_epochs"] = args.n_epochs
    hparams["hidden_size"] = args.hidden_size
    hparams["lr"] = args.lr
    hparams["batch_size"] = args.batch_size

    if args.smoke:
        args.molecules = ["ethanol"]
        split_ids = [1]
        args.methods = ["vanilla"]
        hparams["n_epochs"] = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("BLIS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MLP-pairwise molecular runner (rMD17, canonical 5-split protocol)")
    print(f"Molecules:  {args.molecules}")
    print(f"Splits:     {split_ids}")
    print(f"Methods:    {args.methods}")
    print(f"n_epochs={hparams['n_epochs']}, hidden={hparams['hidden_size']}, "
          f"lr={hparams['lr']}, batch={hparams['batch_size']}")
    print(f"GPU:        {args.gpu}")
    print(f"Results →   {results_dir}")
    total = len(args.molecules) * len(split_ids) * len(args.methods)
    print(f"Total:      {total} runs")
    print("=" * 70)

    n_done = 0
    n_failed = 0
    n_skipped = 0
    t0_all = time.time()

    for molecule in args.molecules:
        for split_id in split_ids:
            for method in args.methods:
                key = f"mlp_md17_{molecule}_split{split_id}_{method}"
                save_path = results_dir / f"{key}.json"

                if args.resume and save_path.exists():
                    n_skipped += 1
                    continue

                print(f"\n--- {key} ---")
                t0 = time.time()
                try:
                    res = train_one(molecule, method, split_id, hparams)
                    res["key"] = key
                    res["timestamp"] = datetime.utcnow().isoformat() + "Z"
                    with open(save_path, "w") as f:
                        json.dump(res, f, indent=2, default=str)
                    print(f"  OK ({time.time()-t0:.1f}s)  E_MSE={res['test_value_mse']:.4e}  "
                          f"dE/d_MSE={res['test_grad_mse']:.4e}  best_epoch={res['best_epoch']}")
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  FAIL: {e}")
                    traceback.print_exc()

    dt = time.time() - t0_all
    print(f"\nDone. {n_done} succeeded, {n_failed} failed, {n_skipped} skipped. "
          f"Wall: {dt/60:.1f} min")


if __name__ == "__main__":
    main()
