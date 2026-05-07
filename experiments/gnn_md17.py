#!/usr/bin/env python3
"""
GNN for rMD17 — GATv2 energy+force prediction with DML comparison.

This implements a proper GNN for molecular property prediction on rMD17,
following the standard ML force field protocol:

  Model:      GATv2-based GNN with edge features (interatomic distances)
  Energy:     Sum pooling over atom embeddings → scalar energy
  Forces:     -∂E/∂positions via torch.autograd.grad (conservative force field)
  Training:   vanilla (energy only) vs DML (energy + force matching)

This is the natural DML setting in computational chemistry — the "derivatives"
are forces, and the energy-force co-training paradigm is the standard approach
in SchNet, NequIP, MACE, etc.

Key design choices:
  - GATv2Conv (Brody et al. 2022): attention-based message passing with
    dynamic attention — more expressive than GATConv for molecular graphs
  - Radius graph with r_cut=5.0 Å: standard cutoff for organic molecules
  - RBF edge features: Gaussian basis expansion of interatomic distances
  - Per-atom embedding via nuclear charge lookup (Z → learnable vector)
  - Conservative forces via autograd: ensures energy conservation

Reference protocol from Chmiela et al. (2018) / Batzner et al. (2022):
  - 1000 train, 1000 val, 1000 test frames (random split)
  - Evaluate energy MAE (meV) and force MAE (meV/Å) on test set
  - Seeds: {42, 123, 456, 789, 1000} for statistical robustness

Usage:
  python experiments/gnn_md17.py --gpu 0
  python experiments/gnn_md17.py --gpu 0 --molecules ethanol aspirin --seeds 42,123
  python experiments/gnn_md17.py --gpu 0 --resume
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyG imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool

# DML-Bench shared loss balancing — single source of truth
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dml_benchmark.loss_balancing import (
    GradNormDmlLoss, SoftmaxBalanceDmlLoss, ReLoBRaLoDmlLoss,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

METHODS = [
    "vanilla",
    "dml_fixed",            # GATv2 convention: 0.5/0.5 convex combination (lambda=1)
    "dml_fixed_half",       # E-H2 (2026-04-13): alias of dml_fixed for GATv2
                            #   (explicit for cross-arch parity with MLP's dml_fixed_half)
    "dml_gradnorm",
    "dml_softmax_balance",  # 2026-04-13 rename of legacy "dml_relobralo" (simplified)
    "dml_relobralo",        # 2026-04-13: now refers to FAITHFUL Eq.11 (Bischof & Kraus 2022)
    "dml_warmup",
]
WARMUP_FRACTION = 0.5  # for dml_warmup: epochs 0..N/2 vanilla, N/2..N use GradNorm
DEFAULT_MOLECULES = ["ethanol", "aspirin"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]
KCAL_TO_MEV = 43.3641  # 1 kcal/mol = 43.3641 meV

HPARAMS = {
    "n_epochs": 1000,
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 0.0,            # M1 (2026-04-13): standardized to 0.0 across all 3 archs
    "patience": 50,         # early stopping patience
    "min_lr": 1e-6,
    "r_cut": 5.0,           # radius cutoff in Angstrom
    "n_rbf": 20,            # number of RBF basis functions
    "hidden_dim": 128,      # hidden dimension
    "n_heads": 4,           # attention heads
    "n_layers": 4,          # message passing layers
    "lambda_force": 1.0,    # force loss weight (for DML)
    "max_z": 100,           # max atomic number for embedding
}


# ============================================================================
# GAUSSIAN RBF EXPANSION
# ============================================================================

class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion for edge distances."""

    def __init__(self, n_rbf: int = 20, r_cut: float = 5.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.r_cut = r_cut
        # Evenly spaced centers from 0 to r_cut
        centers = torch.linspace(0.0, r_cut, n_rbf)
        self.register_buffer("centers", centers)
        # Width parameter (spacing between centers)
        self.width = (r_cut / n_rbf) * 0.5

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (n_edges,) interatomic distances
        Returns:
            rbf: (n_edges, n_rbf) RBF features
        """
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))


# ============================================================================
# COSINE CUTOFF ENVELOPE
# ============================================================================

class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, r_cut: float = 5.0):
        super().__init__()
        self.r_cut = r_cut

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(np.pi * dist / self.r_cut)) * (dist <= self.r_cut).float()


# ============================================================================
# GATv2 ENERGY MODEL
# ============================================================================

class GATv2EnergyModel(nn.Module):
    """
    GATv2-based GNN for molecular energy prediction with conservative forces.

    Architecture:
        1. Atom embedding: Z → learnable vector
        2. Edge features: RBF(d_ij) with cosine cutoff envelope
        3. GATv2 message passing layers with edge features
        4. Sum pooling → molecular energy
        5. Forces via -∂E/∂R (autograd)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_rbf: int = 20,
        r_cut: float = 5.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.r_cut = r_cut

        # Atom embedding
        self.atom_embedding = nn.Embedding(max_z, hidden_dim)
        nn.init.xavier_uniform_(self.atom_embedding.weight)

        # Edge feature processing
        self.rbf = GaussianRBF(n_rbf, r_cut)
        self.cutoff = CosineCutoff(r_cut)
        self.edge_proj = nn.Linear(n_rbf, hidden_dim)

        # GATv2 layers with residual connections
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(n_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=hidden_dim,
                concat=True,
                add_self_loops=True,   # M7 (2026-04-13): aligned with PyG default
            )
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output head: atom features → per-atom energy contribution
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z, pos, edge_index, edge_attr_dist, batch):
        """
        Forward pass — returns energy (and optionally forces).

        Args:
            z: (n_atoms,) atomic numbers (long)
            pos: (n_atoms, 3) positions — MUST have requires_grad=True for forces
            edge_index: (2, n_edges) connectivity
            edge_attr_dist: (n_edges,) interatomic distances
            batch: (n_atoms,) graph membership

        Returns:
            energy: (n_graphs, 1) predicted energy
        """
        # Atom embedding
        h = self.atom_embedding(z)  # (n_atoms, hidden_dim)

        # Edge features: RBF expansion + cosine cutoff
        rbf_feat = self.rbf(edge_attr_dist)                  # (n_edges, n_rbf)
        envelope = self.cutoff(edge_attr_dist).unsqueeze(-1)  # (n_edges, 1)
        edge_attr = self.edge_proj(rbf_feat * envelope)       # (n_edges, hidden_dim)

        # Message passing with residual connections
        for conv, ln in zip(self.convs, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = ln(h + h_new)  # residual + layer norm

        # Per-atom energy contributions → sum pooling
        atom_energies = self.output_head(h)  # (n_atoms, 1)
        energy = global_add_pool(atom_energies, batch)  # (n_graphs, 1)

        return energy

    def forward_with_forces(self, z, pos, edge_index, edge_attr_dist, batch):
        """
        Compute energy and conservative forces (-∂E/∂pos).

        This is the key DML component: forces are analytically the negative
        gradient of energy w.r.t. atomic positions.
        """
        # Ensure positions track gradients
        pos.requires_grad_(True)

        # Recompute distances from positions (so autograd connects them)
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

        energy = self.forward(z, pos, edge_index, dist, batch)

        # Forces = -∂E/∂pos (conservative)
        grad_outputs = torch.ones_like(energy)
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            grad_outputs=None,
            create_graph=self.training,  # need graph for backprop through forces
            retain_graph=True,
        )[0]  # (n_atoms, 3)

        return energy, forces


# ============================================================================
# DATA LOADING — rMD17 → PyG graphs
# ============================================================================

def build_radius_graph(pos, r_cut):
    """Build radius graph (all pairs within cutoff). Vectorized numpy."""
    n_atoms = pos.shape[0]
    # Compute all pairwise distances at once
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
    dist_sq = (diff ** 2).sum(axis=-1)          # (N, N)
    # Mask: within cutoff and not self-loops
    mask = (dist_sq < r_cut ** 2) & (dist_sq > 0)
    row, col = np.where(mask)
    return np.array([row, col], dtype=np.int64)


def load_rmd17_graphs(
    molecule: str,
    data_dir: str = "data/rmd17/rmd17/npz_data",
    splits_dir: str = "data/rmd17/rmd17/splits",
    n_train: int = 1000,
    n_val: int = 50,
    n_test: int = 1000,
    split_id: int = 1,
    seed: int = 42,
    r_cut: float = 5.0,
    use_canonical_splits: bool = True,
):
    """
    Load rMD17 as PyG Data objects.

    P3 + V3/P8 (2026-04-13): defaults to CANONICAL Figshare splits (split_id 1..5)
    with rMD17 1k-train / 50-val / 1k-test convention. Set use_canonical_splits=False
    to fall back to random per-seed split (legacy behavior).

    Returns:
        train_data, val_data, test_data: lists of PyG Data objects
        metadata: dict with dataset info
    """
    npz_path = Path(data_dir) / f"rmd17_{molecule}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"rMD17 data not found at {npz_path}. "
            f"Download from https://figshare.com/articles/dataset/12672038"
        )

    raw = np.load(str(npz_path))
    coords = raw["coords"]          # (N, n_atoms, 3)
    energies = raw["energies"]      # (N,)
    forces = raw["forces"]          # (N, n_atoms, 3)
    nuclear_charges = raw["nuclear_charges"]  # (n_atoms,)

    n_frames, n_atoms, _ = coords.shape

    # Standardize energies (shift to zero mean for stability)
    energy_mean = energies.mean()
    energy_std = energies.std()
    energies_shifted = energies - energy_mean

    print(f"  Loaded {molecule}: {n_frames} frames, {n_atoms} atoms")
    print(f"  Energy: mean={energy_mean:.2f}, std={energy_std:.4f} kcal/mol")
    print(f"  Force RMSE: {np.sqrt((forces**2).mean()):.4f} kcal/mol/Å")

    if use_canonical_splits:
        # P3 (2026-04-13): canonical Figshare splits for literature comparability
        train_csv = Path(splits_dir) / f"index_train_0{split_id}.csv"
        test_csv = Path(splits_dir) / f"index_test_0{split_id}.csv"
        if not train_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"Canonical splits not found at {train_csv} / {test_csv}. "
                f"Set use_canonical_splits=False to use random splits."
            )
        train_idx_full = np.loadtxt(str(train_csv)).flatten().astype(int)
        test_idx_full = np.loadtxt(str(test_csv)).flatten().astype(int)
        rng = np.random.RandomState(seed)
        if n_train + n_val > len(train_idx_full):
            raise ValueError(
                f"Requested n_train={n_train} + n_val={n_val} = {n_train + n_val} "
                f"> canonical train size {len(train_idx_full)}."
            )
        perm = rng.permutation(len(train_idx_full))
        train_idx = train_idx_full[perm[:n_train]]
        val_idx = train_idx_full[perm[n_train:n_train + n_val]]
        if n_test < len(test_idx_full):
            test_perm = rng.permutation(len(test_idx_full))
            test_idx = test_idx_full[test_perm[:n_test]]
        else:
            test_idx = test_idx_full
        print(f"  Using canonical Figshare split {split_id}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        rng = np.random.RandomState(seed)
        n_total = n_train + n_val + n_test
        indices = rng.permutation(n_frames)[:n_total]
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:n_train + n_val + n_test]
        print(f"  Using random split (seed={seed}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build PyG graphs
    # Precompute edge index from first frame (molecular topology is fixed)
    # For rMD17, all frames have the same atoms — distances change but connectivity
    # within cutoff is similar. We recompute per-frame for correctness.
    
    def make_pyg_data(frame_indices, desc=""):
        data_list = []
        for i, idx in enumerate(frame_indices):
            pos = coords[idx]  # (n_atoms, 3)
            
            # Build edges within cutoff
            edge_index = build_radius_graph(pos, r_cut)
            
            # Compute edge distances
            row, col = edge_index
            diff = pos[row] - pos[col]
            dist = np.sqrt((diff ** 2).sum(axis=-1))
            
            data = Data(
                z=torch.tensor(nuclear_charges, dtype=torch.long),
                pos=torch.tensor(pos, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr_dist=torch.tensor(dist, dtype=torch.float32),
                y=torch.tensor([[energies_shifted[idx]]], dtype=torch.float32),
                forces=torch.tensor(forces[idx], dtype=torch.float32),
                batch_idx=torch.zeros(n_atoms, dtype=torch.long),
            )
            data_list.append(data)
            
            if (i + 1) % 500 == 0:
                print(f"    {desc}: {i+1}/{len(frame_indices)} graphs built")
        
        return data_list

    print(f"  Building PyG graphs (r_cut={r_cut} Å)...")
    train_data = make_pyg_data(train_idx, "train")
    val_data = make_pyg_data(val_idx, "val")
    test_data = make_pyg_data(test_idx, "test")

    # Check connectivity
    avg_edges = np.mean([d.edge_index.shape[1] for d in train_data])
    print(f"  Avg edges per graph: {avg_edges:.0f}")
    print(f"  Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    metadata = {
        "molecule": molecule,
        "n_atoms": int(n_atoms),
        "nuclear_charges": nuclear_charges.tolist(),
        "n_frames_total": int(n_frames),
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "r_cut": r_cut,
        "avg_edges": float(avg_edges),
    }

    return train_data, val_data, test_data, metadata


# ============================================================================
# TRAINING LOOP
# ============================================================================

def _train_gnn_md17_warmup(
    model, train_data, val_data, test_data,
    n_epochs, batch_size, lr, weight_decay, patience, min_lr,
    lambda_force, warmup_fraction, device, seed=42, n_layers=4,
):
    """
    Two-phase training for GATv2 on rMD17.
      Phase 1: vanilla (energy only) for warmup_fraction * n_epochs.
      Phase 2: dml_gradnorm for the remaining epochs.
    Mirrors experiments/unified_comparison/run_unified_experiment.py:train_warmup
    so the molecular and synthetic flows use the SAME warmup convention.
    """
    warmup_epochs = int(n_epochs * warmup_fraction)
    finetune_epochs = n_epochs - warmup_epochs
    print(f"    Warmup: phase 1 = {warmup_epochs} vanilla epochs, "
          f"phase 2 = {finetune_epochs} dml_gradnorm epochs (lr/10 drop).")

    # Phase 1 — vanilla. M1 (2026-04-13): ES counter resets in phase 2 (handled
    # naturally because phase 2 is a separate train_gnn_md17 call with a fresh
    # best_val_loss=inf inside that call).
    # 2026-04-14 (phase-1 ES ablation, 75 runs): ES in phase 1 enabled with
    # patience=50 (same as phase 2). Ablation showed identical test metrics
    # across off/pat50/pat20 with 15-55% compute savings. See EVIDENCE/ablation_outcomes.md.
    p1_metrics = train_gnn_md17(
        model=model, train_data=train_data, val_data=val_data, test_data=test_data,
        method="vanilla", n_epochs=warmup_epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, patience=50,  # enable ES in P1 (pat50)
        min_lr=min_lr, lambda_force=lambda_force, device=device,
        seed=seed, n_layers=n_layers,
    )

    # Phase 2 — dml_gradnorm with reduced LR.
    # 2026-04-14: lr drop factor changed from /5 → /10 after warmup-LR ablation
    # (105 runs across 3 targets). See EVIDENCE/ablation_outcomes.md.
    p2_metrics = train_gnn_md17(
        model=model, train_data=train_data, val_data=val_data, test_data=test_data,
        method="dml_gradnorm", n_epochs=finetune_epochs, batch_size=batch_size,
        lr=lr / 10.0, weight_decay=weight_decay, patience=patience,
        min_lr=min_lr / 10.0, lambda_force=lambda_force, device=device,
        seed=seed, n_layers=n_layers,
    )

    # Return P2 test metrics (final model state); preserve combined log.
    p2_metrics["training_logs"] = (
        p1_metrics.get("training_logs", []) + p2_metrics.get("training_logs", [])
    )
    p2_metrics["best_epoch_p1"] = p1_metrics.get("best_epoch", -1)
    p2_metrics["best_epoch_p2"] = p2_metrics.get("best_epoch", -1)
    p2_metrics["n_epochs_actual"] = (
        p1_metrics.get("n_epochs_actual", warmup_epochs)
        + p2_metrics.get("n_epochs_actual", finetune_epochs)
    )
    return p2_metrics


def _make_balancing_loss(method, input_dim_proxy, lambda_force, n_layers=4, seed=42):
    """
    Construct the balancing loss object for a given DML method.

    `input_dim_proxy` is a placeholder; the GNN flow doesn't have a fixed input_dim
    (number of atoms varies per molecule), but loss_balancing.py's GradNorm only
    uses input_dim for `DimNormGradNormDmlLoss` (not used here).

    D-M3 (2026-04-14): GradNorm "shared layer" is the last GATv2Conv's message
    linear (`convs.{n-1}.lin_l`) — the last shared backbone layer before the
    Sequential output_head per GradNorm paper Alg. 1.

    M-C-R2-1 (2026-04-14): seed threaded to stochastic balancing classes
    (SoftmaxBalance reference-step draw; ReLoBRaLo saudade Bernoulli).
    """
    gradnorm_shared = f"convs.{n_layers - 1}.lin_l"
    if method == "dml_gradnorm":
        return GradNormDmlLoss(input_dim=input_dim_proxy, shared_layer_name=gradnorm_shared)
    elif method == "dml_softmax_balance":
        # Legacy simplified ReLoBRaLo (was misnamed in v2). See DEVIATIONS D012.
        return SoftmaxBalanceDmlLoss(input_dim=input_dim_proxy, seed=seed)
    elif method == "dml_relobralo":
        # Faithful Bischof & Kraus 2022 Eq. 11
        return ReLoBRaLoDmlLoss(input_dim=input_dim_proxy, seed=seed)
    else:
        raise ValueError(f"Unknown balanced method: {method}")


def train_gnn_md17(
    model,
    train_data,
    val_data,
    test_data,
    method="vanilla",
    n_epochs=1000,
    batch_size=32,
    lr=5e-4,
    weight_decay=1e-5,
    patience=50,
    min_lr=1e-6,
    lambda_force=1.0,
    warmup_fraction=WARMUP_FRACTION,
    device="cuda",
    seed=42,
    n_layers=4,
):
    """
    Train GATv2 on rMD17 with one of:
      - vanilla        : energy only (existing path, untouched)
      - dml_fixed      : energy + force, fixed lambda (existing path, untouched)
      - dml_gradnorm   : energy + force with GradNorm (Chen et al. 2018)
      - dml_relobralo  : energy + force with ReLoBRaLo (Bischof & Kraus 2022)
      - dml_warmup     : two-phase — vanilla for warmup_fraction*n_epochs,
                         then GradNorm for the rest

    For dml_warmup, the test evaluation always uses the model from end of phase 2.

    Args:
        method: see above
        warmup_fraction: only used for dml_warmup

    Returns:
        dict with test metrics, training logs, etc.
    """
    # ----- Special-case 2-phase training for warmup -----
    if method == "dml_warmup":
        return _train_gnn_md17_warmup(
            model, train_data, val_data, test_data,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, min_lr=min_lr,
            lambda_force=lambda_force, warmup_fraction=warmup_fraction,
            device=device, seed=seed, n_layers=n_layers,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # L1 (2026-04-13): scheduler patience aligned with packet (20 epochs).
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=min_lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Method dispatch
    # E-H2 (2026-04-13): dml_fixed_half is an alias of dml_fixed for GATv2
    # (both already use 0.5/0.5 — the rename exists for cross-arch naming parity
    # with MLP's dml_fixed_half which contrasts with MLP's default HS-weighted dml_fixed).
    use_forces = method in (
        "dml_fixed", "dml_fixed_half",
        "dml_gradnorm", "dml_softmax_balance", "dml_relobralo",
    )
    is_balanced = method in ("dml_gradnorm", "dml_softmax_balance", "dml_relobralo")

    # Construct balancing loss (input_dim_proxy = arbitrary; not used for these methods)
    balancing_loss = _make_balancing_loss(
        method, input_dim_proxy=1, lambda_force=lambda_force,
        n_layers=n_layers, seed=seed,
    ) if is_balanced else None
    if balancing_loss is not None:
        balancing_loss = balancing_loss.to(device)
        balancing_loss.train()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    training_logs = []

    for epoch in range(n_epochs):
        # === TRAIN ===
        model.train()
        train_energy_loss = 0.0
        train_force_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if use_forces:
                # DML mode: compute energy + forces
                pos = batch.pos.clone().detach().requires_grad_(True)
                energy, pred_forces = model.forward_with_forces(
                    batch.z, pos, batch.edge_index, batch.edge_attr_dist, batch.batch
                )

                energy_loss = F.mse_loss(energy, batch.y)
                force_loss = F.mse_loss(pred_forces, batch.forces)

                if is_balanced:
                    # Use shared loss_balancing class (single source of truth).
                    # Forward shapes: y=(n_graphs,1), F=(n_atoms,3); MSE inside the
                    # class handles arbitrary shapes.
                    lc = balancing_loss(
                        y_pred=energy, y_true=batch.y,
                        dydx_pred=pred_forces, dydx_true=batch.forces,
                        model=model,
                    )
                    loss = lc.total
                else:
                    # dml_fixed: existing fixed-lambda formula (UNCHANGED)
                    lam_prime = lambda_force / (1.0 + lambda_force)
                    loss = (1.0 - lam_prime) * energy_loss + lam_prime * force_loss

                train_force_loss += force_loss.item()
            else:
                # Vanilla mode: energy only
                energy = model(
                    batch.z, batch.pos, batch.edge_index, batch.edge_attr_dist, batch.batch
                )
                energy_loss = F.mse_loss(energy, batch.y)
                loss = energy_loss

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_energy_loss += energy_loss.item()
            n_train_batches += 1

        train_energy_loss /= n_train_batches
        if use_forces:
            train_force_loss /= n_train_batches

        # === VALIDATION ===
        model.eval()
        val_energy_loss = 0.0
        val_force_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            # For validation we only evaluate energy (no force autograd needed)
            for batch in val_loader:
                batch = batch.to(device)
                energy = model(
                    batch.z, batch.pos, batch.edge_index, batch.edge_attr_dist, batch.batch
                )
                val_energy_loss += F.mse_loss(energy, batch.y).item()
                n_val_batches += 1

        val_energy_loss /= n_val_batches

        # Also evaluate force MAE on validation if DML
        # J8 (2026-04-16): removed dead vars `force_mae_sum`/`n_force_atoms`
        # (F-L2); they were never wired into any aggregation.
        if use_forces:
            for batch in val_loader:
                batch = batch.to(device)
                pos = batch.pos.clone().detach().requires_grad_(True)
                _, pred_forces = model.forward_with_forces(
                    batch.z, pos, batch.edge_index, batch.edge_attr_dist, batch.batch
                )
                val_force_loss += F.mse_loss(pred_forces, batch.forces).item()
            val_force_loss /= n_val_batches

        # M3 (2026-04-13): adaptive ES across all 3 architectures.
        # ES & scheduler monitor the SAME convex combination the loss_fn uses
        # at val time (live training objective evaluated on val set). This matches
        # the standard ML practice (Bishop §5.5.2; Goodfellow §7.8) and is what
        # MLP and PaiNN do.
        if use_forces:
            if is_balanced and balancing_loss is not None:
                # Live adaptive weights from the balancing class
                if hasattr(balancing_loss, "task_weights"):
                    # GradNorm: read current task_weights (renormalized to sum=2)
                    w = balancing_loss.task_weights.detach().cpu().numpy()
                    w_sum = w.sum() + 1e-8
                    w_E, w_F = 2.0 * w[0] / w_sum, 2.0 * w[1] / w_sum
                elif hasattr(balancing_loss, "running_weights"):
                    # SoftmaxBalance / legacy ReLoBRaLo
                    w = balancing_loss.running_weights.detach().cpu().numpy()
                    w_E, w_F = float(w[0]), float(w[1])
                elif hasattr(balancing_loss, "lambda_current"):
                    # Faithful ReLoBRaLo
                    w = balancing_loss.lambda_current.detach().cpu().numpy()
                    w_E, w_F = float(w[0]), float(w[1])
                else:
                    w_E, w_F = 0.5, 0.5
                val_total = w_E * val_energy_loss + w_F * val_force_loss
            else:
                # dml_fixed: same convex combination as the train loss
                lam_prime = lambda_force / (1.0 + lambda_force)
                val_total = (1.0 - lam_prime) * val_energy_loss + lam_prime * val_force_loss
        else:
            val_total = val_energy_loss

        scheduler.step(val_total)

        # Early stopping on total loss (was: val_energy_loss only — fixed 2026-04-13
        # to match dml_benchmark/trainer.py convention).
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Log
        log_entry = {
            "epoch": epoch,
            "train_energy_mse": train_energy_loss,
            "val_energy_mse": val_energy_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if use_forces:
            log_entry["train_force_mse"] = train_force_loss
            log_entry["val_force_mse"] = val_force_loss
        training_logs.append(log_entry)

        if epoch % 50 == 0 or epoch == n_epochs - 1 or patience_counter == patience:
            force_str = f", force_mse={val_force_loss:.4e}" if use_forces else ""
            print(f"    Epoch {epoch:4d}: train_E={train_energy_loss:.4e}, "
                  f"val_E={val_energy_loss:.4e}{force_str}, "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}, "
                  f"best@{best_epoch}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # === TEST EVALUATION ===
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    all_pred_energy = []
    all_true_energy = []
    all_pred_forces = []
    all_true_forces = []

    for batch in test_loader:
        batch = batch.to(device)
        pos = batch.pos.clone().detach().requires_grad_(True)
        energy, pred_forces = model.forward_with_forces(
            batch.z, pos, batch.edge_index, batch.edge_attr_dist, batch.batch
        )
        all_pred_energy.append(energy.detach().cpu())
        all_true_energy.append(batch.y.cpu())
        all_pred_forces.append(pred_forces.detach().cpu())
        all_true_forces.append(batch.forces.cpu())

    pred_energy = torch.cat(all_pred_energy, dim=0)
    true_energy = torch.cat(all_true_energy, dim=0)
    pred_forces = torch.cat(all_pred_forces, dim=0)
    true_forces = torch.cat(all_true_forces, dim=0)

    # Metrics  
    energy_mse = F.mse_loss(pred_energy, true_energy).item()
    energy_mae = (pred_energy - true_energy).abs().mean().item()
    force_mse = F.mse_loss(pred_forces, true_forces).item()
    force_mae = (pred_forces - true_forces).abs().mean().item()

    # Convert to meV and meV/Å for comparison with literature
    energy_mae_mev = energy_mae * KCAL_TO_MEV
    force_mae_mev = force_mae * KCAL_TO_MEV

    print(f"\n    === TEST RESULTS ({method}) ===")
    print(f"    Energy MSE:  {energy_mse:.6e} (kcal/mol)²")
    print(f"    Energy MAE:  {energy_mae:.4f} kcal/mol = {energy_mae_mev:.1f} meV")
    print(f"    Force MSE:   {force_mse:.6e} (kcal/mol/Å)²")
    print(f"    Force MAE:   {force_mae:.4f} kcal/mol/Å = {force_mae_mev:.1f} meV/Å")

    return {
        "test_energy_mse": energy_mse,
        "test_energy_mae": energy_mae,
        "test_energy_mae_mev": energy_mae_mev,
        "test_force_mse": force_mse,
        "test_force_mae": force_mae,
        "test_force_mae_mev": force_mae_mev,
        "best_epoch": best_epoch,
        "n_epochs_actual": len(training_logs),
        "training_logs": training_logs,
    }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def set_deterministic(seed):
    """Set all random seeds for reproducibility.

    I-L12 (2026-04-16): delegate to dml_benchmark.trainer.set_deterministic
    so the GNN runner inherits PYTHONHASHSEED + CUBLAS_WORKSPACE_CONFIG +
    torch.use_deterministic_algorithms(warn_only=True) — matching the MLP
    runner's determinism setup for cross-arch consistency.
    """
    from dml_benchmark.trainer import set_deterministic as _canonical
    _canonical(seed)


def run_gnn_experiments(
    molecules,
    seeds,
    gpu=0,
    resume=False,
    results_dir=None,
    split_ids=None,
    use_canonical_splits=True,
):
    """Run GNN MD17 experiments for all molecule × seed × method combinations."""

    if results_dir is None:
        results_dir = Path("results/gnn_md17")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load existing results for resume
    existing = {}
    if resume:
        for f in results_dir.glob("*.json"):
            try:
                d = json.load(open(f))
                existing[d.get("key", f.stem)] = d
            except Exception:
                pass
        print(f"Found {len(existing)} existing results")

    all_results = {}

    # P3 (2026-04-13): canonical Figshare splits. Iterate over split_ids; seed=split_id.
    if split_ids is None:
        split_ids = [1, 2, 3, 4, 5] if use_canonical_splits else seeds

    for molecule in molecules:
        print(f"\n{'=' * 70}")
        print(f"MOLECULE: {molecule}")
        print(f"{'=' * 70}")

        for split_id in split_ids:
            seed = split_id  # one seed per split
            print(f"\n--- Split {split_id} (seed={seed}) ---")

            # Load data once per molecule+split
            set_deterministic(seed)
            try:
                train_data, val_data, test_data, metadata = load_rmd17_graphs(
                    molecule,
                    n_train=950,     # rMD17 paper / SchNetPack canonical / PaiNN paper Table 5
                    n_val=50,        # canonical rMD17 val size (total 1000)
                    n_test=1000,
                    split_id=split_id,
                    seed=seed,
                    r_cut=HPARAMS["r_cut"],
                    use_canonical_splits=use_canonical_splits,
                )
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            for method in METHODS:
                key = f"gnn_md17_{molecule}_split{split_id}_{method}"

                if resume and key in existing:
                    print(f"  SKIP (exists): {key}")
                    all_results[key] = existing[key]
                    continue

                print(f"\n  Training {method} (seed={seed})...")
                set_deterministic(seed)
                t0 = time.time()

                try:
                    model = GATv2EnergyModel(
                        hidden_dim=HPARAMS["hidden_dim"],
                        n_heads=HPARAMS["n_heads"],
                        n_layers=HPARAMS["n_layers"],
                        n_rbf=HPARAMS["n_rbf"],
                        r_cut=HPARAMS["r_cut"],
                        max_z=HPARAMS["max_z"],
                    )
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"    Model parameters: {n_params:,}")

                    metrics = train_gnn_md17(
                        model=model,
                        train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        method=method,
                        n_epochs=HPARAMS["n_epochs"],
                        batch_size=HPARAMS["batch_size"],
                        lr=HPARAMS["lr"],
                        weight_decay=HPARAMS["weight_decay"],
                        patience=HPARAMS["patience"],
                        min_lr=HPARAMS["min_lr"],
                        lambda_force=HPARAMS["lambda_force"],
                        device=device,
                        seed=seed,
                        n_layers=HPARAMS["n_layers"],
                    )

                    elapsed = time.time() - t0

                    result_dict = {
                        "key": key,
                        "method": method,
                        "model": "GATv2",
                        "dataset": f"md17_{molecule}",
                        "molecule": molecule,
                        "n_atoms": metadata["n_atoms"],
                        "seed": seed,
                        "split_id": split_id,
                        "split_source": metadata.get("split_source", "?"),
                        "lambda": HPARAMS["lambda_force"],
                        "test_value_mse": metrics["test_energy_mse"],
                        "test_grad_mse": metrics["test_force_mse"],
                        "test_energy_mae": metrics["test_energy_mae"],
                        "test_energy_mae_mev": metrics["test_energy_mae_mev"],
                        "test_force_mae": metrics["test_force_mae"],
                        "test_force_mae_mev": metrics["test_force_mae_mev"],
                        "best_epoch": metrics["best_epoch"],
                        "n_epochs_actual": metrics["n_epochs_actual"],
                        "time_s": round(elapsed, 2),
                        "n_params": n_params,
                        "metadata": metadata,
                        "hparams": HPARAMS,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Save (omit training logs from JSON to keep files small)
                    save_path = results_dir / f"{key}.json"
                    with open(save_path, "w") as f:
                        json.dump(result_dict, f, indent=2, default=str)
                    print(f"    Saved: {save_path}")

                    all_results[key] = result_dict

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"    FAILED ({elapsed:.1f}s): {e}")
                    traceback.print_exc()

    # === SUMMARY ===
    print(f"\n{'=' * 70}")
    print("SUMMARY: GNN MD17 Results")
    print(f"{'=' * 70}")

    # Group by molecule+seed for comparison
    comparisons = {}
    for key, res in all_results.items():
        mol = res["molecule"]
        seed = res["seed"]
        group_key = f"{mol}_s{seed}"
        comparisons.setdefault(group_key, {})[res["method"]] = res

    for group_key in sorted(comparisons):
        methods = comparisons[group_key]
        print(f"\n  {group_key}:")
        for m in METHODS:
            if m in methods:
                r = methods[m]
                print(f"    {m:15s}  E_MAE={r['test_energy_mae_mev']:7.1f} meV  "
                      f"F_MAE={r['test_force_mae_mev']:7.1f} meV/Å  "
                      f"E_MSE={r['test_value_mse']:.4e}  "
                      f"F_MSE={r['test_grad_mse']:.4e}  "
                      f"t={r['time_s']:.0f}s")

        # DML improvement
        if "vanilla" in methods and "dml_fixed" in methods:
            van_e = methods["vanilla"]["test_value_mse"]
            dml_e = methods["dml_fixed"]["test_value_mse"]
            van_f = methods["vanilla"]["test_grad_mse"]
            dml_f = methods["dml_fixed"]["test_grad_mse"]
            print(f"    → DML improvement: energy {van_e/dml_e:.1f}×, force {van_f/dml_f:.1f}×")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GNN (GATv2) for rMD17 — DML vs Vanilla")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--molecules", nargs="+", default=DEFAULT_MOLECULES,
                        help="Molecules to test (default: ethanol aspirin)")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seeds (only used if --use_random_splits)")
    parser.add_argument("--split_ids", type=str, default="1,2,3,4,5",
                        help="Canonical Figshare split IDs (1..5). Default: 5-fold CV.")
    parser.add_argument("--use_random_splits", action="store_true",
                        help="Use random per-seed splits instead of canonical (legacy).")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    split_ids = [int(s.strip()) for s in args.split_ids.split(",")]

    print("=" * 70)
    print("GNN (GATv2) for rMD17 — DML vs Vanilla")
    print("=" * 70)
    print(f"Molecules:  {args.molecules}")
    print(f"Seeds:      {seeds}")
    print(f"Methods:    {METHODS}")
    print(f"GPU:        {args.gpu}")
    print(f"Hparams:    hidden={HPARAMS['hidden_dim']}, heads={HPARAMS['n_heads']}, "
          f"layers={HPARAMS['n_layers']}, r_cut={HPARAMS['r_cut']}")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results = run_gnn_experiments(
        molecules=args.molecules,
        seeds=seeds,
        gpu=0,  # always 0 after CUDA_VISIBLE_DEVICES
        resume=args.resume,
        split_ids=split_ids,
        use_canonical_splits=not args.use_random_splits,
    )

    print(f"\nTotal experiments: {len(results)}")
    print("Done!")


if __name__ == "__main__":
    main()
