#!/usr/bin/env python3
"""
PaiNN on rMD17 — faithful reproduction via SchNetPack v2.2.0 programmatic API.

This runner is the literature-standard equivariant baseline for the molecular
pillar (Phase 4). It mirrors the canonical SchNetPack configuration:
  - configs/experiment/rmd17.yaml
  - configs/model/representation/painn.yaml
in the installed schnetpack 2.2.0 package. We assemble the same components
programmatically so we can capture metrics in our results-JSON schema.

NO conceptual modification to the published method — only minimal glue to fit
our results format. Per AGENT_PRINCIPLES §1: non-novel = authenticity.

Methods supported:
  - "native_EF"   : canonical joint energy+force loss (loss_weight: E=0.01, F=0.99).
                    This is the de-facto standard PaiNN-on-rMD17 setup in modern
                    molecular-ML papers (MACE, NequIP, Allegro all reference it).
  - "energy_only" : ablation — same architecture and training, but force loss
                    weight = 0. Only energy is supervised. Tests "what does PaiNN
                    learn about forces purely via autograd from energy alone?"

Reference: Schütt, Unke, Gastegger, "Equivariant message passing for the
prediction of tensorial properties and molecular spectra", ICML 2021,
arXiv:2102.03150. Implementation: https://github.com/atomistic-machine-learning/schnetpack
(v2.2.0 tag).

Usage:
  python experiments/molecular/run_painn.py --gpu 1 --molecules ethanol --seeds 42 --smoke
  python experiments/molecular/run_painn.py --gpu 1                              # full grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add repo root to path BEFORE importing torch/schnetpack so we can read EVIDENCE etc.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================================
# CONFIGURATION (mirrors SchNetPack rmd17.yaml + painn.yaml)
# ============================================================================

# Methods aligned with MLP+GATv2 cross-architecture comparison.
# - vanilla       : energy-only (= SchNetPack "energy_only" — keep for back-compat)
# - dml_fixed     : energy + force, fixed lambda (NOT the canonical 0.01/0.99 weights;
#                   uses the same lambda=1.0 convex combination as MLP/GATv2 for parity).
# - native_EF     : SchNetPack canonical (loss weights 0.01 energy / 0.99 force).
#                   Kept as a separate method so we can compare the canonical PaiNN
#                   weight choice to the benchmark-standard lambda=1.
# - dml_gradnorm  : energy + force with GradNorm (Chen et al. 2018); via
#                   dml_benchmark.loss_balancing.GradNormDmlLoss (same code as MLP/GATv2).
# - dml_relobralo : energy + force with ReLoBRaLo (Bischof & Kraus 2022); same source.
# - dml_warmup    : two-phase — vanilla for first warmup_fraction*N epochs then
#                   dml_gradnorm for the rest. Same convention as MLP/GATv2.
METHODS = [
    "vanilla",
    "dml_fixed",            # 0.5/0.5 convex combination (cross-arch parity)
    "dml_fixed_half",       # E-H2 (2026-04-13): alias of dml_fixed for PaiNN
                            #   (explicit for cross-arch parity with MLP's dml_fixed_half)
    "native_EF",            # SchNetPack canonical (0.01/0.99)
    "dml_gradnorm",
    "dml_softmax_balance",  # legacy simplified ReLoBRaLo (renamed)
    "dml_relobralo",        # FAITHFUL Bischof & Kraus 2022 Eq.11
    "dml_warmup",
]
WARMUP_FRACTION = 0.5
DEFAULT_MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]

# SchNetPack v2.2.0's rMD17 datamodule keys salicylic acid as
# "salicylic_acid" (rmd17.py:57: salicylic_acid="rmd17_salicylic.npz"),
# but the rMD17 paper, our rMD17 .npz file, our pre-built ASE database
# (rmd17_salicylic.db), our pairwise-MLP and GATv2 result files, and
# our split-file naming all use "salicylic" without the suffix. We
# translate at the rMD17(...) call site only so that result-file
# naming, split-file naming, db-path naming, and aggregator keys all
# remain consistent with the other nine molecules and across
# architectures. Other molecules pass through unchanged.
_SCHNETPACK_MOL_KEY = {"salicylic": "salicylic_acid"}
# P3 (2026-04-13): use canonical Figshare split IDs (1..5) instead of arbitrary seeds.
DEFAULT_SPLIT_IDS = [1, 2, 3, 4, 5]
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]  # legacy / random-split fallback only

# Canonical rMD17 + PaiNN config from SchNetPack v2.2.0 configs/
HPARAMS_CANONICAL = {
    "cutoff": 5.0,                  # Å (rmd17.yaml: globals.cutoff)
    "n_atom_basis": 128,            # painn.yaml
    "n_interactions": 3,            # painn.yaml
    "n_rbf": 20,                    # standard
    "lr": 1e-3,                     # rmd17.yaml: globals.lr
    "n_epochs": 1000,               # standard rMD17 protocol
    "batch_size": 10,               # rmd17.yaml: data.batch_size
    "num_train": 950,               # rMD17 paper / SchNetPack canonical / PaiNN paper Table 5
    "num_val": 50,                  # canonical SchNetPack 50 (total train+val=1000)
    "num_test": 1000,               # P2 (2026-04-13): 1k for cross-arch parity (was: rest≈99K)
    "loss_weight_energy_native": 0.01,    # rmd17.yaml: task.outputs[0].loss_weight
    "loss_weight_forces_native": 0.99,    # rmd17.yaml: task.outputs[1].loss_weight
    # D-M1 (2026-04-14): ES patience set to SchNetPack canonical 200.
    # (The prior "PaiNN Table 5 = 150" citation was incorrect — the PaiNN paper
    # arXiv:2102.03150 contains no documented ES patience. SchNetPack v2.2.0
    # canonical configs use 200 for rMD17 protocols.)
    "es_patience": 200,
    # P5 (2026-04-13) + D-M2 (2026-04-14): LR scheduler canonical kwargs.
    # cooldown=10, threshold=0.0, threshold_mode='rel' match SchNetPack's
    # configs/callbacks/lr_monitor.yaml default for rMD17. Prior code used
    # torch defaults (cooldown=0, threshold=1e-4) which decay sooner.
    "scheduler_factor": 0.5,
    "scheduler_patience": 75,       # SchNetPack default
    "scheduler_min_lr": 0.0,
    "scheduler_cooldown": 10,
    "scheduler_threshold": 0.0,
    # Warmup phase 2 LR drop factor (matches MLP/GATv2 convention).
    # 2026-04-14: changed from 5.0 → 10.0 after warmup-LR ablation (105 runs).
    # See EVIDENCE/ablation_outcomes.md.
    "warmup_lr_drop": 10.0,
}


# ============================================================================
# CUSTOM LIGHTNING TASKS — module-level (L4 refactor 2026-04-13)
#
# Each task uses dml_benchmark.loss_balancing classes for the loss computation
# (single source of truth across MLP, GATv2, PaiNN). All tasks override
# `configure_optimizers` to FILTER OUT the balancing-loss parameters from
# the AdamW optimizer (P1 fix 2026-04-13: prevents double-update of
# GradNorm.task_weights via SchNetPack's outer optimizer).
# ============================================================================

def _filtered_optimizer_factory(task, scheduler_cls=None, scheduler_args=None):
    """P1 (2026-04-13): build optimizer over task.parameters() EXCLUDING any
    parameter whose name starts with '_balancing_loss' or '_gradnorm' or
    '_softmax_balance' or '_faithful_relobralo'. These are auto-registered
    submodules of the LightningModule because we assigned them as attributes
    after super().__init__(); without this filter, the AdamW optimizer would
    pick up GradNorm's task_weights and apply a second uncontrolled update
    per step (Reviewer A H1)."""
    excluded_prefixes = ("_balancing_loss", "_gradnorm", "_softmax_balance", "_faithful_relobralo")
    main_params = [
        p for n, p in task.named_parameters()
        if not any(n.startswith(prefix) for prefix in excluded_prefixes)
    ]
    optimizer_cls = task.optimizer_cls
    optimizer_args = task.optimizer_kwargs or {}   # SchNetPack stores as `_kwargs`
    optimizer = optimizer_cls(main_params, **optimizer_args)
    if scheduler_cls is not None:
        scheduler = scheduler_cls(optimizer, **(scheduler_args or {}))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",   # ReduceLROnPlateau needs a metric
                "interval": "epoch",
                "frequency": 1,
            },
        }
    return optimizer


def _make_atomistic_task_class():
    """L4 (2026-04-13): build module-level task classes lazily so that the
    `import schnetpack` happens only when this module is actually called
    (avoids the import at module-load time which slows down `python -c`
    smokes that don't need PaiNN)."""
    import torch
    import schnetpack as spk
    from dml_benchmark.loss_balancing import (
        GradNormDmlLoss, SoftmaxBalanceDmlLoss, ReLoBRaLoDmlLoss,
    )

    class DmlBalancedTask(spk.task.AtomisticTask):
        """AtomisticTask whose loss uses a dml_benchmark balancing class.

        The architecture and forward path are unchanged (PaiNN representation
        + Atomwise + Forces). Only the loss-combining step is overridden.
        """
        def __init__(self, *args, balancing_loss=None,
                     scheduler_cls=None, scheduler_args=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._balancing_loss = balancing_loss
            self._scheduler_cls = scheduler_cls
            self._scheduler_args = scheduler_args

        def loss_fn(self, pred, targets):
            lc = self._balancing_loss(
                y_pred=pred["energy"], y_true=targets["energy"],
                dydx_pred=pred["forces"], dydx_true=targets["forces"],
                model=self.model,
            )
            return lc.total

        def configure_optimizers(self):
            return _filtered_optimizer_factory(
                self, self._scheduler_cls, self._scheduler_args,
            )

    class DmlWarmupTask(spk.task.AtomisticTask):
        """Two-phase warmup task.

        Phase 1 (epoch < warmup_epochs): vanilla energy-only MSE.
        Phase 2 (epoch >= warmup_epochs): GradNorm on (energy, forces).

        H-C-R2-1/2 (2026-04-14): eager _gradnorm construction so state_dict
        always contains `_gradnorm.task_weights` (phase-1 checkpoints would
        otherwise miss this key and crash strict_loading at test time); AND
        unified val/test-time loss = MSE(E)+MSE(F) across both phases so
        ModelCheckpoint(monitor="val_loss") compares apples-to-apples (phase-1
        checkpoints would otherwise look artificially better than phase-2 simply
        because phase-1 loss excludes the force term, reducing dml_warmup to
        vanilla).

        M9 (2026-04-13): at the phase boundary, multiply the optimizer's LR
        by 1/lr_drop_factor. Matches MLP/GATv2 convention.

        Mirrors experiments/unified_comparison/run_unified_experiment.py:train_warmup.
        """
        def __init__(self, *args, warmup_epochs=0, lr_drop_factor=10.0,
                     scheduler_cls=None, scheduler_args=None,
                     gradnorm_shared_layer_name=None, **kwargs):
            # K-M3 (2026-04-16): default lr_drop_factor=10.0 per D022 ablation
            # (was 5.0, stale pre-ablation value). Callers still override via
            # HPARAMS_CANONICAL["warmup_lr_drop"].
            super().__init__(*args, **kwargs)
            self._warmup_epochs = warmup_epochs
            self._lr_drop_factor = lr_drop_factor
            self._scheduler_cls = scheduler_cls
            self._scheduler_args = scheduler_args
            # H-C-R2-1: eager construction so `_gradnorm.task_weights` is always
            # in state_dict (phase-1 checkpoints would otherwise miss this key).
            self._gradnorm = GradNormDmlLoss(
                input_dim=1, shared_layer_name=gradnorm_shared_layer_name,
            )
            self._phase2_setup_done = False
            # F-H1/G-H4 (2026-04-16): cross-arch warmup parity. MLP+GATv2 create
            # a fresh AdamW and restore best-phase-1 weights at the phase boundary.
            # We mirror that here by (a) caching the best-phase-1 state on val
            # improvement and (b) restoring + clearing Adam m_t/v_t on phase 2 entry.
            self._phase1_best_val_loss = float("inf")
            self._phase1_best_state = None
            # J-H3 (2026-04-16): capture base lr from optimizer_kwargs so
            # phase-2 LR = base_lr / lr_drop_factor (matching MLP/GATv2 which
            # build a fresh optimizer at base_lr/lr_drop_factor). Prior code
            # used grp["lr"] /= lr_drop_factor, which divides the possibly-
            # ReduceLROnPlateau-decayed LR, yielding a smaller phase-2 LR than
            # MLP/GATv2 and silently handicapping PaiNN's phase 2.
            self._base_lr = (self.optimizer_kwargs or {}).get("lr", None)

        def on_validation_epoch_end(self):
            # Cache best-phase-1 state_dict based on unified val_loss (F-H1).
            # During phase 1 the val_loss = MSE(E)+MSE(F); since force MSE stays
            # near its untrained level while energy MSE drops, the trend mirrors
            # energy quality (the phase-1 objective).
            super_fn = getattr(super(), "on_validation_epoch_end", None)
            if super_fn is not None:
                super_fn()
            # J-M6 (2026-04-16): skip Lightning's sanity-check val pass — it
            # fires before training starts, so capturing state there saves an
            # untrained model as "phase-1 best".
            if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
                return
            if self.current_epoch < self._warmup_epochs and self.trainer is not None:
                cm = self.trainer.callback_metrics.get("val_loss")
                if cm is not None:
                    v = float(cm.item()) if hasattr(cm, "item") else float(cm)
                    if v < self._phase1_best_val_loss:
                        self._phase1_best_val_loss = v
                        # Cache state_dict. Some values may be dicts (from
                        # get_extra_state in balancing losses) not tensors —
                        # only .detach().cpu().clone() tensor values.
                        self._phase1_best_state = {}
                        for k, t in self.state_dict().items():
                            if isinstance(t, torch.Tensor):
                                self._phase1_best_state[k] = t.detach().cpu().clone()
                            else:
                                import copy
                                self._phase1_best_state[k] = copy.deepcopy(t)

        def on_train_epoch_start(self):
            # F-H1 (2026-04-16): at phase-2 transition, restore best-phase-1
            # state, clear Adam m_t/v_t, and drop LR by lr_drop_factor.
            super_fn = getattr(super(), "on_train_epoch_start", None)
            if super_fn is not None:
                super_fn()
            if (
                self.current_epoch == self._warmup_epochs
                and not self._phase2_setup_done
            ):
                if self._phase1_best_state is not None:
                    self.load_state_dict(self._phase1_best_state, strict=False)
                    # J-L2 (2026-04-16): free the phase-1 state after restoring
                    # it, so the ~500KB blob is not kept for the rest of training.
                    self._phase1_best_state = None
                if hasattr(self, "optimizers"):
                    opts = self.optimizers(use_pl_optimizer=False)
                    if not isinstance(opts, (list, tuple)):
                        opts = [opts]
                    for opt in opts:
                        if opt is None:
                            continue
                        # Clear Adam state (m_t, v_t) so phase-1 momentum
                        # (tuned for value-only gradients) doesn't pollute phase 2.
                        for s in list(opt.state.values()):
                            s.clear()
                        # J-H3 (2026-04-16): set from BASE lr, not current
                        # (possibly-decayed) LR. This matches MLP/GATv2.
                        new_lr = (self._base_lr / self._lr_drop_factor
                                   if self._base_lr is not None
                                   else (opt.param_groups[0]["lr"] / self._lr_drop_factor))
                        for grp in opt.param_groups:
                            grp["lr"] = new_lr
                # J-M3 (2026-04-16): reset scheduler state so phase-1 plateau
                # accountancy doesn't leak into phase 2.
                if hasattr(self, "trainer") and self.trainer is not None:
                    for lr_conf in getattr(self.trainer, "lr_scheduler_configs", []):
                        sch = lr_conf.scheduler
                        if hasattr(sch, "best"):
                            sch.best = float("inf")
                        if hasattr(sch, "num_bad_epochs"):
                            sch.num_bad_epochs = 0
                        if hasattr(sch, "cooldown_counter"):
                            sch.cooldown_counter = 0
                self._phase2_setup_done = True

        def loss_fn(self, pred, targets):
            # H-C-R2-2: unified val/test loss across phases so ModelCheckpoint's
            # monitor="val_loss" selects phase-2-trained weights (which actually
            # learn forces) over phase-1-trained weights.
            if not self.training:
                return (torch.nn.functional.mse_loss(pred["energy"], targets["energy"])
                        + torch.nn.functional.mse_loss(pred["forces"], targets["forces"]))
            if self.current_epoch < self._warmup_epochs:
                # Phase 1 (train) — vanilla energy-only
                return torch.nn.functional.mse_loss(pred["energy"], targets["energy"])
            # Phase 2 (train) — GradNorm. The LR drop + state reset happens in
            # on_train_epoch_start; loss_fn here is a pure loss computation.
            lc = self._gradnorm(
                y_pred=pred["energy"], y_true=targets["energy"],
                dydx_pred=pred["forces"], dydx_true=targets["forces"],
                model=self.model,
            )
            return lc.total

        def configure_optimizers(self):
            return _filtered_optimizer_factory(
                self, self._scheduler_cls, self._scheduler_args,
            )

    return (
        DmlBalancedTask, DmlWarmupTask,
        GradNormDmlLoss, SoftmaxBalanceDmlLoss, ReLoBRaLoDmlLoss,
    )


def _make_balanced_task(nnp, method, lr, warmup_fraction, n_epochs,
                        scheduler_cls=None, scheduler_args=None,
                        n_interactions=3, seed=42):
    """
    Build a custom AtomisticTask subclass whose loss is computed via
    dml_benchmark.loss_balancing classes.

    For vanilla / dml_fixed / native_EF, we fall back to standard SchNetPack
    AtomisticTask (linear loss combination).

    For dml_warmup, the task switches loss class at warmup_epochs.

    D-M3 (2026-04-14): GradNorm "shared layer" is the final PaiNN mixing block's
    intra-atomic linear — the last shared backbone layer before the Atomwise head.
    Name depends on n_interactions: `representation.mixing.{n-1}.intraatomic_context_net.1`.

    M-C-R2-1 (2026-04-14): seed is threaded to balancing-loss classes so the
    stochastic balancing RNG (saudade/reference-step draws) varies across
    experimental seeds.
    """
    import torch
    import schnetpack as spk

    DmlBalancedTask, DmlWarmupTask, GradNormDmlLoss, SoftmaxBalanceDmlLoss, ReLoBRaLoDmlLoss = \
        _make_atomistic_task_class()

    gradnorm_shared = f"representation.mixing.{n_interactions - 1}.intraatomic_context_net.1"

    # Build outputs with metrics (used for logging only when method uses
    # SchNetPack's standard loss; balanced methods override loss_fn).
    import torchmetrics
    output_e = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,  # placeholder; not used when our loss_fn override is active
        metrics={
            "mae": torchmetrics.regression.MeanAbsoluteError(),
            "rmse": torchmetrics.regression.MeanSquaredError(squared=False),
        },
    )
    output_f = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={
            "mae": torchmetrics.regression.MeanAbsoluteError(),
            "rmse": torchmetrics.regression.MeanSquaredError(squared=False),
        },
    )

    common_kwargs = dict(
        model=nnp, outputs=[output_e, output_f],
        optimizer_cls=torch.optim.AdamW, optimizer_args={"lr": lr, "weight_decay": 0.0},
        scheduler_cls=scheduler_cls, scheduler_args=scheduler_args,
    )

    if method == "dml_gradnorm":
        balancing = GradNormDmlLoss(input_dim=1, shared_layer_name=gradnorm_shared)
        return DmlBalancedTask(balancing_loss=balancing, **common_kwargs)
    if method == "dml_softmax_balance":
        balancing = SoftmaxBalanceDmlLoss(input_dim=1, seed=seed)
        return DmlBalancedTask(balancing_loss=balancing, **common_kwargs)
    if method == "dml_relobralo":
        # 2026-04-13: faithful Bischof & Kraus 2022 Eq.11
        balancing = ReLoBRaLoDmlLoss(input_dim=1, seed=seed)
        return DmlBalancedTask(balancing_loss=balancing, **common_kwargs)
    if method == "dml_warmup":
        warmup_epochs = int(n_epochs * warmup_fraction)
        return DmlWarmupTask(
            warmup_epochs=warmup_epochs,
            lr_drop_factor=HPARAMS_CANONICAL.get("warmup_lr_drop", 5.0),
            gradnorm_shared_layer_name=gradnorm_shared,
            **common_kwargs,
        )

    # vanilla / dml_fixed / dml_fixed_half / native_EF — use standard SchNetPack AtomisticTask
    if method == "vanilla":
        output_e.loss_weight = 1.0
        output_f.loss_weight = 0.0
    elif method in ("dml_fixed", "dml_fixed_half"):
        # E-H2 (2026-04-13): both aliases → 0.5/0.5 convex combination matching
        # MLP's dml_fixed_half. Kept as separate method names for cross-arch
        # naming parity (MLP's dml_fixed uses H&S weights; dml_fixed_half is
        # the symmetric variant).
        output_e.loss_weight = 0.5
        output_f.loss_weight = 0.5
    elif method == "native_EF":
        # SchNetPack canonical (NOT faithful to Schütt et al. paper which uses ρ=0.95)
        output_e.loss_weight = 0.01
        output_f.loss_weight = 0.99
    else:
        raise ValueError(f"Unknown method: {method}")

    # P5 (2026-04-13): SchNetPack standard task supports scheduler kwargs natively
    task_kwargs = dict(common_kwargs)
    task_kwargs.pop("scheduler_cls", None)
    task_kwargs.pop("scheduler_args", None)
    if scheduler_cls is not None:
        task_kwargs["scheduler_cls"] = scheduler_cls
        task_kwargs["scheduler_args"] = scheduler_args
        task_kwargs["scheduler_monitor"] = "val_loss"
    return spk.task.AtomisticTask(**task_kwargs)


# ============================================================================
# TRAIN ONE
# ============================================================================

def train_one(molecule, method, seed, hparams, smoke=False, gpu=1, split_id=None,
               warmup_fraction=None):
    """Train one (molecule, method, split) PaiNN configuration.

    P3 (2026-04-13): if `split_id` is provided (1..5), uses SchNetPack's
    canonical Figshare split via the `split_id` argument of rMD17. Otherwise
    falls back to per-seed random split (legacy).
    """
    import torch
    import pytorch_lightning as pl
    import schnetpack as spk
    import schnetpack.transform as trn
    from schnetpack.datasets import rMD17

    # 2026-04-14: torch>=2.6 default `weights_only=True` breaks PL's
    # checkpoint loader on `trainer.test(ckpt_path=best)` for SchNetPack
    # pickled hyperparameters. Lightning's loader calls `torch.load(weights_only=True)`
    # and Lightning's allowlist does not include SchNetPack classes. Rather
    # than try to enumerate every class, monkey-patch `torch.load` inside
    # this process to default to `weights_only=False` (the ckpt is produced
    # by our own code this run, so trust is established).
    # J-L4 (2026-04-16): restore on exit so the monkey-patch doesn't leak into
    # subsequent training runs, tests, or analysis scripts that share this
    # Python process.
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    import atexit
    atexit.register(lambda: setattr(torch, "load", _orig_torch_load))

    # Determinism (note: not bit-perfect on GPU; Lightning sets seeds internally too)
    pl.seed_everything(seed, workers=True)

    spk_data_dir = ROOT / "data" / "schnetpack_rmd17"
    spk_data_dir.mkdir(parents=True, exist_ok=True)
    db_path = spk_data_dir / f"rmd17_{molecule}.db"

    # P3 (2026-04-13): use canonical Figshare split via SchNetPack's split_id
    # parameter (0..4 in SchNetPack, 1..5 in the rMD17 Figshare CSV naming).
    if split_id is not None:
        # Convert 1..5 → 0..4 for SchNetPack's split_id index
        split_path = spk_data_dir / f"split_{molecule}_canonical_{split_id}.npz"
        canonical_split_id = split_id - 1
    else:
        split_path = spk_data_dir / f"split_{molecule}_s{seed}.npz"
        canonical_split_id = None

    # ----- Data -----
    # num_test=1000 for cross-architecture parity with MLP/GATv2.
    dataset = rMD17(
        datapath=str(db_path),
        molecule=_SCHNETPACK_MOL_KEY.get(molecule, molecule),
        batch_size=hparams["batch_size"],
        num_train=hparams["num_train"] if not smoke else 100,
        num_val=hparams["num_val"] if not smoke else 20,
        num_test=hparams.get("num_test", 1000) if not smoke else 100,
        split_file=str(split_path),
        split_id=canonical_split_id,
        distance_unit="Ang",
        property_units={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
        transforms=[
            trn.SubtractCenterOfMass(),
            trn.RemoveOffsets(property="energy", remove_mean=True),
            trn.MatScipyNeighborList(cutoff=hparams["cutoff"]),
            trn.CastTo32(),
        ],
        num_workers=0,
    )
    dataset.prepare_data()
    dataset.setup()

    # ----- Model -----
    radial = spk.nn.GaussianRBF(n_rbf=hparams["n_rbf"], cutoff=hparams["cutoff"])
    cutoff_fn = spk.nn.CosineCutoff(cutoff=hparams["cutoff"])
    representation = spk.representation.PaiNN(
        n_atom_basis=hparams["n_atom_basis"],
        n_interactions=hparams["n_interactions"],
        radial_basis=radial,
        cutoff_fn=cutoff_fn,
    )
    pred_energy = spk.atomistic.Atomwise(
        output_key="energy",
        n_in=hparams["n_atom_basis"],
        aggregation_mode="sum",
    )
    pred_forces = spk.atomistic.Forces(
        energy_key="energy",
        force_key="forces",
    )

    nnp = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(property="energy", add_mean=True),
        ],
    )

    # ----- Task: dispatch on method, use single source of truth for balancing -----
    n_epochs = 5 if smoke else hparams["n_epochs"]

    # P5 (2026-04-13) + D-M2 (2026-04-14): canonical SchNetPack kwargs incl.
    # cooldown=10 + threshold=0.0 (torch default cooldown=0, threshold=1e-4
    # decays earlier than canonical SchNetPack runs).
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_args = {
        "mode": "min",
        "factor": hparams.get("scheduler_factor", 0.5),
        "patience": hparams.get("scheduler_patience", 75),
        "min_lr": hparams.get("scheduler_min_lr", 0.0),
        "cooldown": hparams.get("scheduler_cooldown", 10),
        "threshold": hparams.get("scheduler_threshold", 0.0),
        "threshold_mode": "rel",
    }

    tau = WARMUP_FRACTION if warmup_fraction is None else float(warmup_fraction)
    task = _make_balanced_task(
        nnp=nnp, method=method, lr=hparams["lr"],
        warmup_fraction=tau, n_epochs=n_epochs,
        scheduler_cls=scheduler_cls, scheduler_args=scheduler_args,
        n_interactions=hparams["n_interactions"],
        seed=seed,
    )

    # Nominal output weights for JSON reporting
    if method == "dml_warmup":
        w_e, w_f = 1.0, 0.0  # phase 1 reporting
    else:
        w_e = {
            "vanilla": 1.0, "dml_fixed": 0.5,
            "native_EF": hparams["loss_weight_energy_native"],
            "dml_gradnorm": None, "dml_softmax_balance": None, "dml_relobralo": None,
        }.get(method, None)
        w_f = {
            "vanilla": 0.0, "dml_fixed": 0.5,
            "native_EF": hparams["loss_weight_forces_native"],
            "dml_gradnorm": None, "dml_softmax_balance": None, "dml_relobralo": None,
        }.get(method, None)

    # ----- Trainer -----
    # D-M1 (2026-04-14): ES patience = SchNetPack canonical 200.
    # (Prior "PaiNN Table 5 = 150" comment was incorrect — the PaiNN paper
    # arXiv:2102.03150 has no documented ES patience.)
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    es_callback = EarlyStopping(
        monitor="val_loss",   # SchNetPack's AtomisticTask logs total weighted loss as "val_loss"
        mode="min",
        patience=hparams.get("es_patience", 200),
        min_delta=0.0,
        verbose=False,
    )

    # E-H4 (2026-04-13): save best-val-loss checkpoint so test runs on the
    # best model (previously test used the ES-stopped or last-epoch state —
    # systematic late-epoch overfit bias). Restored via trainer.test(ckpt_path="best").
    import tempfile
    ckpt_dir = tempfile.mkdtemp(prefix=f"painn_ckpt_{molecule}_{method}_")
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    callbacks = []
    if not smoke:
        callbacks.extend([es_callback, ckpt_callback])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=not smoke,
        enable_progress_bar=False,
        enable_model_summary=False,
        # Required for SchNetPack autograd forces (Forces module computes
        # F=-dE/dx; in inference_mode=True the dE/dx call has no grad graph).
        inference_mode=False,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
    )

    t0 = time.time()
    trainer.fit(task, datamodule=dataset)
    train_time = time.time() - t0

    # ----- Test -----
    # E-H4 (2026-04-13): use best-val-loss checkpoint for test (not the last state).
    if not smoke and ckpt_callback.best_model_path:
        test_results = trainer.test(
            task, datamodule=dataset, verbose=False,
            ckpt_path=ckpt_callback.best_model_path,
        )
    else:
        test_results = trainer.test(task, datamodule=dataset, verbose=False)
    metrics = test_results[0] if test_results else {}

    # SchNetPack metric keys: e.g. "test_energy_mae", "test_energy_rmse",
    # "test_forces_mae", "test_forces_rmse". Convert RMSE → MSE for our schema.
    energy_mae = float(metrics.get("test_energy_mae", float("nan")))
    energy_rmse = float(metrics.get("test_energy_rmse", float("nan")))
    forces_mae = float(metrics.get("test_forces_mae", float("nan")))
    forces_rmse = float(metrics.get("test_forces_rmse", float("nan")))

    energy_mse = energy_rmse ** 2 if energy_rmse == energy_rmse else float("nan")
    forces_mse = forces_rmse ** 2 if forces_rmse == forces_rmse else float("nan")
    KCAL_TO_MEV = 43.3641

    key_prefix = f"painn_md17_{molecule}"
    if split_id is not None:
        key_str = f"{key_prefix}_split{split_id}_{method}"
    else:
        key_str = f"{key_prefix}_s{seed}_{method}"
    out = {
        "key": key_str,
        "method": method,
        "model": "PaiNN_schnetpack_v2.2.0",
        "dataset": f"md17_{molecule}",
        "molecule": molecule,
        "seed": seed,
        "split_id": split_id,
        "test_value_mse": energy_mse,
        "test_grad_mse": forces_mse,
        "test_energy_mae_kcal": energy_mae,
        "test_force_mae_kcal": forces_mae,
        "test_energy_mae_mev": energy_mae * KCAL_TO_MEV,
        "test_force_mae_mev": forces_mae * KCAL_TO_MEV,
        "loss_weight_energy": w_e,
        "loss_weight_forces": w_f,
        "n_epochs_actual": n_epochs,
        "time_s": round(train_time, 2),
        "hparams": hparams,
        "all_metrics": {k: float(v) if hasattr(v, "__float__") else str(v)
                         for k, v in metrics.items()},
        "timestamp": datetime.now().isoformat(),
        "source": "schnetpack 2.2.0; canonical rMD17 + PaiNN config; arXiv:2102.03150",
    }
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--molecules", nargs="+", default=DEFAULT_MOLECULES)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Legacy: only used if --use_random_splits.")
    parser.add_argument("--split_ids", type=str, default=",".join(map(str, DEFAULT_SPLIT_IDS)),
                        help="Canonical Figshare split IDs (1..5). Default: 5-fold CV.")
    parser.add_argument("--use_random_splits", action="store_true",
                        help="Use random per-seed splits (legacy).")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--results_dir", default="results/molecular_painn")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size (canonical=10; use 128 for speed)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Per AGENT_PRINCIPLES §11
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
    os.environ.setdefault("BLIS_NUM_THREADS", "6")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")
    import torch
    torch.set_num_threads(6)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"PaiNN runner; gpu={args.gpu}; smoke={args.smoke}; results_dir={results_dir}")

    existing = set()
    if args.resume:
        for p in results_dir.glob("*.json"):
            try:
                existing.add(json.load(open(p))["key"])
            except Exception:
                pass

    # P3 (2026-04-13): canonical splits by default
    split_ids = [int(s.strip()) for s in args.split_ids.split(",")]

    # Override batch_size if requested (default: canonical 10)
    hparams = dict(HPARAMS_CANONICAL)
    if args.batch_size is not None:
        hparams["batch_size"] = args.batch_size
        print(f"  batch_size overridden to {args.batch_size}")

    n_done, n_failed = 0, 0
    for mol in args.molecules:
        if args.use_random_splits:
            iters = [(None, s) for s in seeds]
        else:
            iters = [(sid, sid) for sid in split_ids]   # seed = split_id for repro
        for split_id, seed in iters:
            for method in args.methods:
                if split_id is not None:
                    key = f"painn_md17_{mol}_split{split_id}_{method}"
                else:
                    key = f"painn_md17_{mol}_s{seed}_{method}"
                if args.resume and key in existing:
                    print(f"SKIP {key}")
                    continue
                print(f"\n--- {key} ---")
                try:
                    result = train_one(mol, method, seed, hparams,
                                       smoke=args.smoke, gpu=args.gpu,
                                       split_id=split_id)
                    save_path = results_dir / f"{key}.json"
                    with open(save_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"  OK ({result['time_s']:.1f}s)  E_MAE={result['test_energy_mae_mev']:.1f} meV  "
                          f"F_MAE={result['test_force_mae_mev']:.1f} meV/Å")
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  FAIL: {e}")
                    traceback.print_exc()

    print(f"\nDone. {n_done} succeeded, {n_failed} failed.")


if __name__ == "__main__":
    main()
