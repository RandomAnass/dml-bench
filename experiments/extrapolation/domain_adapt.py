"""
Domain-adaptation arm for the SIREN periodic-extrap pilot.

Implements the §6.3 sub-bullet "domain-adaptation training that recomputes
gradient labels at test-time inputs" as an additional ablation arm on top of
M1's SIREN cell.  Compared to the existing DML arm of `pilot_periodic_extrap`:

  - vanilla:    value MSE on x_in (24 pts).                            [M1 baseline]
  - dml_train:  value MSE + λ·grad MSE on x_in (24 pts).               [M1 DML]
  - dml_da:     value MSE on x_in (24 pts) +
                λ·grad MSE on x_in ∪ x_out (24+24 = 48 pts).           [NEW]

x_out is drawn from [-1.5, -1] ∪ [1, 1.5] (the near-extrap region used by
`region_masks`) and gradient labels at those points come from the **closed-form
analytic formula** `target_grad`.  Noise is added at the same `sigma_rel`
scale as the in-support arm using the same `g_scale` (so the σ grids are
directly comparable across arms).

Reproducibility-protection clause (binding):

  - This file ONLY imports from `pilot_periodic_extrap` — no edits or
    monkey-patching of that module.
  - Vanilla and dml_train arms are produced by `pilot_periodic_extrap.train_model`
    unchanged, so they bit-for-bit replicate M1 SIREN rows when run with the
    same seeds.  This is the harness's built-in regression test.
  - Only the dml_da arm uses a locally re-implemented training loop with
    weighted (mask) MSE for value loss.  When `n_da == 0` it reduces exactly
    to dml_train.
  - Output goes to `results/extrapolation_M1_DA/` (sibling of M1, not child).

Run:
  python experiments/extrapolation/domain_adapt.py --smoke   # 1 seed × 3 σ × 3 arms
  python experiments/extrapolation/domain_adapt.py --wide-sigma --seeds 5  # full
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Pure-import from pilot_periodic_extrap — no edits, no monkey-patching.
from pilot_periodic_extrap import (
    Config,
    target_grad,
    make_train_data,
    make_eval_grid,
    standardize_eval,
    region_masks,
    make_grad_noise,
    apply_grad_noise,
    make_model,
    clone_state,
    load_state,
    set_torch_seed,
    stable_seed,
    to_tensor,
    train_model,
    predict_values,
    predict_grads,
    mse_by_region,
    estimate_sigma_star,
)


# -----------------------------------------------------------------------------
# DA data sampling
# -----------------------------------------------------------------------------
def make_da_data(target_seed: int, n_da: int, y_std: float,
                 near_extrap_max: float = 1.5,
                 train_domain=(-1.0, 1.0)) -> dict:
    """Sample n_da points uniformly on [-1.5, -1] ∪ [1, 1.5], compute clean
    closed-form gradient via target_grad, standardise by y_std (the same factor
    used by `make_train_data`).

    Returns:
        {"x":     (n_da, 1) float32 raw inputs in [-1.5, +1.5] excluding [-1, 1],
         "g_raw": (n_da,)   float32 unstandardised closed-form gradient,
         "g":     (n_da, 1) float32 standardised gradient labels (g_raw / y_std)}
    """
    if n_da == 0:
        return {
            "x": np.zeros((0, 1), dtype=np.float32),
            "g_raw": np.zeros((0,), dtype=np.float32),
            "g": np.zeros((0, 1), dtype=np.float32),
        }
    rng = np.random.default_rng(stable_seed("da-sample", target_seed))
    # Half on the negative side, half on the positive side.
    n_neg = n_da // 2
    n_pos = n_da - n_neg
    x_neg = rng.uniform(-near_extrap_max, train_domain[0], size=n_neg)
    x_pos = rng.uniform(train_domain[1], near_extrap_max, size=n_pos)
    x = np.sort(np.concatenate([x_neg, x_pos])).astype(np.float32)
    g = target_grad(x).astype(np.float32)
    return {
        "x": x[:, None],
        "g_raw": g,
        "g": (g / y_std)[:, None].astype(np.float32),
    }


def make_da_grad_noise(g_clean: np.ndarray, target_seed: int) -> tuple:
    """Independent noise stream for DA gradient labels (separate from the
    in-support stream so they don't share random draws)."""
    rng = np.random.default_rng(stable_seed("da-grad-noise", target_seed))
    eps = rng.normal(size=g_clean.shape).astype(np.float32)
    return eps


# -----------------------------------------------------------------------------
# Training loop with value-loss mask (DA arm)
# -----------------------------------------------------------------------------
def train_model_da(model_name: str, init_state: dict,
                   x_in: np.ndarray, y_in: np.ndarray, g_in_noisy: np.ndarray,
                   x_da: np.ndarray, g_da_noisy: np.ndarray,
                   g_scale: np.ndarray, cfg: Config,
                   train_seed: int) -> torch.nn.Module:
    """Train a model with an augmented gradient supervision set.

    Value MSE is computed only on the first `len(x_in)` rows (the in-support
    part).  Gradient MSE is computed on all rows (in-support + DA).

    This re-implements the body of `pilot_periodic_extrap.train_model` with two
    differences:
      1. The gradient set is x_in ∪ x_da, the value set is x_in only.
      2. The value loss uses a 0/1 mask so the implementation reduces exactly
         to dml_train when len(x_da) == 0.

    Everything else (make_model, load_state, AdamW, grad_clip, set_torch_seed,
    cfg.dml_lambda) is imported unchanged.
    """
    n_in = x_in.shape[0]
    n_da = x_da.shape[0]

    # Stack inputs and grad targets; pad y with zeros for the DA rows
    # (mask zeroes them out, so the pad value is irrelevant).
    x_aug = np.concatenate([x_in, x_da], axis=0).astype(np.float32)
    g_aug = np.concatenate([g_in_noisy, g_da_noisy], axis=0).astype(np.float32)
    y_aug = np.concatenate(
        [y_in, np.zeros(n_da, dtype=np.float32)], axis=0
    ).astype(np.float32)
    w_y = np.concatenate(
        [np.ones(n_in, dtype=np.float32), np.zeros(n_da, dtype=np.float32)],
        axis=0,
    )

    model = make_model(model_name, target_seed=train_seed, cfg=cfg)
    load_state(model, init_state, cfg.device)
    model.train()

    x_t = to_tensor(x_aug, cfg.device)
    y_t = to_tensor(y_aug, cfg.device)
    g_t = to_tensor(g_aug, cfg.device)
    w_t = to_tensor(w_y, cfg.device)
    g_scale_t = to_tensor(g_scale, cfg.device)
    n_in_y = max(int(w_t.sum().item()), 1)  # safe denominator for value loss

    lr = cfg.base_lr[model_name]
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg.weight_decay)

    set_torch_seed(train_seed)
    for _ in range(cfg.n_steps):
        xb = x_t.detach().clone().requires_grad_(True)
        pred = model(xb)
        grad_pred = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]

        # Weighted value loss: 0/1 mask, divide by sum(w_y) (= n_in).
        # Identical to F.mse_loss(pred[:n_in], y_in) when w is binary [1]*n_in + [0]*n_da.
        sq_y = (pred - y_t) ** 2
        value_loss = (w_t * sq_y).sum() / n_in_y

        # Unweighted gradient MSE on full augmented set.
        grad_loss = torch.mean(((grad_pred - g_t) / g_scale_t) ** 2)

        loss = value_loss + cfg.dml_lambda * grad_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    return model


# -----------------------------------------------------------------------------
# Driver — runs vanilla, dml_train, dml_da arms per cell
# -----------------------------------------------------------------------------
def run_da(cfg: Config, n_da: int = 24) -> Path:
    """Replicates the cell loop of `run_pilot` but emits three arms per
    (model, seed, sigma) cell: vanilla, dml_train, dml_da.  The vanilla and
    dml_train arms reuse the unchanged `train_model` from pilot_periodic_extrap
    so they bit-for-bit replicate M1.  Only `dml_da` uses the local
    `train_model_da`.

    Schema matches `results/extrapolation_M1/rows.csv` plus an extra `arm`
    column.
    """
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    rows_path = cfg.out_root / "rows.csv"
    rows: list = []
    x_eval, y_raw, g_raw = make_eval_grid(cfg.n_eval, cfg.eval_domain)

    t0_global = time.time()

    for model_name in cfg.models:
        for target_seed in range(cfg.n_seeds):
            t_cell = time.time()
            train = make_train_data(target_seed, cfg.n_train, cfg.train_domain)
            y_eval, g_eval = standardize_eval(y_raw, g_raw,
                                              train["y_mean"], train["y_std"])

            init_model = make_model(model_name, target_seed, cfg)
            init_state = clone_state(init_model)

            # ---- Vanilla arm (M1-identical) ----
            vanilla_seed = stable_seed("vanilla-train", model_name, target_seed)
            t1 = time.time()
            vanilla = train_model(model_name, init_state, train["x"], train["y"],
                                  cfg, mode="vanilla", train_seed=vanilla_seed)
            t_vanilla = time.time() - t1
            pred_v = predict_values(vanilla, x_eval, cfg.device)
            grad_v = predict_grads(vanilla, x_eval, cfg.device)
            mse_v = mse_by_region(x_eval, pred_v, y_eval)
            grad_mse_v = mse_by_region(x_eval, grad_v, g_eval)

            # ---- σ-loop, both DML arms share noise streams ----
            g_scale, eps = make_grad_noise(train["g"], target_seed)
            dml_seed_fixed = stable_seed("dml-train", model_name, target_seed)

            # DA fixed inputs and clean gradient (same across σ within cell).
            da = make_da_data(target_seed, n_da, train["y_std"],
                              near_extrap_max=1.5, train_domain=cfg.train_domain)
            eps_da = make_da_grad_noise(da["g"], target_seed)
            g_scale_full = (np.std(np.concatenate([train["g"], da["g"]], axis=0),
                                   axis=0, keepdims=True) + 1e-12).astype(np.float32)
            # Use the in-support g_scale for the (pred-grad)/g_scale standardisation
            # so the loss scale is identical to dml_train.

            for sigma_rel in cfg.sigma_grid:
                sigma_rel = float(sigma_rel)

                # ---- dml_train arm (M1-identical) ----
                t2 = time.time()
                g_noisy = apply_grad_noise(train["g"], g_scale, eps, sigma_rel)
                dml = train_model(model_name, init_state, train["x"], train["y"],
                                  cfg, g_train=g_noisy, g_scale=g_scale,
                                  mode="dml", train_seed=dml_seed_fixed)
                pred_dt = predict_values(dml, x_eval, cfg.device)
                grad_dt = predict_grads(dml, x_eval, cfg.device)
                mse_dt = mse_by_region(x_eval, pred_dt, y_eval)
                grad_mse_dt = mse_by_region(x_eval, grad_dt, g_eval)
                t_dt = time.time() - t2

                # ---- dml_da arm (NEW) ----
                t3 = time.time()
                # Match noise scaling to in-support arm: use g_scale (1d, in-support
                # std) for the DA noise too — guarantees that sigma_rel=0 means
                # exact gradient labels at x_da, exactly matching the analytic
                # closed-form, and that the σ-grid is comparable across arms.
                g_da_noisy = apply_grad_noise(da["g"], g_scale, eps_da, sigma_rel)
                dml_da = train_model_da(
                    model_name, init_state,
                    x_in=train["x"], y_in=train["y"], g_in_noisy=g_noisy,
                    x_da=da["x"], g_da_noisy=g_da_noisy,
                    g_scale=g_scale, cfg=cfg, train_seed=dml_seed_fixed,
                )
                pred_dda = predict_values(dml_da, x_eval, cfg.device)
                grad_dda = predict_grads(dml_da, x_eval, cfg.device)
                mse_dda = mse_by_region(x_eval, pred_dda, y_eval)
                grad_mse_dda = mse_by_region(x_eval, grad_dda, g_eval)
                t_dda = time.time() - t3

                # ---- Emit rows for all 3 arms × 4 regions ----
                for region in ("interpolation", "near_extrap", "far_extrap", "all"):
                    eps_safe = 1e-30
                    # vanilla "self" row (mse_dml = mse_vanilla, log_ratio = 0).
                    rows.append({
                        "model": model_name,
                        "seed": target_seed,
                        "sigma_rel": sigma_rel,
                        "region": region,
                        "n_train": cfg.n_train,
                        "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region],
                        "mse_dml": mse_v[region],
                        "log_ratio": 0.0,
                        "dml_win": 0,
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_v[region],
                        "log_grad_ratio": 0.0,
                        "t_vanilla": t_vanilla,
                        "t_dml": t_vanilla,
                        "arm": "vanilla",
                    })
                    # dml_train row (M1-equivalent).
                    log_ratio_dt = float(np.log(
                        (mse_dt[region] + eps_safe) / (mse_v[region] + eps_safe)))
                    log_grad_ratio_dt = float(np.log(
                        (grad_mse_dt[region] + eps_safe) / (grad_mse_v[region] + eps_safe)))
                    rows.append({
                        "model": model_name,
                        "seed": target_seed,
                        "sigma_rel": sigma_rel,
                        "region": region,
                        "n_train": cfg.n_train,
                        "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region],
                        "mse_dml": mse_dt[region],
                        "log_ratio": log_ratio_dt,
                        "dml_win": int(mse_dt[region] < mse_v[region]),
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_dt[region],
                        "log_grad_ratio": log_grad_ratio_dt,
                        "t_vanilla": t_vanilla,
                        "t_dml": t_dt,
                        "arm": "dml_train",
                    })
                    # dml_da row (NEW).
                    log_ratio_dda = float(np.log(
                        (mse_dda[region] + eps_safe) / (mse_v[region] + eps_safe)))
                    log_grad_ratio_dda = float(np.log(
                        (grad_mse_dda[region] + eps_safe) / (grad_mse_v[region] + eps_safe)))
                    rows.append({
                        "model": model_name,
                        "seed": target_seed,
                        "sigma_rel": sigma_rel,
                        "region": region,
                        "n_train": cfg.n_train,
                        "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region],
                        "mse_dml": mse_dda[region],
                        "log_ratio": log_ratio_dda,
                        "dml_win": int(mse_dda[region] < mse_v[region]),
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_dda[region],
                        "log_grad_ratio": log_grad_ratio_dda,
                        "t_vanilla": t_vanilla,
                        "t_dml": t_dda,
                        "arm": "dml_da",
                    })

                # one print per σ
                near_dt = next(r for r in reversed(rows)
                               if r["region"] == "near_extrap" and r["arm"] == "dml_train")
                near_dda = next(r for r in reversed(rows)
                                if r["region"] == "near_extrap" and r["arm"] == "dml_da")
                print(f"  {model_name:8s} seed={target_seed} σ={sigma_rel:5.3f} "
                      f"lr_near[dml]={near_dt['log_ratio']:+.3f} "
                      f"lr_near[da]={near_dda['log_ratio']:+.3f} "
                      f"t_dml={t_dt:.1f}s t_da={t_dda:.1f}s",
                      flush=True)

            print(f"[cell done] {model_name} seed={target_seed} "
                  f"t_cell={time.time() - t_cell:.1f}s", flush=True)

            # incremental save
            pd.DataFrame(rows).to_csv(rows_path, index=False)

    print(f"\nTotal wall time: {(time.time() - t0_global) / 60:.2f} min", flush=True)
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    (cfg.out_root / "config.json").write_text(json.dumps(
        {**asdict(cfg), "out_root": str(cfg.out_root),
         "sigma_grid": list(cfg.sigma_grid),
         "models": list(cfg.models),
         "n_da": n_da}, indent=2, default=str))
    return rows_path


def analyze_da(cfg: Config) -> pd.DataFrame:
    """σ* per (model, region, arm) on the DA results CSV."""
    df = pd.read_csv(cfg.out_root / "rows.csv")
    df = df.drop_duplicates(["model", "seed", "sigma_rel", "region", "arm"],
                            keep="last")
    summary = []
    for (model, region, arm), sub in df.groupby(["model", "region", "arm"]):
        if arm == "vanilla":
            continue  # vanilla "self" rows are by construction log_ratio=0
        sigma_star, status = estimate_sigma_star(sub, cfg.lowess_frac)
        sub0 = sub[sub["sigma_rel"] == 0.0]
        sub_max = sub[sub["sigma_rel"] == sub["sigma_rel"].max()]
        summary.append({
            "model": model,
            "region": region,
            "arm": arm,
            "sigma_star": sigma_star,
            "status": status,
            "mean_log_ratio_sigma0": float(sub0["log_ratio"].mean()),
            "std_log_ratio_sigma0": float(sub0["log_ratio"].std(ddof=1))
                if len(sub0) > 1 else float("nan"),
            "mean_log_ratio_max_sigma": float(sub_max["log_ratio"].mean()),
            "n_seeds": int(sub0["seed"].nunique()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cfg.out_root / "sigma_star_summary.csv", index=False)
    print("\n===== σ* SUMMARY (per arm) =====")
    pd.set_option("display.max_rows", 200)
    print(summary_df.sort_values(["model", "region", "arm"]).to_string(index=False))
    return summary_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 seed × 3 σ × 200 steps")
    p.add_argument("--seeds", type=int, default=None,
                   help="override number of seeds (default 5 if --wide-sigma else 4)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--models", type=str, default="siren",
                   help="comma-separated model list (default 'siren')")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--wide-sigma", action="store_true",
                   help="match M1 sigma grid (18 points)")
    p.add_argument("--n-da", type=int, default=24,
                   help="number of out-of-support DA points (default 24)")
    p.add_argument("--shard", type=int, default=0,
                   help="seed-shard index for parallel runs")
    p.add_argument("--n-shards", type=int, default=1,
                   help="total shards (each takes seeds % n_shards == shard)")
    args = p.parse_args()

    cfg = Config()
    cfg.smoke = args.smoke

    if args.smoke:
        cfg.apply_smoke()
        cfg.out_root = Path("results/extrapolation_M1_DA_smoke")
    else:
        cfg.out_root = Path("results/extrapolation_M1_DA")

    cfg.wide_sigma = args.wide_sigma
    cfg.apply_wide_sigma()
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    elif not args.smoke and args.wide_sigma:
        cfg.n_seeds = 5
    if args.steps is not None:
        cfg.n_steps = args.steps
    cfg.models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if args.out_dir is not None:
        cfg.out_root = Path(args.out_dir)

    # Sharding by target_seed: replace the seed-loop range with the subset
    # whose seed % n_shards == shard.  We do this by overriding cfg.n_seeds
    # later — simpler is to filter in run_da, but we keep run_da simple and
    # instead pass a SeedRange object below.
    if args.n_shards > 1:
        # We use a custom run with seed filtering.
        rows_path = run_da_sharded(cfg, args.n_da, args.shard, args.n_shards)
    else:
        rows_path = run_da(cfg, n_da=args.n_da)
    print(f"\n[saved] {rows_path}")

    if args.n_shards == 1 or args.shard == 0:
        # Only main shard (or non-sharded run) does analysis post-hoc.
        # In sharded runs, `merge_shards.py` (or a follow-up script) would do this.
        try:
            analyze_da(cfg)
        except Exception as e:
            print(f"[analyze] skipped: {e}")


def run_da_sharded(cfg: Config, n_da: int, shard: int, n_shards: int) -> Path:
    """Same as run_da but only iterates target_seed values where
    seed % n_shards == shard.  Output rows.csv is written to a shard-specific
    subdirectory (cfg.out_root / f'shard{shard}/') to avoid races."""
    orig_out = cfg.out_root
    cfg.out_root = orig_out / f"shard{shard}"
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    # Filter the seed range manually here by transforming run_da into
    # a per-seed loop.  Easiest: copy the body inline with seed filter.
    rows_path = cfg.out_root / "rows.csv"
    rows: list = []
    x_eval, y_raw, g_raw = make_eval_grid(cfg.n_eval, cfg.eval_domain)
    t0_global = time.time()

    seeds_for_shard = [s for s in range(cfg.n_seeds) if s % n_shards == shard]
    print(f"[shard {shard}/{n_shards}] seeds = {seeds_for_shard}", flush=True)

    for model_name in cfg.models:
        for target_seed in seeds_for_shard:
            t_cell = time.time()
            train = make_train_data(target_seed, cfg.n_train, cfg.train_domain)
            y_eval, g_eval = standardize_eval(y_raw, g_raw,
                                              train["y_mean"], train["y_std"])
            init_model = make_model(model_name, target_seed, cfg)
            init_state = clone_state(init_model)
            vanilla_seed = stable_seed("vanilla-train", model_name, target_seed)
            t1 = time.time()
            vanilla = train_model(model_name, init_state, train["x"], train["y"],
                                  cfg, mode="vanilla", train_seed=vanilla_seed)
            t_vanilla = time.time() - t1
            pred_v = predict_values(vanilla, x_eval, cfg.device)
            grad_v = predict_grads(vanilla, x_eval, cfg.device)
            mse_v = mse_by_region(x_eval, pred_v, y_eval)
            grad_mse_v = mse_by_region(x_eval, grad_v, g_eval)

            g_scale, eps = make_grad_noise(train["g"], target_seed)
            dml_seed_fixed = stable_seed("dml-train", model_name, target_seed)
            da = make_da_data(target_seed, n_da, train["y_std"],
                              near_extrap_max=1.5, train_domain=cfg.train_domain)
            eps_da = make_da_grad_noise(da["g"], target_seed)

            for sigma_rel in cfg.sigma_grid:
                sigma_rel = float(sigma_rel)
                t2 = time.time()
                g_noisy = apply_grad_noise(train["g"], g_scale, eps, sigma_rel)
                dml = train_model(model_name, init_state, train["x"], train["y"],
                                  cfg, g_train=g_noisy, g_scale=g_scale,
                                  mode="dml", train_seed=dml_seed_fixed)
                pred_dt = predict_values(dml, x_eval, cfg.device)
                grad_dt = predict_grads(dml, x_eval, cfg.device)
                mse_dt = mse_by_region(x_eval, pred_dt, y_eval)
                grad_mse_dt = mse_by_region(x_eval, grad_dt, g_eval)
                t_dt = time.time() - t2

                t3 = time.time()
                g_da_noisy = apply_grad_noise(da["g"], g_scale, eps_da, sigma_rel)
                dml_da = train_model_da(
                    model_name, init_state,
                    x_in=train["x"], y_in=train["y"], g_in_noisy=g_noisy,
                    x_da=da["x"], g_da_noisy=g_da_noisy,
                    g_scale=g_scale, cfg=cfg, train_seed=dml_seed_fixed,
                )
                pred_dda = predict_values(dml_da, x_eval, cfg.device)
                grad_dda = predict_grads(dml_da, x_eval, cfg.device)
                mse_dda = mse_by_region(x_eval, pred_dda, y_eval)
                grad_mse_dda = mse_by_region(x_eval, grad_dda, g_eval)
                t_dda = time.time() - t3

                for region in ("interpolation", "near_extrap", "far_extrap", "all"):
                    eps_safe = 1e-30
                    rows.append({
                        "model": model_name, "seed": target_seed,
                        "sigma_rel": sigma_rel, "region": region,
                        "n_train": cfg.n_train, "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region], "mse_dml": mse_v[region],
                        "log_ratio": 0.0, "dml_win": 0,
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_v[region],
                        "log_grad_ratio": 0.0,
                        "t_vanilla": t_vanilla, "t_dml": t_vanilla,
                        "arm": "vanilla",
                    })
                    log_ratio_dt = float(np.log((mse_dt[region] + eps_safe)
                                                / (mse_v[region] + eps_safe)))
                    log_grad_ratio_dt = float(np.log((grad_mse_dt[region] + eps_safe)
                                                     / (grad_mse_v[region] + eps_safe)))
                    rows.append({
                        "model": model_name, "seed": target_seed,
                        "sigma_rel": sigma_rel, "region": region,
                        "n_train": cfg.n_train, "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region], "mse_dml": mse_dt[region],
                        "log_ratio": log_ratio_dt,
                        "dml_win": int(mse_dt[region] < mse_v[region]),
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_dt[region],
                        "log_grad_ratio": log_grad_ratio_dt,
                        "t_vanilla": t_vanilla, "t_dml": t_dt,
                        "arm": "dml_train",
                    })
                    log_ratio_dda = float(np.log((mse_dda[region] + eps_safe)
                                                 / (mse_v[region] + eps_safe)))
                    log_grad_ratio_dda = float(np.log((grad_mse_dda[region] + eps_safe)
                                                      / (grad_mse_v[region] + eps_safe)))
                    rows.append({
                        "model": model_name, "seed": target_seed,
                        "sigma_rel": sigma_rel, "region": region,
                        "n_train": cfg.n_train, "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region], "mse_dml": mse_dda[region],
                        "log_ratio": log_ratio_dda,
                        "dml_win": int(mse_dda[region] < mse_v[region]),
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_dda[region],
                        "log_grad_ratio": log_grad_ratio_dda,
                        "t_vanilla": t_vanilla, "t_dml": t_dda,
                        "arm": "dml_da",
                    })
                near_dt = next(r for r in reversed(rows)
                               if r["region"] == "near_extrap" and r["arm"] == "dml_train")
                near_dda = next(r for r in reversed(rows)
                                if r["region"] == "near_extrap" and r["arm"] == "dml_da")
                print(f"  [shard{shard}] {model_name:8s} seed={target_seed} σ={sigma_rel:5.3f} "
                      f"lr_near[dml]={near_dt['log_ratio']:+.3f} "
                      f"lr_near[da]={near_dda['log_ratio']:+.3f} "
                      f"t_dml={t_dt:.1f}s t_da={t_dda:.1f}s", flush=True)

            print(f"[shard{shard} cell done] {model_name} seed={target_seed} "
                  f"t_cell={time.time() - t_cell:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(rows_path, index=False)

    print(f"\n[shard{shard}] Total wall time: {(time.time() - t0_global) / 60:.2f} min",
          flush=True)
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    (cfg.out_root / "config.json").write_text(json.dumps(
        {**asdict(cfg), "out_root": str(cfg.out_root),
         "sigma_grid": list(cfg.sigma_grid),
         "models": list(cfg.models),
         "n_da": n_da, "shard": shard, "n_shards": n_shards},
        indent=2, default=str))
    return rows_path


if __name__ == "__main__":
    main()
