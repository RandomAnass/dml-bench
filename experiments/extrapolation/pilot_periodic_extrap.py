"""
Periodic extrapolation pilot — sanity check for the SIREN σ* finding.

Question: in the colleague's pilot, SIREN had near-extrap σ* ≈ 0.99 while softplus
and Snake were left-censored. Is that finding stable, or a 4-seed artifact?

This file is a self-contained re-run from scratch — does NOT import the colleague's
code or any external pilot data. Embeds the agreed corrections from
papers/tmp_sonnet/extrapolation_plan_revised.md:
  - Fourier-linear has NO intercept (10 features, not 11).
  - σ* via LOWESS on per-pair log_ratio, not linear-interp on win-rate.
  - Four separate seeds: target / init / noise / dml-train.
  - dml-train seed FIXED across σ within a (model, target_seed) cell.

Run:
  python experiments/extrapolation/pilot_periodic_extrap.py --smoke   # 1 seed × 3 σ
  python experiments/extrapolation/pilot_periodic_extrap.py           # 4 seeds × 15 σ
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    out_root: Path = Path("results/extrapolation_pilot")
    models: tuple = ("softplus", "fourier_linear", "siren", "snake")
    n_train: int = 24
    n_eval: int = 2400
    train_domain: tuple = (-1.0, 1.0)
    eval_domain: tuple = (-3.0, 3.0)
    width: int = 64
    depth: int = 3
    fourier_k: int = 5
    dml_lambda: float = 1.0
    weight_decay: float = 1e-7
    grad_clip: float = 10.0
    lowess_frac: float = 0.20
    base_lr: dict = field(default_factory=lambda: {
        "softplus": 1e-3,
        "fourier_linear": 5e-3,
        "siren": 5e-4,
        "snake": 1e-3,
    })
    sigma_grid: tuple = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075,
                         0.10, 0.15, 0.20, 0.30, 0.40, 0.60,
                         0.80, 1.20, 1.60)
    n_seeds: int = 4
    n_steps: int = 1600
    smoke: bool = False
    wide_sigma: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def apply_smoke(self):
        if self.smoke:
            self.n_seeds = 1
            self.sigma_grid = (0.0, 0.10, 1.20)
            self.n_steps = 200
            self.out_root = self.out_root.parent / "extrapolation_pilot_smoke"

    def apply_wide_sigma(self):
        if self.wide_sigma:
            # Pilot grid + tail extension to σ=3.0 (matches A1 grid).
            # Resolves SIREN's marginal +0.06 log-ratio at σ=1.6.
            self.sigma_grid = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075,
                               0.10, 0.15, 0.20, 0.30, 0.40, 0.60,
                               0.80, 1.20, 1.60, 2.00, 2.50, 3.00)


# -----------------------------------------------------------------------------
# Seed helpers
# -----------------------------------------------------------------------------
def stable_seed(*items, modulo=2**31 - 1) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    digest = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


# -----------------------------------------------------------------------------
# Target — fixed K=5, period-1 (matches the camera-ready theory appendix example)
# -----------------------------------------------------------------------------
def target_value(x: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * x)
        + 0.50 * np.sin(6 * np.pi * x + 0.4)
        + 0.25 * np.cos(10 * np.pi * x - 0.2)
    )


def target_grad(x: np.ndarray) -> np.ndarray:
    return (
        2 * np.pi * np.cos(2 * np.pi * x)
        + 0.50 * 6 * np.pi * np.cos(6 * np.pi * x + 0.4)
        - 0.25 * 10 * np.pi * np.sin(10 * np.pi * x - 0.2)
    )


def make_train_data(target_seed: int, n_train: int, train_domain) -> dict:
    rng = np.random.default_rng(stable_seed("train", target_seed))
    inner = rng.uniform(train_domain[0], train_domain[1], size=n_train - 2)
    x = np.sort(np.concatenate([[train_domain[0]], inner, [train_domain[1]]]))
    y = target_value(x)
    g = target_grad(x)
    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-12)
    return {
        "x": x[:, None].astype(np.float32),
        "y_raw": y.astype(np.float32),
        "g_raw": g.astype(np.float32),
        "y": ((y - y_mean) / y_std).astype(np.float32),
        "g": (g / y_std)[:, None].astype(np.float32),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def make_eval_grid(n: int, eval_domain) -> tuple:
    x = np.linspace(eval_domain[0], eval_domain[1], n)
    return (x[:, None].astype(np.float32),
            target_value(x).astype(np.float32),
            target_grad(x).astype(np.float32))


def standardize_eval(y_raw, g_raw, y_mean, y_std):
    return (((y_raw - y_mean) / y_std).astype(np.float32),
            (g_raw / y_std).astype(np.float32))


def region_masks(x_flat: np.ndarray) -> dict:
    a = np.abs(x_flat)
    return {
        "interpolation": a <= 1.0,
        "near_extrap": (a > 1.0) & (a <= 1.5),
        "far_extrap": a > 1.5,
        "all": np.ones_like(a, dtype=bool),
    }


def make_grad_noise(g_clean: np.ndarray, target_seed: int) -> tuple:
    rng = np.random.default_rng(stable_seed("grad-noise", target_seed))
    g_scale = (np.std(g_clean, axis=0, keepdims=True) + 1e-12).astype(np.float32)
    eps = rng.normal(size=g_clean.shape).astype(np.float32)
    return g_scale, eps


def apply_grad_noise(g_clean, g_scale, eps, sigma_rel) -> np.ndarray:
    return (g_clean + sigma_rel * g_scale * eps).astype(np.float32)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SoftplusMLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.Softplus(beta=2.0)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FourierFeatures(nn.Module):
    """Sin/cos features for k=1..K. NO INTERCEPT — matches theory appendix
    φ_K(x) ∈ R^{2K}; the 11.83 σ* prediction is computed for p=2K."""

    def __init__(self, k_max: int):
        super().__init__()
        freqs = torch.arange(1, k_max + 1, dtype=torch.float32).view(1, -1)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        z = 2 * math.pi * x @ self.freqs
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class FourierLinear(nn.Module):
    """Linear regression on Fourier features. No intercept, no bias."""

    def __init__(self, k_max=5):
        super().__init__()
        self.feat = FourierFeatures(k_max)
        self.linear = nn.Linear(2 * k_max, 1, bias=False)

    def forward(self, x):
        return self.linear(self.feat(x)).squeeze(-1)


class SineLayer(nn.Module):
    """SIREN sine layer (Sitzmann 2020 §3.2)."""

    def __init__(self, in_features, out_features, is_first=False, omega_0=20.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / omega_0)
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, width=64, depth=3, omega_0=20.0):
        super().__init__()
        layers = [SineLayer(1, width, is_first=True, omega_0=omega_0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, is_first=False, omega_0=omega_0))
        final = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / width) / omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SnakeActivation(nn.Module):
    """Snake (Ziyin 2020): x + sin²(αx)/α with learnable per-neuron α.
    log_alpha=0 → α=1 at init."""

    def __init__(self, features):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        alpha = torch.exp(self.log_alpha).view(1, -1) + 1e-6
        return x + torch.sin(alpha * x) ** 2 / alpha


class SnakeMLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), SnakeActivation(width)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_model(model_name: str, target_seed: int, cfg: Config) -> nn.Module:
    set_torch_seed(stable_seed("init", model_name, target_seed))
    if model_name == "softplus":
        m = SoftplusMLP(cfg.width, cfg.depth)
    elif model_name == "fourier_linear":
        m = FourierLinear(cfg.fourier_k)
    elif model_name == "siren":
        m = SIREN(cfg.width, cfg.depth)
    elif model_name == "snake":
        m = SnakeMLP(cfg.width, cfg.depth)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m.to(cfg.device)


def clone_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: dict, device: str) -> nn.Module:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return model


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_model(model_name, init_state, x_train, y_train, cfg: Config,
                g_train=None, g_scale=None, mode="vanilla", train_seed=0):
    assert mode in ("vanilla", "dml")
    model = make_model(model_name, target_seed=train_seed, cfg=cfg)
    load_state(model, init_state, cfg.device)
    model.train()

    x_t = to_tensor(x_train, cfg.device)
    y_t = to_tensor(y_train, cfg.device)
    g_t = to_tensor(g_train, cfg.device) if g_train is not None else None
    g_scale_t = to_tensor(g_scale, cfg.device) if g_scale is not None else None

    lr = cfg.base_lr[model_name]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    set_torch_seed(train_seed)
    for _ in range(cfg.n_steps):
        if mode == "vanilla":
            pred = model(x_t)
            loss = F.mse_loss(pred, y_t)
        else:
            xb = x_t.detach().clone().requires_grad_(True)
            pred = model(xb)
            grad_pred = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]
            value_loss = F.mse_loss(pred, y_t)
            grad_loss = torch.mean(((grad_pred - g_t) / g_scale_t) ** 2)
            loss = value_loss + cfg.dml_lambda * grad_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    return model


def predict_values(model, x, device, batch=4096):
    model.eval()
    x_t = to_tensor(x, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch):
            out.append(model(x_t[i:i + batch]).detach().cpu())
    return torch.cat(out).numpy()


def predict_grads(model, x, device, batch=2048):
    model.eval()
    x_t = to_tensor(x, device)
    out = []
    for i in range(0, len(x_t), batch):
        xb = x_t[i:i + batch].detach().clone().requires_grad_(True)
        pred = model(xb)
        g = torch.autograd.grad(pred.sum(), xb, create_graph=False)[0]
        out.append(g.detach().cpu())
    return torch.cat(out).numpy().squeeze(-1)


def mse_by_region(x_grid, pred, true) -> dict:
    masks = region_masks(x_grid.squeeze())
    return {r: float(np.mean((pred[m] - true[m]) ** 2)) for r, m in masks.items()}


# -----------------------------------------------------------------------------
# Run pipeline
# -----------------------------------------------------------------------------
def run_pilot(cfg: Config) -> Path:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    rows_path = cfg.out_root / "rows.csv"
    rows: list = []
    x_eval, y_raw, g_raw = make_eval_grid(cfg.n_eval, cfg.eval_domain)

    t0_global = time.time()

    for model_name in cfg.models:
        for target_seed in range(cfg.n_seeds):
            t_cell = time.time()
            train = make_train_data(target_seed, cfg.n_train, cfg.train_domain)
            y_eval, g_eval = standardize_eval(y_raw, g_raw, train["y_mean"], train["y_std"])

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

            for sigma_rel in cfg.sigma_grid:
                t2 = time.time()
                g_noisy = apply_grad_noise(train["g"], g_scale, eps, float(sigma_rel))
                dml = train_model(model_name, init_state, train["x"], train["y"], cfg,
                                  g_train=g_noisy, g_scale=g_scale,
                                  mode="dml", train_seed=dml_seed_fixed)
                pred_d = predict_values(dml, x_eval, cfg.device)
                grad_d = predict_grads(dml, x_eval, cfg.device)
                mse_d = mse_by_region(x_eval, pred_d, y_eval)
                grad_mse_d = mse_by_region(x_eval, grad_d, g_eval)
                t_dml = time.time() - t2

                for region in ("interpolation", "near_extrap", "far_extrap", "all"):
                    eps_safe = 1e-30
                    log_ratio = float(np.log((mse_d[region] + eps_safe) / (mse_v[region] + eps_safe)))
                    log_grad_ratio = float(np.log((grad_mse_d[region] + eps_safe) / (grad_mse_v[region] + eps_safe)))
                    rows.append({
                        "model": model_name,
                        "seed": target_seed,
                        "sigma_rel": float(sigma_rel),
                        "region": region,
                        "n_train": cfg.n_train,
                        "n_steps": cfg.n_steps,
                        "mse_vanilla": mse_v[region],
                        "mse_dml": mse_d[region],
                        "log_ratio": log_ratio,
                        "dml_win": int(mse_d[region] < mse_v[region]),
                        "grad_mse_vanilla": grad_mse_v[region],
                        "grad_mse_dml": grad_mse_d[region],
                        "log_grad_ratio": log_grad_ratio,
                        "t_vanilla": t_vanilla,
                        "t_dml": t_dml,
                    })

                # one print per σ
                row_all = rows[-1]
                row_near = rows[-2]
                row_far = rows[-3]
                row_int = rows[-4]
                print(f"  {model_name:14s} seed={target_seed} σ={sigma_rel:5.3f} "
                      f"lr_int={row_int['log_ratio']:+.3f} "
                      f"lr_near={row_near['log_ratio']:+.3f} "
                      f"lr_far={row_far['log_ratio']:+.3f} "
                      f"t_dml={t_dml:.1f}s",
                      flush=True)

            print(f"[cell done] {model_name} seed={target_seed} "
                  f"t_cell={time.time() - t_cell:.1f}s",
                  flush=True)

            # incremental save
            pd.DataFrame(rows).to_csv(rows_path, index=False)

    print(f"\nTotal wall time: {(time.time() - t0_global) / 60:.2f} min", flush=True)
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    (cfg.out_root / "config.json").write_text(json.dumps(
        {**asdict(cfg), "out_root": str(cfg.out_root),
         "sigma_grid": list(cfg.sigma_grid),
         "models": list(cfg.models)}, indent=2, default=str))
    return rows_path


# -----------------------------------------------------------------------------
# Analysis — LOWESS σ* on log_ratio
# -----------------------------------------------------------------------------
def estimate_sigma_star(sub: pd.DataFrame, frac: float) -> tuple:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    g = (sub.groupby("sigma_rel")["log_ratio"].mean().reset_index()
         .sort_values("sigma_rel"))
    x = g["sigma_rel"].to_numpy()
    y = g["log_ratio"].to_numpy()
    if len(x) == 0:
        return float("nan"), "unknown"
    if len(x) >= 8:
        sm = lowess(y, x, frac=frac, return_sorted=True)
        xs, ys = sm[:, 0], sm[:, 1]
    else:
        xs, ys = x, y
    if ys[0] >= 0:
        return float(xs[0]), "left_censored"
    if np.all(ys < 0):
        return float(xs[-1]), "right_censored"
    for i in range(1, len(xs)):
        if ys[i - 1] < 0 <= ys[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)), "observed"
    return float("nan"), "unknown"


def analyze(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.out_root / "rows.csv")
    df = df.drop_duplicates(["model", "seed", "sigma_rel", "region"], keep="last")
    summary = []
    for (model, region), sub in df.groupby(["model", "region"]):
        sigma_star, status = estimate_sigma_star(sub, cfg.lowess_frac)
        sub0 = sub[sub["sigma_rel"] == 0.0]
        sub_max = sub[sub["sigma_rel"] == sub["sigma_rel"].max()]
        summary.append({
            "model": model,
            "region": region,
            "sigma_star": sigma_star,
            "status": status,
            "mean_log_ratio_sigma0": float(sub0["log_ratio"].mean()),
            "std_log_ratio_sigma0": float(sub0["log_ratio"].std(ddof=1)) if len(sub0) > 1 else float("nan"),
            "mean_log_ratio_max_sigma": float(sub_max["log_ratio"].mean()),
            "n_seeds": int(sub0["seed"].nunique()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cfg.out_root / "sigma_star_summary.csv", index=False)
    print("\n===== σ* SUMMARY =====")
    pd.set_option("display.max_rows", 100)
    print(summary_df.sort_values(["model", "region"]).to_string(index=False))
    return summary_df


def plot_raw_curves(cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(cfg.out_root / "rows.csv")
    region_colors = {"interpolation": "#2176ae", "near_extrap": "#f18f01", "far_extrap": "#c73e1d"}
    region_order = ["interpolation", "near_extrap", "far_extrap"]
    models = list(cfg.models)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        for region in region_order:
            sub = df[(df["model"] == model) & (df["region"] == region)]
            # per-seed thin lines
            for seed in sorted(sub["seed"].unique()):
                ss = sub[sub["seed"] == seed].sort_values("sigma_rel")
                ax.plot(ss["sigma_rel"], ss["log_ratio"],
                        color=region_colors[region], alpha=0.25, linewidth=0.8)
            # mean thick line
            mean = (sub.groupby("sigma_rel")["log_ratio"]
                    .mean().reset_index().sort_values("sigma_rel"))
            ax.plot(mean["sigma_rel"], mean["log_ratio"],
                    color=region_colors[region], linewidth=2.2, label=region)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(model)
        ax.set_xlabel("σ_rel (gradient noise)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("log(MSE_DML / MSE_vanilla)")
    axes[-1].legend(loc="best", frameon=False)
    fig.suptitle("Raw mean log-ratio per region (thin lines = per-seed)")
    fig.tight_layout()
    fig.savefig(cfg.out_root / "raw_log_ratio_per_region.pdf")
    fig.savefig(cfg.out_root / "raw_log_ratio_per_region.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {cfg.out_root / 'raw_log_ratio_per_region.pdf'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 seed, 3 σ values, 200 steps — sanity test")
    p.add_argument("--seeds", type=int, default=None,
                   help="override number of seeds (default 4)")
    p.add_argument("--steps", type=int, default=None,
                   help="override training steps (default 1600)")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated subset, e.g. softplus,siren")
    p.add_argument("--out-dir", type=str, default=None,
                   help="override output directory")
    p.add_argument("--wide-sigma", action="store_true",
                   help="extend σ grid to 3.0 (18 points)")
    args = p.parse_args()

    cfg = Config()
    cfg.smoke = args.smoke
    cfg.apply_smoke()
    cfg.wide_sigma = args.wide_sigma
    cfg.apply_wide_sigma()
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.models is not None:
        cfg.models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if args.out_dir is not None:
        cfg.out_root = Path(args.out_dir)

    print(f"[config] device={cfg.device} models={cfg.models} "
          f"n_seeds={cfg.n_seeds} n_steps={cfg.n_steps} "
          f"|sigma_grid|={len(cfg.sigma_grid)} smoke={cfg.smoke}")
    print(f"[config] out={cfg.out_root}")

    rows_path = run_pilot(cfg)
    print(f"\n[saved] {rows_path}")

    summary = analyze(cfg)
    plot_raw_curves(cfg)

    print("\n===== σ*=0.99 SANITY CHECK =====")
    siren_near = summary[(summary["model"] == "siren")
                        & (summary["region"] == "near_extrap")]
    if len(siren_near) > 0:
        print(siren_near.to_string(index=False))
    softplus_near = summary[(summary["model"] == "softplus")
                            & (summary["region"] == "near_extrap")]
    snake_near = summary[(summary["model"] == "snake")
                          & (summary["region"] == "near_extrap")]
    print("\nFor reference (other archs near-extrap):")
    print(pd.concat([softplus_near, snake_near]).to_string(index=False))


if __name__ == "__main__":
    main()
