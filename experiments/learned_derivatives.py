#!/usr/bin/env python3
"""
Learned Derivatives Experiment — DML with Teacher-Network Gradients.

Tests whether DML still helps when derivative labels come from a pretrained
teacher network (autograd) rather than ground truth. This is practically
important: in many real problems, we don't have analytic derivatives, but
we can train a teacher and extract its gradients.

Protocol:
  1. Train a "teacher" vanilla NN on (x, y) only
  2. Compute teacher's gradients: ∂f_teacher/∂x via autograd
  3. Train a "student" with DML using teacher gradients as derivative labels
  4. Compare: student vs vanilla vs DML-with-true-gradients

Expected result: teacher-gradient DML should improve over vanilla if the
teacher's gradient field is close to the true gradient field, but should
underperform DML with exact gradients.

Usage:
    python experiments/learned_derivatives.py --gpu 0
    python experiments/learned_derivatives.py --quick  # Fast test
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dml_benchmark.functions import generate_data, train_test_split
from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.model import DmlFeedForward


def train_teacher(x_train, y_train, x_test, y_test, seed=42,
                  n_epochs=500, hidden_size=256, n_layers=4, lr=0.005,
                  batch_size=256):
    """Train a vanilla teacher network and return it + its gradient predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d = x_train.shape[1]
    model = DmlFeedForward(d, output_dim=1, n_layers=n_layers, hidden_size=hidden_size,
                           activation='softplus').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6)
    
    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_te = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(x_te)
            val_loss = nn.functional.mse_loss(val_pred, y_te).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 40:
                break
    
    model.load_state_dict(best_state)
    return model, best_val_loss


def extract_teacher_gradients(model, x, batch_size=512):
    """Extract gradients from teacher network via autograd."""
    device = next(model.parameters()).device
    model.eval()
    
    all_grads = []
    n = len(x)
    
    for i in range(0, n, batch_size):
        xb = torch.tensor(x[i:i+batch_size], dtype=torch.float32, 
                          device=device, requires_grad=True)
        y_pred = model(xb)
        
        # Compute gradients
        grads = torch.autograd.grad(
            y_pred.sum(), xb, create_graph=False
        )[0]
        all_grads.append(grads.detach().cpu().numpy())
    
    gradients = np.concatenate(all_grads, axis=0)
    # Reshape to match dydx format: (N, 1, d)
    return gradients.reshape(n, 1, -1)


def run_experiment(func_type, dim, n_samples, seed, hparams):
    """Run one learned-derivatives experiment."""
    # Generate data with true gradients
    data = generate_data(func_type, n_dim=dim, n_samples=n_samples, seed=seed)
    train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
    
    x_train = train_data.x
    y_train = train_data.y
    dydx_train_true = train_data.dydx
    x_test = test_data.x
    y_test = test_data.y
    dydx_test = test_data.dydx
    
    results = {}
    
    # 1. Vanilla baseline
    print(f"    Vanilla...", end=" ", flush=True)
    t0 = time.time()
    r_vanilla = train_single_experiment(
        x_train=x_train, y_train=y_train, dydx_train=dydx_train_true,
        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
        method="vanilla", seed=seed, pbar=False, **hparams
    )
    results["vanilla"] = {
        "test_value_mse": float(r_vanilla.test_value_mse),
        "test_grad_mse": float(r_vanilla.test_grad_mse),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"val={r_vanilla.test_value_mse:.4e}, {time.time()-t0:.1f}s")
    
    # 2. DML with true gradients
    print(f"    DML (true grads)...", end=" ", flush=True)
    t0 = time.time()
    r_true = train_single_experiment(
        x_train=x_train, y_train=y_train, dydx_train=dydx_train_true,
        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
        method="dml_fixed", seed=seed, pbar=False, **hparams
    )
    results["dml_true_grads"] = {
        "test_value_mse": float(r_true.test_value_mse),
        "test_grad_mse": float(r_true.test_grad_mse),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"val={r_true.test_value_mse:.4e}, {time.time()-t0:.1f}s")
    
    # 3. Train teacher
    print(f"    Teacher training...", end=" ", flush=True)
    t0 = time.time()
    teacher, teacher_val_loss = train_teacher(
        x_train, y_train, x_test, y_test, seed=seed + 999,
        n_epochs=hparams.get('n_epochs', 500),
        hidden_size=hparams.get('hidden_size', 256),
        n_layers=hparams.get('n_layers', 4),
        lr=hparams.get('lr', 0.005),
        batch_size=hparams.get('batch_size', 256),
    )
    teacher_time = time.time() - t0
    print(f"val_loss={teacher_val_loss:.4e}, {teacher_time:.1f}s")
    
    # 4. Extract teacher gradients
    dydx_train_teacher = extract_teacher_gradients(teacher, x_train)
    
    # Measure teacher gradient quality
    grad_mse_teacher = np.mean((dydx_train_teacher - dydx_train_true) ** 2)
    grad_cos_sim = np.mean([
        np.dot(dydx_train_teacher[i].flatten(), dydx_train_true[i].flatten()) /
        (np.linalg.norm(dydx_train_teacher[i]) * np.linalg.norm(dydx_train_true[i]) + 1e-10)
        for i in range(min(100, len(x_train)))
    ])
    
    results["teacher_quality"] = {
        "teacher_val_loss": float(teacher_val_loss),
        "grad_mse_vs_true": float(grad_mse_teacher),
        "grad_cosine_sim": float(grad_cos_sim),
        "teacher_time_s": round(teacher_time, 2),
    }
    
    # 5. DML with teacher gradients
    print(f"    DML (teacher grads)...", end=" ", flush=True)
    t0 = time.time()
    r_teacher = train_single_experiment(
        x_train=x_train, y_train=y_train, dydx_train=dydx_train_teacher,
        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
        method="dml_fixed", seed=seed, pbar=False, **hparams
    )
    results["dml_teacher_grads"] = {
        "test_value_mse": float(r_teacher.test_value_mse),
        "test_grad_mse": float(r_teacher.test_grad_mse),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"val={r_teacher.test_value_mse:.4e}, {time.time()-t0:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    hparams = {
        "n_epochs": 500,
        "batch_size": 256,
        "n_layers": 4,
        "hidden_size": 256,
        "activation": "softplus",
        "lr": 0.005,
        "lambda_": 1.0,
        "max_grad_norm": 1.0,
        "scheduler_patience": 20,
        "scheduler_factor": 0.5,
    }
    
    if args.quick:
        configs = [("poly_trig", 5, 1024, 42)]
        hparams["n_epochs"] = 200
    else:
        configs = [
            ("poly_trig", 2, 1024, 42),
            ("poly_trig", 5, 1024, 42),
            ("poly_trig", 10, 1024, 42),
            ("poly_trig", 50, 1024, 42),
            ("bachelier", 5, 1024, 42),
            ("bachelier", 10, 1024, 42),
        ]
    
    results_dir = Path("results/learned_derivatives")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("LEARNED DERIVATIVES EXPERIMENT")
    print("=" * 70)
    print("Protocol: Train teacher (vanilla) → extract autograd gradients →")
    print("          train student with DML using teacher gradients")
    print()
    
    all_results = {}
    
    for func, dim, ns, seed in configs:
        key = f"{func}_d{dim}_n{ns}_s{seed}"
        print(f"\n--- {key} ---")
        
        result = run_experiment(func, dim, ns, seed, hparams)
        result["config"] = {
            "func_type": func, "dim": dim, "n_samples": ns, "seed": seed,
            "hparams": hparams,
        }
        result["timestamp"] = datetime.now().isoformat()
        
        all_results[key] = result
        
        # Save individual result
        path = results_dir / f"{key}.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Print comparison
        van = result["vanilla"]["test_value_mse"]
        true_ = result["dml_true_grads"]["test_value_mse"]
        teacher_ = result["dml_teacher_grads"]["test_value_mse"]
        cos_sim = result["teacher_quality"]["grad_cosine_sim"]
        
        print(f"  Summary: vanilla={van:.4e}, dml_true={true_:.4e}, dml_teacher={teacher_:.4e}")
        print(f"  Teacher grad cosine similarity: {cos_sim:.4f}")
        print(f"  True-grad improvement: {van/true_:.2f}×")
        print(f"  Teacher-grad improvement: {van/teacher_:.2f}×")
        print(f"  Teacher captures {(van - teacher_)/(van - true_ + 1e-30)*100:.1f}% of DML benefit")
    
    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Config':<25} {'Vanilla':>12} {'DML-true':>12} {'DML-teacher':>12} {'Cos-sim':>8} {'% Captured':>12}")
    print("-" * 90)
    for key, r in all_results.items():
        van = r["vanilla"]["test_value_mse"]
        true_ = r["dml_true_grads"]["test_value_mse"]
        teacher_ = r["dml_teacher_grads"]["test_value_mse"]
        cos = r["teacher_quality"]["grad_cosine_sim"]
        pct = (van - teacher_) / (van - true_ + 1e-30) * 100
        print(f"{key:<25} {van:12.4e} {true_:12.4e} {teacher_:12.4e} {cos:8.4f} {pct:11.1f}%")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
