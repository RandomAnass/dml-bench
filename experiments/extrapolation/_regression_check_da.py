"""Compare dml_train arm of DA regression test against M1 SIREN seed=0 rows."""
import numpy as np
import pandas as pd

m1 = pd.read_csv("results/extrapolation_M1/rows.csv")
da = pd.read_csv("results/extrapolation_M1_DA_regression_test/rows.csv")

m1_siren = m1[(m1["model"] == "siren") & (m1["seed"] == 0)].copy()
da_dml_train = da[(da["model"] == "siren") & (da["seed"] == 0)
                  & (da["arm"] == "dml_train")].copy()

# Drop the "arm" column from da to align schemas.
da_dml_train = da_dml_train.drop(columns=["arm"])

# Sort both by (sigma_rel, region) for row-aligned comparison.
sort_cols = ["sigma_rel", "region"]
m1_siren = m1_siren.sort_values(sort_cols).reset_index(drop=True)
da_dml_train = da_dml_train.sort_values(sort_cols).reset_index(drop=True)

print(f"M1 SIREN seed=0 rows: {len(m1_siren)}")
print(f"DA dml_train rows: {len(da_dml_train)}")

# Compare numeric columns for row-by-row absolute difference.
num_cols = ["mse_vanilla", "mse_dml", "log_ratio",
            "grad_mse_vanilla", "grad_mse_dml", "log_grad_ratio"]

print("\n--- Per-column max abs diff ---")
worst_log_ratio = 0.0
for c in num_cols:
    a = m1_siren[c].to_numpy()
    b = da_dml_train[c].to_numpy()
    diff = np.abs(a - b)
    print(f"  {c:20s} max_abs_diff={diff.max():.6e}  mean_abs_diff={diff.mean():.6e}")
    if c == "log_ratio":
        worst_log_ratio = diff.max()

# Show worst rows for log_ratio
worst_idx = np.argsort(np.abs(m1_siren["log_ratio"] - da_dml_train["log_ratio"]))[::-1][:5]
print("\n--- Top 5 log_ratio diffs ---")
for i in worst_idx:
    s = m1_siren.iloc[i]["sigma_rel"]; r = m1_siren.iloc[i]["region"]
    a = m1_siren.iloc[i]["log_ratio"]; b = da_dml_train.iloc[i]["log_ratio"]
    print(f"  σ={s:.3f} region={r:14s} M1={a:+.10f} DA={b:+.10f} diff={a-b:+.3e}")

PASS = worst_log_ratio < 1e-6
print(f"\n[REGRESSION TEST] worst log_ratio diff = {worst_log_ratio:.3e}")
print(f"[REGRESSION TEST] {'PASS' if PASS else 'FAIL'} (tolerance = 1e-6)")
