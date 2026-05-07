# Unified Discontinuous-Payoff Comparison — Multi-Seed Results

**Generated:** 2026-04-16 00:03 UTC
**Mode:** multi_seed
**Total experiments:** 550
**Methods:** 11 (5 pathwise + 3 LRM + 3 fuzzy)
**Datasets:** 5
**Seeds:** 10 ([42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999])
**Epochs:** 500, **Architecture:** 4×256 softplus, **Optimizer:** Adam (lr=0.005)

## Digital BS (1D, analytical)

| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |
|---|---|---|---|---|
| Vanilla (no deriv.) | 1.0039e-02 ± 5.7298e-04 | 1.6101e-06 ± 7.0051e-07 | — | baseline |
| DML fixed λ (PW) | 3.5435e-02 ± 1.9308e-03 | 3.1345e-05 ± 3.0436e-06 | +253.0% | 0.1× |
| GradNorm (PW) | 6.6444e-02 ± 2.6302e-02 | 6.3031e-05 ± 2.7392e-05 | +561.8% | 0.0× |
| ReLoBRaLo (PW) | 3.7124e-02 ± 5.4003e-03 | 3.2477e-05 ± 6.0306e-06 | +269.8% | 0.0× |
| Warmup (PW) | 1.0085e-02 ± 5.7562e-04 | 1.3531e-06 ± 1.0358e-06 | +0.5% | 1.2× |
| DML fixed λ (LRM) | 1.0314e-02 ± 5.9237e-04 | 5.8864e-07 ± 2.9043e-07 | +2.7% | 2.7× |
| GradNorm (LRM) | 1.0484e-02 ± 7.9676e-04 | 1.3268e-06 ± 8.7223e-07 | +4.4% | 1.2× |
| Warmup (LRM) | 9.9990e-03 ± 5.2562e-04 | 3.4502e-07 ± 1.5638e-07 | -0.4% | 4.7× |
| DML fixed λ (Fuzzy) | 1.0158e-02 ± 5.1630e-04 | 5.4550e-07 ± 3.7040e-07 | +1.2% | 3.0× |
| GradNorm (Fuzzy) | 1.0210e-02 ± 4.9270e-04 | 6.0637e-07 ± 2.8004e-07 | +1.7% | 2.7× |
| Warmup (Fuzzy) | 9.9776e-03 ± 5.3728e-04 | 3.9656e-07 ± 1.6187e-07 | -0.6% | 4.1× |

**Best (≤10% val penalty):** Warmup (LRM) — val penalty -0.4%, gradient improvement 4.7×

## Barrier BS (1D, analytical)

| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |
|---|---|---|---|---|
| Vanilla (no deriv.) | 3.0251e+01 ± 3.0231e+00 | 9.4920e-03 ± 2.6119e-03 | — | baseline |
| DML fixed λ (PW) | 1.0775e+02 ± 7.6765e+00 | 1.9600e-01 ± 1.3614e-02 | +256.2% | 0.0× |
| GradNorm (PW) | 1.7322e+02 ± 5.9782e+01 | 2.6804e-01 ± 7.7758e-02 | +472.6% | 0.0× |
| ReLoBRaLo (PW) | 1.1531e+02 ± 1.1982e+01 | 1.9988e-01 ± 2.3642e-02 | +281.2% | 0.0× |
| Warmup (PW) | 3.1038e+01 ± 3.3440e+00 | 2.3551e-02 ± 8.3458e-03 | +2.6% | 0.4× |
| DML fixed λ (LRM) | 7.5679e+01 ± 3.5376e+01 | 2.3007e-01 ± 2.6271e-01 | +150.2% | 0.0× |
| GradNorm (LRM) | 9.3066e+01 ± 3.7768e+01 | 1.6877e-01 ± 1.9927e-01 | +207.6% | 0.1× |
| Warmup (LRM) | 3.0929e+01 ± 2.1752e+00 | 7.4577e-02 ± 1.0074e-01 | +2.2% | 0.1× |
| DML fixed λ (Fuzzy) | 3.0633e+01 ± 3.0655e+00 | 1.1251e-02 ± 2.3726e-03 | +1.3% | 0.8× |
| GradNorm (Fuzzy) | 3.0640e+01 ± 2.7571e+00 | 1.1433e-02 ± 1.9101e-03 | +1.3% | 0.8× |
| Warmup (Fuzzy) | 3.0535e+01 ± 2.8923e+00 | 9.0869e-03 ± 1.5644e-03 | +0.9% | 1.0× |

**Best (≤10% val penalty):** Warmup (Fuzzy) — val penalty +0.9%, gradient improvement 1.0×

## Heston Digital (1D, COS method)

| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |
|---|---|---|---|---|
| Vanilla (no deriv.) | 1.4398e-02 ± 1.7804e-03 | 6.3806e-06 ± 2.1983e-06 | — | baseline |
| DML fixed λ (PW) | 3.1309e-02 ± 3.9687e-03 | 6.5262e-05 ± 9.2291e-06 | +117.4% | 0.1× |
| GradNorm (PW) | 4.8940e-02 ± 1.5076e-02 | 1.1851e-04 ± 4.9258e-05 | +239.9% | 0.1× |
| ReLoBRaLo (PW) | 3.1923e-02 ± 4.5657e-03 | 6.5181e-05 ± 1.0272e-05 | +121.7% | 0.1× |
| Warmup (PW) | 1.4240e-02 ± 1.8348e-03 | 3.5096e-06 ± 1.3963e-06 | -1.1% | 1.8× |
| DML fixed λ (LRM) | 1.8089e-02 ± 5.7388e-03 | 2.5111e-05 ± 2.0768e-05 | +25.6% | 0.3× |
| GradNorm (LRM) | 1.8778e-02 ± 4.2054e-03 | 2.7857e-05 ± 1.6163e-05 | +30.4% | 0.2× |
| Warmup (LRM) | 1.4305e-02 ± 1.7886e-03 | 1.5609e-05 ± 1.0963e-05 | -0.6% | 0.4× |
| DML fixed λ (Fuzzy) | 1.4382e-02 ± 2.0316e-03 | 1.4456e-06 ± 5.9837e-07 | -0.1% | 4.4× |
| GradNorm (Fuzzy) | 1.4467e-02 ± 2.0559e-03 | 1.6364e-06 ± 9.1587e-07 | +0.5% | 3.9× |
| Warmup (Fuzzy) | 1.4291e-02 ± 1.9006e-03 | 6.7498e-07 ± 3.6629e-07 | -0.7% | 9.5× |

**Best (≤10% val penalty):** Warmup (Fuzzy) — val penalty -0.7%, gradient improvement 9.5×

## Basket Digital d=1 (analytical)

| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |
|---|---|---|---|---|
| Vanilla (no deriv.) | 2.2254e-02 ± 2.2477e-03 | 5.8656e-06 ± 2.6636e-06 | — | baseline |
| DML fixed λ (PW) | 3.0729e-02 ± 1.9987e-03 | 9.1918e-05 ± 1.8157e-05 | +38.1% | 0.1× |
| GradNorm (PW) | 3.7134e-02 ± 5.1776e-03 | 1.6003e-04 ± 6.1337e-05 | +66.9% | 0.0× |
| ReLoBRaLo (PW) | 3.0444e-02 ± 2.8360e-03 | 8.8223e-05 ± 2.3734e-05 | +36.8% | 0.1× |
| Warmup (PW) | 2.8131e-02 ± 5.8702e-03 | 6.8883e-05 ± 5.3419e-05 | +26.4% | 0.1× |
| DML fixed λ (LRM) | 2.2085e-02 ± 2.1299e-03 | 1.1423e-06 ± 9.2002e-07 | -0.8% | 5.1× |
| GradNorm (LRM) | 2.2049e-02 ± 2.1373e-03 | 1.5366e-06 ± 9.7961e-07 | -0.9% | 3.8× |
| Warmup (LRM) | 2.1923e-02 ± 2.1254e-03 | 9.9450e-07 ± 6.9837e-07 | -1.5% | 5.9× |
| DML fixed λ (Fuzzy) | 2.2115e-02 ± 2.2435e-03 | 1.5924e-06 ± 7.8529e-07 | -0.6% | 3.7× |
| GradNorm (Fuzzy) | 2.2065e-02 ± 2.2250e-03 | 1.3071e-06 ± 3.2716e-07 | -0.8% | 4.5× |
| Warmup (Fuzzy) | 2.1981e-02 ± 2.2099e-03 | 1.2489e-06 ± 5.6090e-07 | -1.2% | 4.7× |

**Best (≤10% val penalty):** Warmup (LRM) — val penalty -1.5%, gradient improvement 5.9×

## Basket Digital d=7 (high-k MC 100K)

| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |
|---|---|---|---|---|
| Vanilla (no deriv.) | 2.4387e-02 ± 1.7305e-03 | 1.3302e-06 ± 5.7617e-07 | — | baseline |
| DML fixed λ (PW) | 2.6693e-02 ± 1.5927e-03 | 3.7449e-06 ± 3.1270e-07 | +9.5% | 0.4× |
| GradNorm (PW) | 2.8069e-02 ± 1.1194e-03 | 1.0226e-05 ± 2.6808e-06 | +15.1% | 0.1× |
| ReLoBRaLo (PW) | 2.4735e-02 ± 1.8875e-03 | 7.9278e-07 ± 2.0663e-07 | +1.4% | 1.7× |
| Warmup (PW) | 2.6675e-02 ± 1.8762e-03 | 1.3019e-05 ± 5.9126e-06 | +9.4% | 0.1× |
| DML fixed λ (LRM) | 2.4171e-02 ± 1.5927e-03 | 3.7366e-07 ± 8.5668e-08 | -0.9% | 3.6× |
| GradNorm (LRM) | 2.4344e-02 ± 1.7578e-03 | 1.2379e-06 ± 6.4344e-07 | -0.2% | 1.1× |
| Warmup (LRM) | 2.4299e-02 ± 1.6802e-03 | 9.4763e-07 ± 6.0295e-07 | -0.4% | 1.4× |
| DML fixed λ (Fuzzy) | 2.3987e-02 ± 1.7176e-03 | 2.1479e-07 ± 5.7259e-08 | -1.6% | 6.2× |
| GradNorm (Fuzzy) | 2.4435e-02 ± 1.7566e-03 | 6.7447e-06 ± 3.3475e-06 | +0.2% | 0.2× |
| Warmup (Fuzzy) | 2.4198e-02 ± 1.8472e-03 | 2.5131e-07 ± 1.5734e-07 | -0.8% | 5.3× |

**Best (≤10% val penalty):** DML fixed λ (Fuzzy) — val penalty -1.6%, gradient improvement 6.2×

## Cross-Dataset Method Ranking (by Gradient MSE)

| Rank | Method | Mean Rank |
|---|---|---|
| 1 | Warmup (Fuzzy) | 1.8 |
| 2 | DML fixed λ (Fuzzy) | 3.0 |
| 3 | Warmup (LRM) | 3.8 |
| 4 | GradNorm (Fuzzy) | 5.0 |
| 5 | DML fixed λ (LRM) | 5.2 |
| 6 | Vanilla (no deriv.) | 5.8 |
| 7 | GradNorm (LRM) | 6.4 |
| 8 | Warmup (PW) | 7.0 |
| 9 | ReLoBRaLo (PW) | 8.2 |
| 10 | DML fixed λ (PW) | 9.0 |
| 11 | GradNorm (PW) | 10.8 |
