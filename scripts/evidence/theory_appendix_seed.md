# Theory appendix seed — sweet-spot extract from technical report

Source: `papers/neurips_DB/theory/DML_Theory (6).pdf` (60-page technical
report, "Derivative-Enhanced Nonparametric Regression: Minimax Rate and
Crossover Analysis with One Noisy Partial Derivative").

The full report has five contributions (C1–C5). For DML-Bench we extract
**only Contribution C2** — the *quantitative crossover analysis* — because
it is the result that directly explains DML-Bench's empirical $\sigma^*$
finding. The remaining contributions (rederivation with explicit constants,
Gaussian-noise lower-bound sharpening, multi-channel extension, true-risk
separation) belong in the standalone technical report and not in the
benchmark paper.

## What C2 says (verbatim from §1.3, condensed)

> **Quantitative crossover analysis** (Theorem 8.1, Theorem 8.3, Corollary
> 8.4). Writing $R_n^{(0)}$ for the value-only minimax risk, we show that
> $R_n^{(1)}/R_n^{(0)} \to 0$ with rate $n^{-\delta}$ where
> $\delta = 2k / ((2k+d-1)(2k+d))$, and that there is an explicit
> sample-size threshold $N_0$ above which the derivative-enhanced upper
> bound beats the value-only upper bound. In the high-derivative-noise
> regime $\sigma_1 \to \infty$ with $\sigma_0, M$ fixed, $N_0$ scales as
> $\sigma_0^2 \cdot (\sigma_1/\sigma_0)^{2(2k+d)}$, making precise how
> large a sample is needed before the derivative channel pays off.

## Why this matters for DML-Bench

DML-Bench measures a related but operational quantity: the *noise level*
$\sigma^*$ above which adding a derivative loss term hurts a fixed-$n$
training run. The two quantities are linked by

$$
N_0\bigl(\sigma_1\bigr) \;=\; n \quad\Longleftrightarrow\quad \sigma_1 \;=\; \sigma^*(n,\sigma_0,k,d)
$$

i.e., $\sigma^*$ is the inverse function of $N_0$ at the chosen training
size. The theory's $N_0 \propto (\sigma_1/\sigma_0)^{2(2k+d)}$ scaling
predicts:

1. **$\sigma^*$ shrinks with $d$.** Higher dimension shifts the
   crossover earlier (smaller $\sigma^*$). DML-Bench's tier-3 sweep
   should observe $\sigma^*$ decreasing in $d$ on the smooth families.
2. **$\sigma^*$ grows with $\sigma_0$.** Noisier *value* labels
   raise the threshold at which derivatives stop being useful.
3. **$\sigma^*$ depends on $k$ (smoothness).** Smoother targets allow
   higher $\sigma^*$ because the value-only baseline converges faster
   so the marginal gain from a noisy derivative shrinks.

## Suggested 1-2 page appendix subsection (drop-in to LaTeX)

```latex
\section{Theoretical Crossover Threshold (Sketch)}
\label{app:theory-crossover}

The empirical noise crossover $\sigma^*$ in \Cref{sec:noise-crossover}
has a corresponding theoretical quantity in classical nonparametric
regression. Consider observing, at each design point $X_i \in [0,1]^d$,
both a noisy value $Y_i = f^*(X_i) + \varepsilon_i$ and a noisy first
partial derivative $Z_i = \partial_1 f^*(X_i) + \eta_i$, with $\varepsilon_i$
and $\eta_i$ centered sub-Gaussian with parameters $\sigma_0$ and
$\sigma_1$ respectively, and $f^* \in \mathcal{F}_k(M)$ the class of
$C^k$ functions with $\|D^\alpha f^*\|_\infty \le M$ for $|\alpha| \le k$.

\paragraph{Minimax rate.} The minimax $L^2$ risk over $\mathcal{F}_k(M)$
satisfies
\begin{equation}
    R_n^{(1)} \;\asymp\; n^{-2k/(2k+d-1)},
\end{equation}
compared to the value-only rate $R_n^{(0)} \asymp n^{-2k/(2k+d)}$
\citep{stone1982optimalrate, hallyatchew2001derivativeobservation}. One
noisy first-order partial derivative thus reduces the effective
dimension from $d$ to $d-1$ — the classical Hall--Yatchew dimension
reduction principle.

\paragraph{Quantitative crossover.} Beyond the rate exponent, the
companion technical report establishes an explicit \emph{sample-size
threshold} $N_0$ above which derivative-enhanced supervision strictly
beats value-only. Writing $\tau^2 = \sigma_0^2 + \sigma_1^2 + M^2$ for
the composite fluctuation parameter, in the high-derivative-noise regime
$\sigma_1 \to \infty$ with $\sigma_0$ and $M$ fixed,
\begin{equation}
    N_0 \;\propto\; \sigma_0^2 \cdot \left(\frac{\sigma_1}{\sigma_0}\right)^{2(2k+d)}.
    \label{eq:N0-scaling}
\end{equation}

\paragraph{Connection to the empirical $\sigma^*$.} For a fixed training
size $n$, the operational threshold $\sigma^*$ satisfies $N_0(\sigma^*) = n$.
Solving \eqref{eq:N0-scaling} gives
\begin{equation}
    \sigma^*(n; \sigma_0, k, d) \;\propto\; \sigma_0 \cdot (n/\sigma_0^2)^{1/(2(2k+d))},
\end{equation}
predicting that $\sigma^*$ (i) shrinks with dimension $d$, (ii) grows
with the value-noise scale $\sigma_0$, and (iii) grows with smoothness
order $k$. The empirical $\sigma^*$ trends across the smooth function
families in \Cref{sec:noise-crossover} (poly\_trig, trig, bachelier;
$d \in \{2,5,10,20,50,100\}$) are qualitatively consistent with these
predictions; quantitative calibration of the proportionality constant
in \eqref{eq:N0-scaling} requires the sub-Gaussian variance floor
$\underline{\sigma}_0^2$ which is dataset-dependent. A full derivation,
matching upper and lower bounds, and a worked example at $k=d=2$ appear
in the technical report companion to this paper.
```

## What to cite

- Stone 1982 — Annals of Statistics (Stone's minimax rate).
- Hall & Yatchew 2007 — JRSSB (one-noisy-derivative dimension reduction).
- Czarnecki et al. 2017 — Sobolev training (general-ML reference).
- Huge & Savine 2020 — Differential ML (finance reference).

## What NOT to include

- The pseudo-observation construction (technical machinery, irrelevant to
  the empirical link).
- The Hilbert-space Rao-Blackwell reduction (proof technique, appendix-only).
- The multi-channel ($p \geq 2$) lower bound (open problem; report's §9).
- Anything from Appendices A–J of the technical report.

The 1-2 page appendix above contains exactly what readers of DML-Bench need
to motivate the empirical $\sigma^*$ finding theoretically.
