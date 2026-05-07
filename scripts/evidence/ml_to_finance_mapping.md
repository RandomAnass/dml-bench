# ML ↔ Finance terminology mapping (for §3.2 of the paper)

NeurIPS reviewers without finance background should not have to decode
pathwise / LRM / fuzzy. This table is a 5-line section that maps each
DML-Bench label paradigm to its general-ML name with the canonical
reference.

| DML-Bench identifier | Finance / SciML term | General-ML name | Canonical reference |
|---|---|---|---|
| `pathwise` | Pathwise differential of a Monte-Carlo estimator | Automatic differentiation through the target (the default in *Sobolev training* and PINNs) | Czarnecki et al. 2017; Huge & Savine 2020 |
| `lrm` | Likelihood Ratio Method (LRM); score-function estimator | REINFORCE / score-function gradient estimator | Glasserman & Karmarkar 2025; Williams 1992 |
| `fuzzy` | Fuzzy call-spread smoothing | Symmetric finite differences with kernel smoothing | Savine 2018 |

## Suggested LaTeX prose for §3.2 (drop-in)

```latex
\subsection{Label-paradigm taxonomy and ML correspondence}
\label{sec:label-paradigms}

We benchmark three derivative-label paradigms. Each has a domain-specific
name in the option-pricing literature and a generic ML name; the two
communities have arrived at the same constructions independently. The
correspondence is summarised in \Cref{tab:label-paradigm-mapping}.

\begin{itemize}
    \item \textbf{Pathwise.} Automatic differentiation through the target
    function. In option pricing this is Adjoint Algorithmic Differentiation
    \citep{huge2020differential}; in ML it is the standard derivative
    label used in Sobolev training \citep{czarnecki2017sobolev} and in
    physics-informed neural networks.

    \item \textbf{Likelihood Ratio Method (LRM).} A score-function
    estimator: $\nabla \mathbb{E}[f(X;\theta)] = \mathbb{E}[f(X;\theta)
    \cdot \nabla \log p(X;\theta)]$. In RL this is REINFORCE
    \citep{williams1992reinforce}; in finance it is the LRM Greek
    \citep{glasserman2025lrm}.

    \item \textbf{Fuzzy.} A symmetric-finite-difference call-spread with
    kernel smoothing of bandwidth $\varepsilon$:
    $\widehat{f}'_\varepsilon(x) = \varepsilon^{-1}\,\mathbf{1}[|x{-}K|<\varepsilon/2]$.
    The construction was popularised by \citet{savine2018fuzzy}; mapped
    to general ML it is just symmetric finite differences with a hat
    kernel.
\end{itemize}
```

## How this maps to the §3.2 challenge grid (cross-ref)

Pathwise is *designed for* smooth integrands and *fails on*
discontinuities (Dirac at the kink). LRM is *unbiased on any payoff*
but *high-variance*. Fuzzy *trades small bias* (the smoothing
bandwidth) *for finite, well-behaved labels at discontinuities*.
Empirically (§4.2 of the paper) only fuzzy survives the digital
indicator test, and §4.3's challenge×method grid is the visible form
of this analysis.
