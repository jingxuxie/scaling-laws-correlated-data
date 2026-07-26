# Spectral Scaling Laws with Correlated Data

This repository develops an exactly solvable theory of scaling laws for
linear regression on persistent data streams.

## Core claim

Let the time-marginal covariance spectrum, mode persistence, and target energy
satisfy

\[
\lambda_j\asymp j^{-a},\qquad
\tau_j\asymp j^r,\qquad
s_j=\lambda_j\theta_j^2\asymp j^{-b}.
\]

To keep the same marginal covariance while mode `j` persists for `tau_j`, new
blocks arrive with probability

\[
q_j\propto \lambda_j/\tau_j\asymp j^{-(a+r)}.
\]

For the coordinate-wise block-averaging estimator, the proof draft establishes

\[
\mathbb E\mathcal R
=\Theta\!\left(
M^{-(b-1)}+B^{-(b-1)/(a+r)}
+\sigma^2\frac{\min\{M,B^{1/(a+r)}\}}{B}
\right).
\]

In the noiseless case, the first two terms are also the exact minimax risk over
a signed source class. Uniform persistence (`r=0`) changes constants only;
mode-dependent persistence changes the data exponent. When deterministic dwell
times have finite variance (`r < a - 1`), the same exponent holds for a
trajectory containing exactly `N` raw observations, not only for a fixed number
of renewal blocks.

## Repository layout

- `paper/main.tex`: conference-length main-paper source.
- `paper/supplement.tex`: separately compiled technical appendix.
- `paper/appendix_content.tex`: full proofs and additional experiments.
- `paper/references.bib`: verified primary references.
- `notes/proof_roadmap.md`: theorem stack, proof details, and open issues.
- `experiments/exact_risk.py`: exact finite-sample curves and phase collapse.
- `experiments/monte_carlo.py`: independent block-sampling validation.
- `experiments/raw_horizon.py`: fixed-raw-trajectory validation.
- `experiments/test_*.py`: deterministic unit tests.
- `results/`: generated CSV, JSON, and figure outputs.

## Reproduce the pilot

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make experiments
make test
make paper  # builds main.pdf and supplement.pdf
```

The experiment suite uses `O(M)` memory and runs on a laptop. No GPU is
required.

## AAAI format

The repository does not redistribute conference style files. Download the
official AAAI-27 author kit and place `aaai2027.sty` and `aaai2027.bst` in
`paper/`. The source automatically uses the official submission style when
those files are present and otherwise compiles in a readable one-column draft
format.

## Current scope

The completed proof is for persistent coordinate or spectral sampling. It
does not yet establish the same exponent for arbitrary dense Gaussian time
series. The paper states this limitation explicitly; dense random-feature and
real sequential-data validations are the next milestones.
