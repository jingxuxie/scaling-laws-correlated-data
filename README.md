# Spectral Scaling Laws with Correlated Data

This repository develops an exactly solvable theory of scaling laws for linear
regression on persistent data streams. The main point is that correlation is
not fully summarized by one effective sample size when different spectral
modes refresh at different rates.

## Main result

Let the marginal covariance mass, dwell time, and target energy satisfy

\[
\lambda_j\asymp j^{-a},\qquad
\tau_j\asymp j^r,\qquad
s_j=\lambda_j\theta_j^2\asymp j^{-b}.
\]

Keeping the same one-time covariance while mode `j` persists for `tau_j`
forces statistically new blocks to arrive according to

\[
q_j\propto\lambda_j/\tau_j\asymp j^{-(a+r)}.
\]

For the block-averaging estimator trained on `B` innovations,

\[
\mathbb E\mathcal R
=\Theta\!\left(
M^{-(b-1)}+B^{-(b-1)/(a+r)}
+\sigma^2\frac{\min\{M,B^{1/(a+r)}\}}{B}
\right).
\]

The noiseless expression is the exact minimax risk over a signed source class.
For a stream observed for exactly `N` raw steps:

- boundary initialization has rate
  `M^{-(b-1)} + N^{-(b-1)/(a+r)}` for every `r >= 0` in the stated source range;
- stationary initialization adds the inspection-paradox phase
  `N^{-(a-1)/r}`;
- a fractional-moment argument transfers the noisy hard-target rate beyond the
  finite-variance dwell regime.

The result is invariant under known well-conditioned dense changes of
representation. A dense Gaussian AR control establishes an equally important
boundary: correlation alone does not universally imply the renewal exponent.

## Empirical package

The repository contains:

- exact finite-sample summation and phase-collapse experiments;
- block and fixed-raw-horizon Monte Carlo;
- infinite-variance dwell and stationary inspection-paradox tests;
- exact noisy model-selection experiments;
- dense-dictionary validation and a dense Gaussian AR negative control;
- a chronological appliance-energy diagnostic comparing the marginal spectrum
  with the persistence-adjusted spectrum;
- deterministic tests, retained CSV/JSON outputs, and CI workflows.

The appliance experiment fits its proxy only on sample sizes 128--1024 and
extrapolates to 2048--8192. In the retained run, mode-wise persistence lowers
contiguous-window extrapolation log-RMSE from 1.099 to 0.951, while it hurts
random-subset prediction, the intended falsification control.

## Repository layout

- `paper/main.tex`: AAAI-27 main-paper source.
- `paper/supplement.tex`: complete technical supplement.
- `paper/appendix/`: proofs and experimental details.
- `experiments/`: exact calculations, simulations, data experiment, and tests.
- `results/`: retained machine-readable outputs.
- `tools/build_checklist.py`: completes the unmodified official AAAI checklist.
- `tools/audit_submission.py`: checks page count, references, fonts, style use,
  and LaTeX diagnostics.
- `SUBMISSION_CHECKLIST.md`: final author-side checks.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make experiments
make real-data
make paper
make checklist
make audit
```

The synthetic suite is laptop-scale and requires no GPU. The real-data command
downloads the public appliance-energy CSV recorded by the experiment script.

## Submission artifacts

The `Submission build` GitHub Action performs the full test suite, reruns the
real-data experiment, regenerates figures, compiles with the unmodified
AAAI-27 author kit, builds the separate reproducibility checklist, audits the
PDFs, and commits the resulting artifacts and JSON audit report to the working
branch.

The repository is a submission-ready research draft, not a substitute for
human accountability. Before submission, the authors must independently audit
every proof and citation, verify the venue-specific AI-use declaration, add
final authorship metadata outside anonymous review mode, and inspect the final
PDF and supplementary archive.
