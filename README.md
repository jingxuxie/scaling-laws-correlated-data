# Spectral Scaling Laws with Correlated Data

This repository develops an exactly solvable theory of scaling laws for linear
regression on persistent data streams. Its central message is that temporal
dependence is not fully summarized by one effective sample size when different
spectral modes refresh at different rates.

## Main result

Let the marginal covariance mass, mode persistence, and predictive target
energy satisfy

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

The noiseless expression is also the exact minimax risk over a signed source
class. For a stream observed for exactly `N` raw steps:

- boundary initialization has rate
  `M^{-(b-1)} + N^{-(b-1)/(a+r)}` for every `r >= 0` in the stated source range;
- stationary initialization adds the inspection-paradox phase
  `N^{-(a-1)/r}`;
- a fractional-moment argument transfers the noisy hard-target rate beyond the
  finite-variance dwell regime.

The theorem is invariant under known well-conditioned dense changes of
representation. A dense Gaussian AR control establishes an equally important
boundary: autocorrelation alone does not universally imply the renewal
exponent.

## Why one effective sample size can fail

The expanded paper gives a direct matched-autocorrelation counterexample.
For the stationary reversible construction, define the trace integrated
autocorrelation time

\[
\tau_{\mathrm{tr}}
=1+2\sum_{\ell\ge1}
\frac{\operatorname{tr} C(\ell)}{\operatorname{tr} C(0)}
=2\sum_j\lambda_j\tau_j-1.
\]

A uniform persistence profile and a spectrally aligned profile can have the
same marginal covariance and exactly the same `tau_tr`, while their innovation
spectra decay as `j^{-a}` and `j^{-(a+r)}`. Their learning exponents are
therefore different:

\[
\frac{b-1}{a}
\qquad\text{versus}\qquad
\frac{b-1}{a+r}.
\]

For a compute budget `C = M B`, the corresponding noiseless optimum is

\[
M_\star\asymp C^{1/(a+r+1)},\qquad
B_\star\asymp C^{(a+r)/(a+r+1)},\qquad
\mathcal R_\star\asymp C^{-(b-1)/(a+r+1)}.
\]

Thus persistence changes not only the data slope but also the compute-optimal
allocation between model resolution and independent innovations.

## Empirical package

The repository contains:

- exact finite-sample summation and model--data phase-collapse experiments;
- block and fixed-raw-horizon Monte Carlo;
- infinite-variance dwell and stationary inspection-paradox tests;
- exact noisy model-selection experiments;
- a matched-global-IAT simulation and a compute-optimal allocation sweep;
- dense-dictionary validation and a dense Gaussian AR negative control;
- a chronological appliance-energy diagnostic comparing the marginal spectrum
  with the persistence-adjusted spectrum;
- deterministic tests, retained CSV/JSON outputs, and CI workflows.

For `(a,b,r)=(2,1.8,0.8)`, the matched-IAT experiment assigns both streams
trace IAT 15. The fitted raw-horizon exponents are 0.4007 for uniform
persistence and 0.2840 for aligned persistence, versus predictions 0.4000 and
0.2857. The compute sweep recovers the predicted risk and allocation exponents
for `r` in `{0, 0.4, 0.8}`.

The appliance experiment fits its proxy only on sample sizes 128--1024 and
extrapolates to 2048--8192. In the retained run, mode-wise persistence lowers
contiguous-window extrapolation log-RMSE from 1.099 to 0.951, while it hurts
random-subset prediction, the intended falsification control.

## Submission package

The official AAAI-format main paper currently has:

- **7 technical-content pages**;
- **2 reference pages**;
- **31 cited references**;
- a separate **9-page technical supplement**;
- a separate **2-page reproducibility checklist**.

The latest local preflight reports US Letter size, no Type 3 fonts, no undefined
references or citations, no overfull boxes, and references beginning on page 8.
The complete deterministic test suite contains 29 passing tests.

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

For the complete release gate, run:

```bash
make submission
```

The synthetic suite is laptop-scale and requires no GPU. The real-data command
downloads the public appliance-energy CSV recorded by the experiment script.

## Submission artifacts and accountability

The `Submission build` GitHub Action runs the full tests and experiments,
regenerates all figures, compiles with the unmodified AAAI-27 author kit, builds
the separate reproducibility checklist, audits the PDFs, and uploads a
submission bundle.

The repository is a submission-ready research draft, not a substitute for
human accountability. Before submission, the authors must independently audit
every proof and citation, verify the venue-specific AI-use declaration,
anonymize the code archive, and inspect the final rendered PDFs.
