# Research status

## Completed in the initial proof pass

- Exact regenerative/block model with the same time-marginal covariance for
  all persistence exponents.
- Exact finite-sample bias--variance decomposition.
- Matching power-law upper and lower rates.
- Exact noiseless minimax characterization over a signed source class.
- Exact asymptotic scaling function and leading Gamma-function constant.
- Stationary semi-Markov interpretation and reversible Markov realization.
- Fixed-raw-horizon theorem under finite-variance dwell times, including
  equilibrium initialization.
- Exact numerical learning curves, phase-collapse plots, block Monte Carlo,
  fixed-horizon Monte Carlo, and automated tests.

## Main scientific claim currently supported

A common persistence time changes constants, but mode-dependent persistence
aligned with a power-law feature spectrum changes the data exponent from

\[
\frac{b-1}{a}
\quad\text{to}\quad
\frac{b-1}{a+r}.
\]

The mechanism is a change from the marginal spectrum \(\lambda_j\) to the
innovation spectrum \(q_j\propto\lambda_j/\tau_j\).

## Scope discipline

The current proof is for persistent coordinate/spectral sampling. It does not
yet prove that arbitrary dense Gaussian AR covariates follow the same law.
That distinction is deliberate and should remain explicit in the abstract,
introduction, theorem statements, and limitations.


## Remaining reviewer-facing risks

- The exact statistical model is sparse spectral sampling; universality for
  dense Gaussian or random-feature designs is not yet proved.
- The fixed-raw-horizon theorem is currently noiseless and assumes
  \(r<a-1\); the heavy-dwell and noisy-stopping regimes may have additional
  phases.
- A real sequential dataset and an independently audited proof pass are still
  needed before describing the manuscript as submission ready.
