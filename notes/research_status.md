# Research status

## Theorem stack completed

- Exact finite-block bias--variance identity.
- Uniform model--innovation--noise scaling law.
- Exact noiseless minimax equality over a signed source class.
- Full model--data scaling function and Gamma-function asymptotic constant.
- Fixed raw-horizon theorem for every dwell exponent `r >= 0`, without a
  finite-variance requirement.
- Stationary-equilibrium inspection-paradox phase
  `N^{-(a-1)/r}` with matching upper and lower bounds.
- Noisy hard-target fixed-horizon theorem using fractional dwell moments and
  the von Bahr--Esseen inequality.
- Exact invariance under known invertible dense representations, with spectral
  preservation under bounded condition number.
- Reversible nonseparable Markov realization.
- Finite-dimensional dense Gaussian AR exact-recovery proposition, which
  rules out the overbroad claim that autocorrelation alone forces the renewal
  exponent.
- Matched-global-autocorrelation counterexample: equal marginal covariance and
  equal trace integrated autocorrelation, but different innovation spectra and
  different learning exponents.
- Compute-optimal allocation law under `C = M B`:
  `M* ~ C^{1/(a+r+1)}`, `B* ~ C^{(a+r)/(a+r+1)}`, and
  `R* ~ C^{-(b-1)/(a+r+1)}`.

## Proof-audit corrections

- Lower-bound annuli now split correctly according to whether model
  approximation already controls the data term.
- The raw-horizon lower bound uses the same uniform model-size split.
- The scaling-function proof includes an explicit uniform binomial-to-Poisson
  remainder bound before passing to the Riemann integral.
- No finite-variance assumption remains in the noiseless boundary-start
  theorem.
- The matched-IAT proposition is derived directly from the complete covariance
  function and calibrated so both processes have exactly the same scalar IAT.
- The compute-optimal lower bound covers every feasible allocation, rather than
  only balancing the two upper-bound terms.

## Experimental package completed

- Exact slopes and phase collapse.
- Independent innovation-block Monte Carlo.
- Fixed raw-horizon Monte Carlo.
- Infinite-variance dwell and stationary inspection-paradox simulation.
- Noisy hard/smooth target model-selection regimes.
- Matched-global-IAT experiment with distinct predicted slopes.
- Compute-optimal risk and resource-allocation sweep.
- Dense representation validation.
- Dense Gaussian AR negative control.
- Sequential appliance-energy proxy with randomized-subset falsification.
- Automated tests, official-format PDF audit, and completed AAAI
  reproducibility-checklist workflow.

## Retained quantitative checks

- Matched trace IAT: both streams equal 15 exactly.
- Uniform-persistence slope: fitted `0.4007`, predicted `0.4000`.
- Aligned-persistence slope: fitted `0.2840`, predicted `0.2857`.
- Compute-optimal risk exponents for `r = 0, 0.4, 0.8`: fitted
  `0.2666, 0.2352, 0.2104`, versus predictions
  `0.2667, 0.2353, 0.2105`.
- Real contiguous-window proxy log-RMSE: `1.099` spatial-only versus `0.951`
  persistence-adjusted; the ordering reverses for randomized subsets.
- Full deterministic suite: 29 tests passed.

## Manuscript status

- Main paper: 9 pages total, consisting of 7 technical pages and 2 reference
  pages.
- Bibliography: 31 cited references, with no uncited BibTeX entries.
- Technical supplement: 9 pages.
- Reproducibility checklist: 2 pages.
- Latest local audit: pass; US Letter, official AAAI style, references on page
  8, no Type 3 fonts, no undefined citations/references, and no overfull boxes.

## Scope discipline

The theorem is exact for persistent spectral renewal sampling. It does not
assert the same exponent for arbitrary dense dependent covariates. The dense AR
counterexample makes this boundary explicit. The real-data result is an
illustrative diagnostic, not evidence of universality. The matched-IAT theorem
shows exactly why a scalar correlation summary is insufficient within the
renewal mechanism; it does not claim that every estimator or every dependent
process is governed by `lambda_j / tau_j`.

## Remaining human-only submission tasks

1. Audit every proof line independently.
2. Check every bibliographic record against its primary source.
3. Review the official PDFs visually after the automated GitHub build.
4. Confirm anonymous code and data packaging rules.
5. Complete the venue-required AI-use disclosure and author-accountability
   checks.
6. Replace anonymous metadata only after acceptance or as required by the
   submission system.
