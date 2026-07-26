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

## Proof-audit corrections

- Lower-bound annuli now split correctly according to whether model
  approximation already controls the data term.
- The raw-horizon lower bound uses the same uniform model-size split.
- The scaling-function proof includes an explicit uniform binomial-to-Poisson
  remainder bound before passing to the Riemann integral.
- No finite-variance assumption remains in the noiseless boundary-start
  theorem.

## Experimental package completed

- Exact slopes and phase collapse.
- Independent innovation-block Monte Carlo.
- Fixed raw-horizon Monte Carlo.
- Infinite-variance dwell and stationary inspection-paradox simulation.
- Noisy hard/smooth target model-selection regimes.
- Dense representation validation.
- Dense Gaussian AR negative control.
- Sequential appliance-energy proxy with randomized-subset falsification.
- Automated tests, official-format PDF audit, and completed AAAI
  reproducibility checklist workflow.

## Scope discipline

The theorem is exact for persistent spectral renewal sampling. It does not
assert the same exponent for arbitrary dense dependent covariates. The dense AR
counterexample makes this boundary explicit. The real-data result is an
illustrative diagnostic, not evidence of universality.

## Remaining human-only submission tasks

1. Audit every proof line independently.
2. Check every bibliographic record against its primary source.
3. Review the official PDF visually after the automated audit.
4. Confirm anonymous code and data packaging rules.
5. Complete the venue-required AI-use disclosure and author accountability
   checks.
6. Replace anonymous metadata only after acceptance or as required by the
   submission system.
