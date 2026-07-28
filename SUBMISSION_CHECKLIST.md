# Submission checklist

## Automated gates completed locally

- [x] All 29 unit tests pass.
- [x] Real sequential experiment reruns from the recorded public source.
- [x] Combined paper figures regenerate from retained CSV/JSON outputs.
- [x] Main paper compiles with the unmodified AAAI-27 style and bibliography.
- [x] Technical supplement compiles without undefined citations or references.
- [x] Separate official reproducibility checklist compiles.
- [x] Main PDF is US Letter, 9 pages total, and references begin on page 8.
- [x] No Type 3 fonts, undefined references/citations, or overfull boxes appear.
- [x] `results/final_audit/build_report.json` reports `status: pass`.
- [x] Main paper contains 7 technical pages and 2 reference pages.
- [x] All 31 BibTeX entries are cited in the manuscript or supplement.

These gates must also pass on the final GitHub Actions run after every source
change.

## Scientific audit

- [ ] Independently verify the reciprocal-binomial lemma.
- [ ] Independently verify both branches of the weighted missing-mass bound.
- [ ] Check the raw-horizon annulus lower bound for all model sizes.
- [ ] Check the stationary residual-life upper and lower bounds.
- [ ] Check the fractional-moment completion estimator and minimax lower bound.
- [ ] Check the scaling-function remainder and Gamma constant.
- [ ] Check representation-invariance and dense-Gaussian rank propositions.
- [ ] Check the matched-trace-IAT calibration and both innovation laws.
- [ ] Check the compute-optimal lower bound for all feasible `M B <= C`.
- [ ] Confirm all numerical values directly from retained outputs.
- [ ] Reassess whether the real-data diagnostic is stated with appropriately
      limited scope.

## Final manuscript review

- [ ] Read the seven technical pages continuously for clarity and notation.
- [ ] Confirm every theorem assumption is repeated where it is used.
- [ ] Confirm figures remain legible at 100% zoom in the submitted PDF.
- [ ] Confirm all references against publisher or author primary sources.
- [ ] Verify that the abstract and introduction do not overclaim universality.
- [ ] Verify that the negative control and randomized-subset control are
      described as falsification tests, not supporting benchmarks.

## Venue and release

- [ ] Confirm that an AAAI abstract was registered for this paper.
- [ ] Upload the main PDF, checklist, supplement, and anonymized code/data bundle
      to the correct submission fields.
- [ ] Remove repository usernames and commit metadata from the anonymous code
      archive.
- [ ] Do not link the public GitHub repository from the anonymous manuscript.
- [ ] Complete AAAI AI-use and author-accountability disclosures.
- [ ] Confirm every author is registered and satisfies reciprocal-review rules.
- [ ] Add final author names and affiliations only when venue instructions allow.
