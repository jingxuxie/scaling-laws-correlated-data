# Submission checklist

## Automated gates

- [ ] All unit tests pass.
- [ ] Real sequential experiment reruns from the recorded public source.
- [ ] Combined paper figures regenerate from retained CSV/JSON outputs.
- [ ] Main paper compiles with the unmodified AAAI-27 style and bibliography.
- [ ] Technical supplement compiles without undefined citations or references.
- [ ] Separate official reproducibility checklist compiles.
- [ ] Main PDF is US Letter, at most 9 pages, and references begin no later than
      page 8.
- [ ] No Type 3 fonts, undefined references/citations, or overfull boxes appear.
- [ ] `results/final_audit/build_report.json` reports `status: pass`.

## Scientific audit

- [ ] Independently verify the reciprocal-binomial lemma.
- [ ] Independently verify both branches of the weighted missing-mass bound.
- [ ] Check the raw-horizon annulus lower bound for all model sizes.
- [ ] Check the stationary residual-life upper and lower bounds.
- [ ] Check the fractional-moment completion estimator and minimax lower bound.
- [ ] Check the scaling-function remainder and Gamma constant.
- [ ] Check representation-invariance and dense-Gaussian rank propositions.
- [ ] Confirm all numerical values directly from retained outputs.
- [ ] Reassess whether the real-data diagnostic is stated with appropriately
      limited scope.

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
