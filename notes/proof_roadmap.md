# Proof roadmap: spectral scaling laws under persistent sampling

## 1. Core statistical experiment

Let \((e_j)_{j\ge 1}\) be an orthonormal basis. The test distribution draws
\(J\sim\lambda\), a Rademacher sign \(S\), and uses \(X=Se_J\). Hence

\[
\mathcal R(\widehat\theta,\theta)
=\mathbb E_X\langle X,\widehat\theta-\theta\rangle^2
=\sum_{j\ge1}\lambda_j(\widehat\theta_j-\theta_j)^2.
\]

A training stream is organized into innovation blocks. Mode \(J_k\) is drawn
independently from

\[
q_j=\frac{\lambda_j/\tau_j}{Z},
\qquad
Z=\sum_{\ell\ge1}\frac{\lambda_\ell}{\tau_\ell}.
\]

Block \(k\), conditional on \(J_k=j\), persists for \(\tau_j\) raw time units.
It supplies a signed block average

\[
\bar Y_k=S_k\theta_j+\bar\varepsilon_k,
\qquad
\mathbb E[\bar\varepsilon_k\mid J_k=j]=0,
\qquad
\operatorname{Var}(\bar\varepsilon_k\mid J_k=j)=\sigma^2/\tau_j.
\]

For integer \(\tau_j\), this is exactly the average of \(\tau_j\) repeated
labels with independent noise. Rounding a regularly varying \(\tau_j\) does
not alter any exponent.

The expected block duration is

\[
\mu=\sum_jq_j\tau_j=1/Z,
\]

and the fraction of raw time occupied by mode \(j\) is

\[
\frac{q_j\tau_j}{\mu}=\lambda_j.
\]

Thus every persistence profile has the same one-time covariance spectrum. The
quantity that changes is the innovation spectrum \(q_j\), not the marginal
spectrum \(\lambda_j\).

## 2. Estimator and exact finite-sample identity

Let

\[
K_j=\sum_{k=1}^B\mathbf 1\{J_k=j\}\sim\operatorname{Binomial}(B,q_j).
\]

For a model with coordinates \(1,\ldots,M\), define

\[
\widehat\theta_j=
\begin{cases}
K_j^{-1}\sum_{k:J_k=j}S_k\bar Y_k,&K_j>0,\\
0,&K_j=0,
\end{cases}
\qquad j\le M,
\]

and \(\widehat\theta_j=0\) for \(j>M\). Write
\(s_j=\lambda_j\theta_j^2\) and

\[
h_B(q)=\mathbb E\left[\frac{\mathbf 1\{K>0\}}{K}\right],
\qquad K\sim\operatorname{Binomial}(B,q).
\]

Conditioning on \(K_j\) gives the exact decomposition

\[
\boxed{
\mathbb E\mathcal R(\widehat\theta,\theta)
=\sum_{j>M}s_j
+\sum_{j\le M}s_j(1-q_j)^B
+\sigma^2\sum_{j\le M}\frac{\lambda_j}{\tau_j}h_B(q_j).
}
\]

The three terms are model approximation, unseen-mode bias, and label-noise
variance. Since \(\lambda_j/\tau_j=Zq_j\), the variance is controlled by the
same innovation spectrum as the coverage bias.

**Status:** proved by one-line conditional expectation; included as Theorem 1.

## 3. Binomial reciprocal lemma

For all \(B\ge1\) and \(q\in(0,1]\),

\[
q h_B(q)\asymp \min\{Bq^2,B^{-1}\}.
\]

### Upper bound

- Since \(h_B(q)\le \Pr(K>0)\le Bq\), we have
  \(q h_B(q)\le Bq^2\).
- For \(k\ge1\), \(1/k\le2/(k+1)\). The exact identity

  \[
  \mathbb E\frac1{K+1}
  =\frac{1-(1-q)^{B+1}}{(B+1)q}
  \]

  yields \(q h_B(q)\le2/(B+1)\).

### Lower bound

- If \(Bq\le1\),

  \[
  h_B(q)\ge\Pr(K=1)=Bq(1-q)^{B-1}\ge cBq.
  \]

- If \(Bq\ge1\), Markov's inequality and
  \(\Pr(K=0)\le e^{-Bq}\) imply

  \[
  \Pr(1\le K\le2Bq)\ge 1/2-e^{-1}>0.
  \]

  On this event, \(1/K\ge1/(2Bq)\), so \(h_B(q)\ge c/(Bq)\).

**Status:** proof complete; included as Lemma 1.

## 4. Power-law theorem

Assume constants bounded away from zero and infinity such that

\[
\lambda_j\asymp j^{-a},\qquad
\tau_j\asymp j^r,\qquad
s_j\asymp j^{-b},
\]

with \(a>1\), \(r\ge0\), and \(b>1\). Put \(p=a+r\). Then

\[
q_j\asymp j^{-p}.
\]

The target theorem is

\[
\boxed{
\mathbb E\mathcal R(\widehat\theta,\theta)
\asymp
M^{-(b-1)}
+B^{-(b-1)/p}
+\sigma^2\frac{\min\{M,B^{1/p}\}}{B}.
}
\]

### Approximation term

The integral test gives

\[
\sum_{j>M}s_j\asymp M^{1-b}.
\]

### Coverage term

Use \((1-q_j)^B\le e^{-Bq_j}\), and split at
\(J_B=B^{1/p}\). The contribution from \(j\gtrsim J_B\) is
\(\Theta(J_B^{1-b})\), while the lower modes are exponentially suppressed.
Uniformly in \(M\), approximation plus coverage is

\[
\Theta\bigl(M^{1-b}+B^{-(b-1)/p}\bigr).
\]

For the lower bound, if \(M\lesssim J_B\), the approximation tail already
controls the sum. If \(M\gtrsim J_B\), sum over a constant-width annulus
\(j\in[cJ_B,CJ_B]\), where \(Bq_j=\Theta(1)\).

### Variance term

The reciprocal lemma reduces the sum to

\[
\sum_{j\le M}\min\{Bq_j^2,B^{-1}\}.
\]

Splitting at \(J_B\) gives

\[
\Theta\left(\frac{\min\{M,J_B\}}{B}\right).
\]

For \(M>J_B\), the tail also has order
\(B\sum_{j>J_B}j^{-2p}=\Theta(J_B/B)\).

**Status:** proof complete at the rate level; detailed constants can be added if
needed.

## 5. Exact noiseless minimax theorem

Let

\[
\Theta_s=\left\{\theta^\omega:
\theta_j^\omega=\omega_j\sqrt{s_j/\lambda_j},\;
\omega_j\in\{-1,+1\}\right\}.
\]

In the noiseless case, the coordinate memorization estimator attains

\[
\sum_{j>M}s_j+\sum_{j\le M}s_j(1-q_j)^B.
\]

For the lower bound, place the product Rademacher prior on \(\omega\). If mode
\(j\) is unseen, the posterior for \(\omega_j\) remains symmetric and every
estimator incurs Bayes risk at least \(s_j\). If \(j>M\), the model restriction
incurs \(s_j\) regardless of the data. Summing proves exact Bayes risk, hence a
minimax lower bound. The upper bound is independent of \(\omega\), so equality
holds.

**Status:** proof complete; included as Theorem 3.

## 6. Scaling function and leading constant

For exact powers

\[
q_j=c_qj^{-p},\qquad s_j=c_sj^{-b},
\]

and \(M/B^{1/p}\to m\in(0,\infty]\), the noiseless risk satisfies

\[
B^{(b-1)/p}\mathcal R_{B,M}
\to c_s\left[
\int_0^m x^{-b}e^{-c_qx^{-p}}\,dx
+\frac{m^{1-b}}{b-1}
\right].
\]

At \(m=\infty\),

\[
\int_0^\infty x^{-b}e^{-c_qx^{-p}}\,dx
=\frac1p c_q^{(1-b)/p}
\Gamma\left(\frac{b-1}{p}\right).
\]

The proof is a Riemann-sum argument after scaling \(j=B^{1/p}x\), with a
small-\(x\) exponential envelope and an integrable \(x^{-b}\) envelope at
infinity.

**Status:** derivation complete; formal domination details are in the paper
appendix.

## 7. Raw-time and stationary-process interpretation

For deterministic integer durations \(\ell_j\asymp j^r\), concatenate the
blocks and initialize the first block from the equilibrium size-biased law.
The resulting semi-Markov stream is stationary and has one-time occupancy
\(\lambda_j\). At renewal epochs,

\[
N_B=\sum_{k=1}^B\ell_{J_k},
\qquad
\mathbb E N_B=\mu B,
\qquad
N_B/B\to\mu\quad\text{a.s.}
\]

Thus replacing \(B\) by expected raw observations changes constants, not
exponents. If \(r<a-1\), the block duration has finite variance, which gives
standard \(\sqrt B\)-scale concentration of \(N_B\).

A reversible Markov realization uses geometric dwell times. On states
\((j,s)\),

\[
P((j,s),(k,t))
=\left(1-\tau_j^{-1}\right)\mathbf 1\{(j,s)=(k,t)\}
+\tau_j^{-1}q_k/2.
\]

Its stationary law is \(\pi_{j,s}=\lambda_j/2\), detailed balance follows from
\(\lambda_j/\tau_j=Zq_j\), and

\[
\mathbb E[X_tX_{t+\ell}^{\top}]
=\sum_j\lambda_j(1-\tau_j^{-1})^\ell e_je_j^\top.
\]

This covariance is nonseparable whenever \(\tau_j\) varies with \(j\).

**Status:** stationarity, reversibility, and covariance proofs complete. The
exact noisy theorem is stated for deterministic/block-averaged durations;
noisy geometric dwell times may introduce finite-run corrections and are not
claimed in the current draft.

## 8. Fixed raw-horizon theorem

For deterministic integer durations, let

\[
T_N=\min\left\{m:\sum_{k=1}^m\tau_{J_k}\ge N\right\}
\]

be the number of blocks touched by exactly \(N\) raw observations. In the
noiseless setting the pathwise risk is the weighted missing mass after
\(T_N\) innovations.

If \(r<a-1\), then

\[
\mathbb E_q[\tau_J^2]
=Z^{-1}\sum_j\lambda_j\tau_j
\asymp\sum_j j^{-(a-r)}<\infty.
\]

Take \(B_-\asymp N/(2\mu)\) and \(B_+\asymp 2N/\mu\). Monotonicity of the
missing mass gives

\[
\mathbb E R_{B_+}-S_0\Pr(T_N>B_+)
\le \mathbb E R_{T_N}
\le \mathbb E R_{B_-}+S_0\Pr(T_N<B_-).
\]

Chebyshev gives \(\Pr(T_N<B_-)=O(N^{-1})\), while a negative-mgf Chernoff
bound gives \(\Pr(T_N>B_+)\le e^{-cN}\). Therefore, whenever
\((b-1)/(a+r)<1\), equivalently \(b<a+r+1\),

\[
\mathbb E R_{T_N,M}
\asymp
M^{-(b-1)}+N^{-(b-1)/(a+r)}.
\]

For stationary equilibrium initialization, the forward recurrence time has
finite mean under the same second-moment condition. The initial size-biased
mode reduces the missing mass by only
\(O(N^{-(a+b-1)/(a+r)})\), which is lower order.

**Status:** proof complete and included as a main theorem; a dedicated
fixed-horizon Monte Carlo experiment recovers the predicted slopes.

## 9. Consequences

1. **Uniform persistence:** \(r=0\) preserves the i.i.d. data exponent
   \((b-1)/a\); persistence changes only a horizontal constant.
2. **Aligned persistence:** \(r>0\) changes the exponent to
   \((b-1)/(a+r)\).
3. **Model-data frontier:** the learned resolution is
   \(J_B\asymp B^{1/(a+r)}\).
4. **Hard-target regime:** if \(1<b<a+r\), coverage bias decays more slowly
   than variance, so the clean two-term scaling law is visible.
5. **Smooth-target regime:** if \(b>a+r\) and \(\sigma>0\), optimizing the
   noisy bound gives
   \(M_\star\asymp(B/\sigma^2)^{1/b}\) and
   risk \(\asymp(\sigma^2/B)^{(b-1)/b}\).

## 10. Remaining work before submission

- Extend the fixed-raw-horizon result to noisy labels and heavy-tailed dwell
  times with \(r\ge a-1\).
- Add a dense-coordinate or random-sketch stress test to determine how broadly
  the mechanism survives beyond one-hot spectral sampling.
- Add noisy simulations across the \(b<a+r\) and \(b>a+r\) regimes.
- Add at least one real sequential regression dataset, estimating
  mode-wise persistence and testing out-of-range learning-curve prediction.
- Check all constants and edge cases independently, especially the
  block-average noise model and the \(b=a+r\) boundary.
- Compress the main text to the AAAI technical-content limit after the result
  set is finalized.
