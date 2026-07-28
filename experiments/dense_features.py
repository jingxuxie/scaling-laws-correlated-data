#!/usr/bin/env python3
"""Dense-representation validation for spectral renewal regression.

The latent renewal theorem is invariant under any known invertible linear
representation.  This script checks that identity numerically and then performs
a deliberately harder stress test: minimum-norm interpolation in a coherent,
well-conditioned dense dictionary, without explicitly decoding the latent
coordinates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

# Embed TrueType fonts in PDFs so AAAI preflight does not report Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
from scipy.special import zeta


@dataclass(frozen=True)
class DenseSummary:
    r: float
    predicted_exponent: float
    dictionary_fitted_exponent: float
    orthogonal_max_relative_error: float
    decoded_max_relative_error: float
    dictionary_max_ratio_to_coverage: float
    dictionary_condition_number: float
    dictionary_max_row_coherence: float
    mixing_strength: float
    model_size: int
    trials: int
    fit_start_block: int
    fit_end_block: int


def spectral_sequences(
    a: float, b: float, r: float, model_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if a <= 1 or b <= 1 or r < 0 or model_size < 2:
        raise ValueError("invalid spectral parameters")
    j = np.arange(1, model_size + 1, dtype=np.float64)
    lam = np.power(j, -a) / zeta(a, 1.0)
    q = np.power(j, -(a + r)) / zeta(a + r, 1.0)
    target = np.power(j, -b) / zeta(b, 1.0)
    tail = float(zeta(b, float(model_size + 1)) / zeta(b, 1.0))
    return lam, q, target, tail


def make_dictionary(model_size: int, mixing_strength: float, seed: int) -> np.ndarray:
    """Return a dense square dictionary with normalized rows.

    ``mixing_strength=0`` is orthogonal.  Positive values perturb a Haar-like
    orthogonal matrix and then normalize rows.  The condition number and actual
    row coherence are measured and retained with the results.
    """
    if not 0.0 <= mixing_strength < 1.0:
        raise ValueError("mixing_strength must lie in [0,1)")
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(model_size, model_size))
    orthogonal, _ = np.linalg.qr(gaussian)
    if mixing_strength == 0:
        return orthogonal
    perturbation = rng.normal(size=(model_size, model_size)) / math.sqrt(model_size)
    dictionary = (
        math.sqrt(1.0 - mixing_strength) * orthogonal
        + math.sqrt(mixing_strength) * perturbation
    )
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("degenerate dictionary row")
    return dictionary / norms


def max_row_coherence(dictionary: np.ndarray) -> float:
    normalized = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    gram = np.abs(normalized @ normalized.T)
    np.fill_diagonal(gram, 0.0)
    return float(np.max(gram))


def exact_coverage_risk(
    blocks: int, q: np.ndarray, target: np.ndarray, tail: float
) -> float:
    return tail + float(np.dot(target, np.exp(blocks * np.log1p(-q))))


def decoded_dense_risk(
    dictionary: np.ndarray,
    theta: np.ndarray,
    lam: np.ndarray,
    tail: float,
    seen_modes: np.ndarray,
) -> float:
    """Prediction risk after exact latent decoding through a known dictionary."""
    latent_hat = np.zeros_like(theta)
    latent_hat[seen_modes] = theta[seen_modes]
    ambient_hat = np.linalg.solve(dictionary, latent_hat)
    prediction_error = dictionary @ ambient_hat - theta
    return tail + float(np.dot(lam, prediction_error**2))


def conditional_min_norm_risk(
    dictionary: np.ndarray,
    theta: np.ndarray,
    teacher: np.ndarray,
    lam: np.ndarray,
    tail: float,
    seen_modes: np.ndarray,
    ridge: float = 1e-10,
) -> float:
    """Risk of the ambient minimum-norm interpolant on observed mode equations."""
    if seen_modes.size == 0:
        prediction_error = -theta
    else:
        observed = dictionary[seen_modes]
        gram = observed @ observed.T
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(seen_modes.size), theta[seen_modes]
        )
        estimate = observed.T @ coefficients
        prediction_error = dictionary @ (estimate - teacher)
    return tail + float(np.dot(lam, prediction_error**2))


def fit_exponent(
    blocks: np.ndarray, risk: np.ndarray, fit_points: int
) -> tuple[float, int, int]:
    if fit_points < 2 or fit_points > blocks.size:
        raise ValueError("invalid fit_points")
    slope, _ = np.polyfit(
        np.log(blocks[-fit_points:].astype(np.float64)),
        np.log(risk[-fit_points:]),
        deg=1,
    )
    return float(-slope), int(blocks[-fit_points]), int(blocks[-1])


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    a: float,
    b: float,
    r_values: Sequence[float],
    model_size: int,
    mixing_strength: float,
    block_powers: Sequence[int],
    trials: int,
    fit_points: int,
    seed: int,
    output_dir: Path,
) -> dict[float, DenseSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=model_size)
    orthogonal = make_dictionary(model_size, 0.0, seed + 1)
    dictionary = make_dictionary(model_size, mixing_strength, seed + 2)
    dictionary_condition = float(np.linalg.cond(dictionary))
    dictionary_coherence = max_row_coherence(dictionary)
    blocks_array = np.asarray([2**power for power in block_powers], dtype=np.int64)
    rows: list[dict[str, float]] = []
    summaries: dict[float, DenseSummary] = {}

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r in r_values:
        lam, q, target, tail = spectral_sequences(a, b, r, model_size)
        theta = signs * np.sqrt(target / lam)
        orthogonal_teacher = np.linalg.solve(orthogonal, theta)
        dictionary_teacher = np.linalg.solve(dictionary, theta)
        exact_values: list[float] = []
        decoded_values: list[float] = []
        orthogonal_values: list[float] = []
        dictionary_values: list[float] = []
        dictionary_stderr: list[float] = []

        for blocks in blocks_array:
            exact = exact_coverage_risk(int(blocks), q, target, tail)
            decoded_risks = np.empty(trials, dtype=np.float64)
            orthogonal_risks = np.empty(trials, dtype=np.float64)
            dictionary_risks = np.empty(trials, dtype=np.float64)
            for trial in range(trials):
                sampled = rng.zipf(a + r, size=int(blocks)) - 1
                seen = np.unique(sampled[sampled < model_size]).astype(np.int64)
                decoded_risks[trial] = decoded_dense_risk(
                    dictionary, theta, lam, tail, seen
                )
                orthogonal_risks[trial] = conditional_min_norm_risk(
                    orthogonal,
                    theta,
                    orthogonal_teacher,
                    lam,
                    tail,
                    seen,
                )
                dictionary_risks[trial] = conditional_min_norm_risk(
                    dictionary,
                    theta,
                    dictionary_teacher,
                    lam,
                    tail,
                    seen,
                )
            decoded_mean = float(np.mean(decoded_risks))
            orthogonal_mean = float(np.mean(orthogonal_risks))
            dictionary_mean = float(np.mean(dictionary_risks))
            dictionary_error = float(
                np.std(dictionary_risks, ddof=1) / math.sqrt(trials)
            )
            exact_values.append(exact)
            decoded_values.append(decoded_mean)
            orthogonal_values.append(orthogonal_mean)
            dictionary_values.append(dictionary_mean)
            dictionary_stderr.append(dictionary_error)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "r": r,
                    "blocks": int(blocks),
                    "model_size": model_size,
                    "mixing_strength": mixing_strength,
                    "condition_number": dictionary_condition,
                    "max_row_coherence": dictionary_coherence,
                    "trials": trials,
                    "exact_coverage_risk": exact,
                    "decoded_dense_mean": decoded_mean,
                    "orthogonal_min_norm_mean": orthogonal_mean,
                    "dictionary_min_norm_mean": dictionary_mean,
                    "dictionary_min_norm_stderr": dictionary_error,
                    "decoded_relative_error": (decoded_mean - exact) / exact,
                    "orthogonal_relative_error": (orthogonal_mean - exact) / exact,
                    "dictionary_ratio_to_exact": dictionary_mean / exact,
                }
            )

        fitted, fit_start, fit_end = fit_exponent(
            blocks_array, np.asarray(dictionary_values), fit_points
        )
        exact_array = np.asarray(exact_values)
        decoded_array = np.asarray(decoded_values)
        orthogonal_array = np.asarray(orthogonal_values)
        dictionary_array = np.asarray(dictionary_values)
        summary = DenseSummary(
            r=r,
            predicted_exponent=(b - 1.0) / (a + r),
            dictionary_fitted_exponent=fitted,
            orthogonal_max_relative_error=float(
                np.max(np.abs((orthogonal_array - exact_array) / exact_array))
            ),
            decoded_max_relative_error=float(
                np.max(np.abs((decoded_array - exact_array) / exact_array))
            ),
            dictionary_max_ratio_to_coverage=float(
                np.max(dictionary_array / exact_array)
            ),
            dictionary_condition_number=dictionary_condition,
            dictionary_max_row_coherence=dictionary_coherence,
            mixing_strength=mixing_strength,
            model_size=model_size,
            trials=trials,
            fit_start_block=fit_start,
            fit_end_block=fit_end,
        )
        summaries[r] = summary
        line, = ax.loglog(
            blocks_array,
            exact_values,
            linewidth=1.4,
            label=rf"exact/decoded, $r={r:g}$",
        )
        ax.errorbar(
            blocks_array,
            dictionary_values,
            yerr=dictionary_stderr,
            marker="o",
            markersize=3,
            linewidth=0,
            capsize=2,
            color=line.get_color(),
            label=(
                rf"ambient min-norm, fit $-{fitted:.3f}$ "
                rf"(pred. $-{summary.predicted_exponent:.3f}$)"
            ),
        )

    ax.set_xlabel("Innovation blocks $B$")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("Dense representations preserve the persistence-controlled slope")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "dense_features.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "dense_features.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {str(r): asdict(summary) for r, summary in summaries.items()},
            handle,
            indent=2,
            sort_keys=True,
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.8)
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.0, 0.8])
    parser.add_argument("--model-size", type=int, default=384)
    parser.add_argument("--mixing-strength", type=float, default=0.2)
    parser.add_argument(
        "--block-powers", type=int, nargs="+", default=[6, 8, 10, 12, 14]
    )
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--fit-points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dense_features")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_experiment(
        a=args.a,
        b=args.b,
        r_values=args.r_values,
        model_size=args.model_size,
        mixing_strength=args.mixing_strength,
        block_powers=args.block_powers,
        trials=args.trials,
        fit_points=args.fit_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    for r, summary in summaries.items():
        print(
            f"r={r:g}: predicted={summary.predicted_exponent:.6f}, "
            f"ambient fit={summary.dictionary_fitted_exponent:.6f}, "
            f"decoded max rel.err={summary.decoded_max_relative_error:.4f}, "
            f"cond={summary.dictionary_condition_number:.2f}, "
            f"row coherence={summary.dictionary_max_row_coherence:.3f}"
        )


if __name__ == "__main__":
    main()
