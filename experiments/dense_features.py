#!/usr/bin/env python3
"""Dense-feature stress tests for spectral renewal scaling.

The theorem is stated in a spectral basis, but its mechanism is not tied to
sparse observed vectors.  We test two dense representations of the first M
spectral modes:

1. an orthogonal random rotation, where normalized Kaczmarz/minimum-norm
   regression is exactly equivalent to coordinate memorization;
2. a coherent near-orthogonal random dictionary, where modes interfere and
   the exact decomposition no longer applies, but the innovation-limited slope
   should remain visible.

An irreducible power-law tail j>M is added to the test risk, matching the model
approximation term in the theorem.
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
import numpy as np
from scipy.special import zeta


@dataclass(frozen=True)
class DenseSummary:
    r: float
    predicted_exponent: float
    dictionary_fitted_exponent: float
    orthogonal_max_relative_error: float
    dictionary_max_ratio_to_coverage: float
    coherence: float
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


def make_dictionary(
    model_size: int,
    coherence: float,
    seed: int,
) -> np.ndarray:
    """Return a square dense dictionary with controlled departure from rotation."""
    if not 0.0 <= coherence < 1.0:
        raise ValueError("coherence must lie in [0,1)")
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(model_size, model_size))
    orthogonal, _ = np.linalg.qr(gaussian)
    if coherence == 0:
        return orthogonal
    perturbation = rng.normal(size=(model_size, model_size)) / math.sqrt(model_size)
    dictionary = (
        math.sqrt(1.0 - coherence) * orthogonal
        + math.sqrt(coherence) * perturbation
    )
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    return dictionary


def conditional_dense_risk(
    dictionary: np.ndarray,
    theta: np.ndarray,
    teacher: np.ndarray,
    lam: np.ndarray,
    tail: float,
    seen_modes: np.ndarray,
    ridge: float = 1e-10,
) -> float:
    """Risk of the minimum-norm interpolant on the observed mode equations."""
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


def exact_coverage_risk(
    blocks: int,
    q: np.ndarray,
    target: np.ndarray,
    tail: float,
) -> float:
    return tail + float(np.dot(target, np.exp(blocks * np.log1p(-q))))


def fit_exponent(blocks: np.ndarray, risk: np.ndarray, fit_points: int) -> tuple[float, int, int]:
    slope, _ = np.polyfit(
        np.log(blocks[-fit_points:].astype(np.float64)),
        np.log(risk[-fit_points:]),
        deg=1,
    )
    return float(-slope), int(blocks[-fit_points]), int(blocks[-1])


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
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
    coherence: float,
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
    dictionary = make_dictionary(model_size, coherence, seed + 2)
    blocks_array = np.asarray([2**power for power in block_powers], dtype=np.int64)
    rows: list[dict[str, float]] = []
    summaries: dict[float, DenseSummary] = {}

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r_index, r in enumerate(r_values):
        lam, q, target, tail = spectral_sequences(a, b, r, model_size)
        theta = signs * np.sqrt(target / lam)
        orthogonal_teacher = orthogonal.T @ theta
        dictionary_teacher = np.linalg.solve(dictionary, theta)
        exact_values: list[float] = []
        orthogonal_values: list[float] = []
        dictionary_values: list[float] = []
        dictionary_stderr: list[float] = []

        for power, blocks in zip(block_powers, blocks_array, strict=True):
            exact = exact_coverage_risk(int(blocks), q, target, tail)
            orthogonal_risks = np.empty(trials, dtype=np.float64)
            dictionary_risks = np.empty(trials, dtype=np.float64)
            for trial in range(trials):
                sampled = rng.zipf(a + r, size=int(blocks)) - 1
                seen = np.unique(sampled[sampled < model_size]).astype(np.int64)
                orthogonal_risks[trial] = conditional_dense_risk(
                    orthogonal,
                    theta,
                    orthogonal_teacher,
                    lam,
                    tail,
                    seen,
                )
                dictionary_risks[trial] = conditional_dense_risk(
                    dictionary,
                    theta,
                    dictionary_teacher,
                    lam,
                    tail,
                    seen,
                )
            orthogonal_mean = float(np.mean(orthogonal_risks))
            dictionary_mean = float(np.mean(dictionary_risks))
            dictionary_error = float(
                np.std(dictionary_risks, ddof=1) / math.sqrt(trials)
            )
            exact_values.append(exact)
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
                    "coherence": coherence,
                    "trials": trials,
                    "exact_coverage_risk": exact,
                    "orthogonal_dense_mean": orthogonal_mean,
                    "dictionary_dense_mean": dictionary_mean,
                    "dictionary_dense_stderr": dictionary_error,
                    "orthogonal_relative_error": (orthogonal_mean - exact) / exact,
                    "dictionary_ratio_to_exact": dictionary_mean / exact,
                }
            )

        fitted, fit_start, fit_end = fit_exponent(
            blocks_array, np.asarray(dictionary_values), fit_points
        )
        summary = DenseSummary(
            r=r,
            predicted_exponent=(b - 1.0) / (a + r),
            dictionary_fitted_exponent=fitted,
            orthogonal_max_relative_error=float(
                np.max(
                    np.abs(
                        (np.asarray(orthogonal_values) - np.asarray(exact_values))
                        / np.asarray(exact_values)
                    )
                )
            ),
            dictionary_max_ratio_to_coverage=float(
                np.max(np.asarray(dictionary_values) / np.asarray(exact_values))
            ),
            coherence=coherence,
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
            label=rf"exact occupancy, $r={r:g}$",
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
                rf"dense dictionary, fit $-{fitted:.3f}$ "
                rf"(pred. $-{summary.predicted_exponent:.3f}$)"
            ),
        )

    ax.set_xlabel("Innovation blocks $B$")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("The spectral-persistence law survives dense mode mixing")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.4, ncol=2)
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
    parser.add_argument("--model-size", type=int, default=512)
    parser.add_argument("--coherence", type=float, default=0.2)
    parser.add_argument(
        "--block-powers", type=int, nargs="+", default=[6, 8, 10, 12, 14]
    )
    parser.add_argument("--trials", type=int, default=120)
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
        coherence=args.coherence,
        block_powers=args.block_powers,
        trials=args.trials,
        fit_points=args.fit_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    for r, summary in summaries.items():
        print(
            f"r={r:g}: predicted={summary.predicted_exponent:.6f}, "
            f"dense dictionary fit={summary.dictionary_fitted_exponent:.6f}, "
            f"orthogonal max relative error={summary.orthogonal_max_relative_error:.4f}, "
            f"dictionary max ratio={summary.dictionary_max_ratio_to_coverage:.3f}"
        )


if __name__ == "__main__":
    main()
