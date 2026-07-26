#!/usr/bin/env python3
"""Dense Gaussian AR negative control.

Every coordinate is present in every sample.  The marginal covariance has a
power-law spectrum and coordinate j has correlation time tau_j ~ j^r, but there
is no renewal/coverage bottleneck.  Batch minimum-norm regression therefore
need not follow the renewal exponent (b-1)/(a+r).  This experiment is included
to delimit, rather than inflate, the scope of the main theorem.
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
class ARSummary:
    process: str
    r: float | None
    fitted_exponent: float
    renewal_formula_exponent: float | None
    largest_n_risk: float
    largest_n_ratio_to_iid: float
    dimension: int
    trials: int
    fit_start_n: int
    fit_end_n: int


def spectral_problem(
    dimension: int, a: float, b: float, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    j = np.arange(1, dimension + 1, dtype=np.float64)
    lam = np.power(j, -a) / zeta(a, 1.0)
    source = np.power(j, -b) / zeta(b, 1.0)
    tail = float(zeta(b, float(dimension + 1)) / zeta(b, 1.0))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=dimension)
    theta = signs * np.sqrt(source / lam)
    return lam, theta, tail


def sample_dense_ar(
    rng: np.random.Generator,
    n: int,
    lam: np.ndarray,
    r: float | None,
) -> np.ndarray:
    dimension = lam.size
    if r is None:
        return rng.normal(size=(n, dimension)) * np.sqrt(lam)[None, :]
    j = np.arange(1, dimension + 1, dtype=np.float64)
    tau = np.power(j, r)
    rho = np.exp(-1.0 / tau)
    innovation_scale = np.sqrt(lam * (1.0 - rho**2))
    x = np.empty((n, dimension), dtype=np.float64)
    x[0] = rng.normal(size=dimension) * np.sqrt(lam)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + innovation_scale * rng.normal(size=dimension)
    return x


def minimum_norm_risk(
    x: np.ndarray,
    theta: np.ndarray,
    lam: np.ndarray,
    tail: float,
    ridge: float,
) -> float:
    y = x @ theta
    gram = x @ x.T
    dual = np.linalg.solve(gram + ridge * np.eye(x.shape[0]), y)
    estimate = x.T @ dual
    error = estimate - theta
    return tail + float(np.dot(lam, error**2))


def fit_exponent(n_values: np.ndarray, risk: np.ndarray, fit_points: int) -> float:
    slope, _ = np.polyfit(
        np.log(n_values[-fit_points:].astype(np.float64)),
        np.log(risk[-fit_points:]),
        deg=1,
    )
    return float(-slope)


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
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
    dimension: int,
    n_values: Sequence[int],
    trials: int,
    fit_points: int,
    ridge: float,
    seed: int,
    output_dir: Path,
) -> dict[str, ARSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lam, theta, tail = spectral_problem(dimension, a, b, seed)
    rng = np.random.default_rng(seed + 1)
    processes: list[tuple[str, float | None]] = [("iid", None)] + [
        (f"ar_r={r:g}", float(r)) for r in r_values
    ]
    n_array = np.asarray(n_values, dtype=np.int64)
    rows: list[dict[str, float | str]] = []
    mean_by_process: dict[str, np.ndarray] = {}
    stderr_by_process: dict[str, np.ndarray] = {}

    for process, r in processes:
        means: list[float] = []
        stderrs: list[float] = []
        for n in n_array:
            trial_risk = np.empty(trials, dtype=np.float64)
            for trial in range(trials):
                x = sample_dense_ar(rng, int(n), lam, r)
                trial_risk[trial] = minimum_norm_risk(x, theta, lam, tail, ridge)
            mean = float(np.mean(trial_risk))
            stderr = float(np.std(trial_risk, ddof=1) / math.sqrt(trials))
            means.append(mean)
            stderrs.append(stderr)
            rows.append(
                {
                    "process": process,
                    "r": "" if r is None else r,
                    "n": int(n),
                    "dimension": dimension,
                    "trials": trials,
                    "mean_risk": mean,
                    "stderr": stderr,
                }
            )
        mean_by_process[process] = np.asarray(means)
        stderr_by_process[process] = np.asarray(stderrs)

    iid_largest = float(mean_by_process["iid"][-1])
    summaries: dict[str, ARSummary] = {}
    for process, r in processes:
        mean = mean_by_process[process]
        fitted = fit_exponent(n_array, mean, fit_points)
        summaries[process] = ARSummary(
            process=process,
            r=r,
            fitted_exponent=fitted,
            renewal_formula_exponent=(
                None if r is None else (b - 1.0) / (a + r)
            ),
            largest_n_risk=float(mean[-1]),
            largest_n_ratio_to_iid=float(mean[-1] / iid_largest),
            dimension=dimension,
            trials=trials,
            fit_start_n=int(n_array[-fit_points]),
            fit_end_n=int(n_array[-1]),
        )

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for process, _ in processes:
        summary = summaries[process]
        ax.errorbar(
            n_array,
            mean_by_process[process],
            yerr=stderr_by_process[process],
            marker="o",
            markersize=3,
            linewidth=1.1,
            capsize=2,
            label=f"{process}, fitted slope -{summary.fitted_exponent:.2f}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dense Gaussian samples $N$")
    ax.set_ylabel("Population excess risk")
    ax.set_title("Negative control: autocorrelation alone does not impose $a+r$")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_dir / "dense_ar_control.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "dense_ar_control.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {name: asdict(summary) for name, summary in summaries.items()},
            handle,
            indent=2,
            sort_keys=True,
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.8)
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=[16, 24, 36, 54, 80, 120, 176]
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--fit-points", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dense_ar_control")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_experiment(
        a=args.a,
        b=args.b,
        r_values=args.r_values,
        dimension=args.dimension,
        n_values=args.n_values,
        trials=args.trials,
        fit_points=args.fit_points,
        ridge=args.ridge,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    for name, summary in summaries.items():
        print(
            f"{name}: fitted={summary.fitted_exponent:.4f}, "
            f"largest-N ratio to iid={summary.largest_n_ratio_to_iid:.3f}, "
            f"renewal formula={summary.renewal_formula_exponent}"
        )


if __name__ == "__main__":
    main()
