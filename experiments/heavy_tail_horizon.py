#!/usr/bin/env python3
"""Heavy-dwell fixed-horizon and stationary-initialization experiments.

The raw-horizon theorem in the first draft used a finite-second-moment
renewal-count sandwich.  A mode-wise hitting-time argument removes that
restriction at a renewal boundary.  Under equilibrium initialization,
however, the forward recurrence time has a polynomial tail and creates an
additional inspection-paradox term.

This script validates both statements for deterministic integer dwell times
ell_j = ceil(j**r):

  boundary start:  R_N ~ N^{-(b-1)/(a+r)};
  stationary start: R_N ~ N^{-(b-1)/(a+r)} + N^{-(a-1)/r}.

All simulations are noiseless.  Seeing any portion of a block reveals its
spectral coordinate.  The stationary initial mode follows the time-occupancy
law lambda_j proportional to j^{-a}; conditional on that mode, the residual
length is uniform on {1,...,ell_j}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt

# Embed TrueType fonts in PDFs so AAAI preflight does not report Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
from scipy.special import zeta

from raw_horizon import (
    conditional_noiseless_risk,
    integer_duration,
    sample_touched_modes,
)


@dataclass(frozen=True)
class HeavyTailSummary:
    a: float
    b: float
    r: float
    p: float
    model_size: int
    trials: int
    boundary_predicted_exponent: float
    boundary_fitted_exponent: float
    stationary_predicted_exponent: float
    stationary_fitted_exponent: float
    residual_predicted_exponent: float
    residual_fitted_exponent: float
    fit_start_horizon: int
    fit_end_horizon: int


def _validate(a: float, b: float, r: float, model_size: int) -> None:
    if a <= 1:
        raise ValueError("a must exceed 1")
    if b <= 1:
        raise ValueError("b must exceed 1")
    if r <= 0:
        raise ValueError("this experiment requires r>0")
    if b >= a + r + 1:
        raise ValueError("require b<a+r+1 for the boundary hitting-time law")
    if model_size < 2:
        raise ValueError("model_size must be at least two")


def sample_stationary_touched_modes(
    rng: np.random.Generator,
    raw_horizon: int,
    a: float,
    r: float,
) -> tuple[np.ndarray, int]:
    """Sample modes touched from the equilibrium renewal initialization.

    Returns the unique initial/future block sequence (duplicates are retained)
    and the residual length of the initial equilibrium block.
    """
    if raw_horizon < 1:
        raise ValueError("raw_horizon must be positive")
    initial_mode = int(rng.zipf(a))
    initial_duration = int(integer_duration(np.asarray([initial_mode]), r)[0])
    residual = int(rng.integers(1, initial_duration + 1))
    if residual >= raw_horizon:
        return np.asarray([initial_mode], dtype=np.int64), residual
    future_modes, _ = sample_touched_modes(
        rng, raw_horizon - residual, a=a, r=r
    )
    return np.concatenate(
        [np.asarray([initial_mode], dtype=np.int64), future_modes]
    ), residual


def simulate_horizon(
    *,
    initialization: Literal["boundary", "stationary"],
    raw_horizon: int,
    trials: int,
    a: float,
    b: float,
    r: float,
    model_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noiseless risk and number of touched blocks for each trial."""
    _validate(a, b, r, model_size)
    if trials < 1:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    risks = np.empty(trials, dtype=np.float64)
    counts = np.empty(trials, dtype=np.int64)
    for trial in range(trials):
        if initialization == "boundary":
            modes, _ = sample_touched_modes(rng, raw_horizon, a, r)
        elif initialization == "stationary":
            modes, _ = sample_stationary_touched_modes(
                rng, raw_horizon, a, r
            )
        else:
            raise ValueError(f"unknown initialization: {initialization}")
        risks[trial] = conditional_noiseless_risk(modes, b, model_size)
        counts[trial] = modes.size
    return risks, counts


def stationary_residual_tail(
    raw_horizon: int,
    a: float,
    r: float,
    *,
    max_mode: int = 2_000_000,
) -> float:
    """Numerically evaluate P(R>=N) under equilibrium initialization.

    The omitted Zipf(a) tail is at most zeta(a,max_mode+1)/zeta(a), which is
    below 4e-7 for the default a=2 and max_mode=2e6.
    """
    if raw_horizon < 1 or a <= 1 or r <= 0 or max_mode < 10:
        raise ValueError("invalid residual-tail parameters")
    j = np.arange(1, max_mode + 1, dtype=np.float64)
    duration = np.ceil(np.power(j, r))
    survival = np.maximum(0.0, (duration - raw_horizon + 1.0) / duration)
    lam = np.power(j, -a) / zeta(a, 1.0)
    return float(np.dot(lam, survival))


def fit_exponent(
    horizons: np.ndarray, values: np.ndarray, fit_points: int
) -> tuple[float, int, int]:
    if fit_points < 2 or fit_points > horizons.size:
        raise ValueError("invalid fit_points")
    if np.any(values[-fit_points:] <= 0):
        raise ValueError("values must be positive")
    slope, _ = np.polyfit(
        np.log(horizons[-fit_points:].astype(np.float64)),
        np.log(values[-fit_points:]),
        deg=1,
    )
    return float(-slope), int(horizons[-fit_points]), int(horizons[-1])


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    a: float,
    b: float,
    r: float,
    model_size: int,
    horizon_powers: Sequence[int],
    trials: int,
    fit_points: int,
    seed: int,
    output_dir: Path,
) -> HeavyTailSummary:
    _validate(a, b, r, model_size)
    horizons = np.asarray([2**power for power in horizon_powers], dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    boundary_means: list[float] = []
    stationary_means: list[float] = []
    boundary_errors: list[float] = []
    stationary_errors: list[float] = []
    residual_tails: list[float] = []

    for index, horizon in enumerate(horizons):
        boundary, boundary_counts = simulate_horizon(
            initialization="boundary",
            raw_horizon=int(horizon),
            trials=trials,
            a=a,
            b=b,
            r=r,
            model_size=model_size,
            seed=seed + 10_000 * index,
        )
        stationary, stationary_counts = simulate_horizon(
            initialization="stationary",
            raw_horizon=int(horizon),
            trials=trials,
            a=a,
            b=b,
            r=r,
            model_size=model_size,
            seed=seed + 10_000 * index + 1,
        )
        boundary_mean = float(np.mean(boundary))
        stationary_mean = float(np.mean(stationary))
        boundary_stderr = float(np.std(boundary, ddof=1) / math.sqrt(trials))
        stationary_stderr = float(
            np.std(stationary, ddof=1) / math.sqrt(trials)
        )
        residual_tail = stationary_residual_tail(int(horizon), a, r)
        boundary_means.append(boundary_mean)
        stationary_means.append(stationary_mean)
        boundary_errors.append(boundary_stderr)
        stationary_errors.append(stationary_stderr)
        residual_tails.append(residual_tail)
        rows.append(
            {
                "a": a,
                "b": b,
                "r": r,
                "raw_horizon": int(horizon),
                "model_size": model_size,
                "trials": trials,
                "boundary_mean_risk": boundary_mean,
                "boundary_stderr": boundary_stderr,
                "boundary_mean_blocks": float(np.mean(boundary_counts)),
                "stationary_mean_risk": stationary_mean,
                "stationary_stderr": stationary_stderr,
                "stationary_mean_blocks": float(np.mean(stationary_counts)),
                "stationary_residual_tail": residual_tail,
                "boundary_predicted_exponent": (b - 1.0) / (a + r),
                "residual_predicted_exponent": (a - 1.0) / r,
            }
        )

    boundary_fit, fit_start, fit_end = fit_exponent(
        horizons, np.asarray(boundary_means), fit_points
    )
    stationary_fit, _, _ = fit_exponent(
        horizons, np.asarray(stationary_means), fit_points
    )
    residual_fit, _, _ = fit_exponent(
        horizons, np.asarray(residual_tails), fit_points
    )
    boundary_prediction = (b - 1.0) / (a + r)
    residual_prediction = (a - 1.0) / r
    stationary_prediction = min(boundary_prediction, residual_prediction)
    summary = HeavyTailSummary(
        a=a,
        b=b,
        r=r,
        p=a + r,
        model_size=model_size,
        trials=trials,
        boundary_predicted_exponent=boundary_prediction,
        boundary_fitted_exponent=boundary_fit,
        stationary_predicted_exponent=stationary_prediction,
        stationary_fitted_exponent=stationary_fit,
        residual_predicted_exponent=residual_prediction,
        residual_fitted_exponent=residual_fit,
        fit_start_horizon=fit_start,
        fit_end_horizon=fit_end,
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(
        horizons,
        boundary_means,
        yerr=boundary_errors,
        marker="o",
        markersize=3,
        linewidth=1.2,
        capsize=2,
        label=(
            rf"renewal boundary: predicted $-{boundary_prediction:.3f}$, "
            rf"fit $-{boundary_fit:.3f}$"
        ),
    )
    ax.errorbar(
        horizons,
        stationary_means,
        yerr=stationary_errors,
        marker="s",
        markersize=3,
        linewidth=1.2,
        capsize=2,
        label=(
            rf"stationary: predicted $-{stationary_prediction:.3f}$, "
            rf"fit $-{stationary_fit:.3f}$"
        ),
    )
    ax.plot(
        horizons,
        residual_tails,
        marker="^",
        markersize=3,
        linewidth=1.1,
        linestyle="--",
        label=(
            rf"forward-recurrence tail: predicted $-{residual_prediction:.3f}$, "
            rf"fit $-{residual_fit:.3f}$"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Raw trajectory length $N$")
    ax.set_ylabel("Expected risk / residual probability")
    ax.set_title("Heavy dwell times create an inspection-paradox phase")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.6)
    fig.tight_layout()
    fig.savefig(output_dir / "heavy_tail_horizon.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "heavy_tail_horizon.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=4.5)
    parser.add_argument("--r", type=float, default=3.0)
    parser.add_argument("--model-size", type=int, default=65_536)
    parser.add_argument(
        "--horizon-powers", type=int, nargs="+", default=[5, 7, 9, 11, 13]
    )
    parser.add_argument("--trials", type=int, default=3_000)
    parser.add_argument("--fit-points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/heavy_tail_horizon")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        a=args.a,
        b=args.b,
        r=args.r,
        model_size=args.model_size,
        horizon_powers=args.horizon_powers,
        trials=args.trials,
        fit_points=args.fit_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(
        "boundary exponent: "
        f"predicted={summary.boundary_predicted_exponent:.6f}, "
        f"fitted={summary.boundary_fitted_exponent:.6f}"
    )
    print(
        "stationary exponent: "
        f"predicted={summary.stationary_predicted_exponent:.6f}, "
        f"fitted={summary.stationary_fitted_exponent:.6f}"
    )
    print(
        "residual exponent: "
        f"predicted={summary.residual_predicted_exponent:.6f}, "
        f"fitted={summary.residual_fitted_exponent:.6f}"
    )


if __name__ == "__main__":
    main()
