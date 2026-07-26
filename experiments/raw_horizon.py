#!/usr/bin/env python3
"""Fixed-raw-horizon validation for spectral renewal regression.

The theorem in the paper is exact for a fixed number of innovation blocks.
This script validates the transfer to a fixed number N of raw observations.
Mode j has deterministic integer duration ell_j = ceil(j**r), while innovation
modes are distributed as q_j proportional to j**(-a) / ell_j.  Rejection
sampling from Zipf(a+r) is exact because

    [j**(-a) / ceil(j**r)] / j**(-(a+r)) = j**r / ceil(j**r) <= 1.

In the noiseless model, seeing any part of a block reveals its coordinate.
The conditional prediction risk is therefore one minus the target energy of
all distinct in-model modes touched before the raw horizon.
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
class RawHorizonSummary:
    a: float
    b: float
    r: float
    p: float
    model_size: int
    trials: int
    predicted_slope: float
    fitted_slope: float
    fit_start_horizon: int
    fit_end_horizon: int


def _validate(a: float, b: float, r: float, model_size: int) -> None:
    if a <= 1:
        raise ValueError("a must exceed 1")
    if b <= 1:
        raise ValueError("b must exceed 1")
    if r < 0:
        raise ValueError("r must be nonnegative")
    if model_size < 1:
        raise ValueError("model_size must be positive")


def integer_duration(modes: np.ndarray, r: float) -> np.ndarray:
    """Return ceil(j**r) as int64, with the r=0 case handled exactly."""
    if r == 0:
        return np.ones_like(modes, dtype=np.int64)
    values = np.ceil(np.power(modes.astype(np.float64), r))
    if np.any(values > np.iinfo(np.int64).max):
        raise OverflowError("sampled duration exceeds int64")
    return values.astype(np.int64)


def sample_innovation_modes(
    rng: np.random.Generator,
    count: int,
    a: float,
    r: float,
) -> np.ndarray:
    """Sample exactly from q_j proportional to j**(-a)/ceil(j**r)."""
    if count < 1:
        return np.empty(0, dtype=np.int64)
    p = a + r
    accepted: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        proposal_count = max(64, int(math.ceil(1.35 * remaining)))
        proposal = rng.zipf(p, size=proposal_count).astype(np.int64)
        if r == 0:
            keep = np.ones(proposal_count, dtype=bool)
        else:
            duration = integer_duration(proposal, r).astype(np.float64)
            acceptance = np.power(proposal.astype(np.float64), r) / duration
            keep = rng.random(proposal_count) < acceptance
        batch = proposal[keep]
        if batch.size:
            accepted.append(batch[:remaining])
            remaining -= min(remaining, int(batch.size))
    return np.concatenate(accepted)


def sample_touched_modes(
    rng: np.random.Generator,
    raw_horizon: int,
    a: float,
    r: float,
) -> tuple[np.ndarray, int]:
    """Sample all innovation modes touched by a trajectory of length N.

    Returns the touched modes, including the partially observed final block,
    and the total duration of those blocks (which is at least raw_horizon).
    """
    if raw_horizon < 1:
        raise ValueError("raw_horizon must be positive")

    chunks: list[np.ndarray] = []
    elapsed = 0
    while elapsed < raw_horizon:
        batch_size = max(128, min(4096, raw_horizon - elapsed))
        modes = sample_innovation_modes(rng, batch_size, a, r)
        durations = integer_duration(modes, r)
        cumulative = elapsed + np.cumsum(durations, dtype=np.int64)
        crossing = np.searchsorted(cumulative, raw_horizon, side="left")
        if crossing < modes.size:
            chunks.append(modes[: crossing + 1])
            elapsed = int(cumulative[crossing])
            break
        chunks.append(modes)
        elapsed = int(cumulative[-1])
    return np.concatenate(chunks), elapsed


def conditional_noiseless_risk(
    touched_modes: np.ndarray,
    b: float,
    model_size: int,
) -> float:
    """Prediction risk conditional on the modes touched by the trajectory."""
    in_model = touched_modes[touched_modes <= model_size]
    if in_model.size == 0:
        return 1.0
    unique_modes = np.unique(in_model).astype(np.float64)
    learned_energy = float(np.sum(np.power(unique_modes, -b)) / zeta(b, 1.0))
    return max(0.0, 1.0 - learned_energy)


def simulate_raw_horizon_risks(
    *,
    raw_horizon: int,
    trials: int,
    a: float,
    b: float,
    r: float,
    model_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return risk and touched-block count for independent trajectories."""
    _validate(a, b, r, model_size)
    if raw_horizon < 1 or trials < 1:
        raise ValueError("raw_horizon and trials must be positive")

    rng = np.random.default_rng(seed)
    risks = np.empty(trials, dtype=np.float64)
    block_counts = np.empty(trials, dtype=np.int64)
    for trial in range(trials):
        modes, _ = sample_touched_modes(rng, raw_horizon, a, r)
        risks[trial] = conditional_noiseless_risk(modes, b, model_size)
        block_counts[trial] = modes.size
    return risks, block_counts


def fit_log_slope(
    horizons: np.ndarray, risks: np.ndarray, fit_points: int
) -> tuple[float, int, int]:
    if fit_points < 2 or fit_points > horizons.size:
        raise ValueError("fit_points must lie between 2 and the curve length")
    slope, _ = np.polyfit(
        np.log(horizons[-fit_points:].astype(np.float64)),
        np.log(risks[-fit_points:]),
        deg=1,
    )
    return float(slope), int(horizons[-fit_points]), int(horizons[-1])


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
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
    horizon_powers: Sequence[int],
    trials: int,
    fit_points: int,
    seed: int,
    output_dir: Path,
) -> dict[float, RawHorizonSummary]:
    horizons = np.asarray([2**power for power in horizon_powers], dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    summaries: dict[float, RawHorizonSummary] = {}

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r_index, r in enumerate(r_values):
        means: list[float] = []
        stderrs: list[float] = []
        for power, horizon in zip(horizon_powers, horizons, strict=True):
            risks, block_counts = simulate_raw_horizon_risks(
                raw_horizon=int(horizon),
                trials=trials,
                a=a,
                b=b,
                r=r,
                model_size=model_size,
                seed=seed + 100_000 * r_index + power,
            )
            mean = float(np.mean(risks))
            stderr = float(np.std(risks, ddof=1) / math.sqrt(trials))
            means.append(mean)
            stderrs.append(stderr)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "r": r,
                    "raw_horizon": int(horizon),
                    "model_size": model_size,
                    "trials": trials,
                    "mean_risk": mean,
                    "risk_stderr": stderr,
                    "mean_touched_blocks": float(np.mean(block_counts)),
                    "predicted_slope": -(b - 1.0) / (a + r),
                }
            )

        risk_array = np.asarray(means)
        fitted, fit_start, fit_end = fit_log_slope(horizons, risk_array, fit_points)
        summary = RawHorizonSummary(
            a=a,
            b=b,
            r=r,
            p=a + r,
            model_size=model_size,
            trials=trials,
            predicted_slope=-(b - 1.0) / (a + r),
            fitted_slope=fitted,
            fit_start_horizon=fit_start,
            fit_end_horizon=fit_end,
        )
        summaries[r] = summary
        ax.errorbar(
            horizons,
            means,
            yerr=stderrs,
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            label=rf"$r={r:g}$: predicted {summary.predicted_slope:.3f}, fit {fitted:.3f}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Raw trajectory length $N$")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("The exponent persists at a fixed raw horizon")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_horizon_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "raw_horizon_validation.csv", rows)
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
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    parser.add_argument("--model-size", type=int, default=65_536)
    parser.add_argument("--horizon-powers", type=int, nargs="+", default=[6, 8, 10, 12, 14])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--fit-points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw_horizon_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_experiment(
        a=args.a,
        b=args.b,
        r_values=args.r_values,
        model_size=args.model_size,
        horizon_powers=args.horizon_powers,
        trials=args.trials,
        fit_points=args.fit_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    for r, summary in summaries.items():
        print(
            f"r={r:g}: predicted slope {summary.predicted_slope:.6f}, "
            f"fitted slope {summary.fitted_slope:.6f}"
        )


if __name__ == "__main__":
    main()
