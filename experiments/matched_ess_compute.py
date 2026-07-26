#!/usr/bin/env python3
"""Matched-effective-sample-size and compute-optimal scaling experiments.

This module tests two consequences of the spectral-renewal theory.

1. A uniform-persistence stream and a spectrally aligned stream are calibrated
   to have the same marginal spectrum and the same trace integrated
   autocorrelation time.  Nevertheless, their raw-horizon learning curves have
   different exponents because their innovation spectra are different.
2. Under the stylized dense-training cost C=M*B, the noiseless exact risk is
   minimized at M ~ C^(1/(a+r+1)) and B ~ C^((a+r)/(a+r+1)).

The raw-horizon experiment simulates the stationary reversible refresh chain
blockwise.  Geometric dwell times make the equilibrium residual distribution
memoryless, so the initial mode is drawn from the marginal law lambda and each
subsequent refresh mode is drawn from q proportional to lambda/tau.
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
class MatchedESSSummary:
    a: float
    b: float
    r: float
    trace_iat_target: float
    uniform_trace_iat: float
    aligned_trace_iat: float
    aligned_scale: float
    uniform_mean_block_length: float
    aligned_mean_block_length: float
    uniform_predicted_exponent: float
    uniform_fitted_exponent: float
    aligned_predicted_exponent: float
    aligned_fitted_exponent: float
    model_size: int
    trials: int
    fit_start_horizon: int
    fit_end_horizon: int


@dataclass(frozen=True)
class ComputeSummary:
    r: float
    predicted_risk_exponent: float
    fitted_risk_exponent: float
    predicted_model_exponent: float
    fitted_model_exponent: float
    predicted_innovation_exponent: float
    fitted_innovation_exponent: float
    fit_start_budget: int
    fit_end_budget: int


def _validate(a: float, b: float, r: float, trace_iat_target: float) -> None:
    if a <= 1 or b <= 1:
        raise ValueError("a and b must exceed one")
    if not 0 < r < a - 1:
        raise ValueError("matched finite trace-IAT construction requires 0<r<a-1")
    if trace_iat_target <= 1:
        raise ValueError("trace_iat_target must exceed one")


def matched_profiles(
    a: float, r: float, trace_iat_target: float
) -> dict[str, float]:
    """Return persistence constants for profiles with identical trace IAT.

    The trace IAT of the reversible chain is
        tau_trace = 2 * sum_j lambda_j tau_j - 1.
    We parameterize it by T=(tau_trace+1)/2.  Uniform persistence uses tau_j=T.
    Aligned persistence uses tau_j=c*j^r with c=T/E_lambda[J^r].
    """
    if a <= 1 or not 0 < r < a - 1 or trace_iat_target <= 1:
        raise ValueError("invalid matched-profile parameters")
    target_mean_persistence = 0.5 * (trace_iat_target + 1.0)
    marginal_r_moment = float(zeta(a - r, 1.0) / zeta(a, 1.0))
    aligned_scale = target_mean_persistence / marginal_r_moment
    if aligned_scale < 1.0:
        raise ValueError(
            "target IAT is too small to keep all aligned dwell means at least one"
        )
    uniform_mu = target_mean_persistence
    aligned_mu = float(
        aligned_scale * zeta(a, 1.0) / zeta(a + r, 1.0)
    )
    trace_iat = 2.0 * target_mean_persistence - 1.0
    return {
        "target_mean_persistence": target_mean_persistence,
        "marginal_r_moment": marginal_r_moment,
        "aligned_scale": aligned_scale,
        "uniform_mean_block_length": uniform_mu,
        "aligned_mean_block_length": aligned_mu,
        "uniform_trace_iat": trace_iat,
        "aligned_trace_iat": 2.0 * aligned_scale * marginal_r_moment - 1.0,
    }


def _sample_stationary_seen_modes(
    rng: np.random.Generator,
    *,
    raw_horizon: int,
    a: float,
    r: float,
    profile: str,
    uniform_persistence: float,
    aligned_scale: float,
    aligned_mean_block_length: float,
    model_size: int,
) -> np.ndarray:
    """Sample unique modeled modes touched in a stationary geometric stream."""
    if raw_horizon < 1 or model_size < 1:
        raise ValueError("raw_horizon and model_size must be positive")
    if profile not in {"uniform", "aligned"}:
        raise ValueError(f"unknown profile: {profile}")

    # At stationarity, the time-zero mode follows the common marginal lambda.
    initial_mode = int(rng.zipf(a))
    seen_chunks: list[np.ndarray] = []
    if initial_mode <= model_size:
        seen_chunks.append(np.asarray([initial_mode], dtype=np.int64))

    if profile == "uniform":
        initial_probability = 1.0 / uniform_persistence
        innovation_exponent = a
        mean_block_length = uniform_persistence
    else:
        initial_probability = 1.0 / (aligned_scale * initial_mode**r)
        innovation_exponent = a + r
        mean_block_length = aligned_mean_block_length
    initial_probability = min(1.0, initial_probability)
    elapsed = min(int(rng.geometric(initial_probability)), raw_horizon)

    while elapsed < raw_horizon:
        remaining = raw_horizon - elapsed
        # Oversample slightly relative to the expected renewal count; if a rare
        # batch is insufficient, the loop simply draws another batch.
        guess = max(64, int(1.12 * remaining / mean_block_length) + 64)
        modes = rng.zipf(innovation_exponent, size=guess).astype(np.int64)
        if profile == "uniform":
            probabilities = np.full(guess, 1.0 / uniform_persistence)
        else:
            probabilities = 1.0 / (
                aligned_scale * np.power(modes.astype(np.float64), r)
            )
            probabilities = np.minimum(probabilities, 1.0)
        durations = rng.geometric(probabilities).astype(np.int64)
        cumulative = np.cumsum(durations, dtype=np.int64)
        touched = int(np.searchsorted(cumulative, remaining, side="left")) + 1
        used = modes[: min(touched, guess)]
        modeled = used[used <= model_size]
        if modeled.size:
            seen_chunks.append(modeled)
        if touched <= guess:
            elapsed = raw_horizon
        else:
            elapsed += int(cumulative[-1])

    if not seen_chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(seen_chunks))


def simulate_matched_ess(
    *,
    a: float,
    b: float,
    r: float,
    trace_iat_target: float,
    model_size: int,
    horizons: Sequence[int],
    trials: int,
    fit_points: int,
    seed: int,
) -> tuple[list[dict[str, float]], MatchedESSSummary]:
    _validate(a, b, r, trace_iat_target)
    if trials < 2 or model_size < 2:
        raise ValueError("need at least two trials and two modeled modes")
    horizon_array = np.asarray(horizons, dtype=np.int64)
    if np.any(horizon_array < 2) or np.any(np.diff(horizon_array) <= 0):
        raise ValueError("horizons must be strictly increasing and at least two")
    if fit_points < 2 or fit_points > horizon_array.size:
        raise ValueError("invalid fit_points")

    profile = matched_profiles(a, r, trace_iat_target)
    uniform_persistence = profile["target_mean_persistence"]
    aligned_scale = profile["aligned_scale"]
    aligned_mu = profile["aligned_mean_block_length"]
    source = np.power(
        np.arange(1, model_size + 1, dtype=np.float64), -b
    ) / zeta(b, 1.0)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    means: dict[str, list[float]] = {"uniform": [], "aligned": []}

    for h_index, horizon in enumerate(horizon_array):
        for name in ["uniform", "aligned"]:
            risks = np.empty(trials, dtype=np.float64)
            for trial in range(trials):
                # Separate deterministic streams across scales and profiles.
                trial_rng = np.random.default_rng(
                    rng.integers(0, np.iinfo(np.int64).max)
                    + 10_000 * h_index
                    + trial
                )
                seen = _sample_stationary_seen_modes(
                    trial_rng,
                    raw_horizon=int(horizon),
                    a=a,
                    r=r,
                    profile=name,
                    uniform_persistence=uniform_persistence,
                    aligned_scale=aligned_scale,
                    aligned_mean_block_length=aligned_mu,
                    model_size=model_size,
                )
                observed_energy = float(source[seen - 1].sum()) if seen.size else 0.0
                risks[trial] = 1.0 - observed_energy
            mean = float(np.mean(risks))
            stderr = float(np.std(risks, ddof=1) / math.sqrt(trials))
            means[name].append(mean)
            rows.append(
                {
                    "profile": name,
                    "a": a,
                    "b": b,
                    "r": 0.0 if name == "uniform" else r,
                    "raw_horizon": int(horizon),
                    "model_size": model_size,
                    "trials": trials,
                    "trace_iat": profile[f"{name}_trace_iat"],
                    "mean_risk": mean,
                    "stderr": stderr,
                }
            )

    fitted: dict[str, float] = {}
    for name in ["uniform", "aligned"]:
        slope, _ = np.polyfit(
            np.log(horizon_array[-fit_points:].astype(np.float64)),
            np.log(np.asarray(means[name])[-fit_points:]),
            1,
        )
        fitted[name] = float(-slope)

    summary = MatchedESSSummary(
        a=a,
        b=b,
        r=r,
        trace_iat_target=trace_iat_target,
        uniform_trace_iat=profile["uniform_trace_iat"],
        aligned_trace_iat=profile["aligned_trace_iat"],
        aligned_scale=aligned_scale,
        uniform_mean_block_length=profile["uniform_mean_block_length"],
        aligned_mean_block_length=aligned_mu,
        uniform_predicted_exponent=(b - 1.0) / a,
        uniform_fitted_exponent=fitted["uniform"],
        aligned_predicted_exponent=(b - 1.0) / (a + r),
        aligned_fitted_exponent=fitted["aligned"],
        model_size=model_size,
        trials=trials,
        fit_start_horizon=int(horizon_array[-fit_points]),
        fit_end_horizon=int(horizon_array[-1]),
    )
    return rows, summary


def exact_noiseless_risk(a: float, b: float, r: float, model_size: int, blocks: int) -> float:
    if model_size < 1 or blocks < 1:
        raise ValueError("model_size and blocks must be positive")
    j = np.arange(1, model_size + 1, dtype=np.float64)
    q = np.power(j, -(a + r)) / zeta(a + r, 1.0)
    source = np.power(j, -b) / zeta(b, 1.0)
    tail = float(zeta(b, float(model_size + 1)) / zeta(b, 1.0))
    unseen = np.exp(float(blocks) * np.log1p(-q))
    return tail + float(np.dot(source, unseen))


def optimize_compute_budget(
    *,
    a: float,
    b: float,
    r: float,
    budget: int,
    max_model_size: int,
    grid_size: int,
) -> tuple[int, int, float]:
    if budget < 4 or max_model_size < 2 or grid_size < 8:
        raise ValueError("invalid compute search parameters")
    upper = min(max_model_size, budget)
    candidates = np.unique(
        np.rint(np.geomspace(2, upper, grid_size)).astype(np.int64)
    )
    best: tuple[float, int, int] | None = None
    for model_size in candidates:
        blocks = max(1, budget // int(model_size))
        risk = exact_noiseless_risk(a, b, r, int(model_size), blocks)
        if best is None or risk < best[0]:
            best = (risk, int(model_size), int(blocks))
    assert best is not None
    return best[1], best[2], best[0]


def run_compute_experiment(
    *,
    a: float,
    b: float,
    r_values: Sequence[float],
    budgets: Sequence[int],
    max_model_size: int,
    grid_size: int,
    fit_points: int,
) -> tuple[list[dict[str, float]], dict[float, ComputeSummary]]:
    budget_array = np.asarray(budgets, dtype=np.int64)
    if np.any(budget_array < 4) or np.any(np.diff(budget_array) <= 0):
        raise ValueError("budgets must be increasing")
    if fit_points < 2 or fit_points > budget_array.size:
        raise ValueError("invalid fit_points")
    rows: list[dict[str, float]] = []
    summaries: dict[float, ComputeSummary] = {}
    for r in r_values:
        risks: list[float] = []
        models: list[int] = []
        blocks_list: list[int] = []
        for budget in budget_array:
            model, blocks, risk = optimize_compute_budget(
                a=a,
                b=b,
                r=float(r),
                budget=int(budget),
                max_model_size=max_model_size,
                grid_size=grid_size,
            )
            risks.append(risk)
            models.append(model)
            blocks_list.append(blocks)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "r": float(r),
                    "compute_budget": int(budget),
                    "optimal_model_size": model,
                    "optimal_innovation_blocks": blocks,
                    "optimal_risk": risk,
                }
            )
        x = np.log(budget_array[-fit_points:].astype(np.float64))
        risk_slope, _ = np.polyfit(x, np.log(np.asarray(risks)[-fit_points:]), 1)
        model_slope, _ = np.polyfit(x, np.log(np.asarray(models, dtype=float)[-fit_points:]), 1)
        block_slope, _ = np.polyfit(x, np.log(np.asarray(blocks_list, dtype=float)[-fit_points:]), 1)
        p = a + float(r)
        summaries[float(r)] = ComputeSummary(
            r=float(r),
            predicted_risk_exponent=(b - 1.0) / (p + 1.0),
            fitted_risk_exponent=float(-risk_slope),
            predicted_model_exponent=1.0 / (p + 1.0),
            fitted_model_exponent=float(model_slope),
            predicted_innovation_exponent=p / (p + 1.0),
            fitted_innovation_exponent=float(block_slope),
            fit_start_budget=int(budget_array[-fit_points]),
            fit_end_budget=int(budget_array[-1]),
        )
    return rows, summaries


def _write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
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
    trace_iat_target: float,
    model_size: int,
    horizon_powers: Sequence[int],
    trials: int,
    raw_fit_points: int,
    compute_r_values: Sequence[float],
    compute_powers: Sequence[int],
    compute_max_model_size: int,
    compute_grid_size: int,
    compute_fit_points: int,
    seed: int,
    output_dir: Path,
) -> tuple[MatchedESSSummary, dict[float, ComputeSummary]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = [2**power for power in horizon_powers]
    matched_rows, matched_summary = simulate_matched_ess(
        a=a,
        b=b,
        r=r,
        trace_iat_target=trace_iat_target,
        model_size=model_size,
        horizons=horizons,
        trials=trials,
        fit_points=raw_fit_points,
        seed=seed,
    )
    budgets = [2**power for power in compute_powers]
    compute_rows, compute_summaries = run_compute_experiment(
        a=a,
        b=b,
        r_values=compute_r_values,
        budgets=budgets,
        max_model_size=compute_max_model_size,
        grid_size=compute_grid_size,
        fit_points=compute_fit_points,
    )
    _write_csv(output_dir / "matched_ess.csv", matched_rows)
    _write_csv(output_dir / "compute_optimal.csv", compute_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "matched_ess": asdict(matched_summary),
                "compute": {
                    str(r_value): asdict(summary)
                    for r_value, summary in compute_summaries.items()
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for name in ["uniform", "aligned"]:
        subset = [row for row in matched_rows if row["profile"] == name]
        axes[0].errorbar(
            [row["raw_horizon"] for row in subset],
            [row["mean_risk"] for row in subset],
            yerr=[row["stderr"] for row in subset],
            marker="o" if name == "uniform" else "s",
            markersize=3,
            linewidth=1.1,
            capsize=2,
            label=name,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("raw horizon $N$")
    axes[0].set_ylabel("excess risk")
    axes[0].set_title("Same global IAT, different slopes")
    axes[0].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].legend(frameon=False)

    for r_value in compute_r_values:
        subset = [row for row in compute_rows if row["r"] == float(r_value)]
        summary = compute_summaries[float(r_value)]
        axes[1].loglog(
            [row["compute_budget"] for row in subset],
            [row["optimal_risk"] for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=(
                rf"$r={r_value:g}$: fit $-{summary.fitted_risk_exponent:.3f}$, "
                rf"pred. $-{summary.predicted_risk_exponent:.3f}$"
            ),
        )
    axes[1].set_xlabel("dense compute budget $C=MB$")
    axes[1].set_ylabel("minimum exact risk")
    axes[1].set_title("Persistence shifts compute-optimal scaling")
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "matched_ess_compute.pdf", bbox_inches="tight")
    plt.close(fig)
    return matched_summary, compute_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.8)
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--trace-iat-target", type=float, default=15.0)
    parser.add_argument("--model-size", type=int, default=65_536)
    parser.add_argument(
        "--horizon-powers", type=int, nargs="+", default=list(range(8, 19))
    )
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--raw-fit-points", type=int, default=6)
    parser.add_argument(
        "--compute-r-values", type=float, nargs="+", default=[0.0, 0.4, 0.8]
    )
    parser.add_argument(
        "--compute-powers", type=int, nargs="+", default=list(range(12, 31))
    )
    parser.add_argument("--compute-max-model-size", type=int, default=8192)
    parser.add_argument("--compute-grid-size", type=int, default=300)
    parser.add_argument("--compute-fit-points", type=int, default=7)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/matched_ess_compute")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matched, compute = run_experiment(
        a=args.a,
        b=args.b,
        r=args.r,
        trace_iat_target=args.trace_iat_target,
        model_size=args.model_size,
        horizon_powers=args.horizon_powers,
        trials=args.trials,
        raw_fit_points=args.raw_fit_points,
        compute_r_values=args.compute_r_values,
        compute_powers=args.compute_powers,
        compute_max_model_size=args.compute_max_model_size,
        compute_grid_size=args.compute_grid_size,
        compute_fit_points=args.compute_fit_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(
        "matched IAT: uniform fit "
        f"{matched.uniform_fitted_exponent:.6f} vs "
        f"{matched.uniform_predicted_exponent:.6f}; aligned fit "
        f"{matched.aligned_fitted_exponent:.6f} vs "
        f"{matched.aligned_predicted_exponent:.6f}"
    )
    for r_value, summary in compute.items():
        print(
            f"compute r={r_value:g}: risk fit={summary.fitted_risk_exponent:.6f} "
            f"pred={summary.predicted_risk_exponent:.6f}; "
            f"M fit={summary.fitted_model_exponent:.6f}; "
            f"B fit={summary.fitted_innovation_exponent:.6f}"
        )


if __name__ == "__main__":
    main()
