#!/usr/bin/env python3
"""Exact noisy scaling regimes and optimal model-size experiment.

The finite-block identity contains h_B(q)=E[1{K>0}/K] for a binomial count.
We evaluate h_B by Gauss--Legendre quadrature and minimize the resulting exact
risk over model size.  The experiment contrasts:

* hard targets b<a+r, where coverage bias controls the optimal risk and the
  data exponent is (b-1)/(a+r);
* smooth targets b>a+r, where noise changes the optimal model size to
  M* ~ (B/sigma^2)^(1/b) and the risk exponent to (b-1)/b.
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
from numpy.polynomial.legendre import leggauss
from scipy.special import zeta


@dataclass(frozen=True)
class RegimeSummary:
    name: str
    a: float
    b: float
    r: float
    p: float
    sigma: float
    predicted_risk_exponent: float
    fitted_risk_exponent: float
    predicted_model_exponent: float
    fitted_model_exponent: float
    fit_start_block: int
    fit_end_block: int


def binomial_reciprocal_expectation(
    blocks: int,
    q: np.ndarray,
    *,
    quadrature_order: int = 80,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Return h_B(q)=E[1{K>0}/K], K~Binomial(B,q).

    Uses
      h_B(q)=int_0^1 ((1-q+qt)^B-(1-q)^B)/t dt.
    Gauss--Legendre nodes avoid the removable singularity at zero.  The
    difference of powers is evaluated with expm1 to retain accuracy for Bq<<1.
    """
    if blocks < 1:
        raise ValueError("blocks must be positive")
    q = np.asarray(q, dtype=np.float64)
    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must lie strictly between zero and one")
    nodes, weights = leggauss(quadrature_order)
    t = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    out = np.empty_like(q)
    for start in range(0, q.size, chunk_size):
        qc = q[start : start + chunk_size, None]
        log_base = blocks * np.log1p(-qc)
        log_tilt = blocks * np.log1p(-qc * (1.0 - t[None, :]))
        log_difference = log_tilt - log_base
        delta = np.empty_like(log_difference)
        stable = log_difference < 40.0
        delta[stable] = (
            np.exp(np.broadcast_to(log_base, log_difference.shape)[stable])
            * np.expm1(log_difference[stable])
        )
        delta[~stable] = (
            np.exp(log_tilt[~stable])
            - np.exp(np.broadcast_to(log_base, log_difference.shape)[~stable])
        )
        out[start : start + chunk_size] = np.sum(
            w[None, :] * delta / t[None, :], axis=1
        )
    return out


def direct_h(blocks: int, q: float) -> float:
    """Small-B reference implementation used by unit tests."""
    probability = (1.0 - q) ** blocks
    total = 0.0
    for k in range(1, blocks + 1):
        probability *= (blocks - k + 1) / k * q / (1.0 - q)
        total += probability / k
    return total


def exact_risk_over_models(
    *,
    blocks: int,
    a: float,
    b: float,
    r: float,
    sigma: float,
    max_model_size: int,
    quadrature_order: int = 80,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return exact finite-block risk for every M=1,...,max_model_size."""
    if a <= 1 or b <= 1 or r < 0 or sigma < 0 or max_model_size < 2:
        raise ValueError("invalid model parameters")
    j = np.arange(1, max_model_size + 1, dtype=np.float64)
    p = a + r
    lam = np.power(j, -a) / zeta(a, 1.0)
    q = np.power(j, -p) / zeta(p, 1.0)
    target = np.power(j, -b) / zeta(b, 1.0)
    tau = np.power(j, r)

    unseen_coordinate = target * np.exp(blocks * np.log1p(-q))
    h = binomial_reciprocal_expectation(
        blocks, q, quadrature_order=quadrature_order
    )
    variance_coordinate = sigma**2 * lam / tau * h
    cumulative_unseen = np.cumsum(unseen_coordinate)
    cumulative_variance = np.cumsum(variance_coordinate)
    model_sizes = np.arange(1, max_model_size + 1, dtype=np.int64)
    approximation = np.asarray(
        [zeta(b, float(m + 1)) / zeta(b, 1.0) for m in model_sizes],
        dtype=np.float64,
    )
    total = approximation + cumulative_unseen + cumulative_variance
    pieces = {
        "approximation": approximation,
        "coverage": cumulative_unseen,
        "variance": cumulative_variance,
    }
    return model_sizes, total, pieces


def fit_exponent(x: np.ndarray, y: np.ndarray, fit_points: int) -> float:
    if fit_points < 2 or fit_points > x.size or np.any(y[-fit_points:] <= 0):
        raise ValueError("invalid exponent fit")
    slope, _ = np.polyfit(
        np.log(x[-fit_points:].astype(np.float64)),
        np.log(y[-fit_points:].astype(np.float64)),
        deg=1,
    )
    return float(slope)


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_regime(
    *,
    name: str,
    selection: str,
    a: float,
    b: float,
    r: float,
    sigma: float,
    block_powers: Sequence[int],
    max_model_size: int,
    fit_points: int,
    quadrature_order: int,
) -> tuple[list[dict[str, float]], RegimeSummary]:
    p = a + r
    blocks_array = np.asarray([2**power for power in block_powers], dtype=np.int64)
    rows: list[dict[str, float]] = []
    optimal_risks: list[float] = []
    optimal_models: list[int] = []
    for blocks in blocks_array:
        models, risks, pieces = exact_risk_over_models(
            blocks=int(blocks),
            a=a,
            b=b,
            r=r,
            sigma=sigma,
            max_model_size=max_model_size,
            quadrature_order=quadrature_order,
        )
        if selection == "frontier":
            requested = int(math.ceil(float(blocks) ** (1.0 / p)))
            index = min(requested, max_model_size) - 1
        elif selection == "optimal":
            index = int(np.argmin(risks))
        else:
            raise ValueError(f"unknown model-selection rule: {selection}")
        m_star = int(models[index])
        risk_star = float(risks[index])
        optimal_models.append(m_star)
        optimal_risks.append(risk_star)
        rows.append(
            {
                "regime": name,
                "selection": selection,
                "a": a,
                "b": b,
                "r": r,
                "p": p,
                "sigma": sigma,
                "blocks": int(blocks),
                "optimal_model_size": m_star,
                "optimal_risk": risk_star,
                "approximation_at_optimum": float(pieces["approximation"][index]),
                "coverage_at_optimum": float(pieces["coverage"][index]),
                "variance_at_optimum": float(pieces["variance"][index]),
            }
        )

    fitted_risk_slope = fit_exponent(
        blocks_array, np.asarray(optimal_risks), fit_points
    )
    fitted_model_slope = fit_exponent(
        blocks_array, np.asarray(optimal_models, dtype=np.float64), fit_points
    )
    if selection == "frontier":
        predicted_risk = (b - 1.0) / p
        predicted_model = 1.0 / p
    else:
        predicted_risk = (b - 1.0) / b
        predicted_model = 1.0 / b
    summary = RegimeSummary(
        name=name,
        a=a,
        b=b,
        r=r,
        p=p,
        sigma=sigma,
        predicted_risk_exponent=predicted_risk,
        fitted_risk_exponent=-fitted_risk_slope,
        predicted_model_exponent=predicted_model,
        fitted_model_exponent=fitted_model_slope,
        fit_start_block=int(blocks_array[-fit_points]),
        fit_end_block=int(blocks_array[-1]),
    )
    return rows, summary


def run_experiment(
    *,
    block_powers: Sequence[int],
    max_model_size: int,
    fit_points: int,
    quadrature_order: int,
    output_dir: Path,
) -> dict[str, RegimeSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        dict(
            name="hard", selection="frontier",
            a=2.0, b=1.8, r=0.8, sigma=0.35
        ),
        dict(
            name="smooth", selection="optimal",
            a=2.0, b=4.0, r=0.8, sigma=0.35
        ),
    ]
    all_rows: list[dict[str, float]] = []
    summaries: dict[str, RegimeSummary] = {}
    for config in configurations:
        rows, summary = run_regime(
            **config,
            block_powers=block_powers,
            max_model_size=max_model_size,
            fit_points=fit_points,
            quadrature_order=quadrature_order,
        )
        all_rows.extend(rows)
        summaries[summary.name] = summary

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for name, summary in summaries.items():
        rows = [row for row in all_rows if row["regime"] == name]
        blocks = np.asarray([row["blocks"] for row in rows])
        risks = np.asarray([row["optimal_risk"] for row in rows])
        ax.loglog(
            blocks,
            risks,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=(
                rf"{name}: predicted $-{summary.predicted_risk_exponent:.3f}$, "
                rf"fit $-{summary.fitted_risk_exponent:.3f}$"
            ),
        )
    ax.set_xlabel("Innovation blocks $B$")
    ax.set_ylabel("Minimum exact excess risk over $M$")
    ax.set_title("Noise separates coverage-limited and variance-limited scaling")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "noise_regimes.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "noise_regimes.csv", all_rows)
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
    parser.add_argument(
        "--block-powers", type=int, nargs="+", default=list(range(7, 23))
    )
    parser.add_argument("--max-model-size", type=int, default=32_768)
    parser.add_argument("--fit-points", type=int, default=6)
    parser.add_argument("--quadrature-order", type=int, default=80)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/noise_regimes")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_experiment(
        block_powers=args.block_powers,
        max_model_size=args.max_model_size,
        fit_points=args.fit_points,
        quadrature_order=args.quadrature_order,
        output_dir=args.output_dir,
    )
    for name, summary in summaries.items():
        print(
            f"{name}: risk exponent predicted={summary.predicted_risk_exponent:.6f}, "
            f"fitted={summary.fitted_risk_exponent:.6f}; "
            f"model exponent predicted={summary.predicted_model_exponent:.6f}, "
            f"fitted={summary.fitted_model_exponent:.6f}"
        )


if __name__ == "__main__":
    main()
