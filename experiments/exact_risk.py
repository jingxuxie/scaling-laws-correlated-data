#!/usr/bin/env python3
"""Exact learning curves for spectral renewal regression.

The model is indexed by renewal blocks.  A block chooses spectral mode j with
q_j proportional to lambda_j / tau_j, persists for tau_j raw observations,
and repeatedly observes the same signed coordinate.  The time-marginal
covariance spectrum is lambda_j, while the rate of statistically new arrivals
is q_j.  In the noiseless setting the Bayes/memorization risk is available in
closed form:

    R(B, M) = sum_{j>M} s_j + sum_{j<=M} s_j (1-q_j)^B.

This script evaluates the exact finite-dimensional curve, compares fitted and
predicted exponents, and produces the pilot figures used by the paper draft.
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
class CurveSummary:
    a: float
    b: float
    r: float
    p: float
    model_size: int
    predicted_slope: float
    fitted_slope: float
    fitted_intercept: float
    fit_start_block: int
    fit_end_block: int
    mean_block_length: float


def _validate(a: float, b: float, r: float, model_size: int) -> None:
    if a <= 1:
        raise ValueError("a must exceed 1 so that the covariance trace is finite")
    if b <= 1:
        raise ValueError("b must exceed 1 so that total target energy is finite")
    if r < 0:
        raise ValueError("r must be nonnegative")
    if model_size < 2:
        raise ValueError("model_size must be at least 2")


def power_law_sequences(
    a: float, b: float, r: float, model_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return lambda_j, q_j, s_j for j=1,...,M and E[block length]."""
    _validate(a, b, r, model_size)
    j = np.arange(1, model_size + 1, dtype=np.float64)
    lam = np.power(j, -a) / zeta(a, 1.0)
    p = a + r
    q = np.power(j, -p) / zeta(p, 1.0)
    s = np.power(j, -b) / zeta(b, 1.0)
    mean_block_length = float(zeta(a, 1.0) / zeta(p, 1.0))
    return lam, q, s, mean_block_length


def exact_noiseless_risk(
    blocks: int,
    a: float,
    b: float,
    r: float,
    model_size: int,
) -> tuple[float, float, float]:
    """Compute exact approximation, unseen-mode, and total risk.

    The infinite target tail is evaluated using the Hurwitz zeta function.
    """
    if blocks < 0:
        raise ValueError("blocks must be nonnegative")
    _, q, s, _ = power_law_sequences(a, b, r, model_size)
    tail = float(zeta(b, float(model_size + 1)) / zeta(b, 1.0))
    if blocks == 0:
        unseen = float(np.sum(s))
    else:
        log_survival = blocks * np.log1p(-q)
        unseen = float(np.dot(s, np.exp(log_survival)))
    return tail, unseen, tail + unseen


def asymptotic_data_constant(a: float, b: float, r: float) -> float:
    """Leading constant for the infinite-model noiseless data term.

    If q_j ~ c_q j^{-p} and s_j ~ c_s j^{-b}, then

      sum_j s_j exp(-B q_j)
      ~ (c_s/p) (c_q B)^((1-b)/p) Gamma((b-1)/p).
    """
    p = a + r
    c_q = 1.0 / float(zeta(p, 1.0))
    c_s = 1.0 / float(zeta(b, 1.0))
    return (
        c_s
        / p
        * math.pow(c_q, (1.0 - b) / p)
        * math.gamma((b - 1.0) / p)
    )


def fit_log_slope(
    blocks: np.ndarray, risks: np.ndarray, fit_points: int
) -> tuple[float, float, int, int]:
    if fit_points < 2 or fit_points > len(blocks):
        raise ValueError("fit_points must lie between 2 and the curve length")
    x = np.log(blocks[-fit_points:].astype(np.float64))
    y = np.log(risks[-fit_points:])
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept), int(blocks[-fit_points]), int(blocks[-1])


def evaluate_curve(
    a: float,
    b: float,
    r: float,
    model_size: int,
    block_powers: Sequence[int],
    fit_points: int,
) -> tuple[list[dict[str, float]], CurveSummary]:
    p = a + r
    _, _, _, mean_block_length = power_law_sequences(a, b, r, model_size)
    blocks = np.asarray([2**k for k in block_powers], dtype=np.int64)
    rows: list[dict[str, float]] = []
    total_risks: list[float] = []
    constant = asymptotic_data_constant(a, b, r)
    exponent = (b - 1.0) / p

    for B in blocks:
        approximation, unseen, total = exact_noiseless_risk(
            int(B), a, b, r, model_size
        )
        asymptotic = constant * math.pow(float(B), -exponent)
        rows.append(
            {
                "a": a,
                "b": b,
                "r": r,
                "p": p,
                "blocks": int(B),
                "expected_raw_samples": float(B) * mean_block_length,
                "model_size": model_size,
                "approximation_risk": approximation,
                "unseen_mode_risk": unseen,
                "total_risk": total,
                "asymptotic_data_risk": asymptotic,
                "predicted_slope": -exponent,
            }
        )
        total_risks.append(total)

    fitted_slope, fitted_intercept, fit_start, fit_end = fit_log_slope(
        blocks, np.asarray(total_risks), fit_points
    )
    summary = CurveSummary(
        a=a,
        b=b,
        r=r,
        p=p,
        model_size=model_size,
        predicted_slope=-exponent,
        fitted_slope=fitted_slope,
        fitted_intercept=fitted_intercept,
        fit_start_block=fit_start,
        fit_end_block=fit_end,
        mean_block_length=mean_block_length,
    )
    return rows, summary


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(
    output_path: Path,
    all_rows: dict[float, list[dict[str, float]]],
    summaries: dict[float, CurveSummary],
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r, rows in sorted(all_rows.items()):
        n = np.asarray([row["expected_raw_samples"] for row in rows])
        risk = np.asarray([row["total_risk"] for row in rows])
        predicted = summaries[r].predicted_slope
        fitted = summaries[r].fitted_slope
        ax.loglog(
            n,
            risk,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=rf"$r={r:g}$: predicted {predicted:.3f}, fit {fitted:.3f}",
        )
    ax.set_xlabel("Expected raw observations")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("Mode-dependent persistence changes the data exponent")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_collapse(
    output_path: Path,
    a: float,
    b: float,
    r_values: Sequence[float],
    model_sizes: Sequence[int],
    block_powers: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r in r_values:
        p = a + r
        shared_color: str | None = None
        for model_index, model_size in enumerate(model_sizes):
            x_values: list[float] = []
            y_values: list[float] = []
            for k in block_powers:
                B = 2**k
                _, _, risk = exact_noiseless_risk(B, a, b, r, model_size)
                effective_resolution = min(float(model_size), float(B) ** (1.0 / p))
                x_values.append(effective_resolution)
                y_values.append(risk)
            line, = ax.loglog(
                x_values,
                y_values,
                marker=".",
                linewidth=0.9,
                alpha=0.75,
                color=shared_color,
                label=rf"$r={r:g}$" if model_index == 0 else "_nolegend_",
            )
            if shared_color is None:
                shared_color = line.get_color()
    x_ref = np.logspace(0.6, 3.2, 100)
    y_anchor = 0.35 * np.power(x_ref / x_ref[0], -(b - 1.0))
    ax.loglog(x_ref, y_anchor, linestyle="--", linewidth=1.5, label=rf"slope $-(b-1)={-(b-1):.2f}$")
    ax.set_xlabel(r"Effective spectral resolution $\min\{M,B^{1/(a+r)}\}$")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("Model-data phase collapse")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.8)
    parser.add_argument(
        "--r-values", type=float, nargs="+", default=[0.0, 0.4, 0.8]
    )
    parser.add_argument("--model-size", type=int, default=262_144)
    parser.add_argument("--model-sizes-collapse", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--min-power", type=int, default=6)
    parser.add_argument("--max-power", type=int, default=25)
    parser.add_argument("--fit-points", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/exact_pilot")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_powers = list(range(args.min_power, args.max_power + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[float, list[dict[str, float]]] = {}
    summaries: dict[float, CurveSummary] = {}
    combined_rows: list[dict[str, float]] = []

    for r in args.r_values:
        rows, summary = evaluate_curve(
            args.a,
            args.b,
            r,
            args.model_size,
            block_powers,
            args.fit_points,
        )
        all_rows[r] = rows
        summaries[r] = summary
        combined_rows.extend(rows)

    write_csv(args.output_dir / "learning_curves.csv", combined_rows)
    with (args.output_dir / "slope_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {str(r): asdict(summary) for r, summary in summaries.items()},
            handle,
            indent=2,
            sort_keys=True,
        )

    plot_learning_curves(
        args.output_dir / "learning_curves.pdf", all_rows, summaries
    )
    plot_phase_collapse(
        args.output_dir / "phase_collapse.pdf",
        args.a,
        args.b,
        args.r_values,
        args.model_sizes_collapse,
        list(range(args.min_power, min(args.max_power, 22) + 1)),
    )

    for r, summary in sorted(summaries.items()):
        print(
            f"r={r:g}: predicted slope={summary.predicted_slope:.6f}, "
            f"fitted slope={summary.fitted_slope:.6f}, "
            f"fit B=[{summary.fit_start_block}, {summary.fit_end_block}]"
        )


if __name__ == "__main__":
    main()
