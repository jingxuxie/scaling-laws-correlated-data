#!/usr/bin/env python3
r"""Monte Carlo validation for spectral renewal regression.

Each innovation block selects a coordinate J from a Zipf law
q_j \propto j^{-(a+r)}.  Conditional on J=j, the block has persistence
weight tau_j=j^r and provides a block-averaged label with noise variance
sigma^2/tau_j.  The expanded stream has time occupancy proportional to
q_j tau_j \propto j^{-a}, so every value of r has the same marginal
covariance spectrum while the arrival rate of new coordinates changes.

The script samples block counts, evaluates the conditional prediction risk of
the coordinate-wise averaging estimator, and compares the Monte Carlo mean to
the exact noiseless formula.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

# Embed TrueType fonts in PDFs so AAAI preflight does not report Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
from scipy.special import zeta

from exact_risk import exact_noiseless_risk


def simulate_risk(
    *,
    blocks: int,
    trials: int,
    a: float,
    b: float,
    r: float,
    model_size: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Return one conditional expected risk value per sampled dataset."""
    if blocks < 1 or trials < 1:
        raise ValueError("blocks and trials must be positive")
    if a <= 1 or b <= 1 or r < 0 or model_size < 1 or sigma < 0:
        raise ValueError("invalid model parameters")

    rng = np.random.default_rng(seed)
    j = np.arange(1, model_size + 1, dtype=np.float64)
    lam = np.power(j, -a) / zeta(a, 1.0)
    target_energy = np.power(j, -b) / zeta(b, 1.0)
    persistence = np.power(j, r)
    tail = float(zeta(b, float(model_size + 1)) / zeta(b, 1.0))

    risks = np.empty(trials, dtype=np.float64)
    zipf_exponent = a + r
    for trial in range(trials):
        sampled_modes = rng.zipf(zipf_exponent, size=blocks)
        in_model = sampled_modes[sampled_modes <= model_size]
        counts = np.bincount(in_model, minlength=model_size + 1)[1:]
        unseen = counts == 0
        risk = tail + float(np.sum(target_energy[unseen]))
        if sigma > 0:
            seen = ~unseen
            risk += sigma**2 * float(
                np.sum(lam[seen] / (persistence[seen] * counts[seen]))
            )
        risks[trial] = risk
    return risks


def write_rows(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.8)
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    parser.add_argument("--model-size", type=int, default=16_384)
    parser.add_argument("--block-powers", type=int, nargs="+", default=[6, 8, 10, 12, 14])
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/monte_carlo_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r_index, r in enumerate(args.r_values):
        means: list[float] = []
        exacts: list[float] = []
        block_values: list[int] = []
        for power in args.block_powers:
            blocks = 2**power
            risks = simulate_risk(
                blocks=blocks,
                trials=args.trials,
                a=args.a,
                b=args.b,
                r=r,
                model_size=args.model_size,
                sigma=args.sigma,
                seed=args.seed + 10_000 * r_index + power,
            )
            exact = exact_noiseless_risk(
                blocks, args.a, args.b, r, args.model_size
            )[2]
            mean = float(np.mean(risks))
            stderr = float(np.std(risks, ddof=1) / np.sqrt(args.trials))
            rows.append(
                {
                    "a": args.a,
                    "b": args.b,
                    "r": r,
                    "blocks": blocks,
                    "model_size": args.model_size,
                    "sigma": args.sigma,
                    "trials": args.trials,
                    "monte_carlo_mean": mean,
                    "monte_carlo_stderr": stderr,
                    "exact_noiseless_risk": exact,
                    "relative_error_to_noiseless_exact": (mean - exact) / exact,
                }
            )
            block_values.append(blocks)
            means.append(mean)
            exacts.append(exact)

        ax.loglog(block_values, means, marker="o", linewidth=0, label=rf"MC $r={r:g}$")
        if args.sigma == 0:
            ax.loglog(block_values, exacts, linewidth=1.2, label=rf"exact $r={r:g}$")

    ax.set_xlabel("Innovation blocks $B$")
    ax.set_ylabel("Expected excess risk")
    ax.set_title("Monte Carlo validation of the exact risk identity")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "monte_carlo_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    write_rows(args.output_dir / "monte_carlo_validation.csv", rows)
    summary = {
        "parameters": vars(args) | {"output_dir": str(args.output_dir)},
        "maximum_absolute_relative_error": max(
            abs(row["relative_error_to_noiseless_exact"]) for row in rows
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        "maximum absolute relative error to noiseless exact formula: "
        f"{summary['maximum_absolute_relative_error']:.4f}"
    )


if __name__ == "__main__":
    main()
