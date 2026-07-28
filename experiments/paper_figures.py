#!/usr/bin/env python3
"""Build compact submission figures from exact formulas and retained outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Embed TrueType fonts in PDFs so AAAI preflight does not report Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
from scipy.special import zeta


def exact_curve(a: float, b: float, r: float, blocks: np.ndarray, m: int) -> np.ndarray:
    j = np.arange(1, m + 1, dtype=np.float64)
    q = j ** (-(a + r)) / zeta(a + r, 1.0)
    s = j ** (-b) / zeta(b, 1.0)
    tail = float(zeta(b, float(m + 1)) / zeta(b, 1.0))
    return np.asarray(
        [tail + np.dot(s, np.exp(float(B) * np.log1p(-q))) for B in blocks]
    )


def phase_points(a: float, b: float, r_values: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolutions: list[float] = []
    risks: list[float] = []
    labels: list[float] = []
    for r in r_values:
        p = a + r
        for log_b in [8, 11, 14, 17, 20]:
            B = 2**log_b
            for m in [16, 32, 64, 128, 256, 512, 1024]:
                risk = exact_curve(a, b, r, np.asarray([B]), m)[0]
                resolutions.append(min(float(m), float(B) ** (1.0 / p)))
                risks.append(risk)
                labels.append(r)
    return np.asarray(resolutions), np.asarray(risks), np.asarray(labels)


def load_dense(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            r = float(row["r"])
            grouped.setdefault(r, []).append(
                (
                    float(row["blocks"]),
                    float(row["exact_coverage_risk"]),
                    float(row["dictionary_min_norm_mean"]),
                )
            )
    out: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for r, rows in grouped.items():
        rows.sort()
        out[r] = tuple(np.asarray(col) for col in zip(*rows, strict=True))  # type: ignore[assignment]
    return out


def load_ar(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["process"], []).append(
                (float(row["n"]), float(row["mean_risk"]))
            )
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rows in grouped.items():
        rows.sort()
        n, risk = zip(*rows, strict=True)
        out[name] = (np.asarray(n), np.asarray(risk))
    return out


def load_real_summary(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["summary"] if "summary" in payload else payload


def build_figures(root: Path) -> None:
    figures = root / "paper" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    a, b = 2.0, 1.8
    blocks = np.asarray([2**k for k in range(6, 23)], dtype=np.int64)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    for r in [0.0, 0.4, 0.8]:
        risk = exact_curve(a, b, r, blocks, 2**17)
        axes[0].loglog(
            blocks,
            risk,
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            label=rf"$r={r:g}$, pred. $-{(b-1)/(a+r):.3f}$",
        )
    axes[0].set_xlabel("innovation blocks $B$")
    axes[0].set_ylabel("excess risk")
    axes[0].set_title("(a) Exponent change")
    axes[0].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=7)

    resolution, risk, r_label = phase_points(a, b, [0.0, 0.4, 0.8])
    for r in [0.0, 0.4, 0.8]:
        mask = r_label == r
        axes[1].loglog(
            resolution[mask], risk[mask], "o", markersize=2.3, label=rf"$r={r:g}$"
        )
    xline = np.logspace(1.1, 3.0, 50)
    anchor = 0.3 * xline[0] ** (b - 1)
    axes[1].loglog(xline, anchor * xline ** (-(b - 1)), "--", linewidth=1.0, label=rf"slope $-{b-1:g}$")
    axes[1].set_xlabel(r"effective resolution $\min\{M,B^{1/(a+r)}\}$")
    axes[1].set_ylabel("excess risk")
    axes[1].set_title("(b) Model-data collapse")
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=7)

    audit_points: list[tuple[str, str, float, float]] = []

    def read_json(path: Path) -> object:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    exact_summary = read_json(root / "results" / "exact_pilot" / "slope_summary.json")
    assert isinstance(exact_summary, dict)
    for item in exact_summary.values():
        assert isinstance(item, dict)
        audit_points.append(("exact block", "o", abs(float(item["predicted_slope"])), abs(float(item["fitted_slope"]))))

    raw_summary = read_json(root / "results" / "raw_horizon_pilot" / "summary.json")
    assert isinstance(raw_summary, dict)
    for item in raw_summary.values():
        assert isinstance(item, dict)
        audit_points.append(("raw horizon", "s", abs(float(item["predicted_slope"])), abs(float(item["fitted_slope"]))))

    noise_summary = read_json(root / "results" / "noise_regimes" / "summary.json")
    assert isinstance(noise_summary, dict)
    for item in noise_summary.values():
        assert isinstance(item, dict)
        audit_points.append(("noisy", "^", float(item["predicted_risk_exponent"]), float(item["fitted_risk_exponent"])))

    dense_summary = read_json(root / "results" / "dense_features" / "summary.json")
    assert isinstance(dense_summary, dict)
    for item in dense_summary.values():
        assert isinstance(item, dict)
        audit_points.append(("dense", "D", float(item["predicted_exponent"]), float(item["dictionary_fitted_exponent"])))

    heavy_summary = read_json(root / "results" / "heavy_tail_horizon" / "summary.json")
    assert isinstance(heavy_summary, dict)
    audit_points.append(("heavy boundary", "v", float(heavy_summary["boundary_predicted_exponent"]), float(heavy_summary["boundary_fitted_exponent"])))
    audit_points.append(("stationary heavy", "P", float(heavy_summary["stationary_predicted_exponent"]), float(heavy_summary["stationary_fitted_exponent"])))

    predicted = np.asarray([point[2] for point in audit_points], dtype=np.float64)
    fitted = np.asarray([point[3] for point in audit_points], dtype=np.float64)
    axes[2].plot([0.25, 0.95], [0.25, 0.95], "--", linewidth=1.0, label="prediction")
    seen_labels: set[str] = set()
    for label, marker, pred, fit in audit_points:
        display_label = label if label not in seen_labels else "_nolegend_"
        seen_labels.add(label)
        axes[2].scatter(pred, fit, s=28, marker=marker, label=display_label)
    axes[2].text(
        0.04,
        0.94,
        rf"max $|\widehat{{\alpha}}-\alpha|={np.max(np.abs(fitted-predicted)):.3f}$",
        transform=axes[2].transAxes,
        va="top",
        fontsize=7,
    )
    axes[2].set_xlim(0.24, 0.95)
    axes[2].set_ylim(0.24, 0.95)
    axes[2].set_xlabel("predicted exponent")
    axes[2].set_ylabel("fitted exponent")
    axes[2].set_title("(c) Exponent audit")
    axes[2].grid(True, linewidth=0.3, alpha=0.5)
    axes[2].legend(frameon=False, fontsize=6.2, loc="lower right")

    fig.tight_layout()
    fig.savefig(figures / "theory_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    dense = load_dense(root / "results" / "dense_features" / "dense_features.csv")
    ar = load_ar(root / "results" / "dense_ar_control" / "dense_ar_control.csv")
    real = load_real_summary(root / "results" / "real_sequential" / "summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    for r, (bvals, exact, ambient) in sorted(dense.items()):
        line, = axes[0].loglog(bvals, exact, linewidth=1.1, label=rf"exact $r={r:g}$")
        axes[0].loglog(bvals, ambient, "o", markersize=3, color=line.get_color(), label=rf"dense min-norm $r={r:g}$")
    axes[0].set_xlabel("innovation blocks $B$")
    axes[0].set_ylabel("excess risk")
    axes[0].set_title("(a) Dense representation")
    axes[0].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=6.5)

    for name, (n, risk_values) in ar.items():
        axes[1].loglog(n, risk_values, marker="o", markersize=2.7, linewidth=1.0, label=name)
    axes[1].set_xlabel("dense Gaussian samples $N$")
    axes[1].set_ylabel("population risk")
    axes[1].set_title("(b) Dense AR negative control")
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=6.5)

    labels = ["contiguous\nspatial", "contiguous\npersistence", "random\nspatial", "random\npersistence"]
    values = [
        real["contiguous_spatial_log_rmse"],
        real["contiguous_persistence_log_rmse"],
        real["random_spatial_log_rmse"],
        real["random_persistence_log_rmse"],
    ]
    axes[2].bar(np.arange(4), values)
    axes[2].set_xticks(np.arange(4), labels, fontsize=7)
    axes[2].set_ylabel("out-of-range log-RMSE")
    axes[2].set_title("(c) Appliance-energy diagnostic")
    axes[2].grid(True, axis="y", linewidth=0.3, alpha=0.5)
    axes[2].annotate(
        f"{100*(values[0]-values[1])/values[0]:.1f}% lower",
        xy=(1, values[1]),
        xytext=(1.35, max(values) * .92),
        arrowprops={"arrowstyle": "->", "linewidth": .8},
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(figures / "validation_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    matched_dir = root / "results" / "matched_ess_compute"
    with (matched_dir / "summary.json").open(encoding="utf-8") as handle:
        matched_payload = json.load(handle)
    matched_summary = matched_payload["matched_ess"]

    matched_rows: dict[str, list[dict[str, float]]] = {"uniform": [], "aligned": []}
    with (matched_dir / "matched_ess.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matched_rows[row["profile"]].append(
                {
                    "raw_horizon": float(row["raw_horizon"]),
                    "mean_risk": float(row["mean_risk"]),
                    "stderr": float(row["stderr"]),
                }
            )

    compute_rows: dict[float, list[dict[str, float]]] = {}
    with (matched_dir / "compute_optimal.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            r_value = float(row["r"])
            compute_rows.setdefault(r_value, []).append(
                {
                    "compute_budget": float(row["compute_budget"]),
                    "optimal_risk": float(row["optimal_risk"]),
                    "optimal_model_size": float(row["optimal_model_size"]),
                    "optimal_innovation_blocks": float(row["optimal_innovation_blocks"]),
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    modes = np.arange(1, 400, dtype=np.float64)
    marginal = modes ** (-a) / zeta(a, 1.0)
    aligned_q = modes ** (-(a + 0.8)) / zeta(a + 0.8, 1.0)
    axes[0].loglog(modes, marginal, linewidth=1.4, label=r"marginal $\lambda_j$ (both streams)")
    axes[0].loglog(modes, marginal, "--", linewidth=1.2, label=r"uniform innovation $q_j$")
    axes[0].loglog(modes, aligned_q, "-.", linewidth=1.2, label=r"aligned innovation $q_j$")
    axes[0].set_xlabel("spectral mode $j$")
    axes[0].set_ylabel("probability mass")
    axes[0].set_title("(a) Marginal vs. innovation spectrum")
    axes[0].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].text(
        0.04,
        0.05,
        rf"both trace IAT $={matched_summary['trace_iat_target']:.0f}$",
        transform=axes[0].transAxes,
        fontsize=7,
    )

    for profile_name, marker in [("uniform", "o"), ("aligned", "s")]:
        rows = matched_rows[profile_name]
        rows.sort(key=lambda row: row["raw_horizon"])
        predicted = matched_summary[f"{profile_name}_predicted_exponent"]
        fitted = matched_summary[f"{profile_name}_fitted_exponent"]
        axes[1].errorbar(
            [row["raw_horizon"] for row in rows],
            [row["mean_risk"] for row in rows],
            yerr=[row["stderr"] for row in rows],
            marker=marker,
            markersize=3,
            linewidth=1.1,
            capsize=2,
            label=(
                rf"{profile_name}: fit $-{fitted:.3f}$, "
                rf"pred. $-{predicted:.3f}$"
            ),
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("raw trajectory length $N$")
    axes[1].set_ylabel("excess risk")
    axes[1].set_title("(b) A scalar ESS misses the slope")
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=6.5)

    for r_value, rows in sorted(compute_rows.items()):
        rows.sort(key=lambda row: row["compute_budget"])
        summary = matched_payload["compute"][str(r_value)]
        axes[2].loglog(
            [row["compute_budget"] for row in rows],
            [row["optimal_risk"] for row in rows],
            marker="o",
            markersize=2.7,
            linewidth=1.1,
            label=(
                rf"$r={r_value:g}$: fit $-{summary['fitted_risk_exponent']:.3f}$, "
                rf"pred. $-{summary['predicted_risk_exponent']:.3f}$"
            ),
        )
    axes[2].set_xlabel(r"dense compute budget $C=MB$")
    axes[2].set_ylabel("minimum exact risk")
    axes[2].set_title("(c) Compute-optimal frontier")
    axes[2].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[2].legend(frameon=False, fontsize=6.2)

    fig.tight_layout()
    fig.savefig(figures / "mechanism_planning.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figures(args.root.resolve())


if __name__ == "__main__":
    main()
