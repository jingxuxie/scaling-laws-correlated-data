#!/usr/bin/env python3
"""Real sequential-regression validation on appliance energy use.

The experiment uses the UCI Appliances Energy Prediction time series.  It
compares ridge learning curves from contiguous training windows with curves
from random subsets of the same chronological training pool.  A spectral
proxy is built from PCA eigenvalues, target energy, and mode-wise integrated
autocorrelation times:

    G_persist(N;c) = sum_j s_j exp(-c N lambda_j/tau_j).

The spatial-only comparator replaces lambda_j/tau_j by lambda_j.  Both fit
only a scale c and amplitude on small training sizes; their out-of-range log
RMSE is measured on larger sizes.  This is an empirical diagnostic rather
than a direct theorem test, because the real covariates are dense and the
distribution is not an exact renewal model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

# Embed TrueType fonts in PDFs so AAAI preflight does not report Type 3 fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

DATA_URL = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)


@dataclass(frozen=True)
class ProxyFit:
    name: str
    amplitude: float
    scale: float
    train_log_rmse: float
    extrapolation_log_rmse: float


@dataclass(frozen=True)
class RealDataSummary:
    dataset_rows: int
    train_rows: int
    test_rows: int
    feature_dimension: int
    pca_dimension: int
    ridge_alpha: float
    full_data_test_mse: float
    median_mode_persistence: float
    max_mode_persistence: float
    contiguous_spatial_log_rmse: float
    contiguous_persistence_log_rmse: float
    random_spatial_log_rmse: float
    random_persistence_log_rmse: float
    contiguous_persistence_improvement: float
    fit_sizes: list[int]
    extrapolation_sizes: list[int]


def download_dataset(path: Path, url: str = DATA_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def build_features(
    frame: pd.DataFrame,
    *,
    lags: Sequence[int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct current, lagged, target-history, and calendar predictors."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], utc=False)
    target = data["Appliances"].astype(float)
    excluded = {"date", "Appliances", "rv1", "rv2"}
    base_columns = [column for column in data.columns if column not in excluded]
    features = data[base_columns].astype(float).copy()

    timestamps = data["date"]
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    day_of_week = timestamps.dt.dayofweek
    day_of_year = timestamps.dt.dayofyear
    features["time_sin_day"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    features["time_cos_day"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    features["time_sin_week"] = np.sin(2 * np.pi * day_of_week / 7)
    features["time_cos_week"] = np.cos(2 * np.pi * day_of_week / 7)
    features["time_sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["time_cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Lag all physical predictors and the past target.  At 10-minute cadence,
    # the defaults span 10 minutes through one day.
    for lag in lags:
        shifted = data[base_columns].shift(lag)
        shifted.columns = [f"{column}_lag{lag}" for column in base_columns]
        features = pd.concat([features, shifted], axis=1)
        features[f"Appliances_lag{lag}"] = target.shift(lag)

    valid = features.notna().all(axis=1)
    return features.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True)


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0, ddof=0)
    scale[scale < 1e-10] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def integrated_autocorrelation_times(
    scores: np.ndarray,
    *,
    max_lag: int,
) -> np.ndarray:
    """Estimate mode-wise IAT using Geyer's initial positive sequence."""
    if scores.ndim != 2 or scores.shape[0] < max_lag + 2:
        raise ValueError("scores are too short for requested max_lag")
    centered = scores - np.mean(scores, axis=0, keepdims=True)
    n = centered.shape[0]
    nfft = 1 << int(math.ceil(math.log2(2 * n - 1)))
    spectrum = np.fft.rfft(centered, n=nfft, axis=0)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, axis=0)
    autocov = autocov[: max_lag + 1]
    denominator = (n - np.arange(max_lag + 1, dtype=np.float64))[:, None]
    autocov = autocov / denominator
    variance = np.maximum(autocov[0], 1e-12)
    rho = autocov / variance[None, :]

    tau = np.ones(scores.shape[1], dtype=np.float64)
    for component in range(scores.shape[1]):
        positive_sum = 0.0
        lag = 1
        while lag <= max_lag:
            pair = rho[lag, component]
            if lag + 1 <= max_lag:
                pair += rho[lag + 1, component]
            if pair <= 0:
                break
            positive_sum += pair
            lag += 2
        tau[component] = 1.0 + 2.0 * positive_sum
    return np.clip(tau, 1.0, float(2 * max_lag + 1))


def choose_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Sequence[float],
) -> float:
    split = int(0.8 * x.shape[0])
    x_fit, x_validation = x[:split], x[split:]
    y_fit, y_validation = y[:split], y[split:]
    best_alpha = float(alphas[0])
    best_mse = math.inf
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x_fit, y_fit)
        mse = mean_squared_error(y_validation, model.predict(x_validation))
        if mse < best_mse:
            best_mse = float(mse)
            best_alpha = float(alpha)
    return best_alpha


def learning_curves(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sizes: Sequence[int],
    alpha: float,
    replicates: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n_train = x_train.shape[0]
    for size in sizes:
        if size >= n_train:
            raise ValueError("each learning-curve size must be below train size")
        contiguous_mse = np.empty(replicates, dtype=np.float64)
        random_mse = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            start = int(rng.integers(0, n_train - size + 1))
            contiguous_index = np.arange(start, start + size)
            random_index = rng.choice(n_train, size=size, replace=False)
            contiguous_model = Ridge(alpha=alpha, fit_intercept=True)
            random_model = Ridge(alpha=alpha, fit_intercept=True)
            contiguous_model.fit(x_train[contiguous_index], y_train[contiguous_index])
            random_model.fit(x_train[random_index], y_train[random_index])
            contiguous_mse[replicate] = mean_squared_error(
                y_test, contiguous_model.predict(x_test)
            )
            random_mse[replicate] = mean_squared_error(
                y_test, random_model.predict(x_test)
            )
        rows.append(
            {
                "size": int(size),
                "replicates": replicates,
                "contiguous_mse_mean": float(np.mean(contiguous_mse)),
                "contiguous_mse_stderr": float(
                    np.std(contiguous_mse, ddof=1) / math.sqrt(replicates)
                ),
                "random_mse_mean": float(np.mean(random_mse)),
                "random_mse_stderr": float(
                    np.std(random_mse, ddof=1) / math.sqrt(replicates)
                ),
            }
        )
    return rows


def proxy_curve(
    sizes: np.ndarray,
    rate: np.ndarray,
    energy: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.asarray(
        [np.dot(energy, np.exp(-scale * float(size) * rate)) for size in sizes],
        dtype=np.float64,
    )


def fit_proxy(
    *,
    name: str,
    sizes: np.ndarray,
    excess: np.ndarray,
    rate: np.ndarray,
    energy: np.ndarray,
    fit_count: int,
) -> tuple[ProxyFit, np.ndarray]:
    if fit_count < 2 or fit_count >= sizes.size:
        raise ValueError("fit_count must leave extrapolation points")
    excess = np.maximum(excess, 1e-9)
    best: tuple[float, float, float] | None = None
    best_prediction: np.ndarray | None = None
    for scale in np.logspace(-4, 5, 500):
        unscaled = np.maximum(proxy_curve(sizes, rate, energy, scale), 1e-300)
        log_amplitude = float(
            np.mean(np.log(excess[:fit_count]) - np.log(unscaled[:fit_count]))
        )
        prediction = np.exp(log_amplitude) * unscaled
        train_rmse = float(
            np.sqrt(
                np.mean(
                    (np.log(prediction[:fit_count]) - np.log(excess[:fit_count])) ** 2
                )
            )
        )
        if best is None or train_rmse < best[0]:
            best = (train_rmse, float(scale), float(np.exp(log_amplitude)))
            best_prediction = prediction
    assert best is not None and best_prediction is not None
    extrapolation_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.log(best_prediction[fit_count:])
                    - np.log(excess[fit_count:])
                )
                ** 2
            )
        )
    )
    fit = ProxyFit(
        name=name,
        amplitude=best[2],
        scale=best[1],
        train_log_rmse=best[0],
        extrapolation_log_rmse=extrapolation_rmse,
    )
    return fit, best_prediction


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    data_path: Path,
    output_dir: Path,
    lags: Sequence[int],
    sizes: Sequence[int],
    replicates: int,
    test_fraction: float,
    max_pca_components: int,
    max_autocorrelation_lag: int,
    proxy_fit_count: int,
    seed: int,
) -> RealDataSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(data_path)
    frame = pd.read_csv(data_path)
    features, target = build_features(frame, lags=lags)
    split = int((1.0 - test_fraction) * len(features))
    x_train_raw = features.iloc[:split].to_numpy(dtype=np.float64)
    x_test_raw = features.iloc[split:].to_numpy(dtype=np.float64)
    y_train = target.iloc[:split].to_numpy(dtype=np.float64)
    y_test = target.iloc[split:].to_numpy(dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train_raw, x_test_raw)

    pca_dimension = min(max_pca_components, x_train.shape[1], x_train.shape[0] - 1)
    pca = PCA(n_components=pca_dimension, svd_solver="full", random_state=seed)
    z_train = pca.fit_transform(x_train)
    z_test = pca.transform(x_test)

    alpha = choose_ridge_alpha(
        z_train,
        y_train,
        alphas=np.logspace(-3, 4, 15),
    )
    full_model = Ridge(alpha=alpha, fit_intercept=True)
    full_model.fit(z_train, y_train)
    full_mse = float(mean_squared_error(y_test, full_model.predict(z_test)))

    curve_rows = learning_curves(
        z_train,
        y_train,
        z_test,
        y_test,
        sizes=sizes,
        alpha=alpha,
        replicates=replicates,
        seed=seed,
    )
    sizes_array = np.asarray([row["size"] for row in curve_rows], dtype=np.int64)
    contiguous_mse = np.asarray(
        [row["contiguous_mse_mean"] for row in curve_rows], dtype=np.float64
    )
    random_mse = np.asarray(
        [row["random_mse_mean"] for row in curve_rows], dtype=np.float64
    )
    contiguous_excess = np.maximum(contiguous_mse - full_mse, 1e-9)
    random_excess = np.maximum(random_mse - full_mse, 1e-9)

    eigenvalues = np.maximum(pca.explained_variance_, 1e-15)
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    theta = np.asarray(full_model.coef_, dtype=np.float64)
    energy = eigenvalues * theta**2
    if np.sum(energy) <= 0:
        raise RuntimeError("estimated target energy is zero")
    energy = energy / np.sum(energy)
    persistence = integrated_autocorrelation_times(
        z_train, max_lag=max_autocorrelation_lag
    )
    spatial_rate = eigenvalues / np.max(eigenvalues)
    persistent_rate = eigenvalues / persistence
    persistent_rate = persistent_rate / np.max(persistent_rate)

    fits: dict[str, ProxyFit] = {}
    predictions: dict[str, np.ndarray] = {}
    for sample_type, excess in [
        ("contiguous", contiguous_excess),
        ("random", random_excess),
    ]:
        for rate_name, rate in [
            ("spatial", spatial_rate),
            ("persistence", persistent_rate),
        ]:
            name = f"{sample_type}_{rate_name}"
            fit, prediction = fit_proxy(
                name=name,
                sizes=sizes_array,
                excess=excess,
                rate=rate,
                energy=energy,
                fit_count=proxy_fit_count,
            )
            fits[name] = fit
            predictions[name] = prediction

    for index, row in enumerate(curve_rows):
        row["full_data_test_mse"] = full_mse
        row["contiguous_excess_mse"] = float(contiguous_excess[index])
        row["random_excess_mse"] = float(random_excess[index])
        row["contiguous_spatial_prediction"] = float(
            predictions["contiguous_spatial"][index]
        )
        row["contiguous_persistence_prediction"] = float(
            predictions["contiguous_persistence"][index]
        )
        row["random_spatial_prediction"] = float(
            predictions["random_spatial"][index]
        )
        row["random_persistence_prediction"] = float(
            predictions["random_persistence"][index]
        )
    write_csv(output_dir / "real_learning_curves.csv", curve_rows)

    spectral_rows = [
        {
            "component": int(index + 1),
            "eigenvalue": float(eigenvalues[index]),
            "target_energy": float(energy[index]),
            "integrated_autocorrelation_time": float(persistence[index]),
            "persistence_adjusted_rate": float(persistent_rate[index]),
        }
        for index in range(pca_dimension)
    ]
    write_csv(output_dir / "spectral_diagnostics.csv", spectral_rows)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    contiguous_error = np.asarray(
        [row["contiguous_mse_stderr"] for row in curve_rows]
    )
    random_error = np.asarray([row["random_mse_stderr"] for row in curve_rows])
    ax.errorbar(
        sizes_array,
        contiguous_excess,
        yerr=contiguous_error,
        marker="o",
        markersize=3,
        linewidth=1.2,
        capsize=2,
        label="contiguous windows",
    )
    ax.errorbar(
        sizes_array,
        random_excess,
        yerr=random_error,
        marker="s",
        markersize=3,
        linewidth=1.2,
        capsize=2,
        label="random subsets",
    )
    ax.plot(
        sizes_array,
        predictions["contiguous_spatial"],
        linestyle="--",
        linewidth=1.1,
        label=(
            "contiguous: spatial proxy "
            f"(extrap. RMSE {fits['contiguous_spatial'].extrapolation_log_rmse:.2f})"
        ),
    )
    ax.plot(
        sizes_array,
        predictions["contiguous_persistence"],
        linestyle=":",
        linewidth=1.5,
        label=(
            "contiguous: persistence proxy "
            f"(extrap. RMSE {fits['contiguous_persistence'].extrapolation_log_rmse:.2f})"
        ),
    )
    ax.axvline(
        sizes_array[proxy_fit_count - 1],
        linewidth=0.8,
        linestyle="-.",
        label="end of proxy fit range",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training examples $N$")
    ax.set_ylabel("Test MSE above the full-data ridge floor")
    ax.set_title("Mode-wise persistence improves real learning-curve prediction")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.2)
    fig.tight_layout()
    fig.savefig(output_dir / "real_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    components = np.arange(1, pca_dimension + 1)
    ax.loglog(components, eigenvalues, label="marginal spectrum $\\lambda_j$")
    ax.loglog(
        components,
        persistent_rate / np.sum(persistent_rate),
        label="normalized $\\lambda_j/\\tau_j$",
    )
    ax.set_xlabel("PCA mode")
    ax.set_ylabel("Normalized spectral mass")
    ax.set_title("Marginal and persistence-adjusted spectra in appliance data")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "real_spectral_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = RealDataSummary(
        dataset_rows=len(features),
        train_rows=z_train.shape[0],
        test_rows=z_test.shape[0],
        feature_dimension=x_train.shape[1],
        pca_dimension=pca_dimension,
        ridge_alpha=alpha,
        full_data_test_mse=full_mse,
        median_mode_persistence=float(np.median(persistence)),
        max_mode_persistence=float(np.max(persistence)),
        contiguous_spatial_log_rmse=fits[
            "contiguous_spatial"
        ].extrapolation_log_rmse,
        contiguous_persistence_log_rmse=fits[
            "contiguous_persistence"
        ].extrapolation_log_rmse,
        random_spatial_log_rmse=fits["random_spatial"].extrapolation_log_rmse,
        random_persistence_log_rmse=fits[
            "random_persistence"
        ].extrapolation_log_rmse,
        contiguous_persistence_improvement=(
            fits["contiguous_spatial"].extrapolation_log_rmse
            - fits["contiguous_persistence"].extrapolation_log_rmse
        ),
        fit_sizes=[int(value) for value in sizes_array[:proxy_fit_count]],
        extrapolation_sizes=[int(value) for value in sizes_array[proxy_fit_count:]],
    )
    payload = {
        "summary": asdict(summary),
        "proxy_fits": {name: asdict(fit) for name, fit in fits.items()},
        "data_url": DATA_URL,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/energydata_complete.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/real_sequential")
    )
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 6, 12, 36, 72, 144])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-pca-components", type=int, default=192)
    parser.add_argument("--max-autocorrelation-lag", type=int, default=288)
    parser.add_argument("--proxy-fit-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lags=args.lags,
        sizes=args.sizes,
        replicates=args.replicates,
        test_fraction=args.test_fraction,
        max_pca_components=args.max_pca_components,
        max_autocorrelation_lag=args.max_autocorrelation_lag,
        proxy_fit_count=args.proxy_fit_count,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
