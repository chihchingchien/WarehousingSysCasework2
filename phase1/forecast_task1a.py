"""Task 1(a): auditable demand forecasting from the exported SKU-day table.

Run: python forecast_task1a.py --input /path/to/t1_daily_demand.csv
Only pandas, numpy, scipy, statsmodels and matplotlib are needed.
The numbered sections correspond to the steps explained in README.txt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import nbinom, poisson
import statsmodels
import statsmodels.api as sm


MODELS = ["A_weekday_8weeks", "B_weekday_trend", "C_weekday_trend_season"]
DEFAULT_INPUT = Path("/Users/tiaohu/t1_daily_demand_202610140011.csv")


# 1. Load ONLY raw daily demand. Rolling statistics and full-year summaries
#    are not model features: they would require special leakage controls.
def load_daily(path):
    df = pd.read_csv(path, usecols=["sku_id", "demand_date", "demand_units"])
    df["demand_date"] = pd.to_datetime(df["demand_date"], errors="raise")
    df["demand_units"] = pd.to_numeric(df["demand_units"], errors="raise")
    if df.isna().any().any():
        raise ValueError("原始三个字段存在空值；不能把缺失数据自动当成零需求。")
    if df.duplicated(["sku_id", "demand_date"]).any():
        raise ValueError("存在重复 SKU-date，请先检查导出或 SQL JOIN。")
    y = df["demand_units"].to_numpy()
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError("需求必须是有限、非负的整数。")
    dates = pd.date_range(df.demand_date.min(), df.demand_date.max())
    wide = df.pivot(index="demand_date", columns="sku_id", values="demand_units")
    wide = wide.reindex(dates).sort_index(axis=1)
    if wide.isna().any().any():
        raise ValueError("SKU × 日期不完整；先确认缺失的原因，再决定是否补零。")
    if len(wide) != 365 or len(wide.columns) != 2000:
        raise ValueError(f"本案例预期365天、2000 SKU，实际为{wide.shape}。")
    return wide.astype(float)


# 2. Calendar features. All use the fixed start of history; no future demand.
def features(dates, origin, model):
    t = np.asarray((dates - origin).days, dtype=float) / 365.0
    columns = [np.ones(len(dates))]
    names = ["intercept"]
    for day in range(1, 7):  # Monday=0 is the reference day.
        columns.append((dates.dayofweek == day).astype(float))
        names.append(f"weekday_{day}")
    columns.append(t)
    names.append("trend_per_year")
    if model == MODELS[2]:
        columns.extend([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
        names.extend(["annual_sin", "annual_cos"])
    return np.column_stack(columns), names


# 3. Fit a conditional mean, then estimate count dispersion from training only.
#    NB2 is a plug-in predictive distribution, not a jointly fitted NB MLE.
def fit_predict(y, train_dates, pred_dates, origin, model):
    def weekday_baseline():
        recent_y, recent_dates = y[-56:], train_dates[-56:]
        means = np.array([recent_y[recent_dates.dayofweek == k].mean()
                          for k in range(7)])
        return (means[pred_dates.dayofweek], means[recent_dates.dayofweek],
                recent_y, 7, {f"weekday_mean_{k}": float(v)
                             for k, v in enumerate(means)})

    status = "ok"
    effective_model = model
    if model == MODELS[0]:
        mu, fitted, calibration_y, n_params, params = weekday_baseline()
    elif y.sum() == 0:
        mu = np.zeros(len(pred_dates))
        fitted = np.zeros(len(y))
        calibration_y, n_params, params = y, 0, {}
        status, effective_model = "all_zero_training", "zero"
    else:
        x, names = features(train_dates, origin, model)
        xp, _ = features(pred_dates, origin, model)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = sm.GLM(y, x, family=sm.families.Poisson()).fit(maxiter=100)
            mu = result.predict(xp)
            fitted = result.fittedvalues
            if (not result.converged or not np.isfinite(mu).all()
                    or not np.isfinite(fitted).all() or (mu < 0).any()):
                raise ValueError("nonconvergence or invalid fitted values")
            calibration_y, n_params = y, x.shape[1]
            params = dict(zip(names, map(float, result.params)))
            if caught:
                status = "warning: " + "; ".join(sorted({str(w.message) for w in caught}))
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                FloatingPointError) as exc:
            # An explicit, recorded fallback, not a silently missing SKU.
            mu, fitted, calibration_y, n_params, params = weekday_baseline()
            status = f"fallback: {type(exc).__name__}: {exc}"
            effective_model = MODELS[0]

    n = len(calibration_y)
    # Method-of-moments: E[(Y-mu)^2] = mu + alpha*mu^2.
    # Approximate residual degrees-of-freedom correction, then truncate at zero.
    squared_error = np.square(calibration_y - fitted).sum()
    alpha = max(0.0, (squared_error * n / max(n - n_params, 1) - fitted.sum())
                / max(np.square(fitted).sum(), 1e-12))
    residuals = (calibration_y - fitted) / np.sqrt(
        np.maximum(fitted + alpha * fitted**2, 1e-12))
    lag1 = (float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            if np.std(residuals[:-1]) > 0 and np.std(residuals[1:]) > 0 else np.nan)
    meta = {"effective_model": effective_model, "fit_status": status,
            "alpha": alpha, "residual_lag1": lag1, **params}
    return np.asarray(mu), alpha, meta


# 4. Time-varying standard deviation and DISCRETE predictive quantiles.
#    These describe daily demand, not uncertainty intervals for its mean.
def distribution(mu, alpha):
    std = np.sqrt(mu + alpha * mu**2)
    qs = np.array([0.50, 0.90, 0.99])[:, None]
    if alpha < 1e-8:
        quantiles = poisson.ppf(qs, mu[None, :])
    else:
        size = 1.0 / alpha
        probability = size / (size + mu)
        quantiles = nbinom.ppf(qs, size, probability[None, :])
    quantiles[:, mu == 0] = 0
    if not np.isfinite(quantiles).all():
        raise ValueError("预测分位数出现非有限值。")
    return std, quantiles.astype(np.int64)


def metrics(actual, mu, quantiles):
    error = mu - actual
    safe_mu = np.maximum(mu, 1e-12)
    term = safe_mu - actual
    positive = actual > 0
    term[positive] += actual[positive] * np.log(actual[positive] / safe_mu[positive])
    row = {"mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error**2)),
           "bias": error.mean(), "mean_poisson_deviance": 2 * term.mean(),
           "actual_units": actual.sum(), "absolute_error": np.abs(error).sum(),
           "squared_error": np.square(error).sum(), "n_days": len(actual)}
    for j, q in enumerate([0.5, 0.9, 0.99]):
        name = f"p{int(q * 100)}"
        residual = actual - quantiles[j]
        row[f"coverage_{name}"] = np.mean(actual <= quantiles[j])
        row[f"pinball_{name}"] = np.maximum(q * residual, (q - 1) * residual).mean()
    return row


# 5. Fixed-origin validation and test; no random train/test shuffle.
def evaluate(wide, train_end, pred_end, stage):
    train_dates = wide.index[:train_end]
    pred_dates = wide.index[train_end:pred_end]
    rows, detail, diagnostics = [], [], []
    for model in MODELS:
        for index, sku in enumerate(wide.columns):
            y = wide[sku].to_numpy()
            mu, alpha, meta = fit_predict(y[:train_end], train_dates,
                                         pred_dates, wide.index[0], model)
            std, qs = distribution(mu, alpha)
            rows.append({"stage": stage, "model": model, "sku_id": sku,
                         **metrics(y[train_end:pred_end], mu, qs)})
            detail.append(pd.DataFrame({"stage": stage, "model": model,
                "sku_id": sku, "date": pred_dates, "actual": y[train_end:pred_end],
                "pred_mean": mu, "pred_std": std,
                "p50": qs[0], "p90": qs[1], "p99": qs[2]}))
            diagnostics.append({"stage": stage, "model": model, "sku_id": sku, **meta})
            if (index + 1) % 500 == 0:
                print(f"{stage}: {model}, {index + 1}/{len(wide.columns)}", flush=True)
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True), pd.DataFrame(diagnostics)


def summarize(scores):
    rows = []
    for (stage, model), group in scores.groupby(["stage", "model"]):
        row = {"stage": stage, "model": model,
               "mae": group.absolute_error.sum() / group.n_days.sum(),
               "rmse": np.sqrt(group.squared_error.sum() / group.n_days.sum()),
               "wape": group.absolute_error.sum() / group.actual_units.sum(),
               "bias": group.bias.mean(),
               "mean_poisson_deviance": group.mean_poisson_deviance.mean()}
        for name in ["p50", "p90", "p99"]:
            row[f"coverage_{name}"] = group[f"coverage_{name}"].mean()
            row[f"pinball_{name}"] = group[f"pinball_{name}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# 6. Check the three descriptive exports against the raw daily panel.
#    This is optional audit work, never an input to model fitting.
def audit_exports(wide, directory):
    audit = {}
    files = {
        "summary": "_SELECT_sku_id_AVG_demand_units_AS_mean_daily_demand_STDDEV_SAMP_202610140007.csv",
        "monthly": "_SELECT_sku_id_DATE_TRUNC_month_demand_date_date_AS_month_COUNT__202610140010.csv",
        "distribution": "_SELECT_sku_id_demand_units_COUNT_AS_days_observed_COUNT_numeric_202610140009.csv",
    }
    long = wide.rename_axis(index="demand_date", columns="sku_id").stack().rename("demand_units").reset_index()
    for kind, filename in files.items():
        path = directory / filename
        if not path.exists():
            audit[kind] = "not provided; skipped"
            continue
        source = pd.read_csv(path)
        if kind == "summary":
            actual = source.set_index("sku_id").sort_index()
            expected = pd.DataFrame({"mean_daily_demand": wide.mean(),
                "std_daily_demand": wide.std(), "min_daily_demand": wide.min(),
                "max_daily_demand": wide.max(),
                "zero_demand_share": (wide == 0).mean()})
            for q in [0.5, 0.9, 0.99]:
                expected[f"p{int(q*100)}"] = np.quantile(wide, q, axis=0, method="inverted_cdf")
        elif kind == "monthly":
            long["month"] = long.demand_date.dt.to_period("M").dt.to_timestamp()
            expected = long.groupby(["sku_id", "month"]).demand_units.agg(
                observed_days="size", mean_daily_demand="mean", std_daily_demand="std")
            source["month"] = pd.to_datetime(source["month"])
            actual = source.set_index(["sku_id", "month"]).sort_index()
        else:
            expected = long.groupby(["sku_id", "demand_units"]).size().to_frame("days_observed")
            expected["probability"] = expected.days_observed / len(wide)
            actual = source.set_index(["sku_id", "demand_units"]).sort_index()
        expected = expected.sort_index()
        if not actual.index.equals(expected.index):
            raise ValueError(f"{kind} 的索引与每日表不一致。")
        for column in expected:
            if not np.allclose(actual[column], expected[column], rtol=1e-6,
                               atol=1e-8, equal_nan=True):
                raise ValueError(f"{kind}.{column} 与每日需求表不一致。")
        audit[kind] = {"rows": len(source), "matches_daily_panel": True}
    return audit


# 7. A small, reproducible selection of illustrative plots.
def plot_results(wide, test, summary, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "figure.dpi": 130})
    # Choose example products from TRAINING demand, not test performance.
    train_mean = wide.iloc[:253].mean().sort_values(ascending=False)
    example_skus = [train_mean.index[i] for i in [0, 99, 999]]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, sku in zip(axes, example_skus):
        part = test[(test.sku_id == sku) & (test.model == selected)]
        ax.plot(part.date, part.actual, color="#475569", lw=1, label="Actual")
        ax.plot(part.date, part.pred_mean, color="#087e8b", lw=1.8, label="Predicted mean")
        ax.plot(part.date, part.p90, color="#c67d20", lw=1, ls="--", label="Daily p90")
        ax.set(title=f"{sku} | {selected}", ylabel="Units / day")
    axes[0].legend(ncol=3, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "test_examples.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, stage in zip(axes, ["validation", "test"]):
        part = summary[summary.stage == stage].set_index("model").reindex(MODELS)
        ax.bar(["A: Weekday", "B: +Trend", "C: +Season"], part.mean_poisson_deviance,
               color=["#94a3b8", "#3b82a6", "#087e8b"])
        ax.set(title=stage.title(), ylabel="Mean Poisson deviance (lower is better)")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png")
    plt.close(fig)
    return example_skus


# 8. Run the workflow, freeze model selection before examining the final test.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    wide = load_daily(args.input)
    audit = audit_exports(wide, args.input.parent)
    print(f"Validated {wide.shape[1]} SKUs x {wide.shape[0]} days.", flush=True)

    validation_scores, _, validation_fits = evaluate(wide, 253, 309, "validation")
    validation_summary = summarize(validation_scores)
    # Predeclared criterion: average Poisson deviance, a score for the mean.
    # Within 1% of best validation score, choose the simpler family A -> B -> C.
    best = validation_summary.mean_poisson_deviance.min()
    eligible = validation_summary.loc[
        validation_summary.mean_poisson_deviance <= best * 1.01, "model"]
    selected = next(model for model in MODELS if model in eligible.to_list())
    print(f"Frozen selection before final test: {selected}", flush=True)
    validation_scores.to_csv(args.output / "validation_by_sku.csv", index=False)

    test_scores, test_predictions, test_fits = evaluate(wide, 309, 365, "test")
    summary = pd.concat([validation_summary, summarize(test_scores)], ignore_index=True)
    summary["selected_on_validation"] = summary.model == selected
    summary.to_csv(args.output / "model_comparison.csv", index=False)
    test_scores.to_csv(args.output / "test_by_sku.csv", index=False)
    test_predictions.to_csv(args.output / "test_predictions.csv", index=False, float_format="%.8g")

    # Sensitivity: 4-week vs 8-week weekday baseline on the final test.
    # Diagnostic only: do not use this test to reselect the production model.
    sensitivity = []
    for window in [28, 56]:
        train = wide.iloc[309-window:309]
        means = train.groupby(train.index.dayofweek).mean()
        pred = means.loc[wide.index[309:].dayofweek].to_numpy()
        actual = wide.iloc[309:].to_numpy()
        err = pred - actual
        sensitivity.append({"window_days": window, "mae": np.abs(err).mean(),
                            "wape": np.abs(err).sum() / actual.sum()})
    pd.DataFrame(sensitivity).to_csv(args.output / "weekday_window_sensitivity.csv", index=False)

    future_dates = pd.date_range(wide.index[-1] + pd.Timedelta(days=1), periods=args.horizon)
    forecasts, final_fits = [], []
    for sku in wide.columns:
        mu, alpha, meta = fit_predict(wide[sku].to_numpy(), wide.index, future_dates,
                                     wide.index[0], selected)
        std, qs = distribution(mu, alpha)
        forecasts.append(pd.DataFrame({"sku_id": sku, "date": future_dates,
            "model": selected, "effective_model": meta["effective_model"],
            "pred_mean": mu, "pred_std": std, "dispersion_alpha": alpha,
            "p50": qs[0], "p90": qs[1], "p99": qs[2]}))
        final_fits.append({"stage": "full_year_refit", "model": selected,
                           "sku_id": sku, **meta})
    forecast = pd.concat(forecasts, ignore_index=True)
    assert not forecast.duplicated(["sku_id", "date"]).any()
    assert np.isfinite(forecast[["pred_mean", "pred_std", "p50", "p90", "p99"]]).all().all()
    assert (forecast.p50 <= forecast.p90).all() and (forecast.p90 <= forecast.p99).all()
    forecast.to_csv(args.output / "future_daily_forecast.csv", index=False, float_format="%.8g")
    fits = pd.concat([validation_fits, test_fits, pd.DataFrame(final_fits)], ignore_index=True)
    fits.to_csv(args.output / "model_parameters_and_diagnostics.csv", index=False)
    example_skus = [] if args.no_plots else plot_results(wide, test_predictions, summary, selected, args.output)

    manifest = {"source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "sku_count": len(wide.columns), "days": len(wide), "total_units": int(wide.sum().sum()),
        "descriptive_export_audit": audit,
        "training": [str(wide.index[0].date()), str(wide.index[252].date())],
        "validation": [str(wide.index[253].date()), str(wide.index[308].date())],
        "test": [str(wide.index[309].date()), str(wide.index[-1].date())],
        "forecast": [str(future_dates[0].date()), str(future_dates[-1].date())],
        "selected_model": selected, "selection_rule": "Validation mean Poisson deviance; simpler within 1%",
        "forecast_rows": len(forecast), "example_skus": list(example_skus),
        "fit_status_counts": fits.fit_status.value_counts().to_dict(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "randomness": "None: model fitting, selection and quantiles are deterministic.",
        "limitations": ["Only one historical year; annual seasonality and trend are weakly separable.",
            "Forecast horizon is configurable; 56-day tests do not validate a full-year extrapolation.",
            "Plug-in Poisson/NB2 daily distributions exclude parameter and model uncertainty.",
            "Constant alpha per SKU; variance varies with the predicted daily mean.",
            "Marginal daily quantiles are not cumulative-demand or inventory-service targets.",
            "Temporal and cross-SKU dependence is not a joint model; inspect residual_lag1 before Task 2."]}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(forecast):,} future SKU-day rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
