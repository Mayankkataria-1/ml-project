"""Reproducible analysis for India's annual/seasonal temperature series (1901–2021).

Run from the repository root:
    python temperature_analysis/run_analysis.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TEMP_ANNUAL_SEASONAL_MEAN.csv"
OUT = Path(__file__).resolve().parent / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TARGETS = ["ANNUAL", "JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC"]
BREAK_YEAR = 1981
TRAIN_END = 2001
TEST_START = 2002
FOLDS = [(1987, 1991), (1992, 1996), (1997, 2001)]

for folder in (TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")
    raw = pd.read_csv(SOURCE)
    clean = raw.apply(pd.to_numeric, errors="coerce").dropna(subset=["YEAR", *TARGETS])
    clean["YEAR"] = clean["YEAR"].astype(int)
    clean = clean.sort_values("YEAR").reset_index(drop=True)
    if clean["YEAR"].duplicated().any():
        raise ValueError("Duplicate years found.")
    expected = list(range(int(clean.YEAR.min()), int(clean.YEAR.max()) + 1))
    if clean.YEAR.tolist() != expected:
        raise ValueError("Year coverage must be consecutive.")
    if len(clean) != 121 or clean.YEAR.min() != 1901 or clean.YEAR.max() != 2021:
        raise ValueError("Expected exactly 121 annual observations from 1901 to 2021.")
    clean.to_csv(TABLES / "clean_temperature.csv", index=False)
    return clean


def make_model(name, alpha=1.0):
    if name == "Linear Regression":
        return LinearRegression()
    if name.startswith("Polynomial Ridge"):
        degree = int(name.rsplit(" ", 1)[-1])
        return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                             StandardScaler(), Ridge(alpha=alpha))
    if name == "Piecewise Ridge (1981)":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    raise ValueError(name)


def features(years, name):
    x = np.asarray(years, dtype=float).reshape(-1, 1)
    if name == "Piecewise Ridge (1981)":
        return np.column_stack([x[:, 0] - 1901, np.maximum(0, x[:, 0] - BREAK_YEAR)])
    return x


def candidates():
    return [
        ("Linear Regression", None),
        ("Polynomial Ridge 2", 0.01),
        ("Polynomial Ridge 3", 0.01),
        ("Polynomial Ridge 2", 1.0),
        ("Polynomial Ridge 3", 1.0),
        ("Piecewise Ridge (1981)", 0.1),
        ("Piecewise Ridge (1981)", 1.0),
        ("Piecewise Ridge (1981)", 10.0),
    ]


def fit_predict(name, alpha, train_years, train_y, pred_years):
    if name == "Linear Regression":
        model = LinearRegression().fit(features(train_years, name), train_y)
        return model.predict(features(pred_years, name))
    model = make_model(name, alpha)
    model.fit(features(train_years, name), train_y)
    return model.predict(features(pred_years, name))


def run_eda(df):
    df[TARGETS].describe().T.to_csv(TABLES / "descriptive_statistics.csv")
    df[TARGETS].corr().to_csv(TABLES / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in TARGETS:
        ax.plot(df.YEAR, df[col], label=col, linewidth=1.5)
    ax.set(title="All-India Annual and Seasonal Mean Temperature (1901–2021)",
           xlabel="Year", ylabel="Temperature (°C)")
    ax.legend(ncol=3)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "temperature_series.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[TARGETS].corr(), annot=True, cmap="coolwarm", center=0,
                fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Between Temperature Series")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    trends = []
    for target in TARGETS:
        x = sm.add_constant(df.YEAR - df.YEAR.min())
        fit = sm.OLS(df[target], x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        ci = fit.conf_int().iloc[1]
        trends.append({
            "Series": target,
            "Trend_C_per_decade": fit.params.iloc[1] * 10,
            "CI_95_low": ci.iloc[0] * 10,
            "CI_95_high": ci.iloc[1] * 10,
            "HAC_p_value": fit.pvalues.iloc[1],
        })
    pd.DataFrame(trends).to_csv(TABLES / "full_period_trends.csv", index=False)
    return pd.DataFrame(trends)


def run_models(df):
    train = df[df.YEAR <= TRAIN_END]
    test = df[df.YEAR >= TEST_START]
    cv_rows, selected_rows, holdout_rows, prediction_rows = [], [], [], []

    for target in TARGETS:
        scores = {}
        for name, alpha in candidates():
            fold_scores = []
            for fold_no, (start, end) in enumerate(FOLDS, 1):
                fold_train = train[train.YEAR < start]
                fold_valid = train[train.YEAR.between(start, end)]
                pred = fit_predict(name, alpha, fold_train.YEAR, fold_train[target], fold_valid.YEAR)
                score = rmse(fold_valid[target], pred)
                fold_scores.append(score)
                cv_rows.append({
                    "Series": target, "Model": name, "Alpha": alpha,
                    "Fold": fold_no, "Validation_Start": start,
                    "Validation_End": end, "RMSE_C": score,
                })
            scores[(name, alpha)] = float(np.mean(fold_scores))

        best = min(scores, key=scores.get)
        name, alpha = best
        selected_rows.append({
            "Series": target, "Model": name, "Alpha": alpha,
            "CV_RMSE_mean_C": scores[best],
        })

        pred = fit_predict(name, alpha, train.YEAR, train[target], test.YEAR)
        holdout_rows.append({
            "Series": target, "Model": name,
            "MAE_C": mean_absolute_error(test[target], pred),
            "RMSE_C": rmse(test[target], pred),
            "R2": r2_score(test[target], pred),
        })
        for year, actual, estimate in zip(test.YEAR, test[target], pred):
            prediction_rows.append({
                "Series": target, "Year": int(year),
                "Actual_C": float(actual), "Predicted_C": float(estimate),
            })

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train.YEAR, train[target], label="Training observations", linewidth=1.4)
        ax.plot(test.YEAR, test[target], label="Test observations", linewidth=1.4)
        ax.plot(test.YEAR, pred, "--", label=f"Selected: {name}", linewidth=1.8)
        ax.axvline(TEST_START, color="grey", linestyle=":", label="Test begins")
        ax.set(title=f"{target}: chronological holdout comparison",
               xlabel="Year", ylabel="Temperature (°C)")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"holdout_{target.lower().replace('-', '_')}.png", dpi=180)
        plt.close(fig)

    pd.DataFrame(cv_rows).to_csv(TABLES / "cross_validation_scores.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(TABLES / "model_selection.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(TABLES / "holdout_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(TABLES / "holdout_predictions.csv", index=False)
    return pd.DataFrame(selected_rows), pd.DataFrame(holdout_rows)


def run_diagnostics(df, selected):
    rows = []
    for target in TARGETS:
        x = sm.add_constant(df.YEAR - df.YEAR.min())
        fit = sm.OLS(df[target], x).fit()
        resid = fit.resid
        lb = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
        rows.append({
            "Series": target,
            "Durbin_Watson": durbin_watson(resid),
            "Ljung_Box_p_lag5": lb.loc[5, "lb_pvalue"],
            "Ljung_Box_p_lag10": lb.loc[10, "lb_pvalue"],
            "Residual_SD_C": np.std(resid, ddof=1),
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.YEAR, resid, linewidth=1)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{target}: linear-trend residuals", xlabel="Year", ylabel="Residual (°C)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"residuals_{target.lower().replace('-', '_')}.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES / "residual_diagnostics.csv", index=False)


def run_projection(df, selected):
    years = np.arange(2022, 2036)
    rows = []
    for row in selected.itertuples(index=False):
        y = df[row.Series]
        pred = fit_predict(row.Model, row.Alpha, df.YEAR, y, years)
        for year, value in zip(years, pred):
            rows.append({
                "Series": row.Series, "Year": int(year),
                "Exploratory_prediction_C": float(value),
                "Selected_model": row.Model,
            })
    projection = pd.DataFrame(rows)
    projection.to_csv(TABLES / "exploratory_projection_2022_2035.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in TARGETS:
        hist = df[["YEAR", target]]
        fut = projection[projection.Series == target]
        ax.plot(hist.YEAR, hist[target], linewidth=1, label=f"{target} historical")
        ax.plot(fut.Year, fut.Exploratory_prediction_C, "--", linewidth=1.5,
                label=f"{target} extrapolation")
    ax.axvline(2021, color="black", linestyle=":")
    ax.set(title="Exploratory model extrapolations (not climate forecasts)",
           xlabel="Year", ylabel="Temperature (°C)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "exploratory_extrapolation.png", dpi=180)
    plt.close(fig)


def write_report(df, trends, selected, metrics):
    lines = [
        "# India Temperature Analysis — Results",
        "",
        f"- Dataset coverage: {int(df.YEAR.min())}–{int(df.YEAR.max())} ({len(df)} annual observations).",
        f"- Training: 1901–{TRAIN_END}; test: {TEST_START}–2021.",
        "- Model selection: expanding-window validation on 1987–1991, 1992–1996 and 1997–2001.",
        "- Predictor: year only; targets: annual and four seasonal temperature series.",
        "",
        "## Full-period trends (°C per decade)",
        "",
        "| Series | Trend | 95% HAC CI | HAC p-value |",
        "|---|---:|---:|---:|",
    ]
    for r in trends.itertuples(index=False):
        lines.append(f"| {r.Series} | {r.Trend_C_per_decade:.4f} | [{r.CI_95_low:.4f}, {r.CI_95_high:.4f}] | {r.HAC_p_value:.4g} |")
    lines += ["", "## CV-selected models", "", "| Series | Model | Alpha | Mean CV RMSE (°C) |",
              "|---|---|---:|---:|"]
    for r in selected.itertuples(index=False):
        alpha = "—" if pd.isna(r.Alpha) else f"{r.Alpha:g}"
        lines.append(f"| {r.Series} | {r.Model} | {alpha} | {r.CV_RMSE_mean_C:.4f} |")
    lines += ["", "## Holdout performance (2002–2021)", "",
              "| Series | Model | MAE (°C) | RMSE (°C) | R² |",
              "|---|---|---:|---:|---:|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.Series} | {r.Model} | {r.MAE_C:.4f} | {r.RMSE_C:.4f} | {r.R2:.4f} |")
    lines += [
        "",
        "## Interpretation cautions",
        "",
        "- Full-period trend estimates are descriptive, not evidence of predictive accuracy.",
        "- Negative test R² means the selected model did not beat the test-period mean benchmark.",
        "- Residual autocorrelation may remain; see the diagnostics table.",
        "- The 1981 breakpoint is prespecified, not automatically detected.",
        "- Extrapolations through 2035 are univariate demonstrations using year only. They are not official forecasts or physical climate projections.",
        "",
    ]
    (OUT / "results_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()
    trends = run_eda(df)
    selected, metrics = run_models(df)
    run_diagnostics(df, selected)
    run_projection(df, selected)
    write_report(df, trends, selected, metrics)
    print(f"Analysis complete. Outputs: {OUT}")
    print(f"Clean observations: {len(df)} | Tables: {len(list(TABLES.glob('*.csv')))} | Figures: {len(list(FIGURES.glob('*.png')))}")


if __name__ == "__main__":
    main()
