"""
Temperature Analysis and Prediction for India (1901-2021)
Machine Learning Application Project

Author : Mayank Kataria, Roll No. 24570037
         B.Sc. (H) Computer Science, Ramanujan College, University of Delhi

Script version of "Temperature ML Project.ipynb".
Run:  python temperature_ml_project.py
Needs TEMP_ANNUAL_SEASONAL_MEAN.csv in the same folder.
Figures are saved to ./figures and the hold-out metrics to holdout_results.csv.
"""

# # Temperature Analysis and Prediction for India (1901–2021)
# **Machine Learning Application Project**
#
# **Author:** Mayank Kataria · Roll No. 24570037 · B.Sc. (H) Computer Science, Ramanujan College, University of Delhi
#
# ## 1. Problem framing
# **Question.** How fast has India's mean temperature been rising, has the warming accelerated in recent decades, and how well can simple regression models *forecast* future years?
#
# **Task type.** Supervised regression on a time-ordered series with one predictor (`YEAR`). Targets: `ANNUAL` and four seasonal means (`JAN-FEB`, `MAR-MAY`, `JUN-SEP`, `OCT-DEC`), in °C.
#
# **Two different goals are kept separate in this notebook:**
# 1. *Description / inference* (Sections 4, 6, 7): trend and acceleration estimates fitted on all 121 years.
# 2. *Prediction* (Section 5): models are tuned using only the past and judged once on a later, untouched test period, and they must beat a simple baseline.
#
# **Dataset.** `TEMP_ANNUAL_SEASONAL_MEAN.csv`, 121 yearly rows (1901–2021): all-India annual and seasonal mean temperature in °C.
# - **Source:** Open Government Data (OGD) Platform India, resource *"Seasonal and Annual Mean Temperature Series, period 1901–2021"* — <https://data.gov.in/resource/seasonal-and-annual-mean-temperature-series-period-1901-2021>
# - **Accessed:** [ACCESS DATE]
# - **Licence:** data on the OGD Platform is released under the Government Open Data License – India (GODL-India).
#
# **How to run.** Put the CSV in the same folder as this notebook, install `requirements.txt`, and run all cells from top to bottom. Every number and figure below is produced by the code.


# ==========================================================================
# ## 2. Setup and data loading

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- script-mode helpers -------------------------------------------------
import matplotlib
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")            # headless machine: save figures, do not open windows
import matplotlib.pyplot as plt     # (re-import after choosing the backend)

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

pd.set_option("display.width", 220)       # print full tables in the terminal
pd.set_option("display.max_columns", None)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
_fig_no = [0]

def save_and_show():
    """Save the current figure to ./figures and show it (if a display is available)."""
    _fig_no[0] += 1
    plt.savefig(os.path.join(FIG_DIR, f"figure_{_fig_no[0]:02d}.png"), dpi=150, bbox_inches="tight")
    plt.show()

SEED = 42                      # no randomised step is used below; fixed for completeness
np.random.seed(SEED)
pd.set_option("display.precision", 4)

DATA_FILE = "TEMP_ANNUAL_SEASONAL_MEAN.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Put {DATA_FILE} in the same folder as this script.")

raw = pd.read_csv(DATA_FILE)
print("Raw shape:", raw.shape)
print(raw.tail(4))


# ==========================================================================
# ## 3. Data cleaning
# The raw file ends with an empty row and a repeated header row, so the last rows are not numeric. They are removed once, and the result is checked: no missing values, and exactly one row for every year from 1901 to 2021.

temperature_columns = ["ANNUAL", "JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC"]

df = raw.apply(pd.to_numeric, errors="coerce")
n_dropped = int(df.isna().any(axis=1).sum())
df = df.dropna().sort_values("YEAR").reset_index(drop=True)
df["YEAR"] = df["YEAR"].astype(int)

assert df.isna().sum().sum() == 0
assert df["YEAR"].is_unique
assert (df["YEAR"].diff().dropna() == 1).all(), "years are not consecutive"

print(f"Rows dropped (blank / repeated-header rows): {n_dropped}")
print(f"Clean shape: {df.shape}  |  years {df.YEAR.min()}–{df.YEAR.max()}")
print(df.head())


# ==========================================================================
# ## 4. Exploratory data analysis

plt.figure(figsize=(14, 6))
for col in temperature_columns:
    plt.plot(df["YEAR"], df[col], label=col, linewidth=1.5)
plt.xlabel("Year"); plt.ylabel("Mean temperature (°C)")
plt.title("Annual and seasonal mean temperature, India (1901–2021)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); save_and_show()

summary = df[temperature_columns].describe().T
summary = summary.rename(columns={"mean": "Mean", "std": "Std dev", "min": "Min", "max": "Max"})
print("Descriptive statistics")
display(summary.round(3))

corr = df[temperature_columns].corr()
plt.figure(figsize=(8, 5.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation between annual and seasonal temperatures")
plt.tight_layout(); save_and_show()

# ### 4.1 Full-period linear trend (descriptive)
# Ordinary least squares of temperature on time, with HAC (Newey–West) standard errors because yearly climate data are autocorrelated. These R² values are **in-sample** and only describe how much of the variation a straight line explains; they are *not* a measure of forecast skill (see Section 5).

rows = []
for col in temperature_columns:
    X = sm.add_constant((df["YEAR"] - 1901) / 10)          # time in decades
    fit = sm.OLS(df[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    slope, ci = fit.params.iloc[1], fit.conf_int().iloc[1]
    rows.append({"Series": col,
                 "Trend (°C/decade)": slope,
                 "95% CI low": ci.iloc[0], "95% CI high": ci.iloc[1],
                 "HAC p-value": fit.pvalues.iloc[1],
                 "In-sample R²": fit.rsquared})
full_trends = pd.DataFrame(rows)
display(full_trends.round(4))

fig, axes = plt.subplots(3, 2, figsize=(15, 13)); axes = axes.flatten()
for i, col in enumerate(temperature_columns):
    b = np.polyfit(df["YEAR"], df[col], 1)
    axes[i].scatter(df["YEAR"], df[col], s=16, alpha=0.6, label="Observed")
    axes[i].plot(df["YEAR"], np.polyval(b, df["YEAR"]), color="red", lw=2, label="Linear trend")
    axes[i].set_title(col); axes[i].set_xlabel("Year"); axes[i].set_ylabel("°C")
    axes[i].grid(alpha=0.3); axes[i].legend()
fig.delaxes(axes[5]); plt.suptitle("Linear temperature trends (1901–2021)", fontsize=15)
plt.tight_layout(); save_and_show()


# ==========================================================================
# ## 5. Predictive modelling and validation
# **Design (chosen before looking at test results):**
# - **Chronological split.** Train on 1901–2001, hold out the **last 20 years (2002–2021)** as a test set that is used **once**, at the end. No shuffling, because the data are a time series.
# - **Model selection on the training period only**, with `TimeSeriesSplit` (5 expanding-window folds, 10 validation years each). Candidates: Ridge polynomial regression (degree 1–3, several α) and a continuous piecewise-linear model with a slope change at 1981.
# - **Baseline.** The training-period mean (a "no-trend" model), and a plain linear trend. A model is only useful if it beats these on the test period.
# - **Metrics on the test period:** MAE, RMSE and R². R² is computed against the *test-period* mean, so it can be negative when a model is worse than predicting that mean.
#
# The 1981 breakpoint was chosen from domain knowledge and checked for sensitivity in Section 6; it lies inside the training period.

TEST_YEARS = 20
train, test = df.iloc[:-TEST_YEARS], df.iloc[-TEST_YEARS:]
print(f"Train: {train.YEAR.min()}–{train.YEAR.max()} ({len(train)} yrs)   Test: {test.YEAR.min()}–{test.YEAR.max()} ({len(test)} yrs)")

to_decades = lambda d: ((d[["YEAR"]] - 1901) / 10).values
X_train, X_test = to_decades(train), to_decades(test)

BREAK_YEAR = 1981
KNOT = (BREAK_YEAR - 1901) / 10

def add_hinge(X, knot):
    # columns: time, and extra slope after the knot (continuous piecewise-linear model)
    return np.hstack([X, np.maximum(X - knot, 0)])

def make_model(kind, degree=None, alpha=1.0):
    if kind == "poly":
        return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                             StandardScaler(), Ridge(alpha=alpha))
    if kind == "piecewise":
        return make_pipeline(FunctionTransformer(add_hinge, kw_args={"knot": KNOT}),
                             Ridge(alpha=alpha))
    raise ValueError(kind)

ALPHAS = [0.001, 0.1, 1, 10]
candidates = [("poly", d, a) for d in (1, 2, 3) for a in ALPHAS] + \
             [("piecewise", None, a) for a in (0.001, 0.1, 1)]
label = lambda k, d: f"poly deg {d}" if k == "poly" else f"piecewise ({BREAK_YEAR})"

tscv = TimeSeriesSplit(n_splits=5, test_size=10)

cv_rows = []
for col in temperature_columns:
    y_tr = train[col].values
    for kind, deg, alpha in candidates:
        s = cross_validate(make_model(kind, deg, alpha), X_train, y_tr, cv=tscv,
                           scoring="neg_root_mean_squared_error")
        cv_rows.append({"Series": col, "Model": label(kind, deg), "kind": kind,
                        "degree": deg, "alpha": alpha,
                        "CV RMSE": -s["test_score"].mean(),
                        "CV RMSE std": s["test_score"].std()})
cv_table = pd.DataFrame(cv_rows)

# Model selection uses the CV table only (never the test set)
selected = cv_table.loc[cv_table.groupby("Series")["CV RMSE"].idxmin()].reset_index(drop=True)
selected = selected.set_index("Series").loc[temperature_columns].reset_index()
print("Selected by time-series cross-validation on 1901–2001 only:")
display(selected[["Series", "Model", "alpha", "CV RMSE", "CV RMSE std"]].round(4))

# ---- One-time evaluation on the untouched test period ----
def scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)}

test_rows, test_preds = [], {}
for _, r in selected.iterrows():
    col = r["Series"]
    y_tr, y_te = train[col].values, test[col].values

    sel = make_model(r["kind"], None if pd.isna(r["degree"]) else int(r["degree"]), r["alpha"]).fit(X_train, y_tr)
    base = DummyRegressor(strategy="mean").fit(X_train, y_tr)
    lin = make_model("poly", 1, 0.001).fit(X_train, y_tr)

    preds = {f"Selected: {r['Model']}": sel.predict(X_test),
             "Baseline: training mean": base.predict(X_test),
             "Plain linear trend": lin.predict(X_test)}
    test_preds[col] = preds
    for name, p in preds.items():
        test_rows.append({"Series": col, "Model": name, **scores(y_te, p)})

holdout = pd.DataFrame(test_rows)
holdout.to_csv("holdout_results.csv", index=False)
print(f"HELD-OUT TEST RESULTS ({test.YEAR.min()}–{test.YEAR.max()})")
display(holdout.round(3))

fig, axes = plt.subplots(3, 2, figsize=(15, 13)); axes = axes.flatten()
for i, col in enumerate(temperature_columns):
    axes[i].plot(train["YEAR"], train[col], color="lightgray", label="Train (observed)")
    axes[i].plot(test["YEAR"], test[col], "ko-", ms=4, label="Test (observed)")
    for name, p in test_preds[col].items():
        axes[i].plot(test["YEAR"], p, lw=2, label=name)
    axes[i].axvline(test["YEAR"].min() - 0.5, color="black", ls=":")
    axes[i].set_title(col); axes[i].set_xlabel("Year"); axes[i].set_ylabel("°C")
    axes[i].grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.delaxes(axes[5]); plt.suptitle("Forecasts for the held-out years", fontsize=15)
plt.tight_layout(); save_and_show()

# Summary produced by code (nothing typed by hand)
sel_rows = holdout[holdout["Model"].str.startswith("Selected")].set_index("Series")
base_rows = holdout[holdout["Model"].str.startswith("Baseline")].set_index("Series")
beats = (sel_rows["RMSE"] < base_rows["RMSE"]).loc[temperature_columns]
print(f"Selected model beats the training-mean baseline (test RMSE) in {int(beats.sum())} of {len(beats)} series.")
neg = sel_rows.loc[sel_rows["R²"] < 0, :].index.tolist()
print("Series with negative test R² for the selected model:", neg if neg else "none")

# ### 5.1 How to read these results
# - The **training-mean baseline** is poor because temperature has a clear upward trend, so a model that ignores time under-predicts every recent year.
# - A **plain linear trend** also lags behind in the test period when warming has accelerated, so models that allow the slope to change (piecewise, or a higher-degree polynomial) do better for most series. **MAR-MAY is the exception**: no candidate clearly improves on a straight line there, and its test R² is strongly negative, so forecasts for that season are the least reliable.
# - Test R² is measured within a short 20-year window where the variation is small, so it can be low or negative even when MAE and RMSE are small. MAE and RMSE (in °C) are therefore the more informative numbers here; R² is reported for completeness.
# - Only 121 yearly values and a single predictor (`YEAR`) are available. These are **trend-extrapolation** models, not physical climate models, and they should not be used for long-range forecasts.


# ==========================================================================
# ## 6. Has warming accelerated? (inference on the full period)
# A continuous piecewise-linear regression with a slope change at 1981, fitted on all years, with HAC standard errors. This part answers a *descriptive* question and is separate from the forecasting task in Section 5. A sensitivity check moves the breakpoint to 1971 and 1991 to show how much the conclusion depends on that choice.

def piecewise_fit(series, break_year, robust=True):
    t = (df["YEAR"] - 1901) / 10
    X = sm.add_constant(pd.DataFrame({"TIME_DECADE": t,
                                      "POST_BREAK": np.maximum(df["YEAR"] - break_year, 0) / 10}))
    ols = sm.OLS(series, X)
    return ols.fit(cov_type="HAC", cov_kwds={"maxlags": 5}) if robust else ols.fit()

rows = []
for bp in (1971, 1981, 1991):
    for col in temperature_columns:
        m = piecewise_fit(df[col], bp)
        early, change = m.params["TIME_DECADE"], m.params["POST_BREAK"]
        ci = m.conf_int().loc["POST_BREAK"]
        rows.append({"Breakpoint": bp, "Series": col,
                     "Early slope (°C/dec)": early, "Recent slope (°C/dec)": early + change,
                     "Slope change": change, "CI low": ci.iloc[0], "CI high": ci.iloc[1],
                     "p-value": m.pvalues["POST_BREAK"], "In-sample R²": m.rsquared})
accel = pd.DataFrame(rows)
print("Warming-rate comparison (slope change after the breakpoint)")
display(accel.round(4))

fig, axes = plt.subplots(3, 2, figsize=(15, 13)); axes = axes.flatten()
for i, col in enumerate(temperature_columns):
    m = piecewise_fit(df[col], BREAK_YEAR, robust=False)
    axes[i].scatter(df["YEAR"], df[col], s=15, alpha=0.55, label="Observed")
    axes[i].plot(df["YEAR"], m.predict(m.model.exog), color="red", lw=2, label="Piecewise trend")
    axes[i].axvline(BREAK_YEAR, color="black", ls="--", label=f"{BREAK_YEAR} breakpoint")
    axes[i].set_title(col); axes[i].set_xlabel("Year"); axes[i].set_ylabel("°C")
    axes[i].grid(alpha=0.3); axes[i].legend()
fig.delaxes(axes[5]); plt.suptitle("Historical vs recent temperature trends", fontsize=15)
plt.tight_layout(); save_and_show()


# ==========================================================================
# ## 7. Residual diagnostics for the piecewise model

residual_store, rows = {}, []
for col in temperature_columns:
    m = piecewise_fit(df[col], BREAK_YEAR, robust=False)
    res = m.resid.values
    residual_store[col] = res
    lb = acorr_ljungbox(res, lags=[5, 10], return_df=True)
    rows.append({"Series": col, "Residual std": res.std(ddof=1),
                 "Durbin-Watson": durbin_watson(res),
                 "Ljung-Box p (lag 5)": lb.loc[5, "lb_pvalue"],
                 "Ljung-Box p (lag 10)": lb.loc[10, "lb_pvalue"]})
display(pd.DataFrame(rows).round(4))

fig, axes = plt.subplots(3, 2, figsize=(15, 12)); axes = axes.flatten()
for i, col in enumerate(temperature_columns):
    axes[i].scatter(df["YEAR"], residual_store[col], s=18, alpha=0.65)
    axes[i].axhline(0, color="red", ls="--")
    axes[i].set_title(f"{col}: residuals vs year"); axes[i].set_xlabel("Year"); axes[i].set_ylabel("Residual (°C)")
    axes[i].grid(alpha=0.3)
fig.delaxes(axes[5]); plt.tight_layout(); save_and_show()

fig, axes = plt.subplots(3, 2, figsize=(14, 11)); axes = axes.flatten()
for i, col in enumerate(temperature_columns):
    plot_acf(residual_store[col], lags=25, ax=axes[i], alpha=0.05)
    axes[i].set_title(f"{col}: residual ACF"); axes[i].set_xlabel("Lag (years)")
fig.delaxes(axes[5]); plt.tight_layout(); save_and_show()


# ==========================================================================
# ## 8. Conclusions and limitations
# **What the analysis supports (see the tables above for the numbers):**
# - Every series shows a statistically significant upward trend over 1901–2021 (Section 4.1).
# - Allowing the slope to change after the breakpoint gives a higher in-sample R² than a single straight line (compare Section 4.1 with Section 6), and the estimated slope change is positive for every series at the 1971, 1981 and 1991 breakpoints (Section 6).
# - In the 2002–2021 hold-out period, the CV-selected model is compared directly with a no-trend baseline and a plain linear trend (Section 5). The size of the improvement, and the series where R² stays low or negative (notably MAR-MAY), are in the held-out table.
#
# **Limitations.**
# - Only 121 annual values and a single predictor; the models extrapolate a trend and ignore physical drivers such as greenhouse gases, aerosols or monsoon variability.
# - The test set has only 20 points, so test metrics are noisy. Time-series CV folds of 10 years are also small.
# - Residuals of the annual series show some autocorrelation (Section 7), so p-values rely on HAC corrections and should be read with care.
# - The 1981 breakpoint is a modelling choice; the sensitivity check shows how much the results depend on it.
# - The series is a country-average, so regional differences within India are not captured.
#
# **Possible extensions.** Block bootstrap intervals for the slopes, a rolling-origin evaluation with more test windows, and models with additional predictors (e.g. CO₂ concentration, ENSO index).
