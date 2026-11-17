# Temperature Analysis and Prediction for India (1901–2021)

Machine Learning Application Project — regression analysis of India's annual and seasonal mean temperature.

## What this project does
- Cleans and explores 121 years of annual and seasonal mean temperature (`ANNUAL`, `JAN-FEB`, `MAR-MAY`, `JUN-SEP`, `OCT-DEC`).
- Estimates warming trends (OLS with HAC standard errors) and tests whether warming accelerated after 1981 (piecewise-linear regression, with a breakpoint sensitivity check at 1971 / 1981 / 1991).
- Builds forecasting models (Ridge polynomial regression and a piecewise-linear model) and evaluates them honestly:
  - **Chronological split:** train 1901–2001, test 2002–2021.
  - **Model selection** with `TimeSeriesSplit` cross-validation on the training period only.
  - **Test set used once**, compared against a training-mean baseline and a plain linear trend.
- Checks residuals (Durbin–Watson, Ljung–Box, ACF).

## Repository contents
| File | Description |
|---|---|
| `Temperature ML Project.ipynb` | Main notebook (runs top to bottom, outputs included) |
| `TEMP_ANNUAL_SEASONAL_MEAN.csv` | Dataset (annual and seasonal mean temperature, 1901–2021) |
| `holdout_results.csv` | Hold-out test metrics, written by the notebook |
| `requirements.txt` | Python dependencies |

## How to run
```bash
python -m pip install -r requirements.txt
jupyter notebook "Temperature ML Project.ipynb"
```
Keep the CSV in the same folder as the notebook, then run all cells. In Google Colab the notebook asks you to upload the CSV if it is not found. Tested with Python 3.12.3, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, statsmodels 0.15.0.

## Results (hold-out test period 2002–2021)
These numbers are copied from `holdout_results.csv`, which the notebook generates; re-run the notebook to reproduce them.

| Series | Selected model | MAE (°C) | RMSE (°C) | R² | Baseline RMSE (°C) | Linear-trend RMSE (°C) |
|---|---|---|---|---|---|---|
| ANNUAL | piecewise (1981) | 0.134 | 0.185 | -0.024 | 0.613 | 0.396 |
| JAN-FEB | piecewise (1981) | 0.303 | 0.390 | 0.070 | 0.773 | 0.567 |
| MAR-MAY | poly deg 1 | 0.414 | 0.530 | -1.557 | 0.667 | 0.517 |
| JUN-SEP | piecewise (1981) | 0.153 | 0.191 | 0.135 | 0.483 | 0.369 |
| OCT-DEC | poly deg 2 | 0.169 | 0.212 | 0.127 | 0.776 | 0.407 |

The selected model beats the training-mean baseline in every series. R² is computed against the mean of a short 20-year window and is low or negative for some series (notably `MAR-MAY`, where no candidate clearly improves on a straight line), so MAE and RMSE are the more informative metrics. See Section 5.1 of the notebook for the interpretation.

## Limitations
- Only 121 annual values and one predictor (`YEAR`): these are trend-extrapolation models, not physical climate models, and are not suitable for long-range forecasting.
- The 20-year test set and 10-year CV folds are small, so metrics are noisy.
- Residuals show some autocorrelation, so p-values rely on HAC corrections.
- The 1981 breakpoint is a modelling choice (sensitivity is reported in the notebook).

## Dataset and references
> **TODO (author) — fill in before submission:** original data source / publisher, download URL, access date, and licence or attribution terms.

## Author
> **TODO (author):** name(s), roll number(s), course and college.

## Use of AI assistance
> **TODO (author):** state honestly which parts, if any, were written or reviewed with AI assistance, as required by your course guidelines.
